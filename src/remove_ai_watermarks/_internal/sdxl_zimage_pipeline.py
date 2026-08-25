"""The qwen-zimage recipe with an SDXL global stage.

Only the global regeneration model changes. The face stage is inherited verbatim
from :class:`QwenZImagePipeline` -- same YuNet detection, same SAM masks, same
Z-Image Turbo repair of the original crops, same feathered compositing -- so a
change there cannot silently diverge between the two profiles.

Three pieces cannot be shared, because they are bound to the architecture: the
ControlNet, the four-step distillation LoRA, and the sampler. Strength is bound to
it too, which is the part that is easy to miss: an SDXL global pass leaves SynthID
at the strength Qwen needs. See ``watermark_profiles.SDXL_ZIMAGE_OPENAI_STRENGTH``.
"""

# Modified by remove-ai-watermarks-low-vram: cached selective snapshots,
# sequential CPU offload, attention/VAE slicing, and optional VAE tiling.

# Diffusers and torch expose mostly untyped tensor APIs. Keep the relaxation local
# to this optional ML boundary.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportAttributeAccessIssue=false, reportPrivateUsage=false, reportPrivateImportUsage=false
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from remove_ai_watermarks._internal.qwen_zimage_pipeline import (
    GLOBAL_STEPS,
    QwenZImagePipeline,
    build_canny_control_image,
)
from remove_ai_watermarks._internal.watermark_profiles import (
    CONTROLNET_CANNY_MODEL,
    SDXL_LIGHTNING_MODEL_ID,
    SDXL_LIGHTNING_PATTERN,
    SDXL_MODEL_ID,
)

log = logging.getLogger(__name__)

SDXL_VAE_MODEL_ID = "madebyollin/sdxl-vae-fp16-fix"
DISABLE_VAE_TILING_ENV = "REMOVE_AI_WATERMARKS_DISABLE_VAE_TILING"
# SDXL aligns to an 8-pixel latent grid, against Qwen's 16.
_LATENT_GRID = 8


def _cached_repo_or_id(model_id: str, anchor: str, token: dict[str, str]) -> str:
    """Prefer an already-cached selective snapshot without completing the repo.

    Newer huggingface_hub versions reject an intentionally selective snapshot in
    offline mode because README/demo files are absent, even when every file needed
    by Diffusers is present.  Resolving one cached anchor and passing its directory
    to ``from_pretrained`` avoids both that false-negative and an unnecessary full
    repository download.  A machine without the anchor falls back to the normal Hub
    model id and retains upstream's download behaviour.
    """
    from huggingface_hub import hf_hub_download

    try:
        cached_anchor = hf_hub_download(model_id, anchor, local_files_only=True, **token)
    except OSError:
        return model_id
    return str(Path(cached_anchor).parent)


def sdxl_target_size(width: int, height: int) -> tuple[int, int]:
    """Floor dimensions to SDXL's latent grid without changing aspect."""
    return max(_LATENT_GRID, (width // _LATENT_GRID) * _LATENT_GRID), max(
        _LATENT_GRID, (height // _LATENT_GRID) * _LATENT_GRID
    )


def requested_steps(effective_steps: int, strength: float) -> int:
    """Translate "spend N denoising steps" into what Diffusers has to be asked for.

    The two runtimes truncate differently and it is easy to port this wrong.
    DiffSynth sets ``sigma_start = denoising_strength`` and then runs *every*
    requested step across the shortened sigma range. Diffusers img2img instead
    truncates the step *count* (``init_timestep = int(steps * strength)``), so
    asking it for four steps at strength 0.15 runs **zero** and returns nothing but
    a VAE round-trip. Ask for enough that ``effective_steps`` actually execute.
    """
    return max(1, math.ceil(effective_steps / max(float(strength), 1e-6)))


@dataclass
class SdxlZImagePipeline(QwenZImagePipeline):
    """Lazy runtime for the SDXL global stage plus the inherited face stage."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self._sdxl_pipe: Any = None

    def _load_sdxl(self) -> Any:
        if self._sdxl_pipe is not None:
            return self._sdxl_pipe
        self._require_cuda()
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            EulerDiscreteScheduler,
            StableDiffusionXLControlNetImg2ImgPipeline,
        )
        from huggingface_hub import hf_hub_download

        self._progress("Loading SDXL, Lightning LoRA, and Canny ControlNet...")
        token = {"token": self.hf_token} if self.hf_token else {}
        controlnet_source = _cached_repo_or_id(CONTROLNET_CANNY_MODEL, "config.json", token)
        vae_source = _cached_repo_or_id(SDXL_VAE_MODEL_ID, "config.json", token)
        base_source = _cached_repo_or_id(SDXL_MODEL_ID, "model_index.json", token)
        controlnet = ControlNetModel.from_pretrained(controlnet_source, torch_dtype=torch.float16, **token)
        vae = AutoencoderKL.from_pretrained(vae_source, torch_dtype=torch.float16, **token)
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            base_source,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            add_watermarker=False,
            **token,
        )
        # SDXL's own four-step distillation, at the strength its authors document.
        # The reference graph loads the Qwen LoRA at 0.8; carrying that number to a
        # different LoRA on a different architecture would be imitation, not parity.
        lora_path = Path(hf_hub_download(SDXL_LIGHTNING_MODEL_ID, SDXL_LIGHTNING_PATTERN, **token))
        pipe.load_lora_weights(lora_path.parent, weight_name=lora_path.name)
        pipe.fuse_lora()
        # SDXL-Lightning is distilled against trailing timestep spacing.
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        if self.keep_global_models_on_device is False:
            # The public ``--cpu-offload`` option already maps to this field for
            # both profiles.  Keep the SDXL stack on CPU until each leaf module is
            # needed so cards smaller than the combined UNet + ControlNet weights
            # do not OOM while the pipeline is being constructed.
            pipe.enable_attention_slicing("max")
            pipe.enable_vae_slicing()
            disable_vae_tiling = os.environ.get(DISABLE_VAE_TILING_ENV, "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if disable_vae_tiling:
                self._progress("VAE tiling disabled; using full-frame VAE encode/decode.")
            else:
                pipe.enable_vae_tiling()
            pipe.enable_sequential_cpu_offload(device=self.device)
        else:
            pipe.to(self.device)
        self._sdxl_pipe = pipe
        return pipe

    def preload(self, *, global_only: bool = False) -> None:
        """Eagerly load the mandatory stage and, by default, the face stack."""
        from remove_ai_watermarks._internal.qwen_zimage_pipeline import _yunet_model_path

        self._load_sdxl()
        _yunet_model_path()
        if not global_only:
            self._load_zimage()
            self._load_sam()

    def _run_global(self, image: Image.Image, strength: float, seed: int | None) -> Image.Image:
        import torch

        pipe = self._load_sdxl()
        target = sdxl_target_size(image.width, image.height)
        prepared = image if image.size == target else image.resize(target, Image.Resampling.LANCZOS)
        control = build_canny_control_image(prepared)
        steps = requested_steps(GLOBAL_STEPS, strength)
        self._progress(f"Running SDXL Canny pass: strength={strength:.4f}, steps={GLOBAL_STEPS} of {steps}...")
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        result = pipe(
            prompt=self._global_prompt(),
            negative_prompt=self._global_negative(),
            image=prepared,
            control_image=control,
            controlnet_conditioning_scale=float(self.controlnet_conditioning_scale),
            strength=float(strength),
            num_inference_steps=steps,
            guidance_scale=1.0,
            generator=generator,
        ).images[0]
        if result.size != image.size:
            result = result.resize(image.size, Image.Resampling.LANCZOS)
        return result.convert("RGB")

    @staticmethod
    def _global_prompt() -> str:
        from remove_ai_watermarks._internal.qwen_zimage_pipeline import _GLOBAL_PROMPT

        return _GLOBAL_PROMPT

    @staticmethod
    def _global_negative() -> str:
        from remove_ai_watermarks._internal.qwen_zimage_pipeline import _GLOBAL_NEGATIVE

        return _GLOBAL_NEGATIVE
