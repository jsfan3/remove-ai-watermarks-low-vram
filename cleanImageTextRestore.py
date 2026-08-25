#!/usr/bin/env python3
"""Low-VRAM text restoration for cleanImageNoFaceText.sh.

This is a local hardware adaptation of remove-ai-watermarks' experimental
``vae-glyphs`` stage.  It deliberately does *not* load the 57 GiB Qwen
transformer stack: the already-supported SDXL scrub produces the clean base,
then this helper loads only Qwen's VAE as a fresh-pixel glyph donor and LaMa as
the CPU inpainting backend.
"""

# Added by remove-ai-watermarks-low-vram.

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

QWEN_IMAGE_MODEL_ID = "Qwen/Qwen-Image-2512"


def progress(message: str) -> None:
    print(f"[text] {message}", flush=True)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def line_payload(line: Any) -> dict[str, Any]:
    return {
        "box": [int(value) for value in line.box],
        "text": str(line.text),
        "script": str(line.script),
        "angle": float(getattr(line, "angle", 0.0)),
    }


def text_crop_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Return a context-rich crop whose centre contains one detected text box."""
    x1, y1, x2, y2 = box
    line_height = max(1, y2 - y1)
    pad_x = max(48, round(line_height * 1.5))
    pad_y = max(32, round(line_height * 0.75))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )


def qwen_text_donor(source: Image.Image, lines: list[Any], *, device_name: str) -> Image.Image:
    """Reconstruct only text crops with Qwen's compact VAE.

    A full-frame Qwen VAE pass exceeds the GTX 1050's 4 GiB.  Restoration
    consumes donor pixels only under the selected glyph masks, so reconstructing
    context-padded text crops is equivalent for the composite while avoiding the
    large activation tensors.
    """
    import torch
    from diffusers import AutoencoderKLQwenImage
    from diffusers.image_processor import VaeImageProcessor

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for the Qwen VAE")

    active_device = device_name
    active_dtype = torch.float16 if active_device == "cuda" else torch.float32
    progress(f"Loading only the Qwen VAE on {device_name} ({QWEN_IMAGE_MODEL_ID}/vae)...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        QWEN_IMAGE_MODEL_ID,
        subfolder="vae",
        torch_dtype=active_dtype,
        low_cpu_mem_usage=True,
    ).to(active_device)
    vae.eval()
    vae.enable_slicing()
    processor = VaeImageProcessor(vae_scale_factor=8)

    def execute(crop: Image.Image, *, tiled: bool) -> Image.Image:
        if tiled:
            vae.enable_tiling()
        else:
            vae.disable_tiling()
        crop_width, crop_height = crop.size
        pad_width = (-crop_width) % 8
        pad_height = (-crop_height) % 8
        crop_array = np.asarray(crop)
        if pad_width or pad_height:
            crop_array = np.pad(
                crop_array,
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode="edge",
            )
        padded = Image.fromarray(crop_array)
        tensor = processor.preprocess(padded).unsqueeze(2).to(device=active_device, dtype=active_dtype)
        with torch.inference_mode():
            latents = vae.encode(tensor).latent_dist.mode()
            decoded = vae.decode(latents, return_dict=False)[0][:, :, 0]
        decoded_cpu = decoded.float().cpu()
        result = processor.postprocess(decoded_cpu, output_type="pil")[0]
        del tensor, latents, decoded
        return result.crop((0, 0, crop_width, crop_height)).convert("RGB")

    def is_cuda_oom(error: BaseException) -> bool:
        return isinstance(error, torch.OutOfMemoryError) or "CUDA out of memory" in str(error)

    donor = Image.new("RGB", source.size)
    for index, line in enumerate(lines, start=1):
        crop_box = text_crop_box(line.box, source.width, source.height)
        crop = source.crop(crop_box)
        progress(f"VAE reconstruction of text crop {index}/{len(lines)}: {crop.width}x{crop.height}")
        try:
            reconstructed = execute(crop, tiled=False)
        except RuntimeError as error:
            if active_device != "cuda" or not is_cuda_oom(error):
                raise
            del error
            progress("Insufficient VRAM for this crop; retrying with VAE tiling.")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                reconstructed = execute(crop, tiled=True)
            except RuntimeError as tiled_error:
                if not is_cuda_oom(tiled_error):
                    raise
                del tiled_error
                progress("Still insufficient VRAM; moving the VAE to CPU for this and subsequent crops.")
                vae.to(device="cpu", dtype=torch.float32)
                active_device = "cpu"
                active_dtype = torch.float32
                torch.cuda.empty_cache()
                gc.collect()
                reconstructed = execute(crop, tiled=False)
        donor.paste(reconstructed, crop_box[:2])
        del crop, reconstructed
        gc.collect()
        if active_device == "cuda":
            torch.cuda.empty_cache()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return donor


def save_image(image: Image.Image, output: Path) -> None:
    suffix = output.suffix.casefold()
    kwargs: dict[str, Any] = {}
    if suffix in {".jpg", ".jpeg"}:
        kwargs = {"format": "JPEG", "quality": 100, "subsampling": 0}
    elif suffix == ".webp":
        kwargs = {"format": "WEBP", "lossless": True, "quality": 100}
    elif suffix == ".png":
        kwargs = {"format": "PNG"}
    image.save(output, **kwargs)


def write_report(
    path: Path,
    source: Image.Image,
    lines: list[Any],
    details: dict[str, Any],
) -> None:
    from remove_ai_watermarks._internal.text_restoration import source_pixel_sha256

    payload = {
        "schema_version": 1,
        "verified": bool(lines),
        "verification": details,
        "source_pixel_sha256": source_pixel_sha256(source),
        "width": source.width,
        "height": source.height,
        "lines": [line_payload(line) for line in lines],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def restore(args: argparse.Namespace) -> int:
    from remove_ai_watermarks._internal.text_restoration import (
        load_verified_text_manifest,
        restore_verified_text,
    )

    source = open_rgb(args.source)
    candidate = open_rgb(args.candidate)
    if source.size != candidate.size:
        raise RuntimeError(f"Source and candidate have different sizes: {source.size} versus {candidate.size}")

    raw_manifest = json.loads(args.text_manifest.read_text(encoding="utf-8"))
    if raw_manifest.get("verified") is False and not raw_manifest.get("lines"):
        lines = []
        details = dict(raw_manifest.get("verification") or {})
    else:
        manifest = load_verified_text_manifest(args.text_manifest, source)
        lines = list(manifest.lines)
        details = dict(raw_manifest.get("verification") or {})
        details.setdefault("mode", "manual-manifest")
        if args.text_manifest != args.report:
            details["manifest"] = str(args.text_manifest)

    write_report(args.report, source, lines, details)
    if not lines:
        progress("No altered line selected; keeping the SDXL candidate without further changes.")
        save_image(candidate, args.output)
        return 0

    progress(f"Lines selected for restoration: {len(lines)}")
    donor = qwen_text_donor(source, lines, device_name=args.vae_device)
    progress(f"LaMa + VAE glyphs: {len(lines)} lines grouped by proximity and angle.")
    result = restore_verified_text(source, candidate, donor, tuple(lines))
    save_image(result, args.output)
    progress(f"Text restoration completed: {args.output}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Restore glyphs locally after SDXL scrubbing using stable OCR, the "
            "Qwen VAE, and LaMa without loading the full Qwen transformer."
        )
    )
    result.add_argument("--source", required=True, type=Path)
    result.add_argument("--candidate", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--report", required=True, type=Path)
    result.add_argument("--text-manifest", required=True, type=Path)
    result.add_argument("--vae-device", choices=("auto", "cuda", "cpu"), default="auto")
    return result


def main() -> int:
    args = parser().parse_args()
    for path in (args.source, args.candidate):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")
    if not args.text_manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.text_manifest}")
    try:
        return restore(args)
    except Exception as exc:
        print(f"Text restoration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
