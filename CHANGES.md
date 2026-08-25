# Changes from upstream

This repository is an unofficial derivative of
[`wiltodelta/remove-ai-watermarks`](https://github.com/wiltodelta/remove-ai-watermarks)
under Apache License 2.0. It is based on upstream version `0.28.1`, commit
`f52e40ae245db73600c2cbbf9214e6a597d36598`.

The derivative was prepared on 2026-08-25 for a Linux computer with an NVIDIA
GeForce GTX 1050 (4 GB VRAM), 16 GiB of system RAM, and Python 3.12.

## Modified upstream files

- `src/remove_ai_watermarks/_internal/sdxl_zimage_pipeline.py`
  - prefers already cached selective snapshots when every required component is present;
  - enables sequential CPU offload, maximum attention slicing, VAE slicing, and VAE tiling;
  - permits VAE tiling to be disabled independently for visual-quality experiments;
  - loads the cached Lightning LoRA without completing an unneeded repository snapshot.
- `src/remove_ai_watermarks/_internal/qwen_zimage_pipeline.py`
  - honors `REMOVE_AI_WATERMARKS_SKIP_FACE_STAGE` before running any face detector;
  - skips YuNet, SAM, and Z-Image face refinement when that flag is active.
- `src/remove_ai_watermarks/metadata.py`
  - saves JPEG output at quality 100 with 4:4:4 chroma subsampling to minimize added loss.
- `tests/test_qwen_zimage_pipeline.py`
  - covers face-stage bypass and low-VRAM VAE controls.
- `tests/test_metadata.py`
  - covers the JPEG quality-preservation setting.
- `pyproject.toml`, `.gitignore`, and `README.md`
  - identify and document the derivative and exclude its generated artifacts.

Each modified source or test file carries a notice that it was changed by this
derivative. Git history retains the complete upstream history.

## Added files

- `cleanImage.sh`: safe full-pipeline Bash entry point.
- `cleanImageNoFace.sh`: face-free wrapper recommended for 4 GB VRAM.
- `cleanImageNoFaceText.sh`: face-free workflow with selective small-text restoration.
- `cleanImageTextDraft.py`: isolated PaddleOCR comparison and manifest generation.
- `cleanImageTextRestore.py`: crop-based Qwen-VAE and LaMa text restoration.
- `install.sh`: repeatable dual-environment Linux installer.
- `docs/images/low-vram-example/`: byte-identifiable ChatGPT input/output fixture.

No model weights, Python virtual environments, or user images outside the
documentation fixture are included.
