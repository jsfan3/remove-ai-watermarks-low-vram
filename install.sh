#!/usr/bin/env bash

# Added by remove-ai-watermarks-low-vram.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly MAIN_VENV="$SCRIPT_DIR/.venv"
readonly TEXT_VENV="$SCRIPT_DIR/.venv-text"
readonly CUDA_WHEEL_INDEX="https://download.pytorch.org/whl/cu124"
readonly MIN_FREE_GIB=25

assume_yes=0
install_text=1

usage() {
    printf '%s\n' \
        'Usage:' \
        '  ./install.sh [OPTIONS]' \
        '' \
        'Creates two local Python environments for the tested low-VRAM workflow:' \
        '  .venv       CUDA SDXL, metadata removal, LaMa, and Qwen VAE support' \
        '  .venv-text  isolated CPU PaddleOCR support' \
        '' \
        'Options:' \
        '  -y, --yes      Do not ask for confirmation.' \
        '  --no-text      Skip the PaddleOCR environment. cleanImageNoFaceText.sh' \
        '                 will not work until the normal installation is completed.' \
        '  -h, --help     Show this help.' \
        '' \
        'The environments occupy about 8 GiB on the tested system. Models are not' \
        'bundled and download automatically on first use; the complete face-free' \
        'workflow can require roughly 15 GB of downloads from an empty cache and' \
        'about 18 GiB of environment plus model-cache storage.' \
        '' \
        'Tested runtime: Python 3.12, PyTorch 2.5.1 + CUDA 12.4.'
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        -y|--yes)
            assume_yes=1
            shift
            ;;
        --no-text)
            install_text=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

command -v python3 >/dev/null || die 'python3 was not found'
command -v nvidia-smi >/dev/null || die 'nvidia-smi was not found; an NVIDIA CUDA GPU and driver are required'

python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$python_version" in
    3.11|3.12) ;;
    *) die "Python 3.11 or 3.12 is required by the tested installer; found $python_version" ;;
esac

available_bytes=$(df --output=avail -B1 "$SCRIPT_DIR" | tail -n 1 | tr -d ' ')
minimum_bytes=$((MIN_FREE_GIB * 1024 * 1024 * 1024))
if [[ "$available_bytes" =~ ^[0-9]+$ ]] && ((available_bytes < minimum_bytes)); then
    printf 'Warning: less than %d GiB is free on the repository filesystem.\n' "$MIN_FREE_GIB" >&2
fi

printf '%s\n' \
    'This installs the tested CUDA/Python stack. No model weights are committed' \
    'to the repository; required weights download and cache on first use.' \
    'Allow roughly 15 GB of network transfer from an empty cache.'

if ((assume_yes == 0)); then
    [[ -t 0 ]] || die 'interactive confirmation is unavailable; rerun with --yes'
    read -r -p 'Continue? [y/N] ' answer
    [[ "$answer" == [yY] || "$answer" == [yY][eE][sS] ]] || {
        printf '%s\n' 'Installation cancelled.'
        exit 0
    }
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1

printf '\n%s\n' 'Creating the main CUDA environment...'
python3 -m venv "$MAIN_VENV"
"$MAIN_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$MAIN_VENV/bin/python" -m pip install \
    --index-url "$CUDA_WHEEL_INDEX" \
    'torch==2.5.1' \
    'torchvision==0.20.1'
"$MAIN_VENV/bin/python" -m pip install -e "$SCRIPT_DIR[text-restoration]"

if ((install_text)); then
    printf '\n%s\n' 'Creating the isolated PaddleOCR environment...'
    python3 -m venv "$TEXT_VENV"
    "$TEXT_VENV/bin/python" -m pip install --upgrade pip setuptools wheel
    "$TEXT_VENV/bin/python" -m pip install -e "$SCRIPT_DIR[text-draft]"
fi

chmod +x \
    "$SCRIPT_DIR/cleanImage.sh" \
    "$SCRIPT_DIR/cleanImageNoFace.sh" \
    "$SCRIPT_DIR/cleanImageNoFaceText.sh" \
    "$SCRIPT_DIR/cleanImageTextDraft.py" \
    "$SCRIPT_DIR/cleanImageTextRestore.py"

if ! "$MAIN_VENV/bin/python" -c \
    'import torch; print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}"); raise SystemExit(not torch.cuda.is_available())'; then
    die 'PyTorch was installed, but CUDA is not available in the main environment'
fi
if ((install_text)); then
    "$TEXT_VENV/bin/python" -c \
        'import paddle; print(f"PaddlePaddle {paddle.__version__}")'
fi

printf '\n%s\n' \
    'Installation completed.' \
    'Recommended 4 GB VRAM command:' \
    '  ./cleanImageNoFace.sh image.png' \
    'Text-preserving command:' \
    '  ./cleanImageNoFaceText.sh image.png' \
    'The first run downloads the required models automatically.'
