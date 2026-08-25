#!/usr/bin/env bash

# Added by remove-ai-watermarks-low-vram.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_SCRIPT="$SCRIPT_DIR/cleanImage.sh"

if [[ ! -x "$BASE_SCRIPT" ]]; then
    printf 'Error: base script not found or not executable: %s\n' "$BASE_SCRIPT" >&2
    exit 2
fi

for argument in "$@"; do
    if [[ "$argument" == "-h" || "$argument" == "--help" ]]; then
        printf '%s\n\n' \
            'Face-free variant: runs only the global regeneration stage.' \
            'It never starts YuNet detection, SAM masks, or Z-Image face refinement.'
        base_help=$("$BASE_SCRIPT" --help)
        printf '%s\n' "${base_help//cleanImage.sh/cleanImageNoFace.sh}"
        exit 0
    fi
done

export REMOVE_AI_WATERMARKS_SKIP_FACE_STAGE=1
exec "$BASE_SCRIPT" "$@"
