#!/usr/bin/env bash

# Added by remove-ai-watermarks-low-vram: face-free SDXL scrubbing with
# OCR-guided, low-VRAM restoration of small text.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$SCRIPT_DIR"
readonly CLEANER="$REPO_DIR/.venv/bin/remove-ai-watermarks"
readonly PYTHON="$REPO_DIR/.venv/bin/python"
readonly TEXT_PYTHON="$SCRIPT_DIR/.venv-text/bin/python"
readonly TEXT_DRAFT_HELPER="$SCRIPT_DIR/cleanImageTextDraft.py"
readonly TEXT_HELPER="$SCRIPT_DIR/cleanImageTextRestore.py"
readonly MIN_STRENGTH="0.05"
readonly MAX_STRENGTH="1.00"

suffix="_cleaned_text"
strength="0.15"
seed="0"
min_text_score="0.85"
text_manifest=""
restore_all_text=0
declare -a inputs=()
declare -a temp_paths=()

usage() {
    printf '%s\n' \
        'Usage:' \
        '  cleanImageNoFaceText.sh [OPTIONS] FILE [FILE ...]' \
        '' \
        'Scrubs SynthID with an SDXL pipeline compatible with 4 GB VRAM, skips' \
        'the face stage completely, and selectively repairs small text. Text' \
        'taller than 10% of the image is ignored. It uses three local components:' \
        '  1. PaddleOCR finds oriented boxes and compares source and candidate;' \
        '  2. LaMa erases only the deformed glyphs;' \
        '  3. the Qwen VAE alone reconstructs glyph cores in small crops.' \
        '' \
        'It neither loads nor downloads the full Qwen transformer (~57 GiB), and' \
        'does not load YuNet, SAM, or Z-Image for faces. No external AI service is used.' \
        '' \
        'Options:' \
        '  -s, --suffix SUFFIX    Suffix inserted before the extension.' \
        '                         Default: _cleaned_text' \
        '  --strength VALUE       Denoising from 0.05 (changes less) to 1.00' \
        '                         (changes more). Default: 0.15.' \
        '  --seed NUMBER          SDXL regeneration seed. Default: 0.' \
        '  --min-text-score N     Minimum OCR confidence for detecting and comparing' \
        '                         text, from 0.50 to 1.00. Default: 0.85.' \
        '  --restore-all-text     Repair every stable, small OCR line even when the' \
        '                         candidate text already matches the source.' \
        '  --text-manifest FILE   Use a manually verified manifest. Allowed only' \
        '                         with exactly one input FILE.' \
        '  -h, --help             Show this help.' \
        '  --                     End options.' \
        '' \
        'Originals are never overwritten. Each output also gets a' \
        '<name><suffix>.text-report.json file containing lines and coordinates.' \
        'This mode is experimental: inspect the output visually and use the' \
        'provider detector to verify that the watermark is actually absent.' \
        'The image is processed at its original resolution (--max-resolution 0).' \
        '' \
        'Examples:' \
        '  ./cleanImageNoFaceText.sh "image.png"' \
        '  ./cleanImageNoFaceText.sh --strength 0.10 *.png' \
        '  ./cleanImageNoFaceText.sh --seed 2 --suffix _text "image.jpg"' \
        '  ./cleanImageNoFaceText.sh --text-manifest lines.json "image.png"'
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

decimal_in_range() {
    LC_ALL=C awk -v value="$1" -v minimum="$2" -v maximum="$3" 'BEGIN {
        if (value !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/) exit 1
        exit !(value >= minimum && value <= maximum)
    }'
}

cleanup_temps() {
    local path
    for path in "${temp_paths[@]}"; do
        if [[ -n "$path" && -e "$path" ]]; then
            rm -f -- "$path"
        fi
    done
}

trap cleanup_temps EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -s|--suffix)
            (($# >= 2)) || die "$1 requires a value"
            suffix=$2
            shift 2
            ;;
        --suffix=*)
            suffix=${1#*=}
            shift
            ;;
        --strength)
            (($# >= 2)) || die "$1 requires a value"
            strength=$2
            shift 2
            ;;
        --strength=*)
            strength=${1#*=}
            shift
            ;;
        --seed)
            (($# >= 2)) || die "$1 requires a value"
            seed=$2
            shift 2
            ;;
        --seed=*)
            seed=${1#*=}
            shift
            ;;
        --min-text-score)
            (($# >= 2)) || die "$1 requires a value"
            min_text_score=$2
            shift 2
            ;;
        --min-text-score=*)
            min_text_score=${1#*=}
            shift
            ;;
        --restore-all-text)
            restore_all_text=1
            shift
            ;;
        --text-manifest)
            (($# >= 2)) || die "$1 requires a file"
            text_manifest=$2
            shift 2
            ;;
        --text-manifest=*)
            text_manifest=${1#*=}
            shift
            ;;
        --)
            shift
            inputs+=("$@")
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            inputs+=("$1")
            shift
            ;;
    esac
done

[[ -n "$suffix" ]] || die 'suffix cannot be empty'
[[ "$suffix" != */* ]] || die 'suffix cannot contain /'
decimal_in_range "$strength" "$MIN_STRENGTH" "$MAX_STRENGTH" || \
    die "strength must be between $MIN_STRENGTH and $MAX_STRENGTH"
decimal_in_range "$min_text_score" "0.50" "1.00" || \
    die 'min-text-score must be between 0.50 and 1.00'
[[ "$seed" =~ ^-?[0-9]+$ ]] || die 'seed must be an integer'
((${#inputs[@]} > 0)) || {
    usage >&2
    exit 2
}
if [[ -n "$text_manifest" ]]; then
    [[ -f "$text_manifest" ]] || die "manifest not found: $text_manifest"
    ((${#inputs[@]} == 1)) || die '--text-manifest requires exactly one input file'
fi

[[ -x "$CLEANER" ]] || die "command not found: $CLEANER"
[[ -x "$PYTHON" ]] || die "environment Python not found: $PYTHON"
if [[ -z "$text_manifest" ]]; then
    [[ -x "$TEXT_PYTHON" ]] || die "OCR environment not found: $TEXT_PYTHON"
    [[ -f "$TEXT_DRAFT_HELPER" ]] || die "OCR helper not found: $TEXT_DRAFT_HELPER"
fi
[[ -f "$TEXT_HELPER" ]] || die "text helper not found: $TEXT_HELPER"

export HF_HUB_DISABLE_XET=1
export REMOVE_AI_WATERMARKS_SKIP_FACE_STAGE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

failures=0
processed=0

for input_path in "${inputs[@]}"; do
    if [[ ! -f "$input_path" ]]; then
        printf 'Error: file not found: %s\n' "$input_path" >&2
        ((failures += 1))
        continue
    fi

    input_dir=$(dirname -- "$input_path")
    filename=$(basename -- "$input_path")
    stem=${filename%.*}
    if [[ "$filename" == *.* && -n "$stem" ]]; then
        extension=.${filename##*.}
    else
        stem=$filename
        extension=".png"
    fi
    output_path="$input_dir/$stem$suffix$extension"
    report_path="$input_dir/$stem$suffix.text-report.json"

    if [[ -e "$output_path" || -L "$output_path" ]]; then
        printf 'Error: output already exists; refusing to overwrite it: %s\n' "$output_path" >&2
        ((failures += 1))
        continue
    fi
    if [[ -e "$report_path" || -L "$report_path" ]]; then
        printf 'Error: report already exists; refusing to overwrite it: %s\n' "$report_path" >&2
        ((failures += 1))
        continue
    fi

    read -r source_hash _ < <(sha256sum -- "$input_path")
    candidate=$(mktemp --tmpdir="$input_dir" '.cleanImageText.candidate.XXXXXXXX.png')
    restored=$(mktemp --tmpdir="$input_dir" ".cleanImageText.restored.XXXXXXXX$extension")
    report_temp=$(mktemp --tmpdir="$input_dir" '.cleanImageText.report.XXXXXXXX.json')
    temp_paths+=("$candidate" "$restored" "$report_temp")

    printf '\n=== %s ===\n' "$input_path"
    printf '%s\n' 'Initial test ("unknown" does not prove SynthID is absent):'
    if ! "$CLEANER" identify "$input_path" --json; then
        printf 'Warning: initial test failed; continuing.\n' >&2
    fi

    printf 'Face-free SDXL scrub (strength=%s, seed=%s)...\n' "$strength" "$seed"
    if ! "$CLEANER" invisible "$input_path" \
        --output "$candidate" \
        --pipeline sdxl-zimage \
        --strength "$strength" \
        --seed "$seed" \
        --force \
        --cpu-offload \
        --max-resolution 0 \
        --no-adaptive-polish; then
        printf 'Error: SDXL scrub failed for %s\n' "$input_path" >&2
        ((failures += 1))
        continue
    fi

    manifest_for_restore="$text_manifest"
    if [[ -z "$manifest_for_restore" ]]; then
        draft_args=(
            --source "$input_path"
            --candidate "$candidate"
            --output "$report_temp"
            --min-text-score "$min_text_score"
        )
        if ((restore_all_text)); then
            draft_args+=(--restore-all-text)
        fi
        if ! "$TEXT_PYTHON" "$TEXT_DRAFT_HELPER" "${draft_args[@]}"; then
            printf 'Error: OCR analysis failed for %s\n' "$input_path" >&2
            ((failures += 1))
            continue
        fi
        manifest_for_restore="$report_temp"
    fi

    helper_args=(
        --source "$input_path"
        --candidate "$candidate"
        --output "$restored"
        --report "$report_temp"
        --text-manifest "$manifest_for_restore"
        --vae-device auto
    )

    if ! "$PYTHON" "$TEXT_HELPER" "${helper_args[@]}"; then
        printf 'Error: text restoration failed for %s\n' "$input_path" >&2
        ((failures += 1))
        continue
    fi

    printf '%s\n' 'Removing all metadata from the result...'
    if ! "$CLEANER" metadata "$restored" --remove --remove-all; then
        printf 'Error: metadata removal failed for %s\n' "$input_path" >&2
        ((failures += 1))
        continue
    fi

    read -r source_hash_after _ < <(sha256sum -- "$input_path")
    if [[ "$source_hash" != "$source_hash_after" ]]; then
        printf 'Fatal error: the source file changed; the result will not be published.\n' >&2
        ((failures += 1))
        continue
    fi
    if [[ -e "$output_path" || -L "$output_path" || -e "$report_path" || -L "$report_path" ]]; then
        printf 'Error: an output appeared during processing; refusing to overwrite anything.\n' >&2
        ((failures += 1))
        continue
    fi

    chmod --reference="$input_path" "$restored"
    chmod --reference="$input_path" "$report_temp"
    mv -- "$restored" "$output_path"
    mv -- "$report_temp" "$report_path"
    printf '%s\n' 'Final test:'
    if ! "$CLEANER" identify "$output_path" --json; then
        printf 'Warning: final test failed.\n' >&2
        ((failures += 1))
    fi
    printf 'Created: %s\n' "$output_path"
    printf 'Text report: %s\n' "$report_path"
    ((processed += 1))
done

printf '\nCompleted: %d; errors: %d\n' "$processed" "$failures"
((failures == 0))
