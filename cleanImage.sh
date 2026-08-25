#!/usr/bin/env bash

# Modified for remove-ai-watermarks-low-vram: portable repository layout,
# conservative defaults, full-resolution processing, and English diagnostics.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$SCRIPT_DIR"
readonly CLEANER="$REPO_DIR/.venv/bin/remove-ai-watermarks"
readonly MIN_STRENGTH="0.05"
readonly MAX_STRENGTH="1.00"

suffix="_cleaned"
strength="0.15"
declare -a inputs=()
current_temp=""

usage() {
    printf '%s\n' \
        'Usage:' \
        '  cleanImage.sh [--suffix SUFFIX] [--strength VALUE] FILE [FILE ...]' \
        '' \
        'Regenerates pixels with remove-ai-watermarks, strips all metadata from' \
        'the result, and always leaves the original files untouched.' \
        '' \
        'Options:' \
        '  -s, --suffix SUFFIX    Suffix inserted before the extension.' \
        '                         Default: _cleaned' \
        '  --strength VALUE       Regeneration intensity from 0.05 (minimum,' \
        '                         changes less) to 1.00 (maximum, changes more).' \
        '                         Conservative default: 0.15.' \
        '  -h, --help             Show this help.' \
        '  --                     End options (useful for names beginning with -).' \
        '' \
        'The image is processed at its original resolution (--max-resolution 0).' \
        '' \
        'Examples:' \
        '  ./cleanImage.sh photo.jpg' \
        '  ./cleanImage.sh *.jpg *.png' \
        '  ./cleanImage.sh --strength 0.25 photo.jpg' \
        '  ./cleanImage.sh --suffix _scrubbed "photo one.jpg" photo2.png' \
        '' \
        'Wildcards are expanded by Bash and normally must not be quoted. Existing' \
        'output files are never overwritten.'
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 2
}

strength_is_valid() {
    LC_ALL=C awk \
        -v value="$1" \
        -v minimum="$MIN_STRENGTH" \
        -v maximum="$MAX_STRENGTH" \
        'BEGIN {
            if (value !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/) exit 1
            exit !(value >= minimum && value <= maximum)
        }'
}

cleanup_temp() {
    if [[ -n "$current_temp" && -f "$current_temp" ]]; then
        rm -f -- "$current_temp"
    fi
}

trap cleanup_temp EXIT
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

[[ -n "$suffix" ]] || die 'suffix cannot be empty because that could overwrite the original'
[[ "$suffix" != */* ]] || die 'suffix cannot contain /'
strength_is_valid "$strength" || die "strength must be between $MIN_STRENGTH and $MAX_STRENGTH"
((${#inputs[@]} > 0)) || {
    usage >&2
    exit 2
}

[[ -x "$CLEANER" ]] || die "environment not found: $CLEANER (see README.md)"

# Reuse files already present in the Hugging Face cache. Disabling Xet avoids a
# second transfer method; from_pretrained does not redownload cached weights.
export HF_HUB_DISABLE_XET=1

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
        extension=""
    fi
    output_path="$input_dir/$stem$suffix$extension"

    if [[ -e "$output_path" || -L "$output_path" ]]; then
        printf 'Error: output already exists; refusing to overwrite it: %s\n' "$output_path" >&2
        ((failures += 1))
        continue
    fi

    read -r source_hash _ < <(sha256sum -- "$input_path")

    printf '\n=== %s ===\n' "$input_path"
    printf '%s\n' 'Initial test (an "unknown" result does not prove SynthID is absent):'
    if ! "$CLEANER" identify "$input_path" --json; then
        printf 'Warning: initial test failed; continuing with the scrub.\n' >&2
    fi

    temp_extension=${extension:-.png}
    current_temp=$(mktemp --tmpdir="$input_dir" ".cleanImage.XXXXXXXX$temp_extension")

    printf 'Regenerating pixels (strength=%s)...\n' "$strength"
    if ! "$CLEANER" invisible "$input_path" \
        --output "$current_temp" \
        --pipeline sdxl-zimage \
        --strength "$strength" \
        --seed 0 \
        --force \
        --cpu-offload \
        --max-resolution 0; then
        printf 'Error: pixel regeneration failed for %s\n' "$input_path" >&2
        cleanup_temp
        current_temp=""
        ((failures += 1))
        continue
    fi

    printf '%s\n' 'Removing all metadata from the new file...'
    if ! "$CLEANER" metadata "$current_temp" --remove --remove-all; then
        printf 'Error: metadata removal failed for %s\n' "$input_path" >&2
        cleanup_temp
        current_temp=""
        ((failures += 1))
        continue
    fi

    read -r source_hash_after _ < <(sha256sum -- "$input_path")
    if [[ "$source_hash" != "$source_hash_after" ]]; then
        printf 'Fatal error: the source file changed; the temporary result will not be published.\n' >&2
        cleanup_temp
        current_temp=""
        ((failures += 1))
        continue
    fi

    if [[ -e "$output_path" || -L "$output_path" ]]; then
        printf 'Error: an output appeared during processing; refusing to overwrite it: %s\n' "$output_path" >&2
        cleanup_temp
        current_temp=""
        ((failures += 1))
        continue
    fi

    chmod --reference="$input_path" "$current_temp"
    mv -- "$current_temp" "$output_path"
    current_temp=""

    printf '%s\n' 'Final test:'
    if ! "$CLEANER" identify "$output_path" --json; then
        printf 'Warning: final test failed. The output file was still created.\n' >&2
        ((failures += 1))
    fi

    printf 'Created: %s\n' "$output_path"
    ((processed += 1))
done

printf '\nCompleted: %d; errors: %d\n' "$processed" "$failures"
((failures == 0))
