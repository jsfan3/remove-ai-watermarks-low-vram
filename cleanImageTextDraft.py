#!/usr/bin/env python3
"""Create a stability-gated text manifest in an isolated PaddleOCR environment."""

# Added by remove-ai-watermarks-low-vram.

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

# Prefer the checked-out, locally patched source even in the isolated OCR venv.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

MAX_AUTOMATIC_LINES = 128
MAX_AUTOMATIC_LINE_AREA_FRACTION = 0.12
MAX_AUTOMATIC_TOTAL_AREA_FRACTION = 0.35
MAX_TEXT_HEIGHT_FRACTION = 0.10


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


def check_geometry(lines: list[Any], width: int, height: int) -> None:
    if len(lines) > MAX_AUTOMATIC_LINES:
        raise RuntimeError(
            f"The detector proposed {len(lines)} lines, exceeding the {MAX_AUTOMATIC_LINES} limit."
        )
    image_area = width * height
    total_area = 0
    for index, line in enumerate(lines, start=1):
        x1, y1, x2, y2 = line.box
        area = (x2 - x1) * (y2 - y1)
        if area / image_area > MAX_AUTOMATIC_LINE_AREA_FRACTION:
            raise RuntimeError(f"Line {index} covers too much area ({area / image_area:.1%}).")
        total_area += area
    if total_area / image_area > MAX_AUTOMATIC_TOTAL_AREA_FRACTION:
        raise RuntimeError(
            f"Selected areas cover {total_area / image_area:.1%}, exceeding the "
            f"{MAX_AUTOMATIC_TOTAL_AREA_FRACTION:.0%} limit."
        )


def boxes_touch(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    *,
    margin: int = 6,
) -> bool:
    """Whether two OCR boxes overlap after accounting for glyph-mask dilation."""
    return not (
        left[2] + margin < right[0]
        or right[2] + margin < left[0]
        or left[3] + margin < right[1]
        or right[3] + margin < left[1]
    )


def script_and_language(text: str) -> tuple[str, str]:
    names = [unicodedata.name(character, "") for character in text]
    if any("CJK" in name for name in names):
        return "cjk", "ch"
    if any("CYRILLIC" in name for name in names):
        return "alphabetic", "ru"
    return "alphabetic", "en"


def detector_proposals(detector: Any, source_rgb: np.ndarray) -> list[dict[str, Any]]:
    """Read Paddle's individual boxes without joining text across subjects."""
    height, width = source_rgb.shape[:2]
    proposals: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for page in detector.predict(source_rgb):
        detected = page.get("rec_boxes", None)
        if detected is None or len(detected) == 0:
            detected = page.get("rec_polys", [])
        scores = page.get("rec_scores", [])
        texts = page.get("rec_texts", [])
        polygons = page.get("rec_polys", [])
        for index, raw_box in enumerate(detected):
            if index >= len(scores) or index >= len(texts):
                continue
            score = scores[index]
            text = texts[index]
            points = np.asarray(raw_box, dtype=np.float32).reshape(-1)
            if points.size == 4:
                x1, y1, x2, y2 = points
            else:
                points = points.reshape(-1, 2)
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
            box = (
                max(0, round(float(x1))),
                max(0, round(float(y1))),
                min(width, round(float(x2))),
                min(height, round(float(y2))),
            )
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            angle = 0.0
            if index < len(polygons):
                polygon = np.asarray(polygons[index], dtype=np.float32).reshape(-1, 2)
                if len(polygon) >= 2:
                    dx, dy = polygon[1] - polygon[0]
                    if dx or dy:
                        # text_restoration passes ``-angle`` to OpenCV's
                        # RotatedRect; invert the image-coordinate edge angle
                        # here so the support mask follows Paddle's polygon.
                        angle = -math.degrees(math.atan2(float(dy), float(dx)))
                        while angle <= -90.0:
                            angle += 180.0
                        while angle > 90.0:
                            angle -= 180.0
                        if abs(angle) > 30.0:
                            angle = 0.0
            item = {
                "box": box,
                "text": str(text).strip(),
                "score": float(score),
                "angle": round(angle, 6),
            }
            previous = proposals.get(box)
            if previous is None or item["score"] > previous["score"]:
                proposals[box] = item
    return sorted(proposals.values(), key=lambda item: ((item["box"][1] + item["box"][3]) / 2, item["box"][0]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local text manifest with PaddleOCR.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-text-score", type=float, default=0.85)
    parser.add_argument("--restore-all-text", action="store_true")
    args = parser.parse_args()
    if not 0.5 <= args.min_text_score <= 1.0:
        parser.error("--min-text-score must be between 0.5 and 1.0")

    from remove_ai_watermarks._internal.text_restoration import (
        VerifiedTextLine,
        source_pixel_sha256,
    )
    import remove_ai_watermarks.text_draft as text_draft

    source = open_rgb(args.source)
    candidate = open_rgb(args.candidate)
    if source.size != candidate.size:
        raise RuntimeError(f"Size mismatch: source {source.size}, candidate {candidate.size}")

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    print("[text] Loading local OCR models (PaddleOCR)...", flush=True)
    detector, engines = text_draft._build_engines()
    source_rgb = np.asarray(source)
    proposals = detector_proposals(detector, source_rgb)
    candidate_rgb = np.asarray(candidate)
    selected: list[Any] = []
    unchanged_candidates: list[tuple[Any, dict[str, Any]]] = []
    skipped_large: list[dict[str, Any]] = []
    rejected_low_score: list[dict[str, Any]] = []
    eligible = 0
    for item in proposals:
        if not item["text"] or item["score"] < args.min_text_score:
            rejected_low_score.append(
                {
                    "box": list(item["box"]),
                    "text": item["text"],
                    "source_score": round(item["score"], 6),
                }
            )
            continue
        script, language = script_and_language(item["text"])
        line = VerifiedTextLine(item["box"], item["text"], script, item["angle"])
        line_height_fraction = (line.box[3] - line.box[1]) / source.height
        if line_height_fraction > MAX_TEXT_HEIGHT_FRACTION:
            skipped_large.append(
                {
                    **line_payload(line),
                    "source_score": round(item["score"], 6),
                    "height_fraction": round(line_height_fraction, 6),
                    "reason": "large text: the SDXL stage already preserves it reliably",
                }
            )
            continue
        eligible += 1
        reads = [
            text_draft._recognize(engines[language], candidate_rgb, line.box, script, ratio)
            for ratio in text_draft.JITTER_RATIOS
        ]
        candidate_text = text_draft.stable_recognition(reads, args.min_text_score)
        same = (
            candidate_text is not None
            and text_draft.normalize_text(candidate_text) == text_draft.normalize_text(line.text)
        )
        if args.restore_all_text or not same:
            selected.append(line)
        else:
            unchanged_candidates.append(
                (
                    line,
                    {
                        **line_payload(line),
                        "source_score": round(item["score"], 6),
                        "candidate_text": candidate_text,
                    },
                )
            )

    promoted_neighbors: list[dict[str, Any]] = []
    remaining_unchanged = unchanged_candidates
    while selected:
        promoted_this_pass: list[tuple[Any, dict[str, Any]]] = []
        still_unchanged: list[tuple[Any, dict[str, Any]]] = []
        for line, details in remaining_unchanged:
            if any(boxes_touch(line.box, selected_line.box) for selected_line in selected):
                promoted_this_pass.append((line, details))
            else:
                still_unchanged.append((line, details))
        if not promoted_this_pass:
            remaining_unchanged = still_unchanged
            break
        for line, details in promoted_this_pass:
            selected.append(line)
            promoted_neighbors.append(
                {
                    **details,
                    "reason": "line is already readable but touches a mask that must be restored",
                }
            )
        remaining_unchanged = still_unchanged
    unchanged = [details for _line, details in remaining_unchanged]

    selected.sort(key=lambda line: (line.box[1], line.box[0]))
    check_geometry(selected, source.width, source.height)
    payload = {
        "schema_version": 1,
        "verified": bool(selected),
        "verification": {
            "mode": "individual-paddle-boxes-with-stable-candidate-ocr",
            "minimum_score": args.min_text_score,
            "detected_source_lines": len(proposals),
            "eligible_small_source_lines": eligible,
            "rejected_low_score_lines": rejected_low_score,
            "unchanged_lines": unchanged,
            "promoted_neighbor_lines": promoted_neighbors,
            "skipped_large_lines": skipped_large,
            "maximum_text_height_fraction": MAX_TEXT_HEIGHT_FRACTION,
            "restore_all": args.restore_all_text,
            "box_policy": "individual-oriented-paddle-boxes-no-cross-subject-merging",
            "note": "verified attests filtered geometry; OCR text is informational only",
        },
        "source_pixel_sha256": source_pixel_sha256(source),
        "width": source.width,
        "height": source.height,
        "lines": [line_payload(line) for line in selected],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[text] OCR: {len(proposals)} boxes, {len(rejected_low_score)} below threshold, "
        f"{len(skipped_large)} large lines ignored, {len(unchanged)} small lines already correct, "
        f"{len(promoted_neighbors)} neighbors protected, {len(selected)} small lines to restore.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
