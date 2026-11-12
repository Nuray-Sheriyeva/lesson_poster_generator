"""Spike: choose a text detector for mascot placement.

The mascot pipeline must know where rendered text is.  The verifier only
transcribes text, so this experiment compares box detectors on the actual
candidate crop and reports recall on small text.

Usage::

    python experiments/mascot_text_detector.py
    python experiments/mascot_text_detector.py --json
    python experiments/mascot_text_detector.py --poster outputs/foo/bar.png

The synthetic benchmark renders known text at 12-30 px, then runs each detector
only on the four candidate mascot rectangles.  A ground-truth line counts as
detected when a predicted box overlaps it (IoU >= 0.3) or contains its centre.
The existing-poster section reports how many boxes the chosen detector finds in
the same crops and how long it takes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont

from genlesson.app_config import MascotConfig
from genlesson.mascot.layout import CORNERS, Rect, layout_mascot
from genlesson.mascot.zones import RapidOCRDetector, TextDetector, detect_in_block

FONT = ROOT / "assets" / "fonts" / "NotoSans-Regular.ttf"
TEXT = "Байланыс 22.6% Қазақстан"
SIZES = (8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 30)


class EdgeTextDetector:
    """Cheap morphological-gradient baseline (no model download)."""

    name = "opencv-edges"

    def detect(self, image: Image.Image):
        import cv2
        import numpy as np

        gray = np.asarray(image.convert("L"))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        _, threshold = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[Rect] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width < 8 or height < 7 or width * height < 40:
                continue
            if width / max(1, height) < 1.1:
                continue
            boxes.append(Rect(x, y, width, height))
        return tuple(boxes)


def _iou(a: Rect, b: Rect) -> float:
    overlap = a.overlap_area(b)
    union = a.area + b.area - overlap
    return overlap / union if union else 0.0


def _matches(truth: Rect, boxes: tuple[Rect, ...]) -> bool:
    for box in boxes:
        if _iou(truth, box) >= 0.3:
            return True
        if box.contains(*truth.center):
            return True
    return False


def synthetic_poster() -> tuple[Image.Image, list[tuple[Rect, int]]]:
    image = Image.new("RGB", (1024, 1536), (250, 248, 244))
    draw = ImageDraw.Draw(image)
    truth: list[tuple[Rect, int]] = []
    index = 0
    for row in range(0, 1490, 46):
        for column in range(0, 960, 238):
            size = SIZES[index % len(SIZES)]
            index += 1
            font = ImageFont.truetype(str(FONT), size)
            x = 22 + column
            y = row
            draw.text((x, y), TEXT, font=font, fill=(35, 25, 60))
            bbox = draw.textbbox((x, y), TEXT, font=font)
            rect = Rect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            truth.append((rect, size))
    return image, truth


def _benchmark_detector(
    detector: TextDetector, image: Image.Image, truth: list[tuple[Rect, int]]
) -> dict[str, Any]:
    config = MascotConfig()
    detections: list[tuple[Rect, tuple[Rect, ...]]] = []
    elapsed = 0.0
    for corner in CORNERS:
        layout = layout_mascot(1024, 1536, "Бұл сұрақ нені білдіреді?", corner, config)
        start = time.perf_counter()
        boxes = detect_in_block(image, layout, detector)
        elapsed += time.perf_counter() - start
        detections.append((layout.block, boxes))

    small = [(rect, size) for rect, size in truth if size <= 30]
    hits_by_size: dict[int, list[int]] = {}
    misses: list[dict[str, Any]] = []
    small_hits = 0
    small_total = 0
    for rect, size in small:
        # The line is in the candidate crop when its centre is inside it.
        in_crop = any(block.contains(*rect.center) for block, _ in detections)
        if not in_crop:
            continue
        small_total += 1
        bucket = hits_by_size.setdefault(size, [0, 0])
        bucket[1] += 1
        if any(_matches(rect, boxes) for _, boxes in detections):
            small_hits += 1
            bucket[0] += 1
        else:
            misses.append({"size": size, "box": rect.as_dict()})
    return {
        "detector": detector.name,
        "small_text_recall": round(small_hits / small_total, 4) if small_total else 0.0,
        "small_text_boxes": small_total,
        "small_text_hits": small_hits,
        "by_size": {
            str(size): {
                "hits": hits,
                "total": total,
                "recall": round(hits / total, 3) if total else 0.0,
            }
            for size, (hits, total) in sorted(hits_by_size.items())
        },
        "crop_boxes": sum(len(boxes) for _, boxes in detections),
        "extra_boxes": max(0, sum(len(boxes) for _, boxes in detections) - small_hits),
        "elapsed_s": round(elapsed, 4),
        "misses": misses[:8],
    }


def _existing_posters() -> list[Path]:
    return sorted(path for path in (ROOT / "outputs").rglob("*.png") if "mascot" not in path.name)


def _existing_report(detector: TextDetector, posters: list[Path]) -> list[dict[str, Any]]:
    config = MascotConfig()
    report: list[dict[str, Any]] = []
    for path in posters:
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except OSError:
            continue
        layout = layout_mascot(
            image.width,
            image.height,
            "Бұл суретте не көріп тұрсың?",
            "bottom_right",
            config,
        )
        start = time.perf_counter()
        boxes = detect_in_block(image, layout, detector)
        report.append(
            {
                "poster": str(path.relative_to(ROOT)),
                "size": [image.width, image.height],
                "boxes_in_candidate_crop": len(boxes),
                "elapsed_s": round(time.perf_counter() - start, 4),
            }
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument(
        "--poster",
        type=Path,
        action="append",
        default=None,
        help="Existing poster to scan (repeatable; default: outputs/**/*.png)",
    )
    args = parser.parse_args(argv)

    image, truth = synthetic_poster()
    detectors: list[TextDetector] = [RapidOCRDetector(), EdgeTextDetector()]
    benchmarks = [_benchmark_detector(detector, image, truth) for detector in detectors]
    chosen = max(
        benchmarks,
        key=lambda item: (item["small_text_recall"], -item["extra_boxes"]),
    )

    posters = args.poster or _existing_posters()
    existing = _existing_report(RapidOCRDetector(), posters)
    payload = {
        "candidate": "opencv-db-via-rapidocr",
        "chosen": chosen["detector"],
        "synthetic": benchmarks,
        "existing_posters": existing,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("Synthetic small-text recall by detector")
    print("---------------------------------------")
    for item in benchmarks:
        print(
            f"{item['detector']:14} recall={item['small_text_recall']:.3f} "
            f"({item['small_text_hits']}/{item['small_text_boxes']}) "
            f"boxes={item['crop_boxes']:3} extra={item['extra_boxes']:3} "
            f"time={item['elapsed_s']:.3f}s"
        )
    print(f"\nChosen: {chosen['detector']}")
    print("\nExisting posters (bottom-right candidate crop, DB detector)")
    print("-----------------------------------------------------------")
    for item in existing:
        print(
            f"{item['poster']:70} boxes={item['boxes_in_candidate_crop']:3} "
            f"time={item['elapsed_s']:.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
