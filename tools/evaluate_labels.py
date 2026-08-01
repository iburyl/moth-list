#!/usr/bin/env python
"""Compare predicted YOLO-pose labels against reference labels.

Standalone (no Django server needed). It reuses the pure geometry helpers in
``moths/utils.py`` but does not touch Django settings, so it can run directly::

    python tools/evaluate_labels.py IMAGES_DIR LABELS_DIR REFERENCE_DIR

Definitions
-----------
* An image "has an object" if its label/reference file exists and contains at
  least one YOLO-pose line. Only the first object in each file is used.
* Viewpoint categories come from the F/L/R/B keypoints and their visibilities
  (``classify_annotation``): top-down, side-view, bottom-up, unclear (no fully
  visible wing), or "no object" (missing object or F/B keypoints).
* Angle / center / symmetry errors are computed in normalized image coordinates
  (0..1), only for images where both label and reference have a usable object.

Reported for all images together and per tax_id:
* false found     – label has an object where the reference has none
* false not found – reference has an object where the label has none
* found-object viewpoint distribution (% top-down / side-view / bottom-up)
* mean absolute error of the B->F vector angle (degrees)
* mean distance between the C point (midpoint of F and B)
* mean absolute error of the L/R symmetry coefficient

Plus a confusion matrix over all images (reference vs label viewpoint).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make the repo root importable so ``moths`` resolves when run from anywhere.
# This file lives in ``<repo>/tools/``, so the repo root is one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moths.utils import (  # noqa: E402
    POSE_BOTTOM_UP,
    POSE_SIDE,
    POSE_TOP_DOWN,
    POSE_UNCLEAR,
    annotation_symmetry,
    classify_annotation,
    parse_filename,
    read_annotations_file,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Human labels for the confusion matrix / viewpoint distribution.
NO_OBJECT = "no object"
POSE_LABELS = {
    POSE_TOP_DOWN: "top-down",
    POSE_SIDE: "side-view",
    POSE_BOTTOM_UP: "bottom-up",
    POSE_UNCLEAR: "unclear",
}
CATEGORIES = ["top-down", "side-view", "bottom-up", "unclear", NO_OBJECT]


def category(annotations) -> str:
    """Viewpoint category for a (possibly empty) list of annotations."""
    if not annotations:
        return NO_OBJECT
    return POSE_LABELS.get(classify_annotation(annotations[0]), NO_OBJECT)


def _fb(annotation):
    """Return ((fx, fy), (bx, by)) for a usable F and B keypoint, else None."""
    kps = annotation.keypoints
    if len(kps) < 4:
        return None
    front, back = kps[0], kps[3]
    if front.visibility <= 0 or back.visibility <= 0:
        return None
    return (front.x, front.y), (back.x, back.y)


def fb_angle(annotation) -> float | None:
    """Angle (degrees) of the vector from B to F, or None if unavailable."""
    fb = _fb(annotation)
    if fb is None:
        return None
    (fx, fy), (bx, by) = fb
    return math.degrees(math.atan2(fy - by, fx - bx))


def center_c(annotation):
    """Midpoint C of F and B, or None if unavailable."""
    fb = _fb(annotation)
    if fb is None:
        return None
    (fx, fy), (bx, by) = fb
    return ((fx + bx) / 2, (fy + by) / 2)


def angle_abs_error(a: float, b: float) -> float:
    """Absolute angular difference in [0, 180] degrees."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


class Stats:
    """Accumulator for one scope (overall or a single tax_id)."""

    def __init__(self) -> None:
        self.n_images = 0
        self.false_found = 0
        self.false_not_found = 0
        self.true_found = 0      # object present in both label and reference
        self.wrong_pose = 0      # of those, label viewpoint != reference viewpoint
        self.angle_errors: list[float] = []
        self.center_dists: list[float] = []
        self.symmetry_errors: list[float] = []

    def add(self, label_anns, ref_anns) -> None:
        self.n_images += 1
        has_label = bool(label_anns)
        has_ref = bool(ref_anns)

        if has_label and not has_ref:
            self.false_found += 1
        if has_ref and not has_label:
            self.false_not_found += 1

        if has_label and has_ref:
            self.true_found += 1
            if category(label_anns) != category(ref_anns):
                self.wrong_pose += 1

            la, ra = label_anns[0], ref_anns[0]

            al, ar = fb_angle(la), fb_angle(ra)
            if al is not None and ar is not None:
                self.angle_errors.append(angle_abs_error(al, ar))

            cl, cr = center_c(la), center_c(ra)
            if cl is not None and cr is not None:
                self.center_dists.append(math.hypot(cl[0] - cr[0], cl[1] - cr[1]))

            sl, sr = annotation_symmetry(la), annotation_symmetry(ra)
            if sl is not None and sr is not None:
                self.symmetry_errors.append(abs(sl - sr))


def _mean(values):
    return sum(values) / len(values) if values else None


def _num(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _rate(count, total, digits=4):
    return "n/a" if not total else f"{count / total:.{digits}f}"


# Table columns: (header, right-aligned?).
COLUMNS = [
    ("scope", False),
    ("imgs", True),
    ("false+", True),
    ("false-", True),
    ("wrong_pose", True),
    ("MAE_ang", True),
    ("mean_C", True),
    ("MAE_sym", True),
]


def stats_row(scope: str, s: Stats) -> list[str]:
    return [
        scope,
        str(s.n_images),
        _rate(s.false_found, s.n_images),
        _rate(s.false_not_found, s.n_images),
        _rate(s.wrong_pose, s.true_found),
        _num(_mean(s.angle_errors), 3),
        _num(_mean(s.center_dists), 5),
        _num(_mean(s.symmetry_errors), 5),
    ]


def print_table(rows: list[list[str]]) -> None:
    headers = [h for h, _ in COLUMNS]
    aligns = [r for _, r in COLUMNS]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(cells):
        parts = []
        for cell, width, right in zip(cells, widths, aligns):
            parts.append(f"{cell:>{width}}" if right else f"{cell:<{width}}")
        return "  ".join(parts)

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def print_confusion(matrix) -> None:
    print("\n=== confusion matrix (rows=label, cols=reference) ===")
    col_w = max(9, *(len(c) for c in CATEGORIES))
    header = " " * (col_w + 2) + "".join(f"{c:>{col_w + 2}}" for c in CATEGORIES)
    print(header)
    for row in CATEGORIES:
        cells = "".join(f"{matrix[row][col]:>{col_w + 2}}" for col in CATEGORIES)
        print(f"{row:<{col_w + 2}}{cells}")


def index_txt(directory: Path) -> dict[str, Path]:
    """Map ``stem -> path`` for every ``.txt`` under a directory (recursive)."""
    mapping: dict[str, Path] = {}
    for path in directory.rglob("*.txt"):
        if path.is_file():
            mapping[path.stem] = path
    return mapping


def iter_images(directory: Path):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir", type=Path, help="directory with images")
    parser.add_argument("labels_dir", type=Path, help="directory with (predicted) labels")
    parser.add_argument("reference_dir", type=Path, help="directory with reference labels")
    args = parser.parse_args()

    for label, directory in (
        ("images", args.images_dir),
        ("labels", args.labels_dir),
        ("reference", args.reference_dir),
    ):
        if not directory.is_dir():
            parser.error(f"{label} directory not found: {directory}")

    labels_map = index_txt(args.labels_dir)
    reference_map = index_txt(args.reference_dir)

    overall = Stats()
    per_tax: dict[str, Stats] = defaultdict(Stats)
    confusion = {row: Counter() for row in CATEGORIES}

    for image_path in iter_images(args.images_dir):
        stem = image_path.stem
        parsed = parse_filename(image_path.name)
        tax_id = parsed.tax_id if parsed else "unknown"

        label_anns = read_annotations_file(labels_map[stem]) if stem in labels_map else []
        ref_anns = (
            read_annotations_file(reference_map[stem]) if stem in reference_map else []
        )

        overall.add(label_anns, ref_anns)
        per_tax[tax_id].add(label_anns, ref_anns)
        confusion[category(label_anns)][category(ref_anns)] += 1

    if overall.n_images == 0:
        print("No images found.")
        return 1

    rows = [stats_row("ALL", overall)]
    rows += [stats_row(tax_id, per_tax[tax_id]) for tax_id in sorted(per_tax)]
    print_table(rows)
    print_confusion(confusion)

    print(
        "\nColumns: false+ = false found / images, false- = false not found / images, "
        "wrong_pose = wrong viewpoint / objects found in both, "
        "MAE_ang = mean abs B->F angle error (deg), "
        "mean_C = mean C-point distance (norm), "
        "MAE_sym = mean abs symmetry-coeff error."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
