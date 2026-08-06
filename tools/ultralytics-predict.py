#!/usr/bin/env python3

"""Generate predictions for the Django dataset with the 3-model pipeline.

Images are read from ``MOTHS_IMAGE_DIR`` and results are written to
``MOTHS_PREDICTION_DIR`` (the app's prediction/``test`` directory), mirroring the
``images/<tax_id>/<name>.<ext>`` layout. Both come from the environment, exactly
like the web app (every ``MOTHS_*`` path must be set or ``settings.py`` raises).

The prediction logic itself (the classification + general-pose + side-pose
pipeline and the keypoint visibility rules) lives in :mod:`utils_prediction`,
shared verbatim with ``harvest_top_images.py`` and ``evaluate_model.py``. This
script only wires that pipeline to the Django dataset and writes the label +
``.class`` sidecars. Existing labels are kept unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

import utils_prediction as pred

REPO_ROOT = Path(__file__).resolve().parent.parent


def bootstrap_django():
    """Set up Django from the environment; return ``(settings, moth_utils)``."""
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from django.conf import settings  # noqa: E402  (after django.setup)
    from moths import utils as moth_utils  # noqa: E402  (after django.setup)

    return settings, moth_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict poses for the Django dataset (MOTHS_IMAGE_DIR -> "
            "MOTHS_PREDICTION_DIR, from the environment) with a "
            "classification + pose + side-pose model pipeline."
        )
    )

    parser.add_argument(
        "--classification-model",
        dest="cls_model",
        type=Path,
        required=True,
        help="Box-only viewpoint/stage classification model (.pt).",
    )
    parser.add_argument(
        "--pose-model",
        dest="pose_model",
        type=Path,
        required=True,
        help="General F/L/R/B pose model (.pt).",
    )
    parser.add_argument(
        "--side-model",
        dest="side_model",
        type=Path,
        required=True,
        help="Side-view pose model (F, B, wing; class encodes L/R) (.pt).",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=768,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Minimum detection (box) confidence.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Inference device: 0, 1, cpu, etc.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-predict even if an output label already exists (default skips).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings, moth_utils = bootstrap_django()
    images_root = Path(settings.MOTHS_IMAGE_DIR).resolve()
    output_root = Path(settings.MOTHS_PREDICTION_DIR).resolve()

    for label, model_path in (
        ("classification", args.cls_model),
        ("pose", args.pose_model),
        ("side", args.side_model),
    ):
        if not model_path.exists():
            raise RuntimeError(f"{label} model does not exist: {model_path}")

    if not images_root.is_dir():
        raise RuntimeError(f"Image directory does not exist: {images_root}")

    image_paths = pred.find_images(images_root)
    if not image_paths:
        raise RuntimeError(f"No images found under: {images_root}")

    models = {
        "cls": YOLO(str(args.cls_model)),
        "pose": YOLO(str(args.pose_model)),
        "side": YOLO(str(args.side_model)),
    }

    counts: dict[str, int] = {}
    skipped_images = 0

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(images_root)
        output_txt = output_root / relative_path.with_suffix(".txt")
        output_class = output_root / relative_path.with_suffix(".class")

        if not args.force and output_txt.exists():
            skipped_images += 1
            print(f"[{index}/{len(image_paths)}] Skipped (exists): {relative_path}")
            continue

        try:
            prediction = pred.run_pipeline(models, image_path, args, moth_utils)
            pred.write_prediction(prediction, output_txt, output_class, moth_utils)
            print(f"[{index}/{len(image_paths)}] {prediction.status}: {relative_path}")
        except torch.OutOfMemoryError:
            output_txt.parent.mkdir(parents=True, exist_ok=True)
            output_txt.write_text("", encoding="utf-8")
            pred.remove_file(output_class)
            counts["oom"] = counts.get("oom", 0) + 1
            print(f"[{index}/{len(image_paths)}] CUDA OOM, wrote empty label: {relative_path}")
            pred.clear_cuda_cache(args.device)
            continue
        else:
            counts[prediction.status] = counts.get(prediction.status, 0) + 1
        finally:
            pred.clear_cuda_cache(args.device)

    print()
    print(f"Images found:        {len(image_paths)}")
    print(f"  general pose:      {counts.get('pose', 0)}")
    print(f"  side-view pose:    {counts.get('side_pose', 0)}")
    print(f"  side from pose:    {counts.get('side_from_pose', 0)}")
    print(f"  box only:          {counts.get('box_only', 0)}")
    print(f"  no keypoints:      {counts.get('pose_empty', 0)}")
    print(f"  not classified:    {counts.get('no_detection', 0)}")
    if counts.get("oom"):
        print(f"  CUDA OOM:          {counts['oom']}")
    print(f"  skipped (exists):  {skipped_images}")
    print(f"Images directory:    {images_root}")
    print(f"Output directory:    {output_root}")


if __name__ == "__main__":
    main()
