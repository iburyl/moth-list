#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLO pose-label predictions for every image."
    )

    parser.add_argument(
        "model",
        type=Path,
        help="Path to trained YOLO pose model, e.g. runs/.../weights/best.pt",
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/images"),
        help="Input image directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/test"),
        help="Output label directory.",
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
        help="Minimum detection confidence.",
    )

    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.0,
        help="Keypoints below this confidence are written with visibility 0.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="Inference device: 0, 1, cpu, etc.",
    )

    parser.add_argument(
        "--all-detections",
        action="store_true",
        help="Write all detections. Default writes only the best detection.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-predict even if an output label already exists (default skips).",
    )

    return parser.parse_args()


def find_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def prediction_to_line(
    result,
    detection_index: int,
    keypoint_conf_threshold: float,
) -> str:
    class_id = int(result.boxes.cls[detection_index].item())

    cx, cy, width, height = (
        result.boxes.xywhn[detection_index]
        .detach()
        .cpu()
        .tolist()
    )

    values = [
        str(class_id),
        f"{clamp01(float(cx)):.6f}",
        f"{clamp01(float(cy)):.6f}",
        f"{clamp01(float(width)):.6f}",
        f"{clamp01(float(height)):.6f}",
    ]

    keypoints_xy = (
        result.keypoints.xyn[detection_index]
        .detach()
        .cpu()
        .tolist()
    )

    keypoint_confidences = None
    if result.keypoints.conf is not None:
        keypoint_confidences = (
            result.keypoints.conf[detection_index]
            .detach()
            .cpu()
            .tolist()
        )

    for keypoint_index, (x, y) in enumerate(keypoints_xy):
        if keypoint_confidences is None:
            visibility = 2
        else:
            confidence = float(keypoint_confidences[keypoint_index])
            visibility = 2 if confidence >= keypoint_conf_threshold else 0

        if visibility == 0:
            x = 0.0
            y = 0.0

        values.extend(
            [
                f"{clamp01(float(x)):.6f}",
                f"{clamp01(float(y)):.6f}",
                str(visibility),
            ]
        )

    return " ".join(values)


def clear_cuda_cache(device: str) -> None:
    gc.collect()

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()

    images_root = args.images.resolve()
    output_root = args.output.resolve()

    if not args.model.exists():
        raise RuntimeError(f"Model does not exist: {args.model}")

    if not images_root.is_dir():
        raise RuntimeError(f"Image directory does not exist: {images_root}")

    image_paths = find_images(images_root)

    if not image_paths:
        raise RuntimeError(f"No images found under: {images_root}")

    model = YOLO(str(args.model))

    predicted_images = 0
    empty_images = 0
    skipped_images = 0

    for index, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(images_root)
        output_path = output_root / relative_path.with_suffix(".txt")

        # Only (re)predict when no previous label exists, unless forced.
        if not args.force and output_path.exists():
            skipped_images += 1
            print(
                f"[{index}/{len(image_paths)}] "
                f"Skipped (exists): {relative_path}"
            )
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                max_det=100 if args.all_detections else 1,
                device=args.device,
                batch=1,
                stream=False,
                verbose=False,
            )

            result = results[0]

            if (
                result.boxes is None
                or len(result.boxes) == 0
                or result.keypoints is None
                or len(result.keypoints) == 0
            ):
                output_path.write_text("", encoding="utf-8")
                empty_images += 1
                print(
                    f"[{index}/{len(image_paths)}] "
                    f"No prediction: {relative_path}"
                )
                continue

            if args.all_detections:
                detection_indices = range(len(result.boxes))
            else:
                best_index = int(result.boxes.conf.argmax().item())
                detection_indices = [best_index]

            lines = [
                prediction_to_line(
                    result=result,
                    detection_index=detection_index,
                    keypoint_conf_threshold=args.keypoint_conf,
                )
                for detection_index in detection_indices
            ]

            output_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

            predicted_images += 1

            print(
                f"[{index}/{len(image_paths)}] "
                f"Wrote {len(lines)} prediction(s): {relative_path}"
            )

        except torch.OutOfMemoryError:
            output_path.write_text("", encoding="utf-8")
            empty_images += 1

            print(
                f"[{index}/{len(image_paths)}] "
                f"CUDA OOM, wrote empty label: {relative_path}"
            )

            clear_cuda_cache(args.device)

        finally:
            if "results" in locals():
                del results

            clear_cuda_cache(args.device)

    print()
    print(f"Images found:     {len(image_paths)}")
    print(f"With prediction:  {predicted_images}")
    print(f"Without result:   {empty_images}")
    print(f"Skipped (exists): {skipped_images}")
    print(f"Output directory: {output_root}")


if __name__ == "__main__":
    main()