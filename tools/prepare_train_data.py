#!/usr/bin/env python3

"""Bundle the manually-labelled poses into YOLO training archives.

This walks the Django dataset (whose folders come from the environment, exactly
like the web app: ``MOTHS_IMAGE_DIR`` / ``MOTHS_LABEL_DIR`` / ...) and gathers
every image that has a documented pose label. It then produces three archives in
the output directory:

``all_poses.zip`` — the YOLO **pose** dataset (keypoint labels, unchanged)::

    data.yaml                           pose task config (kpt_shape [4,3])
    train.txt / val.txt                 image lists (~5% held out for val)
    data/<labels>/<tax_id>/<name>.txt   the pose label files (F/L/R/B keypoints)

``side_view_poses.zip`` — a YOLO **pose** dataset restricted to side-view images,
whose labels keep only 3 keypoints: F, B and whichever single wing (L or R) is
present::

    data.yaml                           pose task config (kpt_shape [3,3])
    train.txt / val.txt                 image lists (same split as above)
    data/<labels>/<tax_id>/<name>.txt   ``<class> <box> F B wing`` lines

``pose_classification.zip`` — a YOLO **detection** dataset (box-only labels)
where the box class encodes the viewpoint (:func:`moths.utils.classify_annotation`)
plus two per-image flag overrides::

    data.yaml                           detect task config (7 classes)
    train.txt / val.txt                 image lists (same split as above)
    data/<labels>/<tax_id>/<name>.txt   ``<class> <cx> <cy> <w> <h>`` lines

    class ids:  0 top_down · 1 side_view_l · 2 side_view_r · 3 bottom_up
                4 unclear_pose · 5 top_down_pinned · 6 macro

    A side view splits into side_view_l / side_view_r by the present wing;
    Adult + top-down + Pinned -> top_down_pinned (instead of top_down);
    Adult + Macro -> macro. Images flagged Damaged are excluded from BOTH
    archives (and from the train/val split).

Images are assumed to already exist next to each archive at
``data/<images>/<tax_id>/<name>.<ext>`` (they are *not* copied into the zips).
The ``<images>`` / ``<labels>`` path components mirror the real directory names
(``MOTHS_IMAGE_DIR.name`` / ``MOTHS_LABEL_DIR.name``) so the standard YOLO
``images`` -> ``labels`` path swap keeps working after extraction, and image
list lines are ``./``-prefixed so they resolve relative to the ``*.txt`` file.

The split is simple: ``val.txt`` gets a random ``--val-fraction`` (default 5%)
of the labelled images and ``train.txt`` gets the rest; both archives use the
same split. The Django data directories are only ever read — nothing on disk is
modified; ``train.txt`` / ``val.txt`` / ``data.yaml`` exist only inside the
archives.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

POSE_ARCHIVE_NAME = "all_poses.zip"
CLASSIFICATION_ARCHIVE_NAME = "pose_classification.zip"
SIDE_POSE_ARCHIVE_NAME = "side_view_poses.zip"

# Top-level folder inside the archives that mirrors the dataset root.
DATA_ROOT = "data"

# Class order for the box-only classification dataset. The first four are the
# viewpoints from moths.utils.classify_annotation; the last two are derived from
# the per-image flags: an Adult top-down that is Pinned becomes ``top_down_pinned``
# (instead of ``top_down``), and any Adult flagged Macro becomes ``macro``.
# Images flagged Damaged are excluded from both archives entirely.
CLASSIFICATION_NAMES = [
    "top_down",
    "side_view_l",
    "side_view_r",
    "bottom_up",
    "unclear_pose",
    "top_down_pinned",
    "macro",
]


def bootstrap_django():
    """Set up Django (from the environment) and return ``moths.utils``.

    The dataset layout, label/image path logic and pose classification all live
    in the app, so we import it rather than re-implement it. Every ``MOTHS_*``
    path must be present in the environment; ``settings.py`` raises otherwise.
    """
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from moths import utils as moth_utils  # noqa: E402  (after django.setup)

    return moth_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare all_poses.zip (pose) and pose_classification.zip (box-only "
            "viewpoint classes) from the Django dataset configured via the "
            "environment."
        )
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory the archives are written into (created if needed).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
        help="Fraction of labelled images placed in val.txt (default: 0.05).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Random seed for the val split, for reproducible archives (default: 777).",
    )
    return parser.parse_args()


def collect_documented(moth_utils) -> list[tuple]:
    """Return ``(tax_id, filename, label_path, annotations, stage, flags)`` per image.

    A "documented pose" is an image with a non-empty label file (at least one
    parseable YOLO-pose annotation). Iterating the images (not the label files)
    guarantees every entry points at a real, still-present image. ``stage`` /
    ``flags`` come from the ``<name>.class`` file. Images flagged ``Damaged`` are
    dropped here, so they never reach either archive (or the train/val split).
    """
    documented: list[tuple] = []
    tax_ids = moth_utils.list_tax_ids()
    total = len(tax_ids)
    for index, tax_id in enumerate(tax_ids, start=1):
        print(f"\r[{index} of {total}] scanning {tax_id} ", end="", flush=True)
        for image in moth_utils.scan_tax_images(tax_id):
            label_path = moth_utils.get_label_path(image.filename)
            if not label_path.is_file():
                continue
            annotations = moth_utils.read_annotations_file(label_path)
            if not annotations:
                continue
            stage, flags = moth_utils.get_class_and_flags(image.filename)
            if "Damaged" in flags:  # excluded from all training data
                continue
            documented.append(
                (tax_id, image.filename, label_path, annotations, stage, flags)
            )
    if total:
        print()  # end the progress line
    documented.sort(key=lambda row: (row[0], row[1]))
    return documented


def _image_entry(images_component: str, tax_id: str, filename: str) -> str:
    """Image-list line, ``./``-prefixed so it resolves relative to the txt file."""
    return f"./{DATA_ROOT}/{images_component}/{tax_id}/{filename}"


def _label_arcname(labels_component: str, tax_id: str, filename: str) -> str:
    """In-archive path for a label file (mirrors ``data/<labels>/<tax>/<name>.txt``)."""
    stem = Path(filename).stem
    return f"{DATA_ROOT}/{labels_component}/{tax_id}/{stem}.txt"


def _as_text(lines: list[str]) -> str:
    """Join lines into POSIX-style text (trailing newline if non-empty)."""
    return ("\n".join(lines) + "\n") if lines else ""


def _pose_yaml() -> str:
    return (
        "# YOLO pose dataset — F/L/R/B keypoints. Run training from this folder.\n"
        "path: .\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "kpt_shape: [4, 3]  # 4 keypoints (F, L, R, B), each x/y/visibility\n"
        "flip_idx: [0, 2, 1, 3]  # left-right flip swaps L and R\n"
        "names:\n"
        "  0: moth\n"
    )


def _side_pose_yaml() -> str:
    return (
        "# YOLO pose dataset — side view: F, B and the one present wing (L or R).\n"
        "path: .\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "kpt_shape: [3, 3]  # 3 keypoints (F, B, wing), each x/y/visibility\n"
        "flip_idx: [0, 1, 2]  # horizontal flip keeps the F/B/wing slots\n"
        "names:\n"
        "  0: moth\n"
    )


def _classification_yaml() -> str:
    lines = [
        "# YOLO detection dataset — box class encodes the viewpoint.",
        "# Run training from this folder.",
        "path: .",
        "train: train.txt",
        "val: val.txt",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASSIFICATION_NAMES)]
    return "\n".join(lines) + "\n"


def _pose_label_lines(annotations) -> list[str]:
    """YOLO-pose lines for annotations that carry the full F/L/R/B keypoints.

    Box-only annotations (drawn with Shift+drag — no keypoints) and any object
    without exactly four keypoints are dropped, so the pose dataset never
    contains keypoint-less or ragged lines (``kpt_shape`` is a fixed ``[4, 3]``).
    """
    lines: list[str] = []
    for a in annotations:
        if len(a.keypoints) != 4:
            continue
        parts = [
            str(int(a.class_id)),
            f"{a.cx:.6f}",
            f"{a.cy:.6f}",
            f"{a.width:.6f}",
            f"{a.height:.6f}",
        ]
        for kp in a.keypoints:
            parts += [f"{kp.x:.6f}", f"{kp.y:.6f}", str(int(kp.visibility))]
        lines.append(" ".join(parts))
    return lines


def build_pose_archive(
    zip_path: Path,
    documented: list[tuple],
    val_indices: set[int],
    images_component: str,
    labels_component: str,
) -> tuple[int, int]:
    """Write the keypoint pose archive; return ``(train_count, val_count)``.

    Only keypoint-bearing lines are emitted (see :func:`_pose_label_lines`);
    images left with no such line (e.g. box-only labels) are skipped entirely —
    no label file and no image-list entry — so every listed image has keypoints.
    (Damaged-flagged images were already dropped in :func:`collect_documented`.)
    """
    train_lines: list[str] = []
    val_lines: list[str] = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, (tax_id, filename, _label_path, anns, _stage, _flags) in enumerate(
            documented
        ):
            pose_lines = _pose_label_lines(anns)
            if not pose_lines:
                continue
            archive.writestr(
                _label_arcname(labels_component, tax_id, filename),
                _as_text(pose_lines),
            )
            entry = _image_entry(images_component, tax_id, filename)
            (val_lines if i in val_indices else train_lines).append(entry)
        train_lines.sort()
        val_lines.sort()
        archive.writestr("train.txt", _as_text(train_lines))
        archive.writestr("val.txt", _as_text(val_lines))
        archive.writestr("data.yaml", _pose_yaml())
    return len(train_lines), len(val_lines)


def _side_pose_label_lines(annotations, classify, pose_side) -> list[str]:
    """3-keypoint YOLO-pose lines (F, B, wing) for side-view annotations.

    Only annotations that classify as a side view are kept; each emits exactly
    three keypoints — F, B and whichever single wing (L or R) is present — in
    that fixed order (``kpt_shape [3, 3]``). Class is always ``0`` (moth).
    """
    lines: list[str] = []
    for a in annotations:
        if len(a.keypoints) != 4 or classify(a) != pose_side:
            continue
        front, left, right, back = a.keypoints
        wing = left if left.visibility > 0 else right
        parts = [
            "0",
            f"{a.cx:.6f}",
            f"{a.cy:.6f}",
            f"{a.width:.6f}",
            f"{a.height:.6f}",
        ]
        for kp in (front, back, wing):
            parts += [f"{kp.x:.6f}", f"{kp.y:.6f}", str(int(kp.visibility))]
        lines.append(" ".join(parts))
    return lines


def build_side_pose_archive(
    zip_path: Path,
    documented: list[tuple],
    val_indices: set[int],
    images_component: str,
    labels_component: str,
    classify,
    pose_side,
) -> tuple[int, int]:
    """Write the side-view pose archive (F/B/wing); return ``(train, val)``.

    Only images with at least one side-view annotation are listed; each label
    holds the 3-keypoint side lines (see :func:`_side_pose_label_lines`).
    """
    train_lines: list[str] = []
    val_lines: list[str] = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, (tax_id, filename, _label_path, anns, _stage, _flags) in enumerate(
            documented
        ):
            side_lines = _side_pose_label_lines(anns, classify, pose_side)
            if not side_lines:
                continue
            archive.writestr(
                _label_arcname(labels_component, tax_id, filename),
                _as_text(side_lines),
            )
            entry = _image_entry(images_component, tax_id, filename)
            (val_lines if i in val_indices else train_lines).append(entry)
        train_lines.sort()
        val_lines.sort()
        archive.writestr("train.txt", _as_text(train_lines))
        archive.writestr("val.txt", _as_text(val_lines))
        archive.writestr("data.yaml", _side_pose_yaml())
    return len(train_lines), len(val_lines)


def _classification_label(annotations, stage, flags, class_for) -> list[str]:
    """Box-only YOLO lines (``<class> cx cy w h``) for the classifiable objects.

    ``class_for(annotation, stage, flags)`` returns the class id (folding in the
    per-image stage/flag overrides) or ``None`` to drop the object.
    """
    lines: list[str] = []
    for annotation in annotations:
        class_id = class_for(annotation, stage, flags)
        if class_id is None:  # e.g. POSE_NONE — not one of the classes
            continue
        lines.append(
            f"{class_id} {annotation.cx:.6f} {annotation.cy:.6f} "
            f"{annotation.width:.6f} {annotation.height:.6f}"
        )
    return lines


def build_classification_archive(
    zip_path: Path,
    documented: list[tuple],
    val_indices: set[int],
    images_component: str,
    labels_component: str,
    class_for,
) -> tuple[int, int, dict[int, int]]:
    """Write the box-only viewpoint archive.

    Returns ``(train_count, val_count, per_class_counts)``. Images whose objects
    all map to no class are skipped entirely (no label, no list entry), so listed
    images always carry at least one box. ``class_for`` applies the stage/flag
    overrides (``top_down_pinned`` / ``macro``).
    """
    train_lines: list[str] = []
    val_lines: list[str] = []
    per_class: dict[int, int] = defaultdict(int)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, (tax_id, filename, _label_path, anns, stage, flags) in enumerate(
            documented
        ):
            label_lines = _classification_label(anns, stage, flags, class_for)
            if not label_lines:
                continue
            for line in label_lines:
                per_class[int(line.split(" ", 1)[0])] += 1
            archive.writestr(
                _label_arcname(labels_component, tax_id, filename),
                _as_text(label_lines),
            )
            entry = _image_entry(images_component, tax_id, filename)
            (val_lines if i in val_indices else train_lines).append(entry)
        train_lines.sort()
        val_lines.sort()
        archive.writestr("train.txt", _as_text(train_lines))
        archive.writestr("val.txt", _as_text(val_lines))
        archive.writestr("data.yaml", _classification_yaml())
    return len(train_lines), len(val_lines), dict(per_class)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in [0.0, 1.0).")

    moth_utils = bootstrap_django()

    images_component = moth_utils.get_image_dir().name
    labels_component = moth_utils.get_label_dir().name

    name_to_id = {name: i for i, name in enumerate(CLASSIFICATION_NAMES)}
    base_class_ids = {
        moth_utils.POSE_TOP_DOWN: name_to_id["top_down"],
        moth_utils.POSE_BOTTOM_UP: name_to_id["bottom_up"],
        moth_utils.POSE_UNCLEAR: name_to_id["unclear_pose"],
    }
    top_down_pose = moth_utils.POSE_TOP_DOWN
    side_pose = moth_utils.POSE_SIDE
    classify = moth_utils.classify_annotation

    def class_for(annotation, stage, flags):
        """Class id for one object, folding in the per-image stage/flag overrides.

        Adult + top-down + Pinned -> ``top_down_pinned`` (checked first, most
        specific); otherwise Adult + Macro -> ``macro``; a side view splits into
        ``side_view_l`` / ``side_view_r`` by the present wing; otherwise the plain
        viewpoint class (or ``None`` for POSE_NONE).
        """
        pose = classify(annotation)
        is_adult = stage == "Adult"
        if is_adult and pose == top_down_pose and "Pinned" in flags:
            return name_to_id["top_down_pinned"]
        if is_adult and "Macro" in flags:
            return name_to_id["macro"]
        if pose == side_pose:
            left = annotation.keypoints[1]
            return name_to_id["side_view_l" if left.visibility > 0 else "side_view_r"]
        return base_class_ids.get(pose)

    documented = collect_documented(moth_utils)
    total = len(documented)
    if total == 0:
        print("No documented poses found under the labels directory.")

    # One shared split (by index) so both archives hold out the same images.
    rng = random.Random(args.seed)
    val_count = int(round(total * args.val_fraction))
    val_indices = set(rng.sample(range(total), val_count)) if val_count else set()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_zip = output_dir / POSE_ARCHIVE_NAME
    classification_zip = output_dir / CLASSIFICATION_ARCHIVE_NAME
    side_zip = output_dir / SIDE_POSE_ARCHIVE_NAME

    pose_train, pose_val = build_pose_archive(
        pose_zip, documented, val_indices, images_component, labels_component
    )
    cls_train, cls_val, per_class = build_classification_archive(
        classification_zip,
        documented,
        val_indices,
        images_component,
        labels_component,
        class_for,
    )
    side_train, side_val = build_side_pose_archive(
        side_zip,
        documented,
        val_indices,
        images_component,
        labels_component,
        classify,
        moth_utils.POSE_SIDE,
    )

    _print_summary(
        pose_zip,
        (pose_train, pose_val),
        classification_zip,
        (cls_train, cls_val),
        per_class,
        side_zip,
        (side_train, side_val),
    )


def _print_summary(
    pose_zip: Path,
    pose_counts: tuple[int, int],
    classification_zip: Path,
    cls_counts: tuple[int, int],
    per_class: dict[int, int],
    side_zip: Path,
    side_counts: tuple[int, int],
) -> None:
    pose_train, pose_val = pose_counts
    cls_train, cls_val = cls_counts
    side_train, side_val = side_counts
    print(f"Pose archive: {pose_zip}")
    print(f"  poses: {pose_train + pose_val} (train {pose_train}, val {pose_val})")
    print(f"Classification archive: {classification_zip}")
    print(f"  images: {cls_train + cls_val} (train {cls_train}, val {cls_val})")
    print("  objects per class:")
    for i, name in enumerate(CLASSIFICATION_NAMES):
        print(f"    {i} {name}: {per_class.get(i, 0)}")
    print(f"Side-view pose archive: {side_zip}")
    print(f"  side poses: {side_train + side_val} (train {side_train}, val {side_val})")


if __name__ == "__main__":
    main()
