#!/usr/bin/env python3

"""Evaluate the 3-model prediction pipeline against the hand labels.

Runs the same classification / general-pose / side-pose models used by
``ultralytics-predict.py`` on the images that already carry a hand label, and
scores how well the models reproduce those labels. Nothing is written: the
predicted labels on disk are ignored; every prediction is recomputed live.

Directories come from Django (the environment must set every ``MOTHS_*`` path,
exactly like the web app / the other tools). Only the images and hand labels
are read: ``MOTHS_IMAGE_DIR`` for the photos and ``MOTHS_LABEL_DIR`` for the
reference ``.txt`` pose files; the hand stage sidecar (the ``<name>.class`` in
``MOTHS_LABEL_DIR``) supplies the Larva ground truth.

The three models are passed on the command line::

    python tools/evaluate_model.py \
        --classification-model cls.pt --pose-model pose.pt --side-model side.pt

Scope
-----
Only images with at least a box label are evaluated: the hand label ``.txt``
must contain at least one object (a box, with or without keypoints). A hand
stage class alone is *not* enough — an unboxed Larva sample is skipped, since
there is no box for the classifier's top object to be compared against.

Ground-truth category of an image:

* ``larva``    – hand stage is *Larva*;
* otherwise, from the first labelled object's F/L/R/B keypoints
  (``classify_annotation``): ``top_down`` / ``side`` / ``bottom_up`` /
  ``unclear``; anything else (no object, other stage) is ``other``.

Reported per tax and, at the end, totalled over all images.

Classification (top detected object vs. top labelled object), for each of
``larva`` / ``top_down`` / ``side``:

* ``recall`` = detected_true / labelled            (of the images that truly
  are this category, the fraction the classifier also called this category);
* ``fdr``    = detected_false / detected            (of the images the
  classifier called this category, the fraction that were not).

Pose keypoints:

* ``top-down`` — run the **general** pose model on the ground-truth top-down
  images and report the mean keypoint offset (normalized distance, over the
  keypoints visible in the label) and the share of images the model failed to
  detect an object in;
* ``side`` — same, using the **side-view** pose model on the ground-truth side
  images (comparing F, B and the visible wing).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import utils_prediction as pred

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Django bootstrap --------------------------------------------------------


def bootstrap_django():
    """Set up Django from the environment; return ``(settings, moth_utils)``."""
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from django.conf import settings  # noqa: E402  (after django.setup)
    from moths import utils as moth_utils  # noqa: E402  (after django.setup)

    return settings, moth_utils


# --- Categories --------------------------------------------------------------

LARVA = "larva"
TOP_DOWN = "top_down"
SIDE = "side"
OTHER = "other"

# The three categories carried in the classification report, in order.
CLASS_CATEGORIES = [LARVA, TOP_DOWN, SIDE]


def ground_truth_category(stage, label_anns, moth_utils) -> str:
    """Category of the hand label: larva stage wins, else the pose viewpoint."""
    if stage == "Larva":
        return LARVA
    if label_anns:
        pose = moth_utils.classify_annotation(label_anns[0])
        if pose == moth_utils.POSE_TOP_DOWN:
            return TOP_DOWN
        if pose == moth_utils.POSE_SIDE:
            return SIDE
    return OTHER


def predicted_category(cls_id) -> str:
    """Category the classification model asserts for its top object."""
    if cls_id in pred.CLS_TOP_DOWN_ANY:
        return TOP_DOWN
    if cls_id == pred.CLS_SIDE_VIEW:
        return SIDE
    if cls_id == pred.CLS_LARVA:
        return LARVA
    return OTHER


# --- Keypoint comparison -----------------------------------------------------


def _kp_distance(gt_kp, pred_xy) -> float:
    """Normalized Euclidean distance between a label keypoint and a prediction."""
    px, py = pred_xy
    return math.hypot(gt_kp.x - px, gt_kp.y - py)


def top_down_offsets(gt_ann, pred_raw) -> list[float]:
    """Per-keypoint offsets for the 4 F/L/R/B points visible in the label.

    ``pred_raw`` is the general pose model's ``[(x, y, conf), ...]`` output.
    Only keypoints the label marks visible (visibility > 0) are compared, and
    only when the model produced that keypoint index.
    """
    dists: list[float] = []
    kps = gt_ann.keypoints
    for i in range(min(4, len(kps), len(pred_raw))):
        if kps[i].visibility > 0:
            dists.append(_kp_distance(kps[i], pred_raw[i][:2]))
    return dists


def gt_wing_side(gt_ann) -> str | None:
    """The label's visible wing: ``"L"``, ``"R"`` or ``None`` if neither."""
    kps = gt_ann.keypoints
    if len(kps) < 4:
        return None
    if kps[1].visibility > 0:
        return "L"
    if kps[2].visibility > 0:
        return "R"
    return None


def side_offsets(gt_ann, pred_f, pred_b, pred_wing) -> list[float]:
    """Offsets for a side view's F, B and visible wing against predicted points.

    Each predicted point is an ``(x, y)`` pair or ``None`` (skipped). The
    label's front (index 0), back (index 3) and whichever wing (L index 1 or R
    index 2) is visible are matched against ``pred_f`` / ``pred_b`` /
    ``pred_wing``. This works for either pose model — the caller just passes the
    right F/B/wing points from that model's keypoint layout.
    """
    kps = gt_ann.keypoints
    if len(kps) < 4:
        return []
    front, left, right, back = kps[0], kps[1], kps[2], kps[3]
    wing_gt = left if left.visibility > 0 else right
    dists: list[float] = []
    if pred_f is not None:
        dists.append(_kp_distance(front, pred_f))
    if pred_b is not None:
        dists.append(_kp_distance(back, pred_b))
    if wing_gt.visibility > 0 and pred_wing is not None:
        dists.append(_kp_distance(wing_gt, pred_wing))
    return dists


def general_side_choice(pred_raw):
    """Pick the side (``"L"``/``"R"``) and wing point from a general-model output.

    The general F/L/R/B model has no side class, so — mirroring
    ``build_side_from_pose_model`` — the wing is whichever of L/R has the higher
    keypoint confidence. Returns ``(side, wing_xy)``; ``(None, None)`` when the
    confidences are equal/absent (ambiguous — the pipeline would call it
    unclear) or there are fewer than four keypoints.
    """
    if len(pred_raw) < 4:
        return None, None
    lc, rc = pred_raw[1][2], pred_raw[2][2]
    if lc is not None and rc is not None:
        if lc > rc:
            return "L", pred_raw[1][:2]
        if rc > lc:
            return "R", pred_raw[2][:2]
    return None, None


# --- Accumulator -------------------------------------------------------------


class Stats:
    """Counters for one scope (a single tax_id or the overall total)."""

    def __init__(self) -> None:
        self.n_images = 0
        self.labeled = {cat: 0 for cat in CLASS_CATEGORIES}
        self.tp = {cat: 0 for cat in CLASS_CATEGORIES}
        self.fp = {cat: 0 for cat in CLASS_CATEGORIES}
        # Top-down pose (general model on ground-truth top-down images).
        self.td_total = 0
        self.td_missed = 0
        self.td_dists: list[float] = []
        # Side view via the dedicated side model.
        self.sd_total = 0
        self.sd_missed = 0
        self.sd_dists: list[float] = []
        self.sd_lr_total = 0     # detections where GT side is known
        self.sd_lr_correct = 0   # ... of those, predicted L/R matched
        # Side view via the general pose model (wing chosen by confidence).
        self.gp_total = 0
        self.gp_missed = 0
        self.gp_dists: list[float] = []
        self.gp_lr_total = 0
        self.gp_lr_correct = 0

    def add_classification(self, gt_cat: str, pred_cat: str | None) -> None:
        self.n_images += 1
        for cat in CLASS_CATEGORIES:
            if gt_cat == cat:
                self.labeled[cat] += 1
            if pred_cat == cat:
                if gt_cat == cat:
                    self.tp[cat] += 1
                else:
                    self.fp[cat] += 1

    def add_top_down_pose(self, missed: bool, dists: list[float]) -> None:
        self.td_total += 1
        if missed:
            self.td_missed += 1
        else:
            self.td_dists.extend(dists)

    def _add_side(self, missed, dists, pred_side, gt_side, prefix) -> None:
        setattr(self, f"{prefix}_total", getattr(self, f"{prefix}_total") + 1)
        if missed:
            setattr(self, f"{prefix}_missed", getattr(self, f"{prefix}_missed") + 1)
            return
        getattr(self, f"{prefix}_dists").extend(dists)
        if gt_side is not None:
            setattr(self, f"{prefix}_lr_total", getattr(self, f"{prefix}_lr_total") + 1)
            if pred_side is not None and pred_side == gt_side:
                setattr(
                    self, f"{prefix}_lr_correct",
                    getattr(self, f"{prefix}_lr_correct") + 1,
                )

    def add_side_model(self, missed, dists, pred_side, gt_side) -> None:
        """Side-view *side model* outcome for one ground-truth side image."""
        self._add_side(missed, dists, pred_side, gt_side, "sd")

    def add_general_side(self, missed, dists, pred_side, gt_side) -> None:
        """*General pose model* used as a side detector, same image."""
        self._add_side(missed, dists, pred_side, gt_side, "gp")


# --- Formatting --------------------------------------------------------------


def _mean(values):
    return sum(values) / len(values) if values else None


def _num(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _rate(count, total, digits=4):
    return "n/a" if not total else f"{count / total:.{digits}f}"


# One combined table so per-tax rows can be streamed under a single header as
# each tax finishes (fixed column widths — no need to know all rows up front).
HEADERS = [
    "scope", "imgs",
    "pose_miss",
    "td_miss",
    "side_miss",
    "larva_miss",
    "td_n", "td_nodet", "td_kp",
    "sd_n", "sd_nodet", "sd_kp", "sd_lr",
    "gp_nodet", "gp_kp", "gp_lr",
]
ALIGNS = [False] + [True] * (len(HEADERS) - 1)
WIDTHS = [max(len(h), 7) for h in HEADERS]
WIDTHS[0] = max(WIDTHS[0], 14)  # scope holds tax_ids / "ALL"


def _fmt_row(cells) -> str:
    return "  ".join(
        f"{cell:>{w}}" if right else f"{cell:<{w}}"
        for cell, w, right in zip(cells, WIDTHS, ALIGNS)
    )


def print_header() -> None:
    print(_fmt_row(HEADERS))
    print("  ".join("-" * w for w in WIDTHS))


def stats_row(scope: str, s: Stats) -> list[str]:
    return [
        scope,
        str(s.n_images),
        _rate(
            (s.labeled[TOP_DOWN] - s.tp[TOP_DOWN])
            + (s.labeled[SIDE] - s.tp[SIDE]),
            s.labeled[TOP_DOWN] + s.labeled[SIDE],
        ),
        _rate(s.labeled[TOP_DOWN] - s.tp[TOP_DOWN], s.labeled[TOP_DOWN]),
        _rate(s.labeled[SIDE] - s.tp[SIDE], s.labeled[SIDE]),
        _rate(s.labeled[LARVA] - s.tp[LARVA], s.labeled[LARVA]),
        str(s.td_total),
        _rate(s.td_missed, s.td_total),
        _num(_mean(s.td_dists), 4),
        str(s.sd_total),
        _rate(s.sd_missed, s.sd_total),
        _num(_mean(s.sd_dists), 4),
        _rate(s.sd_lr_total - s.sd_lr_correct, s.sd_lr_total),
        _rate(s.gp_missed, s.gp_total),
        _num(_mean(s.gp_dists), 4),
        _rate(s.gp_lr_total - s.gp_lr_correct, s.gp_lr_total),
    ]


# --- Args --------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--classification-model", dest="cls_model", type=Path, required=True,
        help="Box-only viewpoint/stage classification model (.pt).",
    )
    parser.add_argument(
        "--pose-model", dest="pose_model", type=Path, required=True,
        help="General F/L/R/B pose model (.pt).",
    )
    parser.add_argument(
        "--side-model", dest="side_model", type=Path, required=True,
        help="Side-view pose model (F, B, wing; class encodes L/R) (.pt).",
    )
    parser.add_argument("--imgsz", type=int, default=768, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.10, help="Min box confidence.")
    parser.add_argument("--device", default="0", help="Inference device: 0, cpu, ...")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Also write everything printed to this file (mirrors the screen).",
    )
    return parser.parse_args()


class _Tee:
    """Write-through stream that fans writes out to several underlying streams."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


# --- Main --------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    for label, model_path in (
        ("classification", args.cls_model),
        ("pose", args.pose_model),
        ("side", args.side_model),
    ):
        if not model_path.exists():
            raise SystemExit(f"{label} model does not exist: {model_path}")

    # Mirror everything printed to --output (if given) as well as the screen.
    output_handle = None
    saved_stdout = sys.stdout
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = open(args.output, "w", encoding="utf-8")
        sys.stdout = _Tee(saved_stdout, output_handle)
    try:
        return _run(args)
    finally:
        sys.stdout = saved_stdout
        if output_handle is not None:
            output_handle.close()


def _run(args) -> int:
    settings, moth_utils = bootstrap_django()

    from ultralytics import YOLO

    images_root = Path(settings.MOTHS_IMAGE_DIR)
    labels_root = Path(settings.MOTHS_LABEL_DIR)
    if not images_root.is_dir():
        raise SystemExit(f"Image directory does not exist: {images_root}")

    print(f"Images: {images_root}")
    print(f"Labels: {labels_root}")
    print(f"Loading classification model: {args.cls_model}")
    print(f"Loading pose model:           {args.pose_model}")
    print(f"Loading side-view pose model: {args.side_model}")
    models = {
        "cls": YOLO(str(args.cls_model)),
        "pose": YOLO(str(args.pose_model)),
        "side": YOLO(str(args.side_model)),
    }

    overall = Stats()

    print()
    print_header()

    for tax in moth_utils.list_tax_ids():
        tax_stats = Stats()
        for image in moth_utils.scan_tax_images(tax):
            filename = image.filename

            label_anns = moth_utils.load_poses(filename)
            # Scope: only images with at least a box label (a hand-labelled
            # object). A stage class alone (e.g. an unboxed Larva sample) is not
            # enough — every evaluated image must have a box to compare against.
            if not label_anns:
                continue
            stage = moth_utils.get_image_class(filename)

            gt_cat = ground_truth_category(stage, label_anns, moth_utils)
            image_path = moth_utils.get_image_path(filename)

            # Step 1 — classification: compare the top predicted object's
            # category to the top labelled object's category.
            classification = pred.classify_top(models, image_path, args)
            pred_cat = None if classification is None else predicted_category(
                classification.cls_id
            )
            overall.add_classification(gt_cat, pred_cat)
            tax_stats.add_classification(gt_cat, pred_cat)

            # Step 2 — pose: run the matching pose model on ground-truth
            # top-down / side images and measure keypoint offset + detection
            # rate (raw keypoints, so this isolates the pose model's accuracy).
            if gt_cat == TOP_DOWN:
                pose_result, pose_index = pred.predict_top(
                    models["pose"], image_path, args
                )
                if pose_result is None:
                    overall.add_top_down_pose(True, [])
                    tax_stats.add_top_down_pose(True, [])
                else:
                    raw = pred.detection_keypoints_raw(pose_result, pose_index)
                    dists = top_down_offsets(label_anns[0], raw)
                    overall.add_top_down_pose(False, dists)
                    tax_stats.add_top_down_pose(False, dists)
            elif gt_cat == SIDE:
                gt_side = gt_wing_side(label_anns[0])

                # (a) The dedicated side-view pose model. Its class gives the
                # predicted wing (L/R) directly.
                side_result, side_index = pred.predict_top(
                    models["side"], image_path, args
                )
                if side_result is None:
                    overall.add_side_model(True, [], None, gt_side)
                    tax_stats.add_side_model(True, [], None, gt_side)
                else:
                    raw = pred.detection_keypoints_raw(side_result, side_index)
                    f_xy = raw[0][:2] if len(raw) > 0 else None
                    b_xy = raw[1][:2] if len(raw) > 1 else None
                    wing_xy = raw[2][:2] if len(raw) > 2 else None
                    dists = side_offsets(label_anns[0], f_xy, b_xy, wing_xy)
                    pred_side = (
                        "R"
                        if pred.detection_class(side_result, side_index) == pred.SIDE_R
                        else "L"
                    )
                    overall.add_side_model(False, dists, pred_side, gt_side)
                    tax_stats.add_side_model(False, dists, pred_side, gt_side)

                # (b) The general F/L/R/B pose model used as a side detector:
                # the wing is whichever of L/R has higher keypoint confidence.
                pose_result, pose_index = pred.predict_top(
                    models["pose"], image_path, args
                )
                if pose_result is None:
                    overall.add_general_side(True, [], None, gt_side)
                    tax_stats.add_general_side(True, [], None, gt_side)
                else:
                    raw = pred.detection_keypoints_raw(pose_result, pose_index)
                    pred_side, wing_xy = general_side_choice(raw)
                    f_xy = raw[0][:2] if len(raw) > 0 else None
                    b_xy = raw[3][:2] if len(raw) > 3 else None
                    dists = side_offsets(label_anns[0], f_xy, b_xy, wing_xy)
                    overall.add_general_side(False, dists, pred_side, gt_side)
                    tax_stats.add_general_side(False, dists, pred_side, gt_side)

            pred.clear_cuda_cache(args.device)

        # Stream this tax's results as soon as it finishes (runs are slow).
        if tax_stats.n_images:
            print(_fmt_row(stats_row(tax, tax_stats)), flush=True)

    if overall.n_images == 0:
        print("No labelled images found.")
        return 1

    print("  ".join("-" * w for w in WIDTHS))
    print(_fmt_row(stats_row("ALL", overall)))

    print(
        "\nClassification: *_miss = (labelled - detected_true) / labelled "
        "(fraction of that category's ground-truth images the classifier failed "
        "to call it; 0 is ideal). pose_miss combines td+side (of all ground-"
        "truth top-down or side images, the fraction not called that same "
        "category) — the headline detection/classification quality number."
        "\nPose (all over ground-truth images of that view): *_n = image count; "
        "*_nodet = fraction with no object detected; *_kp = mean keypoint offset "
        "in normalized image coords (label-visible keypoints)."
        "\n  td = general pose model on top-down images."
        "\n  sd = dedicated side model on side images; sd_lr = fraction of "
        "detections whose predicted wing (L/R, from the model's class) did NOT "
        "match the label (wrong-side rate over the same sd_n images; 0 is ideal)."
        "\n  gp = general pose model used as a side detector on the same side "
        "images, picking the wing by higher keypoint confidence; gp_lr = "
        "fraction of detections whose chosen wing did NOT match the label "
        "(equal/absent confidences count as wrong; 0 is ideal)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
