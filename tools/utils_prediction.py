#!/usr/bin/env python3

"""Shared prediction logic for the moth tooling.

This is the single source of truth for the 3-model prediction pipeline used by
``ultralytics-predict.py`` (writes labels for the whole dataset),
``harvest_top_images.py`` (predicts freshly-downloaded candidates) and
``evaluate_model.py`` (scores the models against hand labels). Keeping it here
guarantees all three see identical behaviour.

The pipeline runs three models (produced by ``prepare_train_data.py``):

* a **classification** model — box-only viewpoint/stage detector with classes
  ``0 top_down · 1 side_view · 2 bottom_up · 3 unclear_pose · 4 top_down_pinned
  · 5 macro · 6 larva``;
* a **general** F/L/R/B 4-keypoint pose model;
* a **side-view** pose model — 3 keypoints (F, B, wing) whose class encodes the
  visible wing (``0 side_view_l · 1 side_view_r``).

For one image the flow is:

1. Classify and keep the top (highest-confidence) object (:func:`classify_top`).
2. If it is **larva** or **adult + macro**: keep the box only (no keypoints).
3. If it is **top-down / bottom-up / unclear** (pinned counts as top-down):
   take all four F/L/R/B points from the general pose model regardless of
   keypoint confidence. If the classification asserts a definite viewpoint and
   the raw geometry agrees, all points are visible (2); otherwise all points
   get visibility 1 (reads back as ``unclear``).
4. If it is **side view**: run the side-view pose model and take its top box; F,
   B and the visible wing are visible (2), the missing wing ``0 0 0``. If the
   side model finds nothing, fall back to the general pose model: F and B are
   visible (2), and of L/R only the higher-confidence wing is kept visible (the
   other zeroed); equal confidences write both with visibility 1.

Keypoint visibility is set entirely by these rules, not by a confidence
threshold — the confidence only picks the wing in the side fallback.

The composable steps (:func:`classify_top`, :func:`predict_pose_for_class`) let
a caller stop early or inspect intermediate output; :func:`run_pipeline` chains
them into the full prediction and :func:`write_prediction` writes the label +
class sidecars the app reads. ``moth_utils`` (the Django ``moths.utils`` module)
is passed in so this file needs no Django import of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

# Classification model class ids and names (must match
# prepare_train_data.CLASSIFICATION_NAMES, in order).
CLASSIFICATION_NAMES = [
    "top_down",
    "side_view",
    "bottom_up",
    "unclear_pose",
    "top_down_pinned",
    "macro",
    "larva",
]
CLS_TOP_DOWN = 0
CLS_SIDE_VIEW = 1
CLS_BOTTOM_UP = 2
CLS_UNCLEAR = 3
CLS_TOP_DOWN_PINNED = 4
CLS_MACRO = 5
CLS_LARVA = 6

# Classes routed to the general pose model (adult, keypoints wanted). Larva and
# macro keep the box only; side view uses the dedicated side model.
CLS_TOP_DOWN_ANY = (CLS_TOP_DOWN, CLS_TOP_DOWN_PINNED)
CLS_BOX_ONLY = (CLS_LARVA, CLS_MACRO)

# Side-view pose model class ids (prepare_train_data.SIDE_POSE_NAMES).
SIDE_L = 0
SIDE_R = 1

# Classification class -> (stage, flags) written into the .class sidecar.
CLASS_TO_STAGE_FLAGS: dict[int, tuple[str, list[str]]] = {
    CLS_TOP_DOWN: ("Adult", []),
    CLS_SIDE_VIEW: ("Adult", []),
    CLS_BOTTOM_UP: ("Adult", []),
    CLS_UNCLEAR: ("Adult", []),
    CLS_TOP_DOWN_PINNED: ("Adult", ["Pinned"]),
    CLS_MACRO: ("Adult", ["Macro"]),
    CLS_LARVA: ("Larva", []),
}

# Class id written on the pose/box label lines (the app's pose files use 0).
LABEL_CLASS_ID = 0

ZERO_KP = (0.0, 0.0, 0)


# --- Small helpers -----------------------------------------------------------


def find_images(root: Path) -> list[Path]:
    """All image files under ``root`` (recursive), sorted by path."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def remove_file(path: Path) -> None:
    """Delete ``path`` if it exists, ignoring errors."""
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass


def clear_cuda_cache(device: str) -> None:
    """Run a GC pass and (on GPU) empty the CUDA cache between inferences."""
    import gc

    gc.collect()
    if device != "cpu":
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- Raw model access --------------------------------------------------------


def predict_top(model, image_path: Path, params) -> tuple[object, int] | tuple[None, None]:
    """Run ``model`` on one image; return ``(result, best_index)`` or ``(None, None)``.

    ``best_index`` is the highest-confidence detection. ``params`` supplies the
    shared inference options ``imgsz`` / ``conf`` / ``device``. Returns
    ``(None, None)`` when the model produced no boxes.
    """
    results = model.predict(
        source=str(image_path),
        imgsz=params.imgsz,
        conf=params.conf,
        max_det=1,
        device=params.device,
        batch=1,
        stream=False,
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None, None
    best_index = int(result.boxes.conf.argmax().item())
    return result, best_index


def detection_box(result, index: int) -> tuple[float, float, float, float]:
    """Return the clamped ``(cx, cy, w, h)`` of a detection (normalized)."""
    cx, cy, width, height = result.boxes.xywhn[index].detach().cpu().tolist()
    return clamp01(cx), clamp01(cy), clamp01(width), clamp01(height)


def detection_class(result, index: int) -> int:
    return int(result.boxes.cls[index].item())


def detection_keypoints_raw(result, index: int) -> list[tuple[float, float, float | None]]:
    """Return ``[(x, y, conf), ...]`` for a detection's keypoints (clamped x/y).

    Positions are used as-is regardless of confidence; ``conf`` is ``None`` when
    the model reports no per-keypoint confidence. The caller decides visibility.
    """
    if result.keypoints is None or len(result.keypoints) <= index:
        return []
    xy = result.keypoints.xyn[index].detach().cpu().tolist()
    conf = None
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[index].detach().cpu().tolist()
    out: list[tuple[float, float, float | None]] = []
    for i, (x, y) in enumerate(xy):
        c = None if conf is None else float(conf[i])
        out.append((clamp01(float(x)), clamp01(float(y)), c))
    return out


# --- Label line formatting ---------------------------------------------------


def box_line(box: tuple[float, float, float, float]) -> str:
    """A box-only YOLO line: ``<class> cx cy w h`` (no keypoints)."""
    cx, cy, w, h = box
    return f"{LABEL_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def pose_line(box: tuple[float, float, float, float], keypoints) -> str:
    """A full pose YOLO line: ``<class> cx cy w h (x y v) * n``."""
    cx, cy, w, h = box
    parts = [str(LABEL_CLASS_ID), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for x, y, visibility in keypoints:
        parts += [f"{x:.6f}", f"{y:.6f}", str(int(visibility))]
    return " ".join(parts)


# --- Keypoint reworking (visibility rules) -----------------------------------


def _geometric_pose(frlb, moth_utils) -> str:
    """Viewpoint of raw F/L/R/B points, evaluated as if all four are visible."""
    kps = [moth_utils.Keypoint(x=x, y=y, visibility=2) for x, y, _c in frlb[:4]]
    ann = moth_utils.Annotation(
        class_id=LABEL_CLASS_ID, cx=0.0, cy=0.0, width=0.0, height=0.0, keypoints=kps
    )
    return moth_utils.classify_annotation(ann)


def _classified_target(cls_id: int, moth_utils) -> str | None:
    """The definite viewpoint a classification class asserts (or None if unclear)."""
    if cls_id in CLS_TOP_DOWN_ANY:
        return moth_utils.POSE_TOP_DOWN
    if cls_id == CLS_BOTTOM_UP:
        return moth_utils.POSE_BOTTOM_UP
    return None  # unclear (or anything else) has no asserted viewpoint


def build_full_pose(pose_result, pose_index: int, cls_id: int, moth_utils):
    """F/L/R/B pose for a top-down/bottom-up/unclear classification.

    All four points come from the pose model regardless of confidence. If the
    classification asserts a definite viewpoint (top-down/pinned or bottom-up)
    and the raw geometry agrees, every point is visible (2). Otherwise (unclear
    classification, or geometry that disagrees) every point is written with
    visibility 1, which classifies as ``unclear`` downstream. Returns
    ``(box, [F, L, R, B])`` or ``None`` when there are not four keypoints.
    """
    frlb = detection_keypoints_raw(pose_result, pose_index)
    if len(frlb) < 4:
        return None
    target = _classified_target(cls_id, moth_utils)
    visibility = 2 if target is not None and _geometric_pose(frlb, moth_utils) == target else 1
    keypoints = [(x, y, visibility) for x, y, _c in frlb[:4]]
    return detection_box(pose_result, pose_index), keypoints


def build_side_from_side_model(side_result, side_index: int):
    """Rework a side-model detection (F, B, wing) into generic F/L/R/B form.

    F, B and the wing are all visible (2) regardless of confidence; the class
    picks the wing (L or R) and the other wing is ``0 0 0``. Returns
    ``(box, [F, L, R, B])`` or ``None`` when the detection lacks three keypoints.
    """
    raw = detection_keypoints_raw(side_result, side_index)
    if len(raw) < 3:
        return None
    (fx, fy, _), (bx, by, _), (wx, wy, _) = raw[0], raw[1], raw[2]
    front, back, wing = (fx, fy, 2), (bx, by, 2), (wx, wy, 2)
    if detection_class(side_result, side_index) == SIDE_R:
        left, right = ZERO_KP, wing
    else:
        left, right = wing, ZERO_KP
    return detection_box(side_result, side_index), [front, left, right, back]


def build_side_from_pose_model(pose_result, pose_index: int):
    """Side pose fallback from the general model when the side model finds nothing.

    F and B are visible (2) regardless of confidence. The wing kept is whichever
    of L/R has the higher confidence (visible 2, the other zeroed). If the two
    wing confidences are equal (or unavailable), both wings are written with
    visibility 1 so the result classifies as ``unclear``. Returns
    ``(box, [F, L, R, B])`` or ``None`` when there are not four keypoints.
    """
    frlb = detection_keypoints_raw(pose_result, pose_index)
    if len(frlb) < 4:
        return None
    (fx, fy, _), (lx, ly, lc), (rx, ry, rc), (bx, by, _) = frlb[:4]
    front, back = (fx, fy, 2), (bx, by, 2)
    if lc is not None and rc is not None and lc > rc:
        left, right = (lx, ly, 2), ZERO_KP
    elif lc is not None and rc is not None and rc > lc:
        left, right = ZERO_KP, (rx, ry, 2)
    else:
        # Equal (or missing) confidence: keep both but mark uncertain -> unclear.
        left, right = (lx, ly, 1), (rx, ry, 1)
    return detection_box(pose_result, pose_index), [front, left, right, back]


# --- Composable pipeline steps ----------------------------------------------


@dataclass
class Classification:
    """Top classification-model object, reworked for the .class sidecar."""

    cls_id: int
    cls_name: str
    box: tuple[float, float, float, float]
    stage: str
    flags: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """Full pipeline outcome for one image (nothing written yet).

    ``status`` is one of ``no_detection`` (classifier found nothing),
    ``box_only`` (larva/macro — box, no keypoints), ``pose`` (general model),
    ``side_pose`` (side model), ``side_from_pose`` (side fallback via the
    general model) or ``pose_empty`` (adult pose class but no usable keypoints).
    """

    status: str
    classification: Classification | None = None
    box: tuple[float, float, float, float] | None = None
    keypoints: list[tuple[float, float, int]] | None = None

    @property
    def label_line(self) -> str:
        """The YOLO label line to write (``""`` when there is no object)."""
        if self.box is not None and self.keypoints is not None:
            return pose_line(self.box, self.keypoints)
        if self.status == "box_only" and self.box is not None:
            return box_line(self.box)
        return ""


def classify_top(models, image_path: Path, params) -> Classification | None:
    """Run the classification model; return its top object, or ``None``."""
    result, index = predict_top(models["cls"], image_path, params)
    if result is None:
        return None
    cls_id = detection_class(result, index)
    stage, flags = CLASS_TO_STAGE_FLAGS.get(cls_id, ("Adult", []))
    cls_name = (
        CLASSIFICATION_NAMES[cls_id]
        if 0 <= cls_id < len(CLASSIFICATION_NAMES)
        else str(cls_id)
    )
    return Classification(
        cls_id=cls_id,
        cls_name=cls_name,
        box=detection_box(result, index),
        stage=stage,
        flags=list(flags),
    )


def predict_pose_for_class(models, image_path: Path, params, cls_id: int, moth_utils):
    """Run the pose model(s) for an adult pose classification.

    Handles the side-view branch (side model, then general fallback) and the
    general branch (top-down / bottom-up / unclear). Returns
    ``(status, box, keypoints)`` where ``status`` is ``pose`` / ``side_pose`` /
    ``side_from_pose`` / ``pose_empty`` (the last with ``box``/``keypoints``
    ``None``). Not for larva/macro (box-only) — the caller filters those first.
    """
    if cls_id == CLS_SIDE_VIEW:
        side_result, side_index = predict_top(models["side"], image_path, params)
        if side_result is not None:
            reworked = build_side_from_side_model(side_result, side_index)
            if reworked is not None:
                box, keypoints = reworked
                return "side_pose", box, keypoints
        # Side model found nothing usable: fall back to the general pose model.
        pose_result, pose_index = predict_top(models["pose"], image_path, params)
        reworked = (
            build_side_from_pose_model(pose_result, pose_index)
            if pose_result is not None
            else None
        )
        if reworked is None:
            return "pose_empty", None, None
        box, keypoints = reworked
        return "side_from_pose", box, keypoints

    # Top-down / bottom-up / unclear: the general F/L/R/B pose model.
    pose_result, pose_index = predict_top(models["pose"], image_path, params)
    reworked = (
        build_full_pose(pose_result, pose_index, cls_id, moth_utils)
        if pose_result is not None
        else None
    )
    if reworked is None:
        return "pose_empty", None, None
    box, keypoints = reworked
    return "pose", box, keypoints


def run_pipeline(models, image_path: Path, params, moth_utils) -> Prediction:
    """Full prediction for one image (classification + pose), writing nothing."""
    classification = classify_top(models, image_path, params)
    if classification is None:
        return Prediction(status="no_detection")

    cls_id = classification.cls_id
    if cls_id in CLS_BOX_ONLY:
        return Prediction(
            status="box_only", classification=classification, box=classification.box
        )

    status, box, keypoints = predict_pose_for_class(
        models, image_path, params, cls_id, moth_utils
    )
    return Prediction(
        status=status, classification=classification, box=box, keypoints=keypoints
    )


def write_prediction(prediction: Prediction, output_txt: Path, output_class: Path, moth_utils) -> None:
    """Write a :class:`Prediction`'s label ``.txt`` and ``.class`` sidecars.

    The label file always exists (empty when there is no object). The ``.class``
    file records the predicted stage/flags plus a ``pose:<raw class>`` line; it
    is removed only when nothing was classified.
    """
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    line = prediction.label_line
    output_txt.write_text(line + "\n" if line else "", encoding="utf-8")

    if prediction.classification is None:
        remove_file(output_class)
        return
    c = prediction.classification
    moth_utils._write_class_file(output_class, c.stage, list(c.flags), pose=c.cls_name)
