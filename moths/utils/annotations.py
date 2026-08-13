"""YOLO label/pose/prediction parsing, loading and pose classification."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from django.conf import settings

from .paths import (
    _tax_subdir,
    image_basename,
)
from .classes import (
    _read_class_file,
    _read_class_pose,
    get_class_path,
)


# --- YOLO pose label handling -------------------------------------------------

# Human-readable keypoint names, keyed by 1-based keypoint index.
# TODO: source these from the dataset/model config once available.
KEYPOINT_LABELS = {
    1: "F",
    2: "L",
    3: "R",
    4: "B",
}


@dataclass(frozen=True)
class Keypoint:
    x: float  # normalized 0..1
    y: float  # normalized 0..1
    visibility: int  # 0 = unlabeled, 1 = labeled/occluded, 2 = visible


@dataclass(frozen=True)
class Annotation:
    """A single YOLO-pose object: class id, bounding box and keypoints.

    Coordinates are normalized (0..1). The bounding box is stored as its
    center (``cx``, ``cy``) plus ``width``/``height``.
    """

    class_id: int
    cx: float
    cy: float
    width: float
    height: float
    keypoints: list[Keypoint]

    # Convenience percentage accessors for CSS overlay positioning.
    @property
    def left_pct(self) -> float:
        return (self.cx - self.width / 2) * 100

    @property
    def top_pct(self) -> float:
        return (self.cy - self.height / 2) * 100

    @property
    def width_pct(self) -> float:
        return self.width * 100

    @property
    def height_pct(self) -> float:
        return self.height * 100


def get_label_dir() -> Path:
    """Directory holding the label ``.txt`` files."""
    return Path(settings.MOTHS_LABEL_DIR)


def get_label_path(image_filename: str) -> Path:
    """Return the label file path corresponding to an image filename."""
    name = image_basename(image_filename)
    return _tax_subdir(get_label_dir(), name) / (Path(name).stem + ".txt")


def parse_label_line(line: str) -> Annotation | None:
    """Parse one YOLO-pose line into an ``Annotation`` (or ``None``).

    Expected layout::

        class_id cx cy w h  (kp_x kp_y kp_v) * n
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
    except ValueError:
        return None

    keypoints: list[Keypoint] = []
    kp_values = parts[5:]
    for i in range(0, len(kp_values) - 2, 3):
        try:
            x = float(kp_values[i])
            y = float(kp_values[i + 1])
            visibility = int(float(kp_values[i + 2]))
        except ValueError:
            continue
        keypoints.append(Keypoint(x=x, y=y, visibility=visibility))

    return Annotation(
        class_id=class_id, cx=cx, cy=cy, width=w, height=h, keypoints=keypoints
    )


# --- Annotation file parsing -------------------------------------------------

def _read_annotations(path: Path) -> list[Annotation]:
    """Parse all annotations from a YOLO-pose label/prediction file."""
    if not path.is_file():
        return []
    annotations: list[Annotation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        annotation = parse_label_line(line)
        if annotation is not None:
            annotations.append(annotation)
    return annotations


def read_annotations_file(path) -> list[Annotation]:
    """Public helper: parse a YOLO-pose label/prediction file at ``path``."""
    return _read_annotations(Path(path))


def load_annotations(image_filename: str) -> list[Annotation]:
    """Load all annotations from the label file for the given image."""
    return _read_annotations(get_label_path(image_filename))


# --- Predictions & pose orientation ------------------------------------------

# Pose group identifiers.
POSE_TOP_DOWN = "top_down"
POSE_SIDE = "side"
POSE_BOTTOM_UP = "bottom_up"
POSE_UNCLEAR = "unclear"
POSE_NONE = "none"


def get_pose_dir() -> Path:
    """Directory holding the YOLO-pose files used for pose classification.

    Same as :func:`get_label_dir` — the editable hand labels double as the pose
    source (formerly a separate ``MOTHS_POSE_DIR``, now folded into
    ``MOTHS_LABEL_DIR``).
    """
    return get_label_dir()


def get_pose_path(image_filename: str) -> Path:
    """Return the pose file path corresponding to an image filename."""
    name = image_basename(image_filename)
    return _tax_subdir(get_pose_dir(), name) / (Path(name).stem + ".txt")


def load_poses(image_filename: str) -> list[Annotation]:
    """Load all pose annotations for the given image (for classification)."""
    return _read_annotations(get_pose_path(image_filename))


def get_prediction_dir() -> Path:
    """Directory holding read-only model prediction files (YOLO-pose format)."""
    return Path(settings.MOTHS_PREDICTION_DIR)


def get_prediction_path(image_filename: str) -> Path:
    """Return the prediction file path corresponding to an image filename."""
    name = image_basename(image_filename)
    return _tax_subdir(get_prediction_dir(), name) / (Path(name).stem + ".txt")


def get_prediction_class_path(image_filename: str) -> Path:
    """Path to the predicted ``.class`` sidecar in the prediction directory.

    The prediction pipeline writes a stage/flags ``.class`` file next to each
    prediction (same format as the hand ``classes/`` files) so the species view
    can fall back to it when an image has no hand classification.
    """
    name = image_basename(image_filename)
    return _tax_subdir(get_prediction_dir(), name) / (Path(name).stem + ".class")


def get_predicted_pose_class(image_filename: str) -> str | None:
    """Return the raw predicted class name for an image, or ``None`` if absent.

    Reads the ``pose:`` line the prediction pipeline stores in the predicted
    ``.class`` file (the verbatim classifier class name, e.g. ``top_down`` /
    ``macro`` / ``larva``, before it is collapsed into stage/flags).
    """
    return _read_class_pose(get_prediction_class_path(image_filename))


def get_predicted_class_and_flags(image_filename: str) -> tuple[str | None, list[str]]:
    """Return ``(stage, flags)`` from the predicted ``.class`` sidecar.

    Reads only ``MOTHS_PREDICTION_DIR`` (never the hand ``classes/`` file), so
    the caller can surface what the model predicted regardless of any hand
    classification. ``(None, [])`` when there is no predicted ``.class``.
    """
    return _read_class_file(get_prediction_class_path(image_filename))


def get_class_and_flags_with_source(image_filename: str) -> tuple[str | None, list[str], str | None]:
    """Return ``(stage, flags, source)``, preferring the hand ``.class`` file.

    Reads the hand ``.class`` in ``MOTHS_LABEL_DIR`` first (``source ==
    "class"``); when that file has neither a stage nor flags, falls back to the
    predicted ``.class`` in
    ``MOTHS_PREDICTION_DIR`` (``source == "prediction"``). ``source`` is ``None``
    when neither has any classification. Only the hand file is authoritative:
    once it carries any stage/flag it is used as-is (predictions are not merged
    in). Used by the species view so predicted stage/flags group images too.
    """
    stage, flags = _read_class_file(get_class_path(image_filename))
    if stage or flags:
        return stage, flags, "class"
    pred_stage, pred_flags = _read_class_file(get_prediction_class_path(image_filename))
    if pred_stage or pred_flags:
        return pred_stage, pred_flags, "prediction"
    return None, [], None


def load_predictions(image_filename: str) -> list[Annotation]:
    """Load read-only reference predictions for the given image, if any."""
    return _read_annotations(get_prediction_path(image_filename))


def load_pose_source(image_filename: str) -> tuple[list[Annotation], str | None]:
    """Return the annotations to use for pose/normalization and their source.

    Prefers the hand labels in ``MOTHS_LABEL_DIR`` (source ``"pose"``); if that
    has no data, falls back to ``MOTHS_PREDICTION_DIR`` (source
    ``"prediction"``). When neither has data, returns ``([], None)``.
    """
    poses = load_poses(image_filename)
    if poses:
        return poses, "pose"
    predictions = load_predictions(image_filename)
    if predictions:
        return predictions, "prediction"
    return [], None


def _pose_source_keypoints(image_filename: str):
    """Return ``(keypoints, source)`` for an image's first pose object.

    ``keypoints`` is a list of ``[x, y, v]`` (rounded, JSON-friendly) or ``None``
    when there is no pose data. Used to detect when the underlying keypoints have
    changed since the pose data was cached.
    """
    annotations, source = load_pose_source(image_filename)
    if not annotations:
        return None, None
    keypoints = [
        [round(kp.x, 6), round(kp.y, 6), kp.visibility]
        for kp in annotations[0].keypoints[:4]
    ]
    return keypoints, source


def _valid_frlb(annotation: Annotation):
    """Return ``(front, left, right, back)`` if all four are labeled, else None."""
    keypoints = annotation.keypoints
    if len(keypoints) < 4:
        return None
    front, left, right, back = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
    if min(front.visibility, left.visibility, right.visibility, back.visibility) <= 0:
        return None
    return front, left, right, back


def classify_annotation(annotation: Annotation) -> str:
    """Classify a single annotation's viewpoint from its F/L/R/B keypoints.

    Uses keypoints F (front), L (left), R (right), B (back) and their
    visibilities (0 absent, 1 partial, 2 visible). Requires F and B present to
    define the axis, otherwise ``none``. Any partially-visible (visibility 1)
    keypoint makes the pose ``unclear``. Then, from the wings (L, R):

    * ``side``     – strictly three points: F, B and exactly one wing visible
      (2), the other absent (0)
    * ``unclear``  – neither wing is fully visible (2)
    * otherwise the F→B line splits the plane and the side L/R fall on decides:
      ``top_down`` (L left of F→B, R right), ``bottom_up`` (swapped); if both
      wings land on the same side (or on the line) it is ``unclear``.

    Side of the F→B line uses cross((B-F), (P-F)); with image coords (y down),
    a point left of the F→B heading has a negative cross.
    """
    keypoints = annotation.keypoints
    if len(keypoints) < 4:
        return POSE_NONE
    front, left, right, back = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
    if front.visibility <= 0 or back.visibility <= 0:
        return POSE_NONE

    # A partially-visible (yellow) keypoint means we can't trust the geometry.
    if any(kp.visibility == 1 for kp in (front, left, right, back)):
        return POSE_UNCLEAR

    lv, rv = left.visibility, right.visibility
    # No fully-visible wing: orientation can't be determined.
    if lv != 2 and rv != 2:
        return POSE_UNCLEAR
    # One wing visible, the other absent: a side view.
    if (lv == 2 and rv == 0) or (lv == 0 and rv == 2):
        return POSE_SIDE

    # Both wings present (at least one fully visible): original geometry.
    def side(point: Keypoint) -> float:
        return (back.x - front.x) * (point.y - front.y) - (
            back.y - front.y
        ) * (point.x - front.x)

    cl, cr = side(left), side(right)
    if cl < 0 and cr > 0:
        return POSE_BOTTOM_UP
    if cl > 0 and cr < 0:
        return POSE_TOP_DOWN
    # Both wings on the same side of F→B (or on the line): ambiguous.
    return POSE_UNCLEAR


def classify_pose(image_filename: str) -> str:
    """Classify an image's viewpoint from its first pose object.

    Uses ``MOTHS_LABEL_DIR`` data, falling back to ``MOTHS_PREDICTION_DIR``. The
    classification is source-independent: prediction files already encode the
    model's uncertainty through keypoint visibility (uncertain points are stored
    with visibility 1, which reads as ``unclear``), so predictions and hand
    labels are classified the same way. Whether the data came from a prediction
    is surfaced separately (blue border + "from prediction" note), not here.
    """
    annotations, _source = load_pose_source(image_filename)
    if not annotations:
        return POSE_NONE
    return classify_annotation(annotations[0])
