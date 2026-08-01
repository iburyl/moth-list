"""Helpers for discovering and parsing moth image files.

Images live outside this repo and follow the naming convention::

    {tax_id}_observation_{obs_id}_photo_{photo_id}.{ext}

Files are organized into one subfolder per ``tax_id`` under each base directory
(images, labels, classes and the thumbnail/metric cache), e.g.::

    <images>/<tax_id>/<tax_id>_observation_..._photo_....jpg
    <labels>/<tax_id>/<tax_id>_observation_..._photo_....txt

The filenames themselves are never changed and still begin with the ``tax_id``,
so the correct subfolder for any file is derived from its own name.
"""

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

FILENAME_RE = re.compile(
    r"^(?P<tax_id>.+?)_observation_(?P<obs_id>.+?)"
    r"_photo_(?P<photo_id>.+?)\.(?P<ext>[^.]+)$"
)


@dataclass(frozen=True)
class MothImage:
    filename: str
    tax_id: str
    obs_id: str
    photo_id: str
    ext: str


def get_image_dir() -> Path:
    """Return the configured directory that holds the training images."""
    return Path(settings.MOTHS_IMAGE_DIR)


# --- Taxonomy names (tax_id -> family / species / name) -----------------------

# Cache of the parsed names CSV, invalidated by the file's mtime.
_NAMES_CACHE: dict = {"path": None, "mtime": None, "data": {}}
_EMPTY_NAME_INFO = {"family": "", "species": "", "name": ""}


def get_names_csv_path() -> Path:
    """Return the configured path to the taxonomy names CSV."""
    return Path(settings.MOTHS_NAMES_CSV)


def load_names() -> dict[str, dict]:
    """Return ``{tax_id: {family, species, name}}`` parsed from the names CSV.

    The result is cached and only re-read when the file changes. A missing or
    unreadable file yields an empty mapping (callers fall back to the raw id).
    """
    path = get_names_csv_path()
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    if _NAMES_CACHE["path"] == key and _NAMES_CACHE["mtime"] == mtime:
        return _NAMES_CACHE["data"]

    data: dict[str, dict] = {}
    if mtime is not None:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    tax_id = (row.get("id") or "").strip()
                    if not tax_id:
                        continue
                    data[tax_id] = {
                        "family": (row.get("family") or "").strip(),
                        "species": (row.get("species") or "").strip(),
                        "name": (row.get("name") or "").strip(),
                    }
        except OSError:
            data = {}

    _NAMES_CACHE.update(path=key, mtime=mtime, data=data)
    return data


def get_name_info(tax_id) -> dict:
    """Return ``{family, species, name}`` for a tax_id (empty strings if unknown)."""
    return load_names().get(str(tax_id), _EMPTY_NAME_INFO)


# --- iNaturalist observation metadata ----------------------------------------

# Single-slot cache of the last-loaded observations file, keyed by path+mtime.
_OBSERVATIONS_CACHE: dict = {"path": None, "mtime": None, "data": {}}


def get_observations_path(tax_id: str) -> Path:
    """Path to a tax_id's downloaded observation metadata JSON."""
    return get_image_dir() / f"{tax_id}_observations.json"


def load_observations(tax_id: str) -> dict[str, dict]:
    """Return ``{observation_id: item}`` from ``{tax_id}_observations.json``.

    Cached and re-read only when the file changes. Missing/unreadable → ``{}``.
    """
    path = get_observations_path(tax_id)
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    if _OBSERVATIONS_CACHE["path"] == key and _OBSERVATIONS_CACHE["mtime"] == mtime:
        return _OBSERVATIONS_CACHE["data"]

    data: dict[str, dict] = {}
    if mtime is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raw = None
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("observation_id") is not None:
                    data[str(item["observation_id"])] = item

    _OBSERVATIONS_CACHE.update(path=key, mtime=mtime, data=data)
    return data


def get_observation_info(image_filename: str) -> dict | None:
    """Return the observation metadata for an image, or ``None`` if unavailable."""
    parsed = parse_filename(image_basename(image_filename))
    if parsed is None:
        return None
    return load_observations(parsed.tax_id).get(str(parsed.obs_id))


def parse_filename(filename: str) -> MothImage | None:
    """Parse a filename into a ``MothImage`` or ``None`` if it doesn't match."""
    match = FILENAME_RE.match(filename)
    if not match:
        return None
    return MothImage(filename=filename, **match.groupdict())


def image_basename(image_filename: str) -> str:
    """Return just the file's basename (defends against stray path segments)."""
    return Path(image_filename).name


def tax_id_for_file(image_filename: str) -> str | None:
    """The ``tax_id`` an image (or one of its sidecar files) belongs to.

    Every data file keeps the original image name (``{tax_id}_observation_...``)
    optionally followed by extra suffixes (``.txt``, ``.class``,
    ``.norm.jpg`` ...). ``parse_filename`` recovers the leading ``tax_id`` in all
    of those cases, since it is anchored before ``_observation_``.
    """
    parsed = parse_filename(image_basename(image_filename))
    return parsed.tax_id if parsed else None


def _tax_subdir(base: Path, image_filename: str) -> Path:
    """Return ``base/<tax_id>`` for a file, or ``base`` if the name is unparseable.

    Data is organized into one subfolder per ``tax_id`` under each base
    directory; the ``tax_id`` is derived from the filename itself.
    """
    tax_id = tax_id_for_file(image_filename)
    return base / tax_id if tax_id else base


def get_image_path(image_filename: str) -> Path:
    """Absolute path to an image inside its ``tax_id`` subfolder."""
    name = image_basename(image_filename)
    return _tax_subdir(get_image_dir(), name) / name


def scan_images() -> list[MothImage]:
    """Return every parseable image under the image directory, sorted by name.

    Images live in per-``tax_id`` subfolders, so the tree is walked recursively.
    """
    image_dir = get_image_dir()
    if not image_dir.is_dir():
        return []
    images = []
    for entry in image_dir.rglob("*"):
        if not entry.is_file():
            continue
        parsed = parse_filename(entry.name)
        if parsed is not None:
            images.append(parsed)
    images.sort(key=lambda image: image.filename)
    return images


def group_by_tax_id() -> dict[str, list[MothImage]]:
    """Group all discovered images by their ``tax_id``, sorted by tax_id."""
    groups: dict[str, list[MothImage]] = {}
    for image in scan_images():
        groups.setdefault(image.tax_id, []).append(image)
    return dict(sorted(groups.items()))


# --- Stage classification -----------------------------------------------------

# Life-cycle stages an image can be classified as.
STAGES = ["Egg", "Larva", "Pupa", "Adult"]

# Optional, independent boolean flags an image can carry (order = display order).
FLAGS = ["Pinned", "Macro", "Damaged"]

# Prefix of the flags line inside a ``.class`` file.
_FLAGS_PREFIX = "flags:"


def get_class_dir() -> Path:
    """Directory holding per-image stage classification files."""
    return Path(settings.MOTHS_CLASS_DIR)


def get_class_path(image_filename: str) -> Path:
    """Path to the ``.class`` file for an image (same name, ``.class`` ext)."""
    name = image_basename(image_filename)
    return _tax_subdir(get_class_dir(), name) / (Path(name).stem + ".class")


def _read_class_file(path: Path) -> tuple[str | None, list[str]]:
    """Parse a ``.class`` file into ``(stage, flags)``.

    Format: the stage on the first line (blank when only flags are set),
    optionally followed by a ``flags:Pinned,Macro`` line. Legacy files that hold
    only a bare stage string still parse (first line = stage, no flags). Unknown
    flag names are dropped; the returned flags follow :data:`FLAGS` order.
    """
    if not path.is_file():
        return None, []
    lines = path.read_text(encoding="utf-8").splitlines()
    stage = lines[0].strip() if lines else ""
    present = set()
    for line in lines[1:]:
        line = line.strip()
        if line.startswith(_FLAGS_PREFIX):
            present = {f.strip() for f in line[len(_FLAGS_PREFIX):].split(",")}
    flags = [f for f in FLAGS if f in present]
    return (stage or None), flags


def _write_class_file(path: Path, stage: str | None, flags: list[str]) -> None:
    """Write ``stage`` + ``flags`` to a ``.class`` file (deleting it if empty)."""
    flags = [f for f in FLAGS if f in set(flags)]
    if not stage and not flags:
        if path.is_file():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    body = stage or ""
    if flags:
        body += "\n" + _FLAGS_PREFIX + ",".join(flags)
    path.write_text(body, encoding="utf-8")


def get_image_class(image_filename: str) -> str | None:
    """Return the stored stage for an image, or ``None`` if unclassified."""
    stage, _flags = _read_class_file(get_class_path(image_filename))
    return stage


def set_image_class(image_filename: str, stage: str) -> None:
    """Write the stage classification for an image, preserving any flags."""
    path = get_class_path(image_filename)
    _stage, flags = _read_class_file(path)
    _write_class_file(path, stage, flags)


def clear_image_class(image_filename: str) -> None:
    """Clear the stage classification for an image, preserving any flags.

    Deletes the ``.class`` file only when no flags remain.
    """
    path = get_class_path(image_filename)
    _stage, flags = _read_class_file(path)
    _write_class_file(path, None, flags)


def get_image_flags(image_filename: str) -> list[str]:
    """Return the flags set on an image (subset of :data:`FLAGS`)."""
    _stage, flags = _read_class_file(get_class_path(image_filename))
    return flags


def get_class_and_flags(image_filename: str) -> tuple[str | None, list[str]]:
    """Return ``(stage, flags)`` for an image in a single ``.class`` file read."""
    return _read_class_file(get_class_path(image_filename))


def set_image_flags(image_filename: str, flags: list[str]) -> list[str]:
    """Store the flags for an image, preserving its stage. Returns them back."""
    path = get_class_path(image_filename)
    stage, _flags = _read_class_file(path)
    kept = [f for f in FLAGS if f in set(flags)]
    _write_class_file(path, stage, kept)
    return kept


# --- Starred observations -----------------------------------------------------


def get_starred_path(tax_id: str) -> Path:
    """Path to the per-tax starred-observations file (at the classes root)."""
    return get_class_dir() / f"{tax_id}_starred.txt"


def load_starred(tax_id: str) -> set[str]:
    """Return the set of starred image filenames for a tax_id."""
    path = get_starred_path(tax_id)
    if not path.is_file():
        return set()
    try:
        return {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except OSError:
        return set()


def is_image_starred(image_filename: str) -> bool:
    """Return whether an image (observation) is starred."""
    tax_id = tax_id_for_file(image_filename)
    if not tax_id:
        return False
    return image_basename(image_filename) in load_starred(tax_id)


def set_image_starred(image_filename: str, starred: bool) -> bool:
    """Star or unstar an image; rewrites the tax's starred file (sorted).

    Returns the resulting starred state.
    """
    name = image_basename(image_filename)
    tax_id = tax_id_for_file(name)
    if not tax_id:
        return False
    current = load_starred(tax_id)
    if starred:
        current.add(name)
    else:
        current.discard(name)
    path = get_starred_path(tax_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if current:
            path.write_text("\n".join(sorted(current)) + "\n", encoding="utf-8")
        elif path.is_file():
            path.write_text("", encoding="utf-8")
    except OSError:
        pass
    return starred


# --- Thumbnail cache ----------------------------------------------------------


def get_thumbnail_dir() -> Path:
    """Return the directory where cached thumbnails are stored."""
    return Path(settings.MOTHS_THUMBNAIL_DIR)


def get_or_create_thumbnail(image_filename: str) -> Path | None:
    """Return the cached thumbnail path for an image, generating it if needed.

    The thumbnail keeps the original filename and is stored under
    ``MOTHS_THUMBNAIL_DIR/<tax_id>/``. It is (re)generated when missing or older
    than the source image. Returns ``None`` if the source image can't be found.
    """
    name = image_basename(image_filename)
    image_dir = get_image_dir().resolve()
    src = get_image_path(name).resolve()
    if image_dir not in src.parents or not src.is_file():
        return None

    thumb_dir = get_thumbnail_dir().resolve()
    # Mirror the tax_id subfolder; use only the basename to avoid traversal.
    dest = _tax_subdir(thumb_dir, name) / name

    if dest.is_file() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest

    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.thumbnail(tuple(settings.MOTHS_THUMBNAIL_SIZE))
        save_kwargs: dict = {}
        if dest.suffix.lower() in (".jpg", ".jpeg"):
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            save_kwargs = {"quality": 85, "optimize": True}
        im.save(dest, **save_kwargs)
    return dest


def get_image_size(image_filename: str) -> tuple[int, int] | None:
    """Return the original ``(width, height)`` of an image in pixels.

    PIL reads the dimensions from the image header without decoding pixels, so
    this is cheap and no sidecar cache is kept. The value is persisted per image
    in ``<tax_id>_pose_data.json`` (see :func:`compute_pose_row`) alongside the
    scores. Returns ``None`` if the source image can't be found.
    """
    name = image_basename(image_filename)
    image_dir = get_image_dir().resolve()
    src = get_image_path(name).resolve()
    if image_dir not in src.parents or not src.is_file():
        return None

    from PIL import Image

    with Image.open(src) as im:
        width, height = im.size
    return width, height


def find_image(filename: str) -> MothImage | None:
    """Return the ``MothImage`` for ``filename`` if it exists in the image dir."""
    name = image_basename(filename)
    if parse_filename(name) is None:
        return None
    file_path = get_image_path(name).resolve()
    if get_image_dir().resolve() not in file_path.parents or not file_path.is_file():
        return None
    return parse_filename(name)


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

    Uses ``MOTHS_LABEL_DIR`` data, falling back to ``MOTHS_PREDICTION_DIR``. When
    the pose came from a prediction, only ``top_down`` / ``unclear`` / ``none``
    are trusted: predicted ``side`` and ``bottom_up`` are unreliable so far and
    are collapsed into ``unclear``.
    """
    annotations, source = load_pose_source(image_filename)
    if not annotations:
        return POSE_NONE
    pose = classify_annotation(annotations[0])
    if source == "prediction" and pose in (POSE_SIDE, POSE_BOTTOM_UP):
        return POSE_UNCLEAR
    return pose


def annotation_symmetry(
    annotation: Annotation,
    width: float = 1.0,
    height: float = 1.0,
) -> float | None:
    """L/R symmetry of an annotation about its F→B axis.

    ``L_mir`` is L reflected across the line through F and B. The metric is
    ``||R - L_mir|| / radius`` where ``radius`` is the max distance from C (the
    F/B midpoint) to any keypoint — the same length the normalized crop is
    scaled by (the reference-circle radius). So the value reads directly off the
    normalized view: it is 0 when R is exactly the mirror of L and grows as the
    pair becomes less symmetric, expressed as a fraction of the crop radius.

    Normalizing by the crop radius (rather than ``||R - L||``) keeps the metric
    consistent with what the eye sees: it no longer shrinks just because the
    wings are spread wide (large L↔R) or inflates when they sit close to the
    body.

    Keypoints are stored as fractions of the image width/height, so ``width``
    and ``height`` (pixels) are applied first to work in isotropic pixel space —
    matching ``compute_normalization`` and the normalized overlay. Without them
    a non-square image distorts the reflection and distances. The defaults of
    ``1.0`` keep the (relative) fraction-space behavior for callers that don't
    have the image size. Returns ``None`` when it can't be computed (missing
    keypoints or degenerate geometry).
    """
    kps = _valid_frlb(annotation)
    if kps is None:
        return None
    front, left, right, back = kps

    fx, fy = front.x * width, front.y * height
    lx, ly = left.x * width, left.y * height
    rx, ry = right.x * width, right.y * height
    bx, by = back.x * width, back.y * height

    dx, dy = bx - fx, by - fy
    denom = dx * dx + dy * dy
    if denom == 0:
        return None

    # Reflect L across the F→B line.
    t = ((lx - fx) * dx + (ly - fy) * dy) / denom
    proj_x, proj_y = fx + t * dx, fy + t * dy
    mirror_x, mirror_y = 2 * proj_x - lx, 2 * proj_y - ly

    # Reference length: max distance from C (F/B midpoint) to any keypoint,
    # matching the normalized crop's scale (see compute_normalization).
    cx, cy = (fx + bx) / 2, (fy + by) / 2
    radius = max(
        math.hypot(px - cx, py - cy)
        for px, py in ((fx, fy), (lx, ly), (rx, ry), (bx, by))
    )
    if radius == 0:
        return None
    return math.hypot(rx - mirror_x, ry - mirror_y) / radius


def pose_symmetry_metric(image_filename: str) -> float | None:
    """L/R symmetry of an image's first pose object (see ``annotation_symmetry``).

    Passes the original image's pixel dimensions so the metric is computed in
    isotropic pixel space (non-square images would otherwise skew it).
    """
    annotations, _source = load_pose_source(image_filename)
    if not annotations:
        return None
    size = get_image_size(image_filename)
    if size is None:
        return None
    width, height = size
    return annotation_symmetry(annotations[0], width, height)


def pose_pixel_span(image_filename: str) -> float | None:
    """Largest pixel span among the L↔R and F↔B keypoint pairs (first prediction).

    Normalized keypoint distances are scaled by the original image size, so the
    result approximates how many pixels of detail the pose spans. Returns
    ``None`` when it can't be computed (no pose, missing keypoints, or the
    original image size is unavailable).
    """
    annotations, _source = load_pose_source(image_filename)
    if not annotations:
        return None
    keypoints = annotations[0].keypoints
    if len(keypoints) < 4:
        return None
    front, left, right, back = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
    if min(front.visibility, left.visibility, right.visibility, back.visibility) <= 0:
        return None

    size = get_image_size(image_filename)
    if size is None:
        return None
    width, height = size

    lr = math.hypot((right.x - left.x) * width, (right.y - left.y) * height)
    fb = math.hypot((back.x - front.x) * width, (back.y - front.y) * height)
    return max(lr, fb)


# Neutral grey used to fill areas exposed by rotation/cropping.
NORM_FILL = (128, 128, 128)
# Fraction of the crop side at which the farthest keypoint sits: 0.4 = the
# radius of an 80%-diameter reference circle.
NORM_CIRCLE_RADIUS = 0.4
# Top-down/bottom-up multiplier applied to the max center→keypoint distance to
# size the crop (1 / NORM_CIRCLE_RADIUS, so the farthest point lands on the
# circle).
NORM_CROP_SCALE = 1.0 / NORM_CIRCLE_RADIUS
# Side-view: output-y fraction the horizontal F→B line is placed on (lower-third
# line), leaving room above for the raised wing.
NORM_SIDE_FB_Y = 2.0 / 3.0


def _side_normalization(fx, fy, bx, by, lx, ly, rx, ry, l_present):
    """Full crop geometry ``(u_x, u_y, side, c, f)`` for a side-view image.

    A side view has F, B and exactly one wing (L or R). The F→B line is laid
    horizontal with F facing the wing's side — F on the left for L, on the right
    for R (the R layout is the L one rotated 180°, a *pure rotation*, never a
    mirror, so the moth is never flipped legs-up). The line is placed on the
    lower-third line (``NORM_SIDE_FB_Y``) when the wing points up, or the
    upper-third line when the wing points down, so the wing always gets the
    larger free area. The crop is scaled so the farthest of the three points
    lands on the reference circle (radius ``NORM_CIRCLE_RADIUS``, centred in the
    image).

    ``u_x``/``u_y`` are the input-space unit directions output +x / +y map to;
    ``side`` is the square side in pixels; ``c``/``f`` are the affine
    translation terms (``px = a*ox + b*oy + c``, ``py = d*ox + e*oy + f`` with
    ``(a, d) = u_x`` and ``(b, e) = u_y``).
    """
    wx, wy = (lx, ly) if l_present else (rx, ry)
    mx, my = (fx + bx) / 2, (fy + by) / 2  # F/B midpoint
    length = math.hypot(bx - fx, by - fy) or 1.0
    ux, uy = (bx - fx) / length, (by - fy) / length  # F -> B
    if l_present:
        u_x, u_y = (ux, uy), (-uy, ux)      # F on the left (det +1)
    else:
        u_x, u_y = (-ux, -uy), (uy, -ux)    # F on the right (180° rotation)
    a, d = u_x
    b, e = u_y

    # Place the F→B line so the wing points into the larger free area: if the
    # wing sits above the line (negative output-y offset) keep the line on the
    # lower-third line; if it sits below, move the line to the upper-third line.
    wing_dy = (wx - mx) * b + (wy - my) * e
    fb_y = NORM_SIDE_FB_Y if wing_dy <= 0 else (1.0 - NORM_SIDE_FB_Y)

    # Solve for the scale k = 1/side (output fractions per input pixel). Each
    # point's output-fraction offset from the F/B midpoint is d_i = k·((P-M)·u_x,
    # (P-M)·u_y); F and B have zero perpendicular component, so placing M on the
    # chosen third line puts the whole F→B line there. The point's distance from
    # the image centre is |(k·dx, offset_y + k·dy)| with offset_y the line's
    # signed distance below centre; setting that to the circle radius gives a
    # quadratic in k. The binding (smallest positive) root keeps every point
    # inside the circle with the farthest one exactly on it.
    offset_y = fb_y - 0.5
    r2 = NORM_CIRCLE_RADIUS * NORM_CIRCLE_RADIUS
    ks = []
    for px, py in ((fx, fy), (bx, by), (wx, wy)):
        vx, vy = px - mx, py - my
        dx = vx * a + vy * d
        dy = vx * b + vy * e
        qa = dx * dx + dy * dy
        if qa <= 0:
            continue
        qb = 2 * offset_y * dy
        qc = offset_y * offset_y - r2
        disc = qb * qb - 4 * qa * qc
        if disc < 0:
            continue
        k = (-qb + math.sqrt(disc)) / (2 * qa)
        if k > 0:
            ks.append(k)
    k = min(ks) if ks else 1.0
    side = max(1, int(round(1.0 / k)))

    # M maps to output fraction (0.5, fb_y).
    omx, omy = 0.5 * side, fb_y * side
    c = mx - omx * a - omy * b
    f = my - omx * d - omy * e
    return u_x, u_y, side, c, f


def compute_normalization(image_filename: str) -> dict | None:
    """Geometry of the pose-normalized crop for an image.

    The crop is a square, ``NORM_CROP_SCALE`` times the largest distance from
    the center C to the pose points, centred on C and rotated to a canonical
    orientation. Two layouts are supported:

    * **Top-down / bottom-up** (all four F/B/L/R present): C is the F/B midpoint
      and the F→B line is vertical (F on top, B on bottom).
    * **Side view** (F, B and exactly one wing): see
      :func:`_side_normalization` — the F→B line is laid horizontal with F
      facing the wing's side, on the image's lower-third line.

    Returns ``None`` when it can't be determined (missing prediction, keypoints,
    or original image size). The returned dict has:

    * ``side`` – the square side in pixels
    * ``affine`` – ``(a, b, c, d, e, f)`` mapping *output* (normalized crop) to
      *input* (original image) pixels, for ``Image.transform(..., AFFINE)``
    * ``keypoints`` – the first object's keypoints re-expressed as fractions of
      the normalized crop: ``{"x", "y", "v", "label"}`` (``x``/``y`` are
      ``None`` when the keypoint is unlabeled)
    """
    size = get_image_size(image_filename)
    if size is None:
        return None
    width, height = size

    annotations, _source = load_pose_source(image_filename)
    if not annotations:
        return None
    keypoints = annotations[0].keypoints
    if len(keypoints) < 4:
        return None
    front, left, right, back = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
    if front.visibility <= 0 or back.visibility <= 0:
        return None

    fx, fy = front.x * width, front.y * height
    bx, by = back.x * width, back.y * height
    lx, ly = left.x * width, left.y * height
    rx, ry = right.x * width, right.y * height

    l_present = left.visibility > 0
    r_present = right.visibility > 0
    # Affine samples from the ORIGINAL image (output -> input mapping), so no
    # pixels are clipped by an intermediate rotate step. The linear columns are
    # the input-space directions output +x / +y map to (unit-length, orthonormal
    # with det +1), so it's always a pure rotation — never a mirror/flip. Areas
    # outside the source are filled grey.
    if l_present and r_present:
        # Both wings: top-down layout — F→B vertical, F on top, centred on the
        # F/B midpoint; the farthest keypoint sits on the reference circle.
        cx, cy = (fx + bx) / 2, (fy + by) / 2
        length = math.hypot(bx - fx, by - fy) or 1.0
        ux, uy = (bx - fx) / length, (by - fy) / length
        u_x, u_y = (uy, -ux), (ux, uy)  # output +x perpendicular, +y along F→B
        pts = [(fx, fy), (bx, by), (lx, ly), (rx, ry)]
        max_dist = max(math.hypot(px - cx, py - cy) for px, py in pts)
        side = max(1, int(round(NORM_CROP_SCALE * max_dist)))
        half = side / 2
        a, d = u_x
        b, e = u_y
        c = cx - (a + b) * half
        f = cy - (d + e) * half
    elif l_present != r_present:
        # Exactly one wing: side view (F→B on the lower-third line).
        u_x, u_y, side, c, f = _side_normalization(
            fx, fy, bx, by, lx, ly, rx, ry, l_present
        )
        a, d = u_x
        b, e = u_y
    else:
        return None

    # Inverse (input -> output): the linear part is orthonormal, so its inverse
    # is its transpose. Used to place keypoints on the normalized crop.
    def to_crop(px: float, py: float) -> tuple[float, float]:
        return (a * (px - c) + d * (py - f), b * (px - c) + e * (py - f))

    mapped = []
    for slot, kp in enumerate(keypoints[:4]):
        label = KEYPOINT_LABELS.get(slot + 1, str(slot + 1))
        if kp.visibility <= 0:
            mapped.append({"x": None, "y": None, "v": kp.visibility, "label": label})
            continue
        ox, oy = to_crop(kp.x * width, kp.y * height)
        mapped.append(
            {"x": ox / side, "y": oy / side, "v": kp.visibility, "label": label}
        )

    return {"side": side, "affine": (a, b, c, d, e, f), "keypoints": mapped}


def get_or_create_normalized(image_filename: str):
    """Build a pose-normalized crop and its thumbnail; return ``(norm, thumb)``.

    Geometry comes from :func:`compute_normalization`. Outputs (both in
    ``MOTHS_THUMBNAIL_DIR/<tax_id>/``):

    * ``<name>.norm.jpg`` – the full-resolution normalized crop
    * ``<name>.norm-thumb.jpg`` – a shrunk thumbnail of that crop

    Both depend on the pose keypoints, so they are regenerated when the source
    image or either pose file (``MOTHS_LABEL_DIR`` / ``MOTHS_PREDICTION_DIR``)
    changes. Returns ``None`` when it can't be produced (missing image, pose
    data, or keypoints).
    """
    name = image_basename(image_filename)
    image_dir = get_image_dir().resolve()
    src = get_image_path(name).resolve()
    if image_dir not in src.parents or not src.is_file():
        return None

    geom = compute_normalization(image_filename)
    if geom is None:
        return None

    cache_dir = _tax_subdir(get_thumbnail_dir().resolve(), name)
    norm_path = cache_dir / (name + ".norm.jpg")
    thumb_path = cache_dir / (name + ".norm-thumb.jpg")

    dep_mtime = src.stat().st_mtime
    for source_path in (get_pose_path(image_filename), get_prediction_path(image_filename)):
        if source_path.is_file():
            dep_mtime = max(dep_mtime, source_path.stat().st_mtime)

    if (
        norm_path.is_file()
        and thumb_path.is_file()
        and norm_path.stat().st_mtime >= dep_mtime
        and thumb_path.stat().st_mtime >= dep_mtime
    ):
        return norm_path, thumb_path

    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    side = geom["side"]
    with Image.open(src) as im:
        im = im.convert("RGB")
        canvas = im.transform(
            (side, side),
            Image.AFFINE,
            geom["affine"],
            resample=Image.BICUBIC,
            fillcolor=NORM_FILL,
        )
        canvas.save(norm_path, quality=90, optimize=True)

        thumb = canvas.copy()
        thumb.thumbnail(tuple(settings.MOTHS_THUMBNAIL_SIZE))
        thumb.save(thumb_path, quality=85, optimize=True)

    return norm_path, thumb_path


def clear_normalized(image_filename: str) -> None:
    """Delete the cached normalized crop + thumbnail so they get rebuilt."""
    name = image_basename(image_filename)
    cache_dir = _tax_subdir(get_thumbnail_dir().resolve(), name)
    for suffix in (".norm.jpg", ".norm-thumb.jpg"):
        path = cache_dir / (name + suffix)
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass


def touch_normalized(image_filename: str) -> None:
    """Bump the cached normalized crop + thumbnail mtimes to now.

    Used during a rebuild when the keypoints are unchanged: freshening the
    mtimes lets :func:`get_or_create_normalized` reuse the existing files
    (its cache check is mtime-based) instead of regenerating them.
    """
    name = image_basename(image_filename)
    cache_dir = _tax_subdir(get_thumbnail_dir().resolve(), name)
    for suffix in (".norm.jpg", ".norm-thumb.jpg"):
        path = cache_dir / (name + suffix)
        if path.is_file():
            try:
                path.touch()
            except OSError:
                pass


# Normalized-crop metric settings.
NORM_METRIC_SIZE = 1024      # side of the temporary square used for scoring
SHARPNESS_GRID = 9           # tiles per axis (SHARPNESS_GRID x SHARPNESS_GRID)
SHARPNESS_CENTER_TILES = 5   # central tiles sampled down the middle column
# This metric is cached only in the per-tax ``<tax_id>_pose_data.json``; bump
# POSE_DATA_VERSION when the scoring method below changes so those are rebuilt.


def compute_norm_metrics(image_filename: str) -> dict | None:
    """Compute the pose-normalized crop's scalar metrics (no per-image cache).

    Returns a dict with ``"sharpness"`` (or ``None`` when the normalized crop
    can't be produced). Callers cache the result per tax in the pose-data JSON;
    nothing is written per image.

    * ``sharpness`` – Scharr/Tenengrad gradient energy. Normalization puts the
      F→B axis vertical, so the moth body runs down the centre column; we take
      the median of the middle ``SHARPNESS_CENTER_TILES`` tiles of the centre
      column of a ``SHARPNESS_GRID`` x ``SHARPNESS_GRID`` grid (ignores
      background).
    """
    result = get_or_create_normalized(image_filename)
    if result is None:
        return None
    norm_path, _thumb = result

    import numpy as np
    from PIL import Image

    with Image.open(norm_path) as im:
        gray = im.convert("L").resize(
            (NORM_METRIC_SIZE, NORM_METRIC_SIZE), Image.LANCZOS
        )
        arr = np.asarray(gray, dtype=np.float64)

    # --- Sharpness (Scharr/Tenengrad on the central vertical strip) ---
    gx = (
        3 * (arr[:-2, 2:] - arr[:-2, :-2])
        + 10 * (arr[1:-1, 2:] - arr[1:-1, :-2])
        + 3 * (arr[2:, 2:] - arr[2:, :-2])
    )
    gy = (
        3 * (arr[2:, :-2] - arr[:-2, :-2])
        + 10 * (arr[2:, 1:-1] - arr[:-2, 1:-1])
        + 3 * (arr[2:, 2:] - arr[:-2, 2:])
    )
    energy = gx * gx + gy * gy

    row_blocks = np.array_split(energy, SHARPNESS_GRID, axis=0)
    center_col = SHARPNESS_GRID // 2
    row_start = (SHARPNESS_GRID - SHARPNESS_CENTER_TILES) // 2
    tile_scores = []
    for r in range(row_start, row_start + SHARPNESS_CENTER_TILES):
        tile = np.array_split(row_blocks[r], SHARPNESS_GRID, axis=1)[center_col]
        if tile.size:
            tile_scores.append(float(tile.mean()))
    if not tile_scores:
        return None
    sharpness = float(np.median(tile_scores))

    return {"sharpness": sharpness}


def get_tax_thumbnail(tax_id: str) -> str | None:
    """Return the image filename chosen as ``tax_id``'s representative thumbnail.

    The choice is stored alongside the counts in the tax's summary JSON (see
    :func:`build_summary`). Returns ``None`` when none recorded yet.
    """
    summary = load_summary(tax_id)
    if not summary:
        return None
    thumbnail = summary.get("thumbnail") or {}
    return thumbnail.get("filename") or None


def choose_tax_thumbnail(tax_id: str, per_image: dict | None = None) -> dict | None:
    """Pick ``tax_id``'s representative thumbnail from pose data (no I/O writes).

    Mirrors the poses view's default order: among top-down images with a score,
    starred ones win, then the highest cumulative score. Returns a
    ``{"filename", "score", "starred"}`` dict, or ``None`` when there are no
    scored top-down images yet. Pure w.r.t. the summary cache, so it can be used
    both by :func:`build_summary` and :func:`refresh_tax_thumbnail` without
    recursion.
    """
    if per_image is None:
        data = load_pose_data(tax_id)
        per_image = data.get("images", {}) if data else {}

    starred = load_starred(tax_id)
    candidates = [
        (image_basename(filename) in starred, row.get("score"), filename)
        for filename, row in per_image.items()
        if row.get("pose") == POSE_TOP_DOWN and row.get("score") is not None
    ]
    if not candidates:
        return None

    is_starred, best_score, best_file = max(candidates, key=lambda t: (t[0], t[1]))
    return {"filename": best_file, "score": best_score, "starred": is_starred}


def refresh_tax_thumbnail(tax_id: str, per_image: dict | None = None) -> str | None:
    """Recompute ``tax_id``'s representative thumbnail and store it in the summary.

    Reads the cached pose data (or the freshly built ``per_image`` mapping) plus
    the starred set, so it stays correct when images are starred/unstarred.
    Returns the chosen filename (or ``None`` when there are no scored top-down
    images yet).
    """
    choice = choose_tax_thumbnail(tax_id, per_image)
    if choice is None:
        return get_tax_thumbnail(tax_id)

    summary = load_summary(tax_id)
    if summary is None:
        summary = build_summary(tax_id)

    current = summary.get("thumbnail") or {}
    if current.get("filename") == choice["filename"]:
        return choice["filename"]

    summary["thumbnail"] = choice
    _write_summary(tax_id, summary)
    return choice["filename"]


# --- Cached per-tax pose data ------------------------------------------------

# Version of the cached per-tax pose data. Bump whenever pose classification,
# any metric (symmetry / pixels / sharpness), the cumulative-score formula or
# the row schema changes, so stale caches are flagged for rebuild. v2: added
# per-row ``source`` and ``keypoints``. v3: predicted side/bottom_up collapse to
# unclear. v4: dropped the exposure metric. v5: symmetry computed in isotropic
# pixel space (was skewed for non-square images). v6: store per-image original
# width/height in the row (replaces the ``.size`` sidecar cache). v7: stricter
# classification (any partial keypoint or both wings on one side -> unclear;
# side requires exactly F/B + one wing) and side-view normalization (F→B laid
# horizontal, F facing the wing's side). v9: side-view F→B line placed on the
# lower-third line, scaled so the farthest point sits on the 80% circle. v10:
# F→B line placed on the third line opposite the wing (upper when wing points
# down) so the wing always gets the larger free area.
# v11: cumulative_score switched from the sum of the three sub-scores to their
# minimum (weakest-link), so cached scores must be recomputed.
POSE_DATA_VERSION = 11


def score_components(
    symmetry: float | None,
    pixel_span: float | None,
    sharpness: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Return the three scaled ``0..1`` sub-scores summed by ``cumulative_score``.

    Each element is ``None`` when its input metric is ``None``:

    1. ``1 - symmetry`` (symmetry is 0 when perfectly symmetric).
    2. ``(clamp(pixel_span, 300, 1300) - 300) / 1000`` -> 0..1 over 300..1300 px.
    3. ``min(sharpness, 300000) / 300000`` -> 0..1 (sharpness is Tenengrad energy).
    """
    s_sym = None if symmetry is None else 1 - symmetry
    s_pixels = (
        None
        if pixel_span is None
        else (max(min(pixel_span, 1300), 300) - 300) / 1000
    )
    s_sharp = None if sharpness is None else min(sharpness, 300000) / 300000
    return s_sym, s_pixels, s_sharp


def cumulative_score(
    symmetry: float | None,
    pixel_span: float | None,
    sharpness: float | None,
) -> float | None:
    """Combined quality score (minimum of the three sub-scores); ``None`` if any
    input is missing.

    Using the minimum rather than the sum makes the score a weakest-link
    measure: an image only scores well when *every* metric is good, and a single
    poor metric caps the total.
    """
    if symmetry is None or pixel_span is None or sharpness is None:
        return None
    s_sym, s_pixels, s_sharp = score_components(symmetry, pixel_span, sharpness)
    return min(s_sym, s_pixels, s_sharp)


def compute_pose_row(image_filename: str) -> dict:
    """Compute the pose class, metrics and score for one image (the slow path).

    Records the ``source`` (``pose``/``prediction``/``None``) and the raw
    keypoints used, so a later visit can detect when they've changed. Also
    records the original image ``width``/``height`` (``None`` when unavailable),
    persisting the intrinsic size here instead of a ``.size`` sidecar file.
    Sharpness requires the normalized crop, so it is only computed for the
    top-down group (matching the pose view's thumbnails).
    """
    keypoints, source = _pose_source_keypoints(image_filename)
    pose = classify_pose(image_filename)
    size = get_image_size(image_filename)
    width, height = size if size else (None, None)
    symmetry = pose_symmetry_metric(image_filename)
    pixel_span = pose_pixel_span(image_filename)
    sharpness = None
    if pose == POSE_TOP_DOWN:
        metrics = compute_norm_metrics(image_filename)
        if metrics is not None:
            sharpness = metrics["sharpness"]
    return {
        "pose": pose,
        "source": source,
        "keypoints": keypoints,
        "width": width,
        "height": height,
        "symmetry": symmetry,
        "pixel_span": pixel_span,
        "sharpness": sharpness,
        "score": cumulative_score(symmetry, pixel_span, sharpness),
        "flags": get_image_flags(image_filename),
    }


def get_pose_data_path(tax_id: str) -> Path:
    """Path to the cached pose-data JSON for a tax_id (at the labels root)."""
    return get_label_dir() / f"{tax_id}_pose_data.json"


def load_pose_data(tax_id: str) -> dict | None:
    """Return cached pose data for a tax_id, or ``None`` if missing/stale.

    A file whose ``version`` doesn't match ``POSE_DATA_VERSION`` (or that can't
    be parsed) is treated as absent so it gets rebuilt.
    """
    path = get_pose_data_path(tax_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or data.get("version") != POSE_DATA_VERSION:
        return None
    if not isinstance(data.get("images"), dict):
        return None
    return data


def load_pose_data_raw(tax_id: str) -> dict | None:
    """Return cached pose data *ignoring* the version, or ``None`` if unusable.

    Unlike :func:`load_pose_data`, a version mismatch is not treated as absent:
    the parsed dict is returned so the poses view can still display the images
    (grouped by their cached pose) while flagging them for rebuild. Callers
    check ``data["version"] == POSE_DATA_VERSION`` themselves. Returns ``None``
    only when the file is missing, unparseable, or malformed.
    """
    path = get_pose_data_path(tax_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("images"), dict):
        return None
    return data


def pose_data_version_ok(data: dict | None) -> bool:
    """True when ``data`` is a pose-data dict at the current schema version."""
    return bool(data) and data.get("version") == POSE_DATA_VERSION


def _write_pose_data(tax_id: str, data: dict) -> None:
    """Write a pose-data dict to the tax_id's pose-data JSON (best effort)."""
    path = get_pose_data_path(tax_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def build_pose_data(tax_id: str, images) -> dict:
    """Compute pose data for every image of a tax_id and cache it to JSON.

    For each image, if the keypoints/source match the previously cached row the
    normalized crop + thumbnail are kept (not regenerated); otherwise they are
    cleared so :func:`compute_pose_row` rebuilds them. Also refreshes the tax's
    representative thumbnail. Returns the freshly built data dict.
    """
    prev = load_pose_data_raw(tax_id)
    prev_images = prev.get("images", {}) if prev else {}

    per_image = {}
    for image in images:
        filename = image.filename
        keypoints, source = _pose_source_keypoints(filename)
        prev_row = prev_images.get(filename)
        if prev_row is not None and prev_row.get("keypoints") == keypoints \
                and prev_row.get("source") == source:
            # Keypoints unchanged since the last cache (even across a version
            # bump): reuse the existing crop/thumbnail instead of regenerating.
            touch_normalized(filename)
        elif prev_row is not None:
            # Keypoints changed: force the crop/thumbnail to be rebuilt.
            clear_normalized(filename)
        # No prior row: leave get_or_create_normalized's mtime cache to decide.
        per_image[filename] = compute_pose_row(filename)

    data = {"version": POSE_DATA_VERSION, "images": per_image}
    _write_pose_data(tax_id, data)
    refresh_tax_thumbnail(tax_id, per_image)
    return data


def get_pose_data(tax_id: str, images, rebuild: bool = False) -> dict:
    """Return pose data for a tax_id, building and caching it if needed.

    When a valid cache exists it is returned as-is (nothing is recomputed).
    Pass ``rebuild=True`` to force a fresh computation.
    """
    if not rebuild:
        cached = load_pose_data(tax_id)
        if cached is not None:
            return cached
    return build_pose_data(tax_id, images)


def verify_pose_row(tax_id: str, image_filename: str) -> dict | None:
    """Ensure the cached pose row reflects the image's current keypoints.

    Compares the stored ``source``/``keypoints`` against the live pose source
    (``MOTHS_LABEL_DIR`` then ``MOTHS_PREDICTION_DIR``); when they differ, the
    row and the normalized crop are recomputed and the caches (pose data +
    thumbnail) updated. This is the only place recomputation is triggered on
    change — the poses view reads the cache as-is. Returns the current row.
    """
    data = load_pose_data(tax_id)
    if data is None:
        # No/stale cache: build the whole tax once so the file stays consistent.
        data = build_pose_data(tax_id, scan_tax_images(tax_id))

    images = data.get("images", {})
    row = images.get(image_filename)

    keypoints, source = _pose_source_keypoints(image_filename)
    if (
        row is not None
        and row.get("source") == source
        and row.get("keypoints") == keypoints
    ):
        # Keypoints are actually unchanged; drop any stale flag (e.g. from a
        # re-save with identical points) without recomputing.
        if row.pop("needs_rebuild", None):
            data["images"] = images
            _write_pose_data(tax_id, data)
        return row

    # Keypoints changed (or row missing): recompute just this image. This also
    # clears any pending ``needs_rebuild`` flag since the row is now current.
    clear_normalized(image_filename)
    row = compute_pose_row(image_filename)
    images[image_filename] = row
    data["images"] = images
    _write_pose_data(tax_id, data)
    refresh_tax_thumbnail(tax_id, images)
    return row


def mark_pose_row_stale(tax_id: str, image_filename: str) -> None:
    """Flag one image's cached pose row for rebuild after a manual keypoint edit.

    Sets ``needs_rebuild`` on the row in ``<tax_id>_pose_data.json`` without
    recomputing anything, so the poses view can show a "to be rebuild" note.
    A no-op when no pose data has been cached yet (the first build computes it
    fresh). Works on version-mismatched files too (whole file rebuilds anyway).
    """
    data = load_pose_data_raw(tax_id)
    if data is None:
        return
    images = data.get("images", {})
    row = images.get(image_filename) or {}
    row["needs_rebuild"] = True
    images[image_filename] = row
    data["images"] = images
    _write_pose_data(tax_id, data)


def set_pose_row_flags(tax_id: str, image_filename: str, flags: list[str]) -> None:
    """Update one image's cached ``flags`` in ``<tax_id>_pose_data.json``.

    Writes the flags into the existing pose row without recomputing anything, so
    the cache stays in sync with the ``.class`` file. A no-op when no pose data
    (or no row for this image) has been cached yet — the next build fills it in
    from the ``.class`` file.
    """
    if not tax_id:
        return
    data = load_pose_data_raw(tax_id)
    if data is None:
        return
    images = data.get("images", {})
    row = images.get(image_filename)
    if row is None:
        return
    row["flags"] = [f for f in FLAGS if f in set(flags)]
    images[image_filename] = row
    data["images"] = images
    _write_pose_data(tax_id, data)


# --- Per-tax index summary cache ---------------------------------------------

# Bump when the summary schema/counts change so old caches are ignored.
# v2: added "source_data" (harvest audit CSV life-stage / status tallies).
# v3: counts are plain totals (train/val split removed from the app).
SUMMARY_VERSION = 3

# Life-stage buckets for the harvest "source data" summary, in display order.
# "Adult" also covers Teneral (a freshly emerged adult); anything else
# documented (Nymph, Juvenile, Subimago, numeric ids) or blank -> "Unknown".
SOURCE_STAGES = ["Egg", "Larva", "Pupa", "Adult", "Unknown"]


def list_tax_ids() -> list[str]:
    """Return the tax_id subfolder names under the image directory, sorted.

    This is the cheap source of truth for the index page's rows: it only lists
    directories and never walks their contents.
    """
    image_dir = get_image_dir()
    if not image_dir.is_dir():
        return []
    return sorted(entry.name for entry in image_dir.iterdir() if entry.is_dir())


def scan_tax_images(tax_id: str) -> list[MothImage]:
    """Return parseable images for a single tax_id (its subfolder only)."""
    tax_dir = get_image_dir() / tax_id
    if not tax_dir.is_dir():
        return []
    images = []
    for entry in tax_dir.rglob("*"):
        if entry.is_file():
            parsed = parse_filename(entry.name)
            if parsed is not None:
                images.append(parsed)
    images.sort(key=lambda image: image.filename)
    return images


def get_summary_path(tax_id: str) -> Path:
    """Path to the cached index-summary JSON for a tax_id (at the labels root)."""
    return get_label_dir() / f"{tax_id}_summary.json"


def _compute_summary_counts(images) -> dict:
    """Tally per-stage, unclassified and labeled/unlabeled counts (plain totals)."""
    stages = {stage: 0 for stage in STAGES}
    unclassified = 0
    labeled = 0
    not_labeled = 0
    for image in images:
        stage = get_image_class(image.filename)
        if stage in stages:
            stages[stage] += 1
        else:
            unclassified += 1
        if get_label_path(image.filename).is_file():
            labeled += 1
        else:
            not_labeled += 1
    return {
        "total": len(images),
        "stages": stages,
        "unclassified": unclassified,
        "labeled": labeled,
        "not_labeled": not_labeled,
    }


def _source_stage_bucket(life_stage: str) -> str:
    """Map a documented iNaturalist life stage string to a SOURCE_STAGES bucket.

    ``life_stage`` may list several stages joined by ``/`` (as written by the
    harvest audit log). Adult/Teneral win, then Larva, Pupa, Egg; anything else
    (Nymph, Juvenile, Subimago, numeric ids, blank) falls into "Unknown".
    """
    parts = [p.strip() for p in (life_stage or "").split("/") if p.strip()]
    if any(p in ("Adult", "Teneral") for p in parts):
        return "Adult"
    for stage in ("Larva", "Pupa", "Egg"):
        if stage in parts:
            return stage
    return "Unknown"


def compute_source_data(tax_id: str) -> dict | None:
    """Summarize the harvest audit CSV ``images/<tax>_observations_list.csv``.

    Returns ``{"stages": {stage: count}, "statuses": {status: count}}`` where
    life-stage counts come from real observation rows (those with an
    ``observation_id``) and status counts cover every row's ``status`` flag
    (including terminal markers like ``done``/``corrupted``). Returns ``None``
    when the CSV is missing or unreadable, so the index can show a dash.
    """
    path = get_image_dir() / f"{tax_id}_observations_list.csv"
    if not path.is_file():
        return None
    stages = {stage: 0 for stage in SOURCE_STAGES}
    statuses: dict[str, int] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                status = (row.get("status") or "").strip()
                if status:
                    statuses[status] = statuses.get(status, 0) + 1
                # Rows with an observation_id are real observations; terminal
                # marker rows (done/corrupted/...) carry no life stage.
                if (row.get("observation_id") or "").strip():
                    bucket = _source_stage_bucket(row.get("life_stage") or "")
                    stages[bucket] += 1
    except OSError:
        return None
    return {"stages": stages, "statuses": statuses}


def _write_summary(tax_id: str, data: dict) -> None:
    """Write a summary dict to the tax_id's summary JSON (best effort)."""
    path = get_summary_path(tax_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError:
        pass


def build_summary(tax_id: str, images=None) -> dict:
    """Compute and cache the index summary (names + counts) for a tax_id.

    Reuses the previously stored representative-thumbnail choice when present;
    otherwise (e.g. after a schema bump wipes the old summary) it re-derives it
    from the cached pose data, so a rebuild doesn't drop the thumbnail.
    """
    if images is None:
        images = scan_tax_images(tax_id)
    info = get_name_info(tax_id)
    existing = load_summary(tax_id)
    thumbnail = (existing or {}).get("thumbnail") or choose_tax_thumbnail(tax_id)
    data = {
        "version": SUMMARY_VERSION,
        "names": {
            "family": info["family"],
            "species": info["species"],
            "name": info["name"],
        },
        "counts": _compute_summary_counts(images),
        "source_data": compute_source_data(tax_id),
        "thumbnail": thumbnail or None,
    }
    _write_summary(tax_id, data)
    return data


def load_summary(tax_id: str) -> dict | None:
    """Return the cached summary for a tax_id, or ``None`` if missing/stale."""
    path = get_summary_path(tax_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(data, dict) or data.get("version") != SUMMARY_VERSION:
        return None
    if not isinstance(data.get("counts"), dict):
        return None
    return data


def get_summary(tax_id: str, images=None) -> dict:
    """Return the cached summary for a tax_id, building it once if absent."""
    cached = load_summary(tax_id)
    if cached is not None:
        return cached
    return build_summary(tax_id, images)


def update_summary(tax_id: str | None) -> None:
    """Rebuild a tax_id's cached summary from current files (after an edit)."""
    if tax_id:
        build_summary(tax_id)


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def save_annotations(image_filename: str, objects: list[dict]) -> None:
    """Write annotations to the image's YOLO-pose label file.

    ``objects`` is a list of dicts with keys ``class_id``, ``cx``, ``cy``,
    ``w``, ``h`` and ``keypoints`` (a list of ``{"x", "y", "v"}``). If the list
    is empty, the label file is removed.
    """
    label_path = get_label_path(image_filename)
    if not objects:
        if label_path.is_file():
            label_path.unlink()
        return

    lines = []
    for obj in objects:
        values = [
            str(int(obj.get("class_id", 0))),
            _fmt(obj["cx"]),
            _fmt(obj["cy"]),
            _fmt(obj["w"]),
            _fmt(obj["h"]),
        ]
        for kp in obj.get("keypoints", []):
            values += [_fmt(kp["x"]), _fmt(kp["y"]), str(int(kp["v"]))]
        lines.append(" ".join(values))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
