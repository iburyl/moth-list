"""Helpers for discovering and parsing moth image files.

Images live outside this repo, in a sibling folder, and follow the naming
convention::

    {tax_id}_observation_{obs_id}_photo_{photo_id}.{ext}
"""

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


def parse_filename(filename: str) -> MothImage | None:
    """Parse a filename into a ``MothImage`` or ``None`` if it doesn't match."""
    match = FILENAME_RE.match(filename)
    if not match:
        return None
    return MothImage(filename=filename, **match.groupdict())


def scan_images() -> list[MothImage]:
    """Return every parseable image in the image directory, sorted by name."""
    image_dir = get_image_dir()
    if not image_dir.is_dir():
        return []
    images = []
    for entry in sorted(image_dir.iterdir()):
        if not entry.is_file():
            continue
        parsed = parse_filename(entry.name)
        if parsed is not None:
            images.append(parsed)
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


def get_class_dir() -> Path:
    """Directory holding per-image stage classification files."""
    return Path(settings.MOTHS_CLASS_DIR)


def get_class_path(image_filename: str) -> Path:
    """Path to the ``.class`` file for an image (same name, ``.class`` ext)."""
    return get_class_dir() / (Path(image_filename).stem + ".class")


def get_image_class(image_filename: str) -> str | None:
    """Return the stored stage for an image, or ``None`` if unclassified."""
    path = get_class_path(image_filename)
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def set_image_class(image_filename: str, stage: str) -> None:
    """Write the stage classification for an image."""
    get_class_dir().mkdir(parents=True, exist_ok=True)
    get_class_path(image_filename).write_text(stage, encoding="utf-8")


def clear_image_class(image_filename: str) -> None:
    """Remove the stage classification file for an image, if present."""
    path = get_class_path(image_filename)
    if path.is_file():
        path.unlink()


# --- Thumbnail cache ----------------------------------------------------------


def get_thumbnail_dir() -> Path:
    """Return the directory where cached thumbnails are stored."""
    return Path(settings.MOTHS_THUMBNAIL_DIR)


def get_or_create_thumbnail(image_filename: str) -> Path | None:
    """Return the cached thumbnail path for an image, generating it if needed.

    The thumbnail keeps the original filename and is stored (flat) in
    ``MOTHS_THUMBNAIL_DIR``. It is (re)generated when missing or older than the
    source image. Returns ``None`` if the source image can't be found.
    """
    image_dir = get_image_dir().resolve()
    src = (image_dir / image_filename).resolve()
    if image_dir not in src.parents or not src.is_file():
        return None

    thumb_dir = get_thumbnail_dir().resolve()
    # Use only the basename for the destination to avoid path traversal.
    dest = thumb_dir / Path(image_filename).name

    if dest.is_file() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest

    from PIL import Image

    thumb_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.thumbnail(tuple(settings.MOTHS_THUMBNAIL_SIZE))
        save_kwargs: dict = {}
        if dest.suffix.lower() in (".jpg", ".jpeg"):
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            save_kwargs = {"quality": 85, "optimize": True}
        im.save(dest, **save_kwargs)
    return dest


def find_image(filename: str) -> MothImage | None:
    """Return the ``MothImage`` for ``filename`` if it exists in the image dir."""
    if parse_filename(filename) is None:
        return None
    file_path = (get_image_dir() / filename).resolve()
    if get_image_dir().resolve() not in file_path.parents or not file_path.is_file():
        return None
    return parse_filename(filename)


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
    """Directory holding the label ``.txt`` files.

    Derived from the image directory by swapping the ``images`` path component
    for ``labels`` (e.g. ``.../data/images/train`` -> ``.../data/labels/train``).
    """
    parts = list(get_image_dir().parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts)


def get_label_path(image_filename: str) -> Path:
    """Return the label file path corresponding to an image filename."""
    return get_label_dir() / (Path(image_filename).stem + ".txt")


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


# --- Train / val split membership --------------------------------------------

# Split definition files, relative to the dataset root, keyed by subset name.
SPLIT_FILES = {
    "train": "train.txt",
    "val": "val.txt",
}


def get_dataset_root() -> Path:
    """Return the dataset root (the folder that contains ``data/``).

    The split files (``train.txt`` / ``val.txt``) live here, and their entries
    are paths relative to this folder, e.g. ``data/images/train/<file>.jpg``.
    """
    parts = list(get_image_dir().parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "data":
            return Path(*parts[:i])
    # Fallback: images/<split> -> images -> data -> root
    return get_image_dir().parent.parent.parent


def load_subset_map() -> dict[str, str]:
    """Return a mapping of image basename -> subset ("train"/"val").

    Reads each split file once; use this when classifying many images to avoid
    re-reading the split files per lookup.
    """
    root = get_dataset_root()
    mapping: dict[str, str] = {}
    for subset, split_name in SPLIT_FILES.items():
        split_path = root / split_name
        if not split_path.is_file():
            continue
        for line in split_path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry:
                continue
            mapping[Path(entry).name] = subset
    return mapping


def get_image_subset(image_filename: str) -> str | None:
    """Return the subset ("train"/"val") the image belongs to, or ``None``."""
    return load_subset_map().get(image_filename)


def _split_entry(image_filename: str) -> str:
    """Build the split-file line for an image (POSIX slashes).

    The subset (train/val) is encoded only by which file the line lives in, not
    by a directory in the path.
    """
    rel = get_image_dir().relative_to(get_dataset_root()).as_posix()
    return f"{rel}/{image_filename}"


def set_images_subset(image_filenames, subset: str | None) -> None:
    """Assign one or more images to a subset ("train"/"val") or remove them.

    Rewrites both split files once, de-duplicated and sorted, using Linux-style
    forward slashes. Each image is removed from whichever file(s) it appears in
    before being added to the target subset (``None`` just removes them).
    """
    names = set(image_filenames)
    root = get_dataset_root()
    lines_by_subset: dict[str, list[str]] = {}
    for name, split_name in SPLIT_FILES.items():
        split_path = root / split_name
        if split_path.is_file():
            lines_by_subset[name] = [
                line.strip()
                for line in split_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            lines_by_subset[name] = []

    # Drop any existing entries for these images from every split file.
    for name in lines_by_subset:
        lines_by_subset[name] = [
            line for line in lines_by_subset[name] if Path(line).name not in names
        ]

    if subset in SPLIT_FILES:
        for image_filename in image_filenames:
            lines_by_subset[subset].append(_split_entry(image_filename))

    for name, split_name in SPLIT_FILES.items():
        split_path = root / split_name
        entries = sorted(set(lines_by_subset[name]))
        if entries:
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        elif split_path.is_file():
            split_path.write_text("", encoding="utf-8")


def set_image_subset(image_filename: str, subset: str | None) -> None:
    """Assign a single image to a subset ("train"/"val") or remove it."""
    set_images_subset([image_filename], subset)


def load_annotations(image_filename: str) -> list[Annotation]:
    """Load all annotations from the label file for the given image."""
    label_path = get_label_path(image_filename)
    if not label_path.is_file():
        return []
    annotations: list[Annotation] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        annotation = parse_label_line(line)
        if annotation is not None:
            annotations.append(annotation)
    return annotations


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
