"""Filesystem layout: settings-driven dirs, filename parsing and image discovery."""
from __future__ import annotations

import re

from pathlib import Path
from dataclasses import dataclass

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


# --- Thumbnail cache ----------------------------------------------------------


def get_thumbnail_dir() -> Path:
    """Return the directory where cached thumbnails are stored."""
    return Path(settings.MOTHS_THUMBNAIL_DIR)


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


def _unlink_quiet(path: Path) -> None:
    if path.is_file():
        try:
            path.unlink()
        except OSError:
            pass


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
