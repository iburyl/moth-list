"""Life-cycle stage, the optional-flag table and the ``.class``/starred sidecars."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from django.conf import settings

from .paths import (
    _tax_subdir,
    image_basename,
    tax_id_for_file,
)


# --- Stage classification -----------------------------------------------------

# Life-cycle stages an image can be classified as.
STAGES = ["Egg", "Larva", "Pupa", "Adult"]

@dataclass(frozen=True)
class Flag:
    """One optional per-image flag and how it tailors the views and tools.

    * ``normalize`` – when ``False``, images carrying the flag are never
      pose-normalized: the normalized view/thumbnail falls back to the plain
      image and the poses view shows them in a plain (score-less) subsection.
    * ``train`` – when ``False``, images carrying the flag are excluded from
      every training archive built by ``tools/prepare_train_data.py``.
    * ``stages`` – the life-cycle stages the flag is offered for (a subset of
      :data:`STAGES`); an empty tuple means all stages. Drives the flag picker
      in the edit view and which per-stage flag subsections the poses view
      builds.
    """

    name: str
    normalize: bool = True
    train: bool = True
    stages: tuple[str, ...] = ()


# Single source of truth for the optional flags (order = display order). Every
# view and tool derives its per-flag behaviour from this table; no flag name is
# hardcoded elsewhere, so adding or retuning a flag is a one-line edit here.
FLAG_TABLE: tuple[Flag, ...] = (
    Flag("Pinned", stages=("Adult",)),
    Flag("Macro", stages=("Adult",)),
    Flag("Damaged", train=False),
    Flag("Traces", train=False, stages=("Larva",)),
    Flag("Mating", normalize=False, train=False, stages=("Adult",)),
)

_FLAG_BY_NAME = {flag.name: flag for flag in FLAG_TABLE}

# Flag names in display order (kept as a plain list for templates/JSON/legacy
# callers that only need the ordering).
FLAGS = [flag.name for flag in FLAG_TABLE]

# Derived membership sets for the hot paths.
NO_NORM_FLAGS = {flag.name for flag in FLAG_TABLE if not flag.normalize}
NO_TRAIN_FLAGS = {flag.name for flag in FLAG_TABLE if not flag.train}


def flag_stages(name: str) -> tuple[str, ...]:
    """Stages a flag is offered for (all of :data:`STAGES` when unrestricted)."""
    flag = _FLAG_BY_NAME.get(name)
    if flag is None or not flag.stages:
        return tuple(STAGES)
    return flag.stages


def flag_applies_to_stage(name: str, stage: str | None) -> bool:
    """Whether a flag is offered for ``stage``.

    An unset/unknown stage (anything not in :data:`STAGES`) offers every flag —
    the stage isn't decided yet, so nothing is filtered out.
    """
    if stage not in STAGES:
        return True
    return stage in flag_stages(name)


def flags_for_stage(stage: str | None) -> list[str]:
    """The flag names offered for ``stage`` (in table order)."""
    return [flag.name for flag in FLAG_TABLE if flag_applies_to_stage(flag.name, stage)]


def flags_suppress_normalization(flags) -> bool:
    """Whether any of ``flags`` opts the image out of normalization."""
    return bool(NO_NORM_FLAGS.intersection(flags))


def flags_block_training(flags) -> bool:
    """Whether any of ``flags`` excludes the image from all training data."""
    return bool(NO_TRAIN_FLAGS.intersection(flags))

# Prefix of the flags line inside a ``.class`` file.
_FLAGS_PREFIX = "flags:"

# Prefix of the optional raw predicted-class line inside a ``.class`` file. Only
# the prediction pipeline writes it (the verbatim classifier class name, before
# it is collapsed into stage/flags); hand ``classes/`` files never carry it.
_POSE_PREFIX = "pose:"

# Prefix of the optional 1-5 "details" rating line inside a ``.class`` file. This
# is a throwaway hand annotation (ground truth for tuning the sharpness score);
# it deliberately never feeds the pose-data or summary caches and can be dropped
# later without touching them.
_DETAILS_PREFIX = "details:"


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


def _write_class_file(
    path: Path,
    stage: str | None,
    flags: list[str],
    pose: str | None = None,
    details: int | None = None,
) -> None:
    """Write ``stage`` + ``flags`` (+ optional ``pose``/``details``) to a file.

    Deletes the file when nothing is set. ``pose`` is the raw predicted class
    name; it is written as a trailing ``pose:<name>`` line and only used by the
    prediction pipeline (hand files pass ``None`` and are unaffected).
    ``details`` is the optional 1-5 hand rating; the stage/flag setters read and
    pass it back through so editing a stage or flag never drops the rating.
    """
    flags = [f for f in FLAGS if f in set(flags)]
    if not stage and not flags and not pose and not details:
        if path.is_file():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    body = stage or ""
    if flags:
        body += "\n" + _FLAGS_PREFIX + ",".join(flags)
    if pose:
        body += "\n" + _POSE_PREFIX + pose
    if details:
        body += "\n" + _DETAILS_PREFIX + str(details)
    path.write_text(body, encoding="utf-8")


def _read_class_pose(path: Path) -> str | None:
    """Return the raw ``pose:`` class name from a ``.class`` file, if present."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(_POSE_PREFIX):
            return line[len(_POSE_PREFIX):].strip() or None
    return None


def _read_class_details(path: Path) -> int | None:
    """Return the 1-5 ``details:`` rating from a ``.class`` file, if valid."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(_DETAILS_PREFIX):
            raw = line[len(_DETAILS_PREFIX):].strip()
            try:
                value = int(raw)
            except ValueError:
                return None
            return value if 1 <= value <= 5 else None
    return None


def get_image_class(image_filename: str) -> str | None:
    """Return the stored stage for an image, or ``None`` if unclassified."""
    stage, _flags = _read_class_file(get_class_path(image_filename))
    return stage


def set_image_class(image_filename: str, stage: str) -> None:
    """Write the stage classification for an image, preserving flags/rating."""
    path = get_class_path(image_filename)
    _stage, flags = _read_class_file(path)
    _write_class_file(path, stage, flags, details=_read_class_details(path))


def clear_image_class(image_filename: str) -> None:
    """Clear the stage classification for an image, preserving flags/rating.

    Deletes the ``.class`` file only when no flags or rating remain.
    """
    path = get_class_path(image_filename)
    _stage, flags = _read_class_file(path)
    _write_class_file(path, None, flags, details=_read_class_details(path))


def get_image_flags(image_filename: str) -> list[str]:
    """Return the flags set on an image (subset of :data:`FLAGS`)."""
    _stage, flags = _read_class_file(get_class_path(image_filename))
    return flags


def get_class_and_flags(image_filename: str) -> tuple[str | None, list[str]]:
    """Return ``(stage, flags)`` for an image in a single ``.class`` file read."""
    return _read_class_file(get_class_path(image_filename))


def set_image_flags(image_filename: str, flags: list[str]) -> list[str]:
    """Store the flags for an image, preserving its stage/rating. Returns them."""
    path = get_class_path(image_filename)
    stage, _flags = _read_class_file(path)
    kept = [f for f in FLAGS if f in set(flags)]
    _write_class_file(path, stage, kept, details=_read_class_details(path))
    return kept


def get_image_details(image_filename: str) -> int | None:
    """Return the 1-5 hand ``details`` rating for an image, or ``None``.

    A throwaway ground-truth annotation for tuning the sharpness score; stored
    in the ``.class`` file but never mirrored into the pose-data or summary
    caches.
    """
    return _read_class_details(get_class_path(image_filename))


def set_image_details(image_filename: str, rating: int | None) -> int | None:
    """Store (1-5) or clear (``None``/out-of-range) the details rating.

    Preserves the image's stage and flags. Returns the stored value (``None``
    when cleared). Intentionally does not touch the pose-data/summary caches.
    """
    path = get_class_path(image_filename)
    stage, flags = _read_class_file(path)
    rating = rating if rating in (1, 2, 3, 4, 5) else None
    _write_class_file(path, stage, flags, details=rating)
    return rating


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
