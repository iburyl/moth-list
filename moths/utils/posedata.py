"""The cached ``<tax_id>_pose_data.json`` (rows, scores, tax thumbnail)."""
from __future__ import annotations

import json

from pathlib import Path

from .paths import (
    get_image_size,
    image_basename,
    scan_tax_images,
)
from .classes import (
    FLAGS,
    get_image_flags,
    load_starred,
)
from .annotations import (
    POSE_TOP_DOWN,
    _pose_source_keypoints,
    classify_pose,
    get_label_dir,
)
from .metrics import (
    compute_sharpness,
    cumulative_score,
    pose_pixel_span,
    pose_symmetry_metric,
)
from .normalize import (
    clear_normalized,
    touch_normalized,
)
from .wingstats import (
    build_side_wing_stats,
    build_wing_stats,
)


def get_tax_thumbnail(tax_id: str) -> str | None:
    """Return the image filename chosen as ``tax_id``'s representative thumbnail.

    The choice is stored alongside the counts in the tax's summary JSON (see
    :func:`build_summary`). Returns ``None`` when none recorded yet.
    """
    from .summary import load_summary  # local import: breaks posedata<->summary cycle

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
    from .summary import build_summary, load_summary, _write_summary  # local import: breaks cycle

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
# v12: sharpness measured on the bounding-box centre of the original image
# (central 3x3 of a 5x5 split) instead of the normalized crop, and available
# for any annotated image (not just top-down).
# v13: sharpness sub-score divisor lowered from 300000 to 30000.
# v14: predicted ``side`` views are now trusted (dedicated side-view model) and
# no longer collapsed to ``unclear``; only predicted ``bottom_up`` still is.
# v15: pose classification is fully source-independent — predictions now encode
# uncertainty via keypoint visibility, so no source-based collapse remains.
# v16: sharpness sub-score is now a log-linear fit to the hand details ratings
# (clamp(a*ln(sharpness)+b, 0, 1)) instead of ``min(sharpness, 30000)/30000``.
POSE_DATA_VERSION = 16


def compute_pose_row(image_filename: str) -> dict:
    """Compute the pose class, metrics and score for one image (the slow path).

    Records the ``source`` (``pose``/``prediction``/``None``) and the raw
    keypoints used, so a later visit can detect when they've changed. Also
    records the original image ``width``/``height`` (``None`` when unavailable),
    persisting the intrinsic size here instead of a ``.size`` sidecar file.
    Sharpness is measured on the bounding box centre of the original image, so
    it is computed for any annotated image (see :func:`compute_sharpness`).
    """
    keypoints, source = _pose_source_keypoints(image_filename)
    pose = classify_pose(image_filename)
    size = get_image_size(image_filename)
    width, height = size if size else (None, None)
    symmetry = pose_symmetry_metric(image_filename)
    pixel_span = pose_pixel_span(image_filename)
    # Sharpness only needs a bounding box now, so it is available for any
    # annotated image (not just top-down).
    sharpness = compute_sharpness(image_filename)
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
    representative thumbnail and the wing-position stats image. Returns the
    freshly built data dict.
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
    build_wing_stats(tax_id, per_image)
    build_side_wing_stats(tax_id, per_image)
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
