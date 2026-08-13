"""The cached ``<tax_id>_pose_data.json`` (rows, scores, tax thumbnail)."""
from __future__ import annotations

import json

from pathlib import Path

from .paths import (
    get_image_size,
    image_basename,
)
from .classes import (
    FLAGS,
    load_starred,
)
from .annotations import (
    POSE_NONE,
    _pose_source_keypoints,
    classify_pose,
    get_class_and_flags_with_source,
    get_label_dir,
)
from .groups import target_group_ids, unified_group_order
from .metrics import (
    LEGACY_METRIC_VERSIONS,
    METRIC_VERSIONS,
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

    Follows the poses page's group order (:func:`unified_group_order`):

    A. Among **starred** images, walk the groups top to bottom; the first group
       that holds a scored image wins, and within it the highest-scoring image.
    B. If no starred image has a score, repeat the same walk over the
       **non-starred** images.

    So a starred image always beats any non-starred one, and otherwise the
    earliest group (top-down before side, etc.) decides — matching what you see
    first on the species page. Stage/flags are read live (as the page does) so
    the choice tracks the current grouping; pose and score come from the cached
    row. Returns ``{"filename", "score", "starred"}`` or ``None`` when no image
    has a score yet. Pure w.r.t. the summary cache, so it is safe to call from
    both :func:`build_summary` and :func:`refresh_tax_thumbnail`.
    """
    if per_image is None:
        data = load_pose_data(tax_id)
        per_image = data.get("images", {}) if data else {}

    starred = load_starred(tax_id)
    order = unified_group_order()

    # Best (highest score) scored image per group, split by starred vs not.
    best_starred: dict[str, tuple[float, str]] = {}
    best_other: dict[str, tuple[float, str]] = {}
    for filename, row in per_image.items():
        score = row.get("score")
        if score is None:
            continue
        stage, raw_flags, _src = get_class_and_flags_with_source(filename)
        flags = [f for f in FLAGS if f in set(raw_flags or [])]
        pose = row.get("pose", POSE_NONE)
        target = best_starred if image_basename(filename) in starred else best_other
        for gid in target_group_ids(stage, flags, pose):
            cur = target.get(gid)
            if cur is None or score > cur[0]:
                target[gid] = (score, filename)

    def _first_in_order(best: dict[str, tuple[float, str]]):
        for gid in order:
            if gid in best:
                return best[gid]  # (score, filename)
        return None

    winner = _first_in_order(best_starred)
    is_starred = winner is not None
    if winner is None:
        winner = _first_in_order(best_other)
    if winner is None:
        return None
    score, filename = winner
    return {"filename": filename, "score": score, "starred": is_starred}


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
# v17: sharpness score re-anchored to log(sharpness) 9->0.5, 11->1.0
# (a=0.25, b=-1.75) — a gentler low end than the fitted line.
# NOTE: from here on, changes to an individual metric's computation are tracked
# by that metric's own version (see moths.utils.metrics.METRIC_VERSIONS), which
# is recorded in the file (``metric_versions``) and lets a rebuild recompute
# only the changed metric. Bump POSE_DATA_VERSION only for *structural* row
# schema changes (not for metric formula tweaks). Pre-metric-version files (no
# ``metric_versions`` key) are assumed to hold the v17-current metric values.
POSE_DATA_VERSION = 17


# Sentinel version for a cached metric that must be recomputed on the next
# explicit rebuild (e.g. a heavy metric the light path could not refresh after
# the keypoints changed). It never matches a real (>=1) metric version.
_STALE_METRIC_VERSION = 0


def _pose_inputs(image_filename: str):
    """Snapshot cached per row: keypoints/source plus effective stage/flags.

    Stage and flags are the source-aware values (hand class, else prediction),
    so the cache records what the app actually treats the image as. The metric
    values depend only on the keypoints/box, so metric reuse keys on
    keypoints/source; stage and flags are cached for reference.
    """
    keypoints, source = _pose_source_keypoints(image_filename)
    stage, flags, _src = get_class_and_flags_with_source(image_filename)
    flags = [f for f in FLAGS if f in set(flags or [])]
    return keypoints, source, stage, flags


def _row_metric_version(prev_row, file_versions, metric: str) -> int:
    """Version the cached ``metric`` value in ``prev_row`` was computed at.

    A per-image ``metric_versions`` override wins (stamped when the normalized
    view refreshes a single row); otherwise the file-level ``metric_versions``;
    a cache with neither is assumed to hold the legacy (v1) values (see
    :data:`moths.utils.metrics.LEGACY_METRIC_VERSIONS`) so a later bump still
    forces a recompute.
    """
    if isinstance(prev_row, dict):
        override = prev_row.get("metric_versions")
        if isinstance(override, dict) and metric in override:
            return override[metric]
    if isinstance(file_versions, dict) and metric in file_versions:
        return file_versions[metric]
    # No recorded version: a pre-scheme cache, assumed to hold the legacy (v1)
    # values so a later bump still forces this metric's recompute.
    return LEGACY_METRIC_VERSIONS.get(metric, METRIC_VERSIONS[metric])


def pose_row_needs_rebuild(row, file_versions) -> bool:
    """True when a cached ``row``'s metrics are stale (a rebuild would help).

    Stale when the row is explicitly flagged (``needs_rebuild``, set after a
    manual keypoint edit) or any metric's effective version (per-image override,
    else file-level ``file_versions``, else the legacy baseline) differs from
    the current one. Used by the species view to show a per-plate "rebuild" note.
    """
    if not isinstance(row, dict):
        return False
    if row.get("needs_rebuild"):
        return True
    return any(
        _row_metric_version(row, file_versions, metric) != current
        for metric, current in METRIC_VERSIONS.items()
    )


def _build_pose_row(image_filename, prev_row, file_versions, *, allow_heavy):
    """Compute one pose row, reusing still-valid cached metric values.

    A metric is recomputed only when the pose inputs (keypoints/source) changed
    or its recorded version differs from the current one. The heavy sharpness
    metric is additionally never recomputed unless ``allow_heavy`` (the explicit
    rebuild path): off that path a stale sharpness keeps its old value but is
    stamped with a sentinel version so the next rebuild refreshes it. Also
    records ``stage``/``flags`` and the intrinsic ``width``/``height``. Returns
    ``(row, obj_versions)`` where ``obj_versions`` is the per-metric version the
    row actually holds.
    """
    keypoints, source, stage, flags = _pose_inputs(image_filename)
    pose = classify_pose(image_filename)
    inputs_changed = (
        not isinstance(prev_row, dict)
        or prev_row.get("keypoints") != keypoints
        or prev_row.get("source") != source
    )

    if (
        isinstance(prev_row, dict)
        and not inputs_changed
        and prev_row.get("width") is not None
    ):
        width, height = prev_row.get("width"), prev_row.get("height")
    else:
        size = get_image_size(image_filename)
        width, height = size if size else (None, None)

    obj_versions: dict[str, int] = {}

    def resolve(metric, compute, *, heavy=False):
        recorded = _row_metric_version(prev_row, file_versions, metric)
        stale = inputs_changed or recorded != METRIC_VERSIONS[metric]
        if not stale:
            obj_versions[metric] = recorded
            return prev_row.get(metric) if isinstance(prev_row, dict) else None
        if allow_heavy or not heavy:
            obj_versions[metric] = METRIC_VERSIONS[metric]
            return compute()
        # Stale heavy metric we may not recompute here: keep the old value but
        # record a sentinel so the next explicit rebuild recomputes it.
        obj_versions[metric] = _STALE_METRIC_VERSION
        return prev_row.get(metric) if isinstance(prev_row, dict) else None

    symmetry = resolve("symmetry", lambda: pose_symmetry_metric(image_filename))
    pixel_span = resolve("pixel_span", lambda: pose_pixel_span(image_filename))
    sharpness = resolve(
        "sharpness", lambda: compute_sharpness(image_filename), heavy=True
    )

    row = {
        "pose": pose,
        "source": source,
        "keypoints": keypoints,
        "stage": stage,
        "flags": flags,
        "width": width,
        "height": height,
        "symmetry": symmetry,
        "pixel_span": pixel_span,
        "sharpness": sharpness,
        "score": cumulative_score(symmetry, pixel_span, sharpness),
    }
    return row, obj_versions


def compute_pose_row(image_filename: str) -> dict:
    """Compute a fresh pose row with every metric recomputed at the current
    version (the slow path). Records ``source``/``keypoints``/``stage``/
    ``flags`` and the intrinsic ``width``/``height`` alongside the metrics.
    """
    row, _versions = _build_pose_row(image_filename, None, None, allow_heavy=True)
    return row


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
    the parsed dict is returned so the species view can still display the images
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
    prev_file_versions = prev.get("metric_versions") if prev else None

    per_image = {}
    for image in images:
        filename = image.filename
        prev_row = prev_images.get(filename)
        # Explicit rebuild path: allow_heavy so any stale metric (incl.
        # sharpness) is recomputed, but a metric whose version and keypoints are
        # unchanged is reused untouched. After this pass every row is at the
        # current versions, so no per-row override is kept (file-level records
        # them).
        row, _versions = _build_pose_row(
            filename, prev_row, prev_file_versions, allow_heavy=True
        )
        if prev_row is not None \
                and prev_row.get("keypoints") == row["keypoints"] \
                and prev_row.get("source") == row["source"]:
            # Keypoints unchanged since the last cache (even across a version
            # bump): reuse the existing crop/thumbnail instead of regenerating.
            touch_normalized(filename)
        elif prev_row is not None:
            # Keypoints changed: force the crop/thumbnail to be rebuilt.
            clear_normalized(filename)
        # No prior row: leave get_or_create_normalized's mtime cache to decide.
        per_image[filename] = row

    data = {
        "version": POSE_DATA_VERSION,
        "metric_versions": dict(METRIC_VERSIONS),
        "images": per_image,
    }
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
    """Light-refresh one image's cached pose row for the normalized view.

    Recomputes only the *cheap* metrics (symmetry, pixel span) when the image's
    keypoints changed or a metric's version is stale, and refreshes the
    normalized crop; the heavy **sharpness** metric is never recomputed here
    (only the explicit rebuild does) — a stale sharpness keeps its old value and
    is stamped with a per-row sentinel version so a later rebuild refreshes it.
    The refreshed row's per-metric versions are documented in a per-image
    ``metric_versions`` override when they differ from the file-level baseline.
    Returns the current row. This is the only place recomputation is triggered
    outside an explicit rebuild — the species view reads the cache as-is.
    """
    data = load_pose_data_raw(tax_id)
    if data is None:
        # No cache at all: compute *only* this image for display. Rebuilding the
        # whole tax here would recompute (and normalize) every image of the
        # species as a side effect of opening one — surprising and slow. The
        # explicit Rebuild button (or tools/rebuild_poses.py) builds the full
        # cache, so we don't persist into a missing file.
        return compute_pose_row(image_filename)

    images = data.get("images", {})
    prev_row = images.get(image_filename)
    file_versions = data.get("metric_versions")

    # Refresh the cheap metrics only (allow_heavy=False keeps sharpness cached).
    row, obj_versions = _build_pose_row(
        image_filename, prev_row, file_versions, allow_heavy=False
    )

    # Stamp a per-image override only when this row's actual metric versions
    # differ from the file-level baseline (so e.g. a lagging sharpness is
    # documented per object without touching the other rows).
    baseline = file_versions if isinstance(file_versions, dict) else METRIC_VERSIONS
    if any(
        obj_versions.get(m) != baseline.get(m, METRIC_VERSIONS[m])
        for m in METRIC_VERSIONS
    ):
        row["metric_versions"] = obj_versions

    keypoints_changed = not (
        isinstance(prev_row, dict)
        and prev_row.get("keypoints") == row["keypoints"]
        and prev_row.get("source") == row["source"]
    )
    # Rewrite only when something actually changed (values, versions, or a
    # lingering needs_rebuild flag to clear); avoids needless disk churn.
    had_flag = bool(isinstance(prev_row, dict) and prev_row.get("needs_rebuild"))
    if row != prev_row or had_flag:
        if keypoints_changed:
            clear_normalized(image_filename)
        images[image_filename] = row
        data["images"] = images
        _write_pose_data(tax_id, data)
        refresh_tax_thumbnail(tax_id, images)
    return row


def mark_pose_row_stale(tax_id: str, image_filename: str) -> None:
    """Flag one image's cached pose row for rebuild after a manual keypoint edit.

    Sets ``needs_rebuild`` on the row in ``<tax_id>_pose_data.json`` without
    recomputing anything, so the species view can show a "to be rebuild" note.
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
    # Flags can switch the normalization layout (e.g. a suppress-normalization
    # flag falls back to the bbox crop), so drop the cached crop to force a
    # rebuild on the next view.
    clear_normalized(image_filename)
