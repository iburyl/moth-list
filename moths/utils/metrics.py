"""Per-image quality metrics (symmetry, pixel span, sharpness) and scoring."""
from __future__ import annotations

import math

from .paths import (
    get_image_path,
    get_image_size,
)
from .annotations import (
    Annotation,
    _valid_frlb,
    load_pose_source,
)


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


# Sharpness settings. The metric is measured on the central 3x3 cells of a 5x5
# split of the annotation's bounding box, cropped from the original image and
# resized to a fixed square so the value doesn't depend on the source
# resolution (a bigger image would otherwise have gentler per-pixel gradients).
SHARPNESS_SIZE = 1024             # side the cropped box centre is resized to
SHARPNESS_CENTER_FRAC = 3.0 / 5.0  # central 3 of a 5x5 split, per axis
# This metric is cached only in the per-tax ``<tax_id>_pose_data.json``; bump
# POSE_DATA_VERSION when the scoring method below changes so those are rebuilt.

# Sharpness -> 0..1 score calibration: a straight line in ``log(sharpness)``,
# ``clamp(a*ln(sharpness) + b, 0, 1)``. Fitted against the hand 1-5 *details*
# ratings (see tools/metrics_statistics.py --details-md) so that the sharpness
# typical of a ~2.5-star image scores 0.5 and a ~4.5-star image scores 1.0.
SHARPNESS_SCORE_A = 0.346483
SHARPNESS_SCORE_B = -2.824179


def compute_sharpness(image_filename: str) -> float | None:
    """Scharr/Tenengrad gradient energy over the centre of the bounding box.

    Takes the first annotation's bounding box, keeps its central 3/5 in each
    axis (the middle 3x3 cells of a 5x5 split), crops that region straight from
    the original image, resizes it to :data:`SHARPNESS_SIZE` square (for
    resolution independence) and returns the mean gradient energy. Any
    annotation with a box qualifies — keypoints are not required. Returns
    ``None`` when there is no box or the crop is degenerate.
    """
    annotations, _source = load_pose_source(image_filename)
    if not annotations:
        return None
    ann = annotations[0]
    if ann.width <= 0 or ann.height <= 0:
        return None
    size = get_image_size(image_filename)
    if size is None:
        return None
    width, height = size

    half = SHARPNESS_CENTER_FRAC / 2  # fraction of the box extent on each side
    x0 = max(0.0, ann.cx - ann.width * half)
    x1 = min(1.0, ann.cx + ann.width * half)
    y0 = max(0.0, ann.cy - ann.height * half)
    y1 = min(1.0, ann.cy + ann.height * half)

    left, right = int(round(x0 * width)), int(round(x1 * width))
    top, bottom = int(round(y0 * height)), int(round(y1 * height))
    if right - left < 3 or bottom - top < 3:
        return None

    import numpy as np
    from PIL import Image

    with Image.open(get_image_path(image_filename)) as im:
        crop = im.convert("L").crop((left, top, right, bottom))
        crop = crop.resize((SHARPNESS_SIZE, SHARPNESS_SIZE), Image.LANCZOS)
        arr = np.asarray(crop, dtype=np.float64)

    # --- Sharpness (Scharr/Tenengrad gradient energy over the whole crop) ---
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
    if energy.size == 0:
        return None
    return float(energy.mean())


def sharpness_score(sharpness: float | None) -> float | None:
    """Scale raw Tenengrad ``sharpness`` energy to a ``0..1`` score.

    A straight line in ``log(sharpness)`` (:data:`SHARPNESS_SCORE_A` /
    :data:`SHARPNESS_SCORE_B`) clamped to ``[0, 1]`` and calibrated to the hand
    details ratings. ``None`` passes through; non-positive energy scores ``0``.
    """
    if sharpness is None:
        return None
    if sharpness <= 0:
        return 0.0
    return max(0.0, min(1.0, SHARPNESS_SCORE_A * math.log(sharpness) + SHARPNESS_SCORE_B))


def score_components(
    symmetry: float | None,
    pixel_span: float | None,
    sharpness: float | None,
) -> tuple[float | None, float | None, float | None]:
    """Return the three scaled ``0..1`` sub-scores summed by ``cumulative_score``.

    Each element is ``None`` when its input metric is ``None``:

    1. ``1 - symmetry`` (symmetry is 0 when perfectly symmetric).
    2. ``(clamp(pixel_span, 300, 1300) - 300) / 1000`` -> 0..1 over 300..1300 px.
    3. :func:`sharpness_score` -> ``clamp(a*ln(sharpness) + b, 0, 1)``, a
       log-linear fit to the hand details ratings.
    """
    s_sym = None if symmetry is None else 1 - symmetry
    s_pixels = (
        None
        if pixel_span is None
        else (max(min(pixel_span, 1300), 300) - 300) / 1000
    )
    s_sharp = sharpness_score(sharpness)
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
