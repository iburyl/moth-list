"""Pose normalization geometry and normalized-crop / thumbnail generation."""
from __future__ import annotations

import math

from pathlib import Path

from django.conf import settings

from .paths import (
    _tax_subdir,
    get_image_dir,
    get_image_path,
    get_image_size,
    get_thumbnail_dir,
    image_basename,
)
from .classes import (
    flags_suppress_normalization,
    get_image_flags,
)
from .annotations import (
    KEYPOINT_LABELS,
    get_pose_path,
    get_prediction_path,
    load_pose_source,
)


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


# Neutral grey used to fill areas exposed by rotation/cropping.
NORM_FILL = (128, 128, 128)
# Fraction of the crop side at which the farthest keypoint sits (= the reference
# circle's radius). Top-down/bottom-up keep the original 90% circle (radius
# 0.45); side view uses an 80% circle (radius 0.4).
NORM_TOPDOWN_CIRCLE_RADIUS = 0.45
NORM_SIDE_CIRCLE_RADIUS = 0.4
# Top-down/bottom-up multiplier applied to the max center→keypoint distance to
# size the crop (1 / radius, so the farthest point lands on the circle).
NORM_TOPDOWN_CROP_SCALE = 1.0 / NORM_TOPDOWN_CIRCLE_RADIUS
# Side-view: output-y fraction the horizontal F→B line is placed on (lower-third
# line), leaving room above for the raised wing.
NORM_SIDE_FB_Y = 2.0 / 3.0
# Bump whenever the normalized-crop GEOMETRY changes (crop scale / circle radius,
# centering, side-view placement, ...). The version is embedded in the cached
# crop filename, so old-geometry crops are ignored and regenerated on the next
# view (no manual cache wipe), and stale files are cleaned up on regeneration.
# v1: unified 80% circle. v2: top-down back to the original 90% circle (radius
# 0.45); side view stays on the 80% circle (radius 0.4).
NORM_GEOM_VERSION = 2


def _side_normalization(fx, fy, bx, by, lx, ly, rx, ry, l_present):
    """Full crop geometry ``(u_x, u_y, side, c, f)`` for a side-view image.

    A side view has F, B and exactly one wing (L or R). The F→B line is laid
    horizontal with F facing the wing's side — F on the left for L, on the right
    for R (the R layout is the L one rotated 180°, a *pure rotation*, never a
    mirror, so the moth is never flipped legs-up). The line is placed on the
    lower-third line (``NORM_SIDE_FB_Y``) when the wing points up, or the
    upper-third line when the wing points down, so the wing always gets the
    larger free area. The crop is scaled so the farthest of the three points
    lands on the reference circle (radius ``NORM_SIDE_CIRCLE_RADIUS``, centred in
    the image).

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
    r2 = NORM_SIDE_CIRCLE_RADIUS * NORM_SIDE_CIRCLE_RADIUS
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

    The crop is a square, centred on C and rotated to a canonical orientation.
    Two layouts are supported:

    * **Top-down / bottom-up** (all four F/B/L/R present): C is the F/B midpoint
      and the F→B line is vertical (F on top, B on bottom). The side is
      ``NORM_TOPDOWN_CROP_SCALE`` times the largest C→keypoint distance, so the
      farthest keypoint lands on the 90% reference circle.
    * **Side view** (F, B and exactly one wing): see
      :func:`_side_normalization` — the F→B line is laid horizontal with F
      facing the wing's side, on the image's lower-third line, scaled to the 80%
      reference circle.

    Returns ``None`` when it can't be determined (missing prediction, keypoints,
    or original image size). The returned dict has:

    * ``side`` – the square side in pixels
    * ``affine`` – ``(a, b, c, d, e, f)`` mapping *output* (normalized crop) to
      *input* (original image) pixels, for ``Image.transform(..., AFFINE)``
    * ``circle_radius`` – the reference-circle radius as a fraction of the side
      (0.45 top-down / 0.4 side), so the view can draw the matching circle
    * ``keypoints`` – the first object's keypoints re-expressed as fractions of
      the normalized crop: ``{"x", "y", "v", "label"}`` (``x``/``y`` are
      ``None`` when the keypoint is unlabeled)
    """
    # Flags like "Mating" opt an image out of normalization entirely.
    if flags_suppress_normalization(get_image_flags(image_filename)):
        return None

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
        side = max(1, int(round(NORM_TOPDOWN_CROP_SCALE * max_dist)))
        half = side / 2
        a, d = u_x
        b, e = u_y
        c = cx - (a + b) * half
        f = cy - (d + e) * half
        circle_radius = NORM_TOPDOWN_CIRCLE_RADIUS
    elif l_present != r_present:
        # Exactly one wing: side view (F→B on the lower-third line).
        u_x, u_y, side, c, f = _side_normalization(
            fx, fy, bx, by, lx, ly, rx, ry, l_present
        )
        a, d = u_x
        b, e = u_y
        circle_radius = NORM_SIDE_CIRCLE_RADIUS
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

    return {
        "side": side,
        "affine": (a, b, c, d, e, f),
        "circle_radius": circle_radius,
        "keypoints": mapped,
    }


def get_or_create_normalized(image_filename: str):
    """Build a pose-normalized crop and its thumbnail; return ``(norm, thumb)``.

    Geometry comes from :func:`compute_normalization`. Outputs (both in
    ``MOTHS_THUMBNAIL_DIR/<tax_id>/``):

    * ``<name>.norm.v<N>.jpg`` – the full-resolution normalized crop
    * ``<name>.norm-thumb.v<N>.jpg`` – a shrunk thumbnail of that crop

    (``<N>`` is :data:`NORM_GEOM_VERSION`; crops from an older geometry version
    are ignored and cleaned up here.)

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

    cache_dir, norm_path, thumb_path = _normalized_paths(image_filename)

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

    # Drop any old-geometry (or legacy unversioned) crops for this image now that
    # a current-version crop exists, so the cache doesn't accumulate stale files.
    current = {norm_path.name, thumb_path.name}
    for path in _iter_normalized_files(cache_dir, image_filename):
        if path.name not in current:
            try:
                path.unlink()
            except OSError:
                pass

    return norm_path, thumb_path


def _normalized_paths(image_filename: str):
    """Return ``(cache_dir, norm_path, thumb_path)`` for the current geometry.

    The :data:`NORM_GEOM_VERSION` is baked into the filenames, so a geometry
    change points at fresh names and old-geometry crops are simply ignored.
    """
    name = image_basename(image_filename)
    cache_dir = _tax_subdir(get_thumbnail_dir().resolve(), name)
    v = NORM_GEOM_VERSION
    norm_path = cache_dir / f"{name}.norm.v{v}.jpg"
    thumb_path = cache_dir / f"{name}.norm-thumb.v{v}.jpg"
    return cache_dir, norm_path, thumb_path


def _iter_normalized_files(cache_dir, image_filename):
    """Yield every cached normalized crop/thumb for this image, any version
    (including the legacy unversioned ``<name>.norm.jpg`` names)."""
    name = image_basename(image_filename)
    if not cache_dir.is_dir():
        return
    prefixes = (name + ".norm.", name + ".norm-thumb.")
    for entry in cache_dir.iterdir():
        if entry.is_file() and entry.name.startswith(prefixes):
            yield entry


def clear_normalized(image_filename: str) -> None:
    """Delete the cached normalized crop + thumbnail (all versions) so they get
    rebuilt."""
    cache_dir = _tax_subdir(get_thumbnail_dir().resolve(), image_basename(image_filename))
    for path in _iter_normalized_files(cache_dir, image_filename):
        try:
            path.unlink()
        except OSError:
            pass


def touch_normalized(image_filename: str) -> None:
    """Bump the current-geometry normalized crop + thumbnail mtimes to now.

    Used during a rebuild when the keypoints are unchanged: freshening the
    mtimes lets :func:`get_or_create_normalized` reuse the existing files
    (its cache check is mtime-based) instead of regenerating them. Only the
    current-``NORM_GEOM_VERSION`` files are touched; old-geometry crops are left
    to be cleaned up when the crop is regenerated.
    """
    _cache_dir, norm_path, thumb_path = _normalized_paths(image_filename)
    for path in (norm_path, thumb_path):
        if path.is_file():
            try:
                path.touch()
            except OSError:
                pass
