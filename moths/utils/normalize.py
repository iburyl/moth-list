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
    get_class_path,
)
from .annotations import (
    KEYPOINT_LABELS,
    get_class_and_flags_with_source,
    get_pose_path,
    get_prediction_class_path,
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
# circle's radius). Both pose layouts use the 80% circle (radius 0.4).
NORM_TOPDOWN_CIRCLE_RADIUS = 0.4
NORM_SIDE_CIRCLE_RADIUS = 0.4
# Top-down/bottom-up multiplier applied to the max center→keypoint distance to
# size the crop (1 / radius, so the farthest point lands on the circle).
NORM_TOPDOWN_CROP_SCALE = 1.0 / NORM_TOPDOWN_CIRCLE_RADIUS
# Side-view: output-y fraction where the horizontal F→P (or F→B) line is placed
# (lower third of the crop), or the crop centre when the wing sits close to the
# F→B axis (angle F→W vs F→B below NORM_SIDE_NARROW_DEG). Above
# NORM_SIDE_WIDE_DEG the layout falls back to F→B as the horizontal axis.
NORM_SIDE_FP_Y = 2.0 / 3.0
NORM_SIDE_NARROW_DEG = 30.0
NORM_SIDE_WIDE_DEG = 60.0
# Simplified bounding-box fallback: the box's wider side fills this fraction of
# the (square) crop, centred, with no rotation.
NORM_BBOX_FILL = 0.9
# Bump whenever the normalized-crop GEOMETRY changes (crop scale / circle radius,
# centering, side-view placement, ...). The version is embedded in the cached
# crop filename, so old-geometry crops are ignored and regenerated on the next
# view (no manual cache wipe), and stale files are cleaned up on regeneration.
# v1: unified 80% circle. v2: top-down back to the original 90% circle (radius
# 0.45); side view stays on the 80% circle (radius 0.4). v3: normalization is
# stage-aware — non-adult / normalization-suppressed / F&B-less images get an
# axis-aligned bounding-box crop; adults with F&B but no wing get the vertical
# F→B layout. v5: side view aligns F→P (P = whichever of B/W falls below the
# F→M line), keeps that line on the lower third, and uses the 90% circle. v6:
# side view centres the F→P line when F→W and F→B are within
# NORM_SIDE_NARROW_DEG of each other. v7: above NORM_SIDE_WIDE_DEG fall back to
# F→B as the horizontal axis (wing gets the larger free area). v8: both pose
# layouts use the 80% circle again (radius 0.4).
NORM_GEOM_VERSION = 8


def _angle_between(ax, ay, bx, by) -> float:
    """Angle in degrees between two input-space vectors (0 if either is zero)."""
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na == 0 or nb == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cos))


def _side_normalization(fx, fy, bx, by, lx, ly, rx, ry, l_present):
    """Full crop geometry ``(u_x, u_y, side, c, f)`` for a side-view image.

    A side view has F, B and exactly one wing W (L or R). Layout depends on the
    angle between F→W and F→B:

    * **Wide** (``> NORM_SIDE_WIDE_DEG``): fall back to F→B as the horizontal
      axis. The line sits on the lower- or upper-third line so W always gets the
      larger free area.
    * **Otherwise**: M is the B/W midpoint; in the F→M frame the point of B/W
      that falls *below* the axis becomes P, and F→P is laid horizontal. That
      line is centred when the angle is below ``NORM_SIDE_NARROW_DEG`` (folded
      wing), else on the lower-third line.

    In every case F faces the wing's side — left for L, right for R (the R
    layout is the L one rotated 180°, a *pure rotation*, never a mirror). The
    crop is scaled so the farthest of F, B and W lands on the reference circle
    (radius ``NORM_SIDE_CIRCLE_RADIUS``, centred in the image).

    ``u_x``/``u_y`` are the input-space unit directions output +x / +y map to;
    ``side`` is the square side in pixels; ``c``/``f`` are the affine
    translation terms (``px = a*ox + b*oy + c``, ``py = d*ox + e*oy + f`` with
    ``(a, d) = u_x`` and ``(b, e) = u_y``).
    """
    wx, wy = (lx, ly) if l_present else (rx, ry)
    wing_angle = _angle_between(wx - fx, wy - fy, bx - fx, by - fy)

    if wing_angle > NORM_SIDE_WIDE_DEG:
        # Wide wing: classic F→B horizontal.
        px, py = bx, by
        mx, my = (fx + bx) / 2, (fy + by) / 2
        length = math.hypot(bx - fx, by - fy) or 1.0
        ux, uy = (bx - fx) / length, (by - fy) / length
        if l_present:
            u_x, u_y = (ux, uy), (-uy, ux)
        else:
            u_x, u_y = (-ux, -uy), (uy, -ux)
        a, d = u_x
        b, e = u_y
        # Put the line on the third opposite the wing so W gets the free area.
        wing_dy = (wx - mx) * b + (wy - my) * e
        fp_y = NORM_SIDE_FP_Y if wing_dy <= 0 else (1.0 - NORM_SIDE_FP_Y)
    else:
        # Frame of the F→M axis decides which of B/W is the lower point P.
        bwx, bwy = (bx + wx) / 2, (by + wy) / 2
        axis_len = math.hypot(bwx - fx, bwy - fy) or 1.0
        aux, auy = (bwx - fx) / axis_len, (bwy - fy) / axis_len
        axis_y = (-auy, aux) if l_present else (auy, -aux)
        b_below = (bx - bwx) * axis_y[0] + (by - bwy) * axis_y[1]
        w_below = (wx - bwx) * axis_y[0] + (wy - bwy) * axis_y[1]
        px, py = (bx, by) if b_below >= w_below else (wx, wy)

        mx, my = (fx + px) / 2, (fy + py) / 2
        length = math.hypot(px - fx, py - fy) or 1.0
        ux, uy = (px - fx) / length, (py - fy) / length
        if l_present:
            u_x, u_y = (ux, uy), (-uy, ux)
        else:
            u_x, u_y = (-ux, -uy), (uy, -ux)
        a, d = u_x
        b, e = u_y
        # Folded wing: centre. Mildly raised: lower third.
        fp_y = 0.5 if wing_angle < NORM_SIDE_NARROW_DEG else NORM_SIDE_FP_Y

    # Solve for the scale k = 1/side (output fractions per input pixel). Each
    # point's output-fraction offset from the F/P (or F/B) midpoint is
    # d_i = k·((Q-M)·u_x, (Q-M)·u_y); F and the far axis end have zero
    # perpendicular component, so placing M on the chosen line puts the whole
    # axis there. The point's distance from the image centre is
    # |(k·dx, offset_y + k·dy)|; setting that to the circle radius gives a
    # quadratic in k. The binding (smallest positive) root keeps every point
    # inside the circle with the farthest one exactly on it.
    offset_y = fp_y - 0.5
    r2 = NORM_SIDE_CIRCLE_RADIUS * NORM_SIDE_CIRCLE_RADIUS
    ks = []
    for qx, qy in ((fx, fy), (bx, by), (wx, wy)):
        vx, vy = qx - mx, qy - my
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

    # M maps to output fraction (0.5, fp_y).
    omx, omy = 0.5 * side, fp_y * side
    c = mx - omx * a - omy * b
    f = my - omx * d - omy * e
    return u_x, u_y, side, c, f


def _bbox_normalization(ann, width: int, height: int):
    """Axis-aligned square crop centred on the bounding box (no rotation).

    The simplified fallback for images that don't get a pose layout (non-adult,
    normalization-suppressed, or adult without usable F/B keypoints). Returns
    ``(a, b, c, d, e, f, side)`` — the affine linear/translation terms plus the
    square side in pixels — sized so the box's wider side fills
    :data:`NORM_BBOX_FILL` of the crop and the box centre sits at the crop
    centre. ``None`` when the box is degenerate.
    """
    bw, bh = ann.width * width, ann.height * height
    if bw <= 0 or bh <= 0:
        return None
    cx, cy = ann.cx * width, ann.cy * height
    side = max(1, int(round(max(bw, bh) / NORM_BBOX_FILL)))
    a, d = 1.0, 0.0   # output +x -> input +x (no rotation)
    b, e = 0.0, 1.0   # output +y -> input +y
    half = side / 2
    c = cx - (a + b) * half
    f = cy - (d + e) * half
    return a, b, c, d, e, f, side


def compute_normalization(image_filename: str) -> dict | None:
    """Geometry of the normalized crop for an image (a square, canonical frame).

    The layout depends on the image's (effective) stage, flags and keypoints:

    * **Side view** (Adult, F+B and exactly one wing): see
      :func:`_side_normalization` — F→P (or F→B when the wing angle is wide)
      horizontal with F facing the wing's side, scaled to the 80% reference
      circle.
    * **Vertical F→B** (Adult, F+B present, both wings or neither): C is the F/B
      midpoint and F→B is vertical (F on top). The side is
      ``NORM_TOPDOWN_CROP_SCALE`` times the largest C→(present keypoint)
      distance, so the farthest known point lands on the 80% reference circle.
      (Both wings = the classic top-down layout; neither wing = the same layout
      sized on F/B alone.)
    * **Bounding box** (non-adult, normalization-suppressed by a flag, or adult
      without usable F/B): a simplified axis-aligned crop centred on the box,
      scaled so its wider side fills 90% of the square — see
      :func:`_bbox_normalization`. No reference circle.

    Returns ``None`` only when there is no annotation/box at all or the original
    image size is unavailable. The returned dict has:

    * ``side`` – the square side in pixels
    * ``affine`` – ``(a, b, c, d, e, f)`` mapping *output* (normalized crop) to
      *input* (original image) pixels, for ``Image.transform(..., AFFINE)``
    * ``circle_radius`` – the reference-circle radius as a fraction of the side
      (0.4 for both pose layouts), or ``None`` for the bounding-box layout
    * ``mode`` – ``"side"`` / ``"vertical"`` / ``"top-down"`` / ``"bbox"``
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
    ann = annotations[0]
    keypoints = ann.keypoints

    # Effective stage/flags (hand-preferred, else prediction) decide the layout,
    # matching how the species view groups the image.
    stage, flags, _cls_source = get_class_and_flags_with_source(image_filename)
    is_adult = stage == "Adult"
    suppressed = flags_suppress_normalization(flags)
    have_fb = (
        len(keypoints) >= 4
        and keypoints[0].visibility > 0
        and keypoints[3].visibility > 0
    )

    # Affine samples from the ORIGINAL image (output -> input mapping), so no
    # pixels are clipped by an intermediate rotate step. The pose layouts use
    # orthonormal (det +1) linear columns — always a pure rotation, never a
    # mirror/flip. Areas outside the source are filled grey.
    if is_adult and not suppressed and have_fb:
        front, left, right, back = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
        fx, fy = front.x * width, front.y * height
        bx, by = back.x * width, back.y * height
        lx, ly = left.x * width, left.y * height
        rx, ry = right.x * width, right.y * height
        l_present = left.visibility > 0
        r_present = right.visibility > 0

        if l_present != r_present:
            # Exactly one wing: side view (F→lower(B/W) on the lower third).
            u_x, u_y, side, c, f = _side_normalization(
                fx, fy, bx, by, lx, ly, rx, ry, l_present
            )
            a, d = u_x
            b, e = u_y
            circle_radius = NORM_SIDE_CIRCLE_RADIUS
            mode = "side"
        else:
            # Both wings (top-down) or neither (other adult): F→B vertical, F on
            # top, centred on the F/B midpoint; the farthest *present* keypoint
            # sits on the reference circle.
            cx, cy = (fx + bx) / 2, (fy + by) / 2
            length = math.hypot(bx - fx, by - fy) or 1.0
            ux, uy = (bx - fx) / length, (by - fy) / length
            u_x, u_y = (uy, -ux), (ux, uy)  # output +x perpendicular, +y along F→B
            pts = [(fx, fy), (bx, by)]
            if l_present:
                pts.append((lx, ly))
            if r_present:
                pts.append((rx, ry))
            max_dist = max(math.hypot(px - cx, py - cy) for px, py in pts)
            side = max(1, int(round(NORM_TOPDOWN_CROP_SCALE * max_dist)))
            half = side / 2
            a, d = u_x
            b, e = u_y
            c = cx - (a + b) * half
            f = cy - (d + e) * half
            circle_radius = NORM_TOPDOWN_CIRCLE_RADIUS
            mode = "top-down" if (l_present and r_present) else "vertical"
    else:
        # Simplified bounding-box crop (non-adult / suppressed / no F&B).
        bbox = _bbox_normalization(ann, width, height)
        if bbox is None:
            return None
        a, b, c, d, e, f, side = bbox
        circle_radius = None
        mode = "bbox"

    # Inverse (input -> output): the linear part is orthonormal (pose) or the
    # identity (bbox), so its inverse is its transpose. Places keypoints on the
    # normalized crop.
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
        "mode": mode,
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

    # Depend on the pose sources *and* the class sidecars: the layout is now
    # stage/flag-aware, so a stage or flag change must invalidate the crop too.
    dep_mtime = src.stat().st_mtime
    for source_path in (
        get_pose_path(image_filename),
        get_prediction_path(image_filename),
        get_class_path(image_filename),
        get_prediction_class_path(image_filename),
    ):
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
