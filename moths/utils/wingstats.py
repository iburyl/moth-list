"""Wing-position scatter images built from a tax's pose data."""
from __future__ import annotations

from pathlib import Path

from .paths import (
    _unlink_quiet,
    get_thumbnail_dir,
)
from .annotations import (
    POSE_SIDE,
    POSE_TOP_DOWN,
)


# --- Wing-position statistics images -----------------------------------------
# When a tax has at least WING_STATS_MIN usable poses, build_pose_data renders
# a scatter of every wing position with each pose aligned so the F/B midpoint
# sits at the image centre and the F→B line runs along a centre axis. The scale
# is data-driven: the farthest point (by x or y) is placed just inside the
# border (WING_STATS_MARGIN). Two variants are cached as PNGs for the poses-view
# sidebar and visualise the spread/clusters of wing placement:
#   * top-down: F→B down the vertical centre line (F above), both L and R drawn.
#   * side view: F→B along the horizontal centre line (F left, B right); the one
#     visible wing is drawn, and R-views are mirrored across the F→B axis so they
#     overlay the L-views (i.e. all wings land on the same side).
WING_STATS_MIN = 3
WING_STATS_SIZE = 512
WING_STATS_DOT_RADIUS = 4
WING_STATS_ALPHA = 0.45
WING_STATS_MARGIN = 0.06              # fraction of the side kept clear at the border
WING_STATS_BG = (255, 255, 255)       # white — keep it a light, unobtrusive aside
WING_STATS_AXIS = (200, 200, 205)     # faint central F→B axis line
WING_STATS_MARK = (40, 40, 40)        # F/B markers + labels (dark on white)
WING_STATS_L_COLOR = (59, 130, 246)   # blue
WING_STATS_R_COLOR = (239, 68, 68)    # red
WING_STATS_WING_COLOR = (124, 58, 237)  # purple — single overlaid side-view wing


def get_wing_stats_path(tax_id: str) -> Path:
    """Path to the cached top-down wing-position scatter PNG for a tax_id."""
    return get_thumbnail_dir().resolve() / f"{tax_id}_wing_stats.png"


def get_side_wing_stats_path(tax_id: str) -> Path:
    """Path to the cached side-view wing-position scatter PNG for a tax_id."""
    return get_thumbnail_dir().resolve() / f"{tax_id}_side_wing_stats.png"


def _top_down_wing_points(per_image: dict):
    """Yield ``(L, R)`` wing positions in a common F/B-normalized frame.

    Each top-down pose is aligned by the unique similarity (rotation + uniform
    scale, no reflection — so left/right is preserved) that maps F to
    ``(0, -0.5)`` and B to ``(0, +0.5)``: the F/B midpoint is the origin, the
    F→B line is vertical (F above), and the F↔B distance is 1. L and R are then
    in units of the F↔B length relative to the midpoint, so poses of different
    sizes are directly comparable. Working in original-pixel space keeps the
    transform a true similarity on non-square images. Flagged images
    (Pinned/Macro/Damaged/...) and poses without four visible keypoints or a
    known original size are skipped.
    """
    for row in per_image.values():
        if row.get("pose") != POSE_TOP_DOWN:
            continue
        # Flagged images (Pinned/Macro/Damaged/...) are excluded from the stats.
        if row.get("flags"):
            continue
        kps = row.get("keypoints")
        width = row.get("width")
        height = row.get("height")
        if not kps or len(kps) < 4 or not width or not height:
            continue
        if min(kp[2] for kp in kps[:4]) <= 0:
            continue
        f, l, r, b = (complex(kp[0] * width, kp[1] * height) for kp in kps[:4])
        if b == f:
            continue
        a = 1j / (b - f)          # maps F->-0.5j, B->+0.5j (midpoint at origin)
        t = -0.5j - a * f
        yield (a * l + t, a * r + t)


def _side_wing_points(per_image: dict):
    """Yield single wing positions for side views in a common F/B frame.

    Each side pose (F, B and exactly one of L/R visible) is aligned by the
    unique similarity that maps F to ``-0.5`` and B to ``+0.5`` on the real
    axis: the F/B midpoint is the origin, the F→B line is horizontal (F left, B
    right) and the F↔B distance is 1. The one visible wing is yielded in that
    frame; R-views are reflected across the F→B axis (negate the imaginary part)
    so they overlay the L-views — all wings then land on the same side. Flagged
    images and poses without a known original size are skipped.
    """
    for row in per_image.values():
        if row.get("pose") != POSE_SIDE:
            continue
        if row.get("flags"):
            continue
        kps = row.get("keypoints")
        width = row.get("width")
        height = row.get("height")
        if not kps or len(kps) < 4 or not width or not height:
            continue
        f = complex(kps[0][0] * width, kps[0][1] * height)
        l = complex(kps[1][0] * width, kps[1][1] * height)
        r = complex(kps[2][0] * width, kps[2][1] * height)
        b = complex(kps[3][0] * width, kps[3][1] * height)
        if kps[0][2] <= 0 or kps[3][2] <= 0 or b == f:
            continue
        l_vis, r_vis = kps[1][2] > 0, kps[2][2] > 0
        if l_vis and not r_vis:
            wing, is_r = l, False
        elif r_vis and not l_vis:
            wing, is_r = r, True
        else:
            continue
        a = 1.0 / (b - f)         # maps F->-0.5, B->+0.5 on the real axis
        t = -0.5 - a * f
        w = a * wing + t
        if is_r:
            w = complex(w.real, -w.imag)   # mirror R across F→B to overlay L
        yield w


def _render_wing_scatter(out_path: Path, colored_points, f_norm: complex,
                         b_norm: complex, orientation: str) -> Path | None:
    """Render a wing-position scatter to ``out_path`` and return it (or None).

    ``colored_points`` is a list of ``(complex_point, rgb)`` in the F/B frame.
    ``f_norm``/``b_norm`` are the F and B positions in that frame; ``orientation``
    is ``"v"`` (F→B vertical) or ``"h"`` (F→B horizontal) and only picks which
    centre axis line to draw. Scale is data-driven so the farthest point sits
    ``WING_STATS_MARGIN`` in from the border.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    side = WING_STATS_SIZE
    radius = WING_STATS_DOT_RADIUS
    alpha = WING_STATS_ALPHA

    extent = 0.0
    for w in (f_norm, b_norm):
        extent = max(extent, abs(w.real), abs(w.imag))
    for w, _c in colored_points:
        extent = max(extent, abs(w.real), abs(w.imag))
    extent = extent or 0.5
    scale = (0.5 - WING_STATS_MARGIN) / extent

    def to_px(w: complex) -> tuple[float, float]:
        return ((0.5 + w.real * scale) * side, (0.5 + w.imag * scale) * side)

    canvas = np.empty((side, side, 3), dtype=float)
    canvas[:, :] = WING_STATS_BG

    def stamp(px: float, py: float, color) -> None:
        x0, x1 = max(0, int(px - radius)), min(side, int(px + radius) + 1)
        y0, y1 = max(0, int(py - radius)), min(side, int(py + radius) + 1)
        if x0 >= x1 or y0 >= y1:
            return
        ys, xs = np.ogrid[y0:y1, x0:x1]
        mask = (xs + 0.5 - px) ** 2 + (ys + 0.5 - py) ** 2 <= radius * radius
        sub = canvas[y0:y1, x0:x1]
        sub[mask] = sub[mask] * (1 - alpha) + np.array(color, dtype=float) * alpha

    for w, color in colored_points:
        stamp(*to_px(w), color)

    img = Image.fromarray(canvas.clip(0, 255).astype("uint8"), "RGB")
    draw = ImageDraw.Draw(img)
    c = side / 2
    if orientation == "v":
        draw.line([(c, 0), (c, side)], fill=WING_STATS_AXIS, width=1)
    else:
        draw.line([(0, c), (side, c)], fill=WING_STATS_AXIS, width=1)
    rr = 5
    for w, label in ((f_norm, "F"), (b_norm, "B")):
        px, py = to_px(w)
        draw.ellipse([px - rr, py - rr, px + rr, py + rr], outline=WING_STATS_MARK, width=2)
        draw.text((px + rr + 3, py - 6), label, fill=WING_STATS_MARK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        img.save(out_path, optimize=True)
    except OSError:
        return None
    return out_path


def build_wing_stats(tax_id: str, per_image: dict) -> Path | None:
    """Render/refresh the top-down wing-position scatter PNG for a tax_id.

    Needs at least :data:`WING_STATS_MIN` usable top-down poses; otherwise any
    stale image is deleted and ``None`` returned. Returns the PNG path on write.
    """
    points = list(_top_down_wing_points(per_image))
    out_path = get_wing_stats_path(tax_id)
    if len(points) < WING_STATS_MIN:
        _unlink_quiet(out_path)
        return None
    colored = []
    for l_pt, r_pt in points:
        colored.append((l_pt, WING_STATS_L_COLOR))
        colored.append((r_pt, WING_STATS_R_COLOR))
    return _render_wing_scatter(out_path, colored, -0.5j, 0.5j, "v")


def build_side_wing_stats(tax_id: str, per_image: dict) -> Path | None:
    """Render/refresh the side-view wing-position scatter PNG for a tax_id.

    Needs at least :data:`WING_STATS_MIN` usable side poses; otherwise any
    stale image is deleted and ``None`` returned. Returns the PNG path on write.
    """
    points = list(_side_wing_points(per_image))
    out_path = get_side_wing_stats_path(tax_id)
    if len(points) < WING_STATS_MIN:
        _unlink_quiet(out_path)
        return None
    colored = [(w, WING_STATS_WING_COLOR) for w in points]
    return _render_wing_scatter(out_path, colored, complex(-0.5, 0),
                                complex(0.5, 0), "h")
