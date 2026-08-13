#!/usr/bin/env python3

"""Plot metric distributions (pixel span / sharpness / symmetry) as histograms.

For the whole dataset this reads the *already-computed* per-tax pose caches
(``labels/{tax_id}_pose_data.json``) — nothing is recomputed from images — and
renders, into a target directory, one PNG per (metric, image group):

    pixels_top_down.png          sharpness_top_down.png          symmetry_top_down.png
    pixels_top_down_pinned.png   sharpness_top_down_pinned.png   symmetry_top_down_pinned.png
    pixels_side.png              sharpness_side.png              symmetry_side.png

Each PNG overlays two histograms on shared bins: every image in that group
(blue) and the *starred* subset of that same group (red), so you can see where
the curated/starred images sit within the overall distribution. Histograms are
area-normalised (density) so the small starred subset stays comparable to the
much larger population.

Image groups (classified from the cached ``pose`` + ``flags``, matching the
species view):

    top_down          pose == top-down and not Pinned
    top_down_pinned   pose == top-down and Pinned
    side              pose == side view

Metric sources (cached values):

    pixels      ``pixel_span`` for top-down; for side views (one wing absent, so
                ``pixel_span`` is null) the largest pixel distance among the
                visible keypoints, so side images still get an x value.
    sharpness   ``sharpness`` (Scharr/Tenengrad gradient energy), shown in 10^6.
                Also gets an extra ``sharpness_*_zoom.png`` restricted to
                ``[0, 0.2]`` (x10^6) for a finer look at where images cluster.
    symmetry    ``symmetry`` (||R - L_mir|| as a fraction of the crop radius);
                only defined when both wings are present, so it is empty for
                side views (that PNG is skipped).

With ``--details-md PATH`` it also (or instead) writes a Markdown report
correlating the hand 1-5 *details* rating (ground truth stored in the ``.class``
sidecar) against the cached metrics, for every tax_id that has at least one
rated image, plus a pooled "All" row. It reports the Pearson (linear)
correlation ``r`` of the rating vs each metric, with sharpness shown both raw
and log-transformed (a higher ``sharp r(log)`` means a log straightens
sharpness into a more linear predictor):

    pixel r                the pixel-span metric (raw px)
    sharp r / sharp r(log) raw / log sharpness (Scharr/Tenengrad energy)
    min r                  min of the 0-1 *scaled* pixel & sharpness sub-scores
                           (``score_components``) — the weakest link the ranking uses

The dataset layout comes from the environment, exactly like the web app
(``MOTHS_IMAGE_DIR`` / ``MOTHS_LABEL_DIR`` / ...); every ``MOTHS_*`` path must be
set or ``settings.py`` raises. Taxa whose pose cache is missing or stale (a
``POSE_DATA_VERSION`` mismatch) are skipped and reported — run
``tools/rebuild_poses.py`` first if you want them included.

Usage::

    python tools/metrics_statistics.py OUTPUT_DIR [--bins N]
    python tools/metrics_statistics.py --details-md report.md   # report only
    python tools/metrics_statistics.py OUTPUT_DIR --details-md report.md
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def bootstrap_django():
    """Set up Django (from the environment) and return ``moths.utils``."""
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from moths import utils as moth_utils  # noqa: E402  (after django.setup)

    return moth_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render pixel-span / sharpness / symmetry histograms (population vs "
            "starred) for top-down, top-down-pinned and side-view images, using "
            "the cached pose data of the Django dataset configured via the "
            "environment."
        )
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help=(
            "Directory to write the histogram PNGs into (created if needed). "
            "Optional when --details-md is given (then no PNGs are produced)."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of histogram bins (default: 40).",
    )
    parser.add_argument(
        "--details-md",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Also write a Markdown report of per-tax correlations between the "
            "hand 1-5 details rating and the pixel / sharpness / scaled-min "
            "metrics to this path."
        ),
    )
    return parser.parse_args()


# Image groups: (key, human label, predicate over a pose-data row).
def _is_pinned(row: dict) -> bool:
    return "Pinned" in (row.get("flags") or [])


def _make_groups(moth_utils):
    top_down = moth_utils.POSE_TOP_DOWN
    side = moth_utils.POSE_SIDE
    return [
        (
            "top_down",
            "Top-down (no Pinned)",
            lambda r: r.get("pose") == top_down and not _is_pinned(r),
        ),
        (
            "top_down_pinned",
            "Top-down (Pinned)",
            lambda r: r.get("pose") == top_down and _is_pinned(r),
        ),
        (
            "side",
            "Side view",
            lambda r: r.get("pose") == side,
        ),
    ]


def _row_pixels(row: dict) -> float | None:
    """Pixel-span metric for a row (side-view fallback to visible keypoints).

    Uses the cached ``pixel_span`` when present (top-down / all four keypoints
    visible). For a side view — one wing absent, so ``pixel_span`` is ``None`` —
    it falls back to the largest pixel distance among the visible keypoints so
    side images still get a value. ``None`` when nothing can be derived.
    """
    span = row.get("pixel_span")
    if span is not None:
        return span
    kps = row.get("keypoints")
    width = row.get("width")
    height = row.get("height")
    if not kps or len(kps) < 4 or not width or not height:
        return None
    pts = [(kp[0] * width, kp[1] * height) for kp in kps[:4] if kp[2] > 0]
    if len(pts) < 2:
        return None
    best = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            best = max(best, math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
    return best or None


# Metrics: (key, human label, x-axis label, value extractor, display scale,
# zoom). ``zoom`` is an optional (lo, hi) window (in display units) for an extra
# zoomed-in "_zoom" chart rendered over just that range for finer detail.
METRICS = [
    ("pixels", "Pixel span", "pixel span (px)", _row_pixels, 1.0, None),
    ("sharpness", "Sharpness", "sharpness (x10^6)", lambda r: r.get("sharpness"), 1e-6, (0.0, 0.2)),
    ("symmetry", "Symmetry", "symmetry (frac of crop radius)", lambda r: r.get("symmetry"), 1.0, None),
]

POP_COLOR = "#3b82f6"      # blue — the whole group
STARRED_COLOR = "#ef4444"  # red  — the starred subset


def collect(moth_utils, groups):
    """Gather metric values per (group, metric), split into all vs starred.

    Returns ``(buckets, stats)`` where ``buckets[gkey][mkey] = {"all": [...],
    "starred": [...]}`` (display-scaled floats) and ``stats`` tracks how many
    taxa/images were used or skipped.
    """
    buckets = {
        gkey: {mkey: {"all": [], "starred": []} for mkey, *_rest in METRICS}
        for gkey, *_rest in groups
    }
    stats = {"tax_used": 0, "tax_skipped": 0, "images": 0}

    for tax_id in moth_utils.list_tax_ids():
        data = moth_utils.load_pose_data(tax_id)
        if data is None:
            stats["tax_skipped"] += 1
            continue
        stats["tax_used"] += 1
        starred = moth_utils.load_starred(tax_id)
        for filename, row in data["images"].items():
            group = next((gkey for gkey, _label, pred in groups if pred(row)), None)
            if group is None:
                continue
            stats["images"] += 1
            is_starred = moth_utils.image_basename(filename) in starred
            for mkey, _label, _xlabel, extract, scale, _zoom in METRICS:
                value = extract(row)
                if value is None:
                    continue
                slot = buckets[group][mkey]
                slot["all"].append(value * scale)
                if is_starred:
                    slot["starred"].append(value * scale)
    return buckets, stats


def render_hist(plt, np, out_path, title, xlabel, all_vals, starred_vals, bins,
                value_range=None):
    """Render one overlaid (all vs starred) density histogram to ``out_path``.

    ``value_range`` fixes the bin span to ``(lo, hi)`` (for the zoomed charts);
    when ``None`` the span is taken from the data.
    """
    if value_range is not None:
        lo, hi = value_range
    else:
        lo = min(all_vals)
        hi = max(all_vals)
        if hi <= lo:  # single distinct value — pad so hist() has a real range
            pad = abs(lo) * 0.5 or 1.0
            lo, hi = lo - pad, hi + pad
    edges = np.linspace(lo, hi, bins + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=120)
    ax.hist(
        all_vals, bins=edges, density=True, color=POP_COLOR, alpha=0.55,
        label=f"all (n={len(all_vals)})",
    )
    ax.hist(  # outline for the population so it reads even under the overlay
        all_vals, bins=edges, density=True, histtype="step",
        color=POP_COLOR, linewidth=1.2,
    )
    if starred_vals:
        ax.hist(
            starred_vals, bins=edges, density=True, color=STARRED_COLOR, alpha=0.55,
            label=f"starred (n={len(starred_vals)})",
        )
        ax.hist(
            starred_vals, bins=edges, density=True, histtype="step",
            color=STARRED_COLOR, linewidth=1.2,
        )
    else:
        # Keep a legend entry so the empty-starred case is explicit.
        ax.plot([], [], color=STARRED_COLOR, label="starred (n=0)")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --- Details-rating correlations (Markdown report) ---------------------------


def _pearson(xs, ys) -> float | None:
    """Pearson r of two equal-length samples; ``None`` if undefined.

    Undefined (returns ``None``) when there are fewer than 2 points or either
    variable has zero variance (e.g. every image got the same rating).
    """
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(sx * sy)


# --- Experimental sharpness proxies (computed live, never stored) ------------
#
# Two "what if" sharpness measures used only to compare correlations against the
# hand details ratings. Neither is cached anywhere.
#
#  * fb1  - contrast-normalised gradient along the F->B body axis, sampled on a
#           thin band directly on the original image (no crop/greyscale/resize);
#           the F-B keypoint distance supplies the length scale.
#  * box  - the same Scharr/Tenengrad gradient energy as the cached sharpness
#           metric, but measured on the original image at native resolution over
#           the moth's bounding box (no resize), to gauge what the cached
#           metric's 1024^2 resize costs.
FB_SAMPLES = 512       # samples along F->B (fixed => scale-invariant in body units)
FB_LINES = 5           # parallel lines forming a thin band around the axis
FB_BAND_FRAC = 0.02    # half-band width as a fraction of the F->B length


def _bilinear(np, arr, xs, ys):
    """Bilinearly sample 2-D ``arr`` at float coords (edge-clamped)."""
    h, w = arr.shape
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = np.clip(xs - x0, 0.0, 1.0)
    fy = np.clip(ys - y0, 0.0, 1.0)
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    top = arr[y0, x0] * (1 - fx) + arr[y0, x1] * fx
    bot = arr[y1, x0] * (1 - fx) + arr[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _load_luma(np, moth_utils, filename):
    """Green channel of the original image as a float 2-D array, or ``None``.

    Uses green (not a greyscale conversion) and does no crop/resize; the whole
    frame is decoded by PIL but nothing is transformed.
    """
    from PIL import Image

    path = moth_utils.get_image_path(filename)
    if not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            arr = np.asarray(im)
    except (OSError, ValueError):
        return None
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.ndim == 3 and arr.shape[2] >= 2:
        return arr[:, :, 1].astype(np.float64)
    return None


def _scharr_energy(np, arr):
    """Mean Scharr/Tenengrad gradient energy of a 2-D array (``None`` if empty)."""
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return None
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


def _fb_gradient(np, luma, f_px, b_px):
    """Contrast-normalised gradient along F->B, or ``None``.

    Mean squared first-difference of the axis profile divided by its variance
    (contrast/exposure invariant "acutance"), averaged over a thin band of
    parallel lines. ``None`` when the F-B segment is degenerate.
    """
    (fx, fy), (bx, by) = f_px, b_px
    dx, dy = bx - fx, by - fy
    length = math.hypot(dx, dy)
    if length < 4:
        return None
    perp_x, perp_y = -dy / length, dx / length
    t = np.linspace(0.0, 1.0, FB_SAMPLES)
    base_x, base_y = fx + t * dx, fy + t * dy
    offsets = ([0.0] if FB_LINES <= 1 else
               np.linspace(-FB_BAND_FRAC * length, FB_BAND_FRAC * length, FB_LINES))

    vals = []
    for off in offsets:
        profile = _bilinear(np, luma, base_x + off * perp_x, base_y + off * perp_y)
        var = float(profile.var())
        if var <= 1e-9:
            continue
        diffs = np.diff(profile)
        vals.append(float((diffs * diffs).mean()) / var)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _image_fb_gradient(np, moth_utils, filename, row):
    """:func:`_fb_gradient` for a pose row, or ``None`` if not measurable.

    Needs both F (keypoint 0) and B (keypoint 3) visible and a known image
    size; the normalized keypoints are scaled to pixels using the row's
    ``width``/``height``.
    """
    kps = row.get("keypoints") or []
    w, h = row.get("width"), row.get("height")
    if len(kps) < 4 or not w or not h:
        return None
    f_kp, b_kp = kps[0], kps[3]
    if f_kp[2] <= 0 or b_kp[2] <= 0:
        return None
    luma = _load_luma(np, moth_utils, filename)
    if luma is None:
        return None
    f_px = (f_kp[0] * w, f_kp[1] * h)
    b_px = (b_kp[0] * w, b_kp[1] * h)
    return _fb_gradient(np, luma, f_px, b_px)


def _box_sharpness(np, moth_utils, filename):
    """Scharr/Tenengrad energy over the moth box on the ORIGINAL image.

    Mirrors the cached sharpness metric's greyscale gradient operator but keeps
    the image at native resolution and covers the whole bounding box (no
    central-crop, no resize). ``None`` when there is no box or the crop is too
    small.
    """
    from PIL import Image

    annotations, _source = moth_utils.load_pose_source(filename)
    if not annotations:
        return None
    ann = annotations[0]
    if ann.width <= 0 or ann.height <= 0:
        return None
    path = moth_utils.get_image_path(filename)
    if not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            w, h = im.size
            left = max(0, int(round((ann.cx - ann.width / 2) * w)))
            right = min(w, int(round((ann.cx + ann.width / 2) * w)))
            top = max(0, int(round((ann.cy - ann.height / 2) * h)))
            bottom = min(h, int(round((ann.cy + ann.height / 2) * h)))
            if right - left < 3 or bottom - top < 3:
                return None
            crop = im.convert("L").crop((left, top, right, bottom))
            arr = np.asarray(crop, dtype=np.float64)
    except (OSError, ValueError):
        return None
    return _scharr_energy(np, arr)


def _norm7_sharpness(np, moth_utils, filename):
    """Scharr/Tenengrad energy of the centre cell of the normalized thumbnail.

    Splits the pose-normalized thumbnail (``<name>.norm-thumb`` — the body
    centred/rotated to a fixed frame) into a 7x7 grid and measures the middle
    cell, i.e. the moth's core. ``None`` when the image has no normalized crop
    (needs keypoints) or the thumbnail is too small to grid.
    """
    from PIL import Image

    result = moth_utils.get_or_create_normalized(filename)
    if result is None:
        return None
    _norm_path, thumb_path = result
    try:
        with Image.open(thumb_path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float64)
    except (OSError, ValueError):
        return None
    h, w = arr.shape
    if h < 7 or w < 7:
        return None
    cell = arr[(3 * h) // 7:(4 * h) // 7, (3 * w) // 7:(4 * w) // 7]
    return _scharr_energy(np, cell)


def _corr(values, details) -> float | None:
    """Pearson r over the pairs where ``values`` is not ``None``."""
    pairs = [(v, d) for v, d in zip(values, details) if v is not None]
    if len(pairs) < 2:
        return None
    return _pearson([p[0] for p in pairs], [p[1] for p in pairs])


# --- log(sharpness) -> 0..1 score calibration --------------------------------
#
# ``sharp r(log)`` is the strongest linear predictor of the hand rating, so the
# score is a straight line in log(sharpness), pinned to two rating anchors: an
# image that rates like SCORE_STAR_LOW stars gets SCORE_LOW, one that rates like
# SCORE_STAR_HIGH stars gets SCORE_HIGH (result clamped to [0, 1]). The anchor
# log-sharpness values come from regressing log(sharpness) on the rating over
# the pooled rated images, then reading the fit at those two star levels.
SCORE_STAR_LOW = 2.5
SCORE_LOW = 0.5
SCORE_STAR_HIGH = 4.5
SCORE_HIGH = 1.0


def _linfit(xs, ys):
    """OLS ``(slope, intercept)`` for ``ys ~ slope*xs + intercept``, or ``None``.

    ``None`` when there are fewer than 2 points or ``xs`` has zero variance.
    """
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def sharp_score_calibration(ratings, log_sharps) -> dict | None:
    """Fit the log(sharpness) -> score line from paired ratings & log-sharpness.

    Regresses ``log(sharpness) = p*rating + q``, reads the anchor log-sharpness
    at :data:`SCORE_STAR_LOW`/:data:`SCORE_STAR_HIGH` stars, then solves the
    score line ``score = a*log(sharpness) + b`` so those anchors map to
    :data:`SCORE_LOW`/:data:`SCORE_HIGH`. Returns the fit + anchors + ``a``/``b``
    (and raw-sharpness anchors ``s_low``/``s_high``), or ``None`` if degenerate.
    """
    fit = _linfit(ratings, log_sharps)
    if fit is None:
        return None
    p, q = fit
    log_low = p * SCORE_STAR_LOW + q
    log_high = p * SCORE_STAR_HIGH + q
    if log_high == log_low:
        return None
    a = (SCORE_HIGH - SCORE_LOW) / (log_high - log_low)
    b = SCORE_LOW - a * log_low
    return {
        "n": len(ratings),
        "p": p,
        "q": q,
        "log_low": log_low,
        "log_high": log_high,
        "s_low": math.exp(log_low),
        "s_high": math.exp(log_high),
        "a": a,
        "b": b,
    }


# --- bad(1-2)/good(3-5) threshold separability -------------------------------
#
# Treat each metric as a one-threshold classifier: images rated 1-2 are "bad",
# 3-5 are "good", and (all metrics correlate positively) a value *above* the cut
# is predicted "good". For each metric we pick the cut that minimises the total
# misjudged (bad above the cut + good below it) — the fewest mistakes a single
# threshold can make, i.e. how well that metric alone would gate a harvest.
# ``sharp_log`` is omitted: a monotone transform can't change the cut, so it
# scores identically to ``sharp``.
SEP_METRICS = ("pixel", "sharp", "min", "fb1", "box", "norm7")
SEP_GOOD_MIN = 3  # rating >= this is "good"; below it is "bad"


def _best_split(values, labels) -> dict | None:
    """Threshold that best separates good (higher) from bad, ``None`` if empty.

    ``labels`` is truthy for "good". Pairs with a ``None`` value are dropped, so
    each metric is judged over whatever subset it covers. Returns
    ``{threshold, miss, bad_as_good, good_as_bad, n, bad, good}`` where the rule
    is "predict good when value > threshold" (``-inf`` = predict all good).
    """
    pairs = sorted((v, bool(l)) for v, l in zip(values, labels) if v is not None)
    n = len(pairs)
    if n == 0:
        return None
    total_bad = sum(1 for _, good in pairs if not good)
    total_good = n - total_bad

    # Start below every value: everything predicted good, so all bad are wrong.
    bad_as_good, good_as_bad = total_bad, 0
    best = {"threshold": float("-inf"), "miss": total_bad,
            "bad_as_good": total_bad, "good_as_bad": 0}

    # Sweep the cut upward; each value that drops below it flips to "bad".
    i = 0
    while i < n:
        value = pairs[i][0]
        j = i
        while j < n and pairs[j][0] == value:
            if pairs[j][1]:
                good_as_bad += 1   # a good image now (wrongly) below the cut
            else:
                bad_as_good -= 1   # a bad image now (correctly) below the cut
            j += 1
        miss = bad_as_good + good_as_bad
        if miss < best["miss"]:
            best = {"threshold": value, "miss": miss,
                    "bad_as_good": bad_as_good, "good_as_bad": good_as_bad}
        i = j

    best.update({"n": n, "bad": total_bad, "good": total_good})
    return best


def _separability(sample: dict) -> dict:
    """Best bad/good split per metric for a collected sample (see SEP_METRICS)."""
    labels = [d >= SEP_GOOD_MIN for d in sample["details"]]
    return {metric: _best_split(sample[metric], labels) for metric in SEP_METRICS}


def _correlations(sample: dict) -> dict:
    """Pearson r of each metric vs the details rating for a collected sample.

    Sharpness is shown raw and log-transformed (a higher r(log) means a log
    straightens it into a more linear predictor). ``fb1``/``box``/``norm7`` are
    the experimental proxies; they may be ``None`` per image (missing image,
    box, keypoints or normalized crop), so their correlation is over whatever
    subset is available.
    """
    d = sample["details"]
    return {
        "pixel_r": _corr(sample["pixel"], d),
        "sharp_r": _corr(sample["sharp"], d),
        "sharp_r_log": _corr(sample["sharp_log"], d),
        "min_r": _corr(sample["min"], d),
        "fb1_r": _corr(sample["fb1"], d),
        "box_r": _corr(sample["box"], d),
        "norm7_r": _corr(sample["norm7"], d),
    }


def collect_details(moth_utils):
    """Pair the hand details rating with the cached metrics, per tax_id.

    Iterates the cached pose rows (stale/missing caches are skipped) and, for
    every image carrying a 1-5 ``details`` rating that also has both a pixel
    span and a sharpness value, records the rating alongside the raw pixel/
    sharpness metrics and ``min`` of their scaled sub-scores. It also computes
    two throwaway experimental sharpness proxies live off the original image
    (``fb1`` = contrast-normalised gradient along F->B, ``box`` = Scharr energy
    over the moth box at native resolution; never stored). Returns
    ``(per_tax, overall, stats)``; ``per_tax`` is sorted by rated-image count
    (desc) then tax_id.
    """
    # numpy is only needed for the experimental F->B metrics; without it (or
    # Pillow) those two columns stay blank and the rest of the report is fine.
    try:
        import numpy as np
    except ImportError:
        np = None

    per_tax = []
    keys = ("details", "pixel", "sharp", "sharp_log", "min", "fb1", "box", "norm7")
    pooled = {key: [] for key in keys}
    stats = {"tax_rated": 0, "tax_skipped": 0, "rated": 0, "no_metric": 0,
             "fb_ok": 0, "box_ok": 0, "norm7_ok": 0, "exp_enabled": np is not None}
    # Per-metric wall-clock timing, each ``[total_seconds, calls]``. pixel and
    # sharpness are recomputed from scratch (never read from the cache) so the
    # four metrics are timed on equal footing; the image file is warmed first so
    # the numbers reflect CPU work, not cold-disk reads.
    timings = {key: [0.0, 0] for key in ("pixel", "sharp", "fb1", "box", "norm7")}

    def timed(key, fn, *a):
        t0 = time.perf_counter()
        value = fn(*a)
        timings[key][0] += time.perf_counter() - t0
        timings[key][1] += 1
        return value

    for tax_id in moth_utils.list_tax_ids():
        data = moth_utils.load_pose_data(tax_id)
        if data is None:
            stats["tax_skipped"] += 1
            continue
        sample = {key: [] for key in keys}
        for filename, row in data["images"].items():
            rating = moth_utils.get_image_details(filename)
            if rating is None:
                continue
            stats["rated"] += 1

            # Warm the OS page cache so timing measures decode/compute, not I/O.
            img_path = moth_utils.get_image_path(filename)
            try:
                if img_path.is_file():
                    img_path.read_bytes()
            except OSError:
                pass

            # Recompute pixel & sharpness from scratch (do not reuse the cache).
            pixel_raw = timed("pixel", moth_utils.pose_pixel_span, filename)
            sharp_raw = timed("sharp", moth_utils.compute_sharpness, filename)
            # Experimental proxies, computed live off the original / normalized image.
            fb1 = box = norm7 = None
            if np is not None:
                fb1 = timed("fb1", _image_fb_gradient, np, moth_utils, filename, row)
                box = timed("box", _box_sharpness, np, moth_utils, filename)
                norm7 = timed("norm7", _norm7_sharpness, np, moth_utils, filename)
                stats["fb_ok"] += fb1 is not None
                stats["box_ok"] += box is not None
                stats["norm7_ok"] += norm7 is not None

            if pixel_raw is None or sharp_raw is None:
                stats["no_metric"] += 1
                continue
            _s_sym, s_pixels, s_sharp = moth_utils.score_components(
                None, pixel_raw, sharp_raw
            )
            sample["details"].append(float(rating))
            sample["pixel"].append(float(pixel_raw))
            sample["sharp"].append(float(sharp_raw))
            # Guard log against a (rare) zero/negative energy.
            sample["sharp_log"].append(math.log(sharp_raw) if sharp_raw > 0 else math.log(1e-6))
            sample["min"].append(min(s_pixels, s_sharp))
            sample["fb1"].append(fb1)
            sample["box"].append(box)
            sample["norm7"].append(norm7)

        if not sample["details"]:
            continue
        stats["tax_rated"] += 1
        species = (moth_utils.get_name_info(tax_id).get("species") or "").strip()
        n_bad = sum(1 for d in sample["details"] if d < SEP_GOOD_MIN)
        per_tax.append({
            "tax_id": tax_id,
            "species": species,
            "n": len(sample["details"]),
            "bad": n_bad,
            "good": len(sample["details"]) - n_bad,
            "corr": _correlations(sample),
            "sep": _separability(sample),
        })
        for key in pooled:
            pooled[key].extend(sample[key])

    stats["timings"] = timings
    pooled_bad = sum(1 for d in pooled["details"] if d < SEP_GOOD_MIN)
    overall = {
        "n": len(pooled["details"]),
        "tax_count": stats["tax_rated"],
        "bad": pooled_bad,
        "good": len(pooled["details"]) - pooled_bad,
        "corr": _correlations(pooled),
        "sep": _separability(pooled),
        "sharp_calib": sharp_score_calibration(pooled["details"], pooled["sharp_log"]),
    }
    per_tax.sort(key=lambda r: (-r["n"], r["tax_id"]))
    return per_tax, overall, stats


def _fmt_corr(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}"


def _md_table(headers, aligns, rows) -> list[str]:
    """Render an aligned Markdown table (padded so the raw text lines up).

    ``aligns`` is one ``"l"``/``"r"`` per column; ``rows`` is a list of rows of
    pre-formatted cell strings (any ``**bold**``/``_italic_`` markup is counted
    in the width, so emphasised cells still align in raw monospace text).
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def pad(cell, i):
        return cell.rjust(widths[i]) if aligns[i] == "r" else cell.ljust(widths[i])

    def sep(i):
        return "-" * (widths[i] - 1) + ":" if aligns[i] == "r" else "-" * widths[i]

    out = ["| " + " | ".join(pad(h, i) for i, h in enumerate(headers)) + " |"]
    out.append("| " + " | ".join(sep(i) for i in range(len(headers))) + " |")
    for row in rows:
        out.append("| " + " | ".join(pad(cell, i) for i, cell in enumerate(row)) + " |")
    return out


def render_details_md(per_tax, overall, stats) -> str:
    """Build the Markdown correlation report from :func:`collect_details`."""
    lines = [
        "# Details-rating vs. metric correlations",
        "",
        (
            "Pearson (linear) correlation `r` of the hand 1&ndash;5 *details* "
            "rating (ground truth, from the `.class` sidecar) against the image "
            "metrics, for every `tax_id` with at least one rated image. The pixel "
            "and sharpness metrics are recomputed from scratch here (not read from "
            "the cache), matching the timings below. Sharpness is shown both raw "
            "and log-transformed, so a higher `sharp r(log)` means a log straightens "
            "sharpness into a more linear predictor of the rating. Only rated images "
            "that have **both** a pixel-span and a sharpness value are counted."
        ),
        "",
        (
            f"Rated images: {stats['rated']} "
            f"({stats['no_metric']} without both metrics, excluded); "
            f"taxa with ratings: {stats['tax_rated']}; "
            f"stale/missing caches skipped: {stats['tax_skipped']}."
        ),
        "",
        "- **pixel r** — pixel-span metric (raw px; side views fall back to the "
        "largest visible-keypoint distance).",
        "- **sharp r / sharp r(log)** — sharpness metric (Scharr/Tenengrad "
        "gradient energy), raw and `log`-transformed.",
        "- **min r** — `min` of the 0&ndash;1 *scaled* pixel & sharpness "
        "sub-scores (`score_components`); the weakest link the site's ranking uses.",
        "- **fb1 r** *(experimental)* — contrast-normalised gradient energy along "
        "the F&rarr;B body axis, measured live on the **original** image (no "
        "crop/greyscale/resize); nothing stored.",
        "- **box r** *(experimental)* — the same Scharr/Tenengrad gradient energy "
        "as the cached `sharp` metric, but on the **original** image at native "
        "resolution over the whole moth box (no resize); nothing stored.",
        "- **n7 r** *(experimental)* — Scharr/Tenengrad energy of the **centre "
        "cell of a 7×7 grid** over the pose-normalized thumbnail (the moth's "
        "core, in a fixed frame); nothing stored.",
        "",
    ]

    if stats["exp_enabled"]:
        lines.append(
            f"Experimental metrics: `fb1` computed for {stats['fb_ok']} image(s), "
            f"`box` for {stats['box_ok']}, `n7` for {stats['norm7_ok']} (others "
            "skipped for a missing image, box, F/B keypoints or normalized crop)."
        )
    else:
        lines.append(
            "_Experimental metrics disabled: NumPy/Pillow not available, so the "
            "`fb1 r`/`box r`/`n7 r` columns are blank._"
        )
    lines.append("")

    if not per_tax:
        lines.append("_No images have a details rating yet._")
        lines.append("")
        return "\n".join(lines)

    headers = ["tax_id", "species", "n",
               "pixel r", "sharp r", "sharp r(log)", "min r", "fb1 r", "box r", "n7 r"]
    aligns = ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r"]
    corr_keys = ["pixel_r", "sharp_r", "sharp_r_log", "min_r", "fb1_r", "box_r", "norm7_r"]

    def data_row(tax_id, species, n, corr, bold=False):
        emph = (lambda s: f"**{s}**") if bold else (lambda s: s)
        cells = [emph(tax_id), species, emph(str(n))]
        cells += [emph(_fmt_corr(corr[key])) for key in corr_keys]
        return cells

    rows = [
        data_row(r["tax_id"], r["species"].replace("|", "\\|"), r["n"], r["corr"])
        for r in per_tax
    ]
    rows.append(
        data_row("All", f"_{overall['tax_count']} taxa_", overall["n"],
                 overall["corr"], bold=True)
    )

    lines.extend(_md_table(headers, aligns, rows))
    lines.append("")

    lines.extend(_render_separability(per_tax, overall))
    lines.extend(_render_sharp_calibration(overall.get("sharp_calib")))
    lines.extend(_render_timing(stats.get("timings")))
    return "\n".join(lines)


def _fmt_thr(metric: str, sep) -> str:
    """Format a metric's best-split threshold ("predict good when value > t")."""
    if sep is None:
        return "—"
    t = sep["threshold"]
    if t == float("-inf"):
        return "all"  # no cut beats predicting everything good
    if metric == "sharp":
        return f"{t / 1e6:.3f}M"
    if metric in ("pixel", "box", "norm7"):
        return f"{t:.0f}"
    return f"{t:.4f}"


def _render_separability(per_tax, overall) -> list[str]:
    """Two Markdown tables: per-taxon misjudged counts and the pooled cuts.

    The first mirrors the correlation table (one cell per taxon x metric = the
    fewest images a single best cut misjudges, each taxon at its own optimum);
    the second lists the pooled optimum for every metric with the concrete cut
    and its bad/good error split.
    """
    metric_labels = {"pixel": "pixel", "sharp": "sharp", "min": "min",
                     "fb1": "fb1", "box": "box", "norm7": "n7"}

    lines = [
        "## Bad (1-2) vs good (3-5) separability",
        "",
        (
            f"Each image is *bad* (rating 1-2) or *good* (rating {SEP_GOOD_MIN}-5). "
            "For every metric we pick the single threshold that **minimises the "
            "total misjudged** (bad above the cut + good below it), assuming a "
            "higher value is better. Cells below are that minimum misjudged count "
            "(lower = the metric separates bad from good better); `bad`/`good` are "
            "the class sizes, so the trivial baseline is `min(bad, good)`. "
            "`fb1`/`box` are judged over the subset where they exist. `sharp r(log)` "
            "is omitted — a monotone transform cannot change the cut."
        ),
        "",
    ]

    # Table 1: per-taxon misjudged counts (each taxon at its own optimum).
    headers = ["tax_id", "species", "n", "bad", "good",
               *[metric_labels[m] for m in SEP_METRICS]]
    aligns = ["l", "l", "r", "r", "r"] + ["r"] * len(SEP_METRICS)

    def miss_cells(entry, bold=False):
        emph = (lambda s: f"**{s}**") if bold else (lambda s: s)
        cells = [emph(str(entry["n"])), emph(str(entry["bad"])), emph(str(entry["good"]))]
        for metric in SEP_METRICS:
            sep = entry["sep"].get(metric)
            cells.append(emph("—" if sep is None else str(sep["miss"])))
        return cells

    rows = []
    for r in per_tax:
        rows.append([r["tax_id"], r["species"].replace("|", "\\|"), *miss_cells(r)])
    rows.append(["All", f"_{overall['tax_count']} taxa_", *miss_cells(overall, bold=True)])
    lines.extend(_md_table(headers, aligns, rows))
    lines.append("")

    # Table 2: the pooled optimum per metric, with the concrete cut + error split.
    lines += [
        "Pooled optimum over all rated images (the cut you'd actually harvest on):",
        "",
    ]
    t_headers = ["metric", "cut (value >)", "misjudged", "bad→good", "good→bad", "n", "err %"]
    t_aligns = ["l", "r", "r", "r", "r", "r", "r"]
    t_rows = []
    for metric in SEP_METRICS:
        sep = overall["sep"].get(metric)
        if sep is None:
            t_rows.append([metric_labels[metric], "—", "—", "—", "—", "—", "—"])
            continue
        err = 100.0 * sep["miss"] / sep["n"] if sep["n"] else 0.0
        t_rows.append([
            metric_labels[metric],
            _fmt_thr(metric, sep),
            str(sep["miss"]),
            str(sep["bad_as_good"]),
            str(sep["good_as_bad"]),
            str(sep["n"]),
            f"{err:.1f}",
        ])
    lines.extend(_md_table(t_headers, t_aligns, t_rows))
    lines.append("")
    return lines


def _fmt_m(value: float) -> str:
    """Format a raw sharpness energy as a fraction of a million (e.g. ``0.123M``)."""
    return f"{value / 1e6:.3f}M"


def _render_sharp_calibration(calib) -> list[str]:
    """Markdown "## Sharpness → score" section: the fitted log-sharp score line.

    Emits the fit, the two star anchors (in raw and log sharpness) and a
    copy-paste ``score = a*log(sharp) + b`` (clamped to [0, 1]) ready to drop
    into ``moths/utils/metrics.py``.
    """
    if not calib:
        return []
    a, b = calib["a"], calib["b"]
    lines = [
        "## Sharpness → score",
        "",
        (
            f"Linear score in `log(sharpness)`, fit over {calib['n']} rated image(s) "
            f"so **{SCORE_STAR_LOW:g}★ → {SCORE_LOW:g}** and "
            f"**{SCORE_STAR_HIGH:g}★ → {SCORE_HIGH:g}** (clamped to `[0, 1]`). "
            "Anchors come from regressing `log(sharpness)` on the rating: "
            f"`log(sharpness) = {calib['p']:+.4f}·rating {calib['q']:+.4f}`."
        ),
        "",
    ]
    lines.extend(_md_table(
        ["anchor", "sharpness", "log(sharpness)", "score"],
        ["l", "r", "r", "r"],
        [
            [f"{SCORE_STAR_LOW:g}★", _fmt_m(calib["s_low"]),
             f"{calib['log_low']:.4f}", f"{SCORE_LOW:g}"],
            [f"{SCORE_STAR_HIGH:g}★", _fmt_m(calib["s_high"]),
             f"{calib['log_high']:.4f}", f"{SCORE_HIGH:g}"],
        ],
    ))
    lines += [
        "",
        f"So the score is `clamp(a·ln(sharpness) + b, 0, 1)` with "
        f"**a = {a:.6f}**, **b = {b:.6f}**:",
        "",
        "```python",
        "import math",
        "",
        "",
        "def sharpness_score(sharpness):",
        '    """0..1 sharpness score, calibrated to the hand details ratings."""',
        "    if sharpness is None or sharpness <= 0:",
        "        return None",
        f"    return max(0.0, min(1.0, {a:.6g} * math.log(sharpness) + {b:.6g}))",
        "```",
        "",
        "Equivalent anchor form (independent of the log base):",
        "",
        "```",
        f"score = 0.5 + 0.5 * ln(sharpness / {calib['s_low']:.1f}) "
        f"/ ln({calib['s_high']:.1f} / {calib['s_low']:.1f})",
        "```",
        "",
    ]
    return lines


def _render_timing(timings) -> list[str]:
    """Markdown "## Performance" section: average wall time per metric.

    ``pixel`` and ``sharp`` are recomputed from scratch (not read from the
    cache) and every image is warmed first, so these averages compare the four
    metrics' compute cost on equal footing.
    """
    if not timings:
        return []
    labels = {
        "pixel": "pixel span",
        "sharp": "sharpness (cached-method, 1024² resize)",
        "fb1": "fb1 (F→B gradient)",
        "box": "box (native-res Scharr)",
        "norm7": "norm7 (norm-thumb centre cell)",
    }
    rows = []
    for key in ("pixel", "sharp", "fb1", "box", "norm7"):
        total, calls = timings.get(key, (0.0, 0))
        avg_ms = (total / calls * 1000.0) if calls else None
        rows.append([
            labels[key],
            "—" if avg_ms is None else f"{avg_ms:.2f}",
            str(calls),
            "—" if calls == 0 else f"{total:.2f}",
        ])
    return [
        "## Performance",
        "",
        (
            "Average wall time per image for each metric. `pixel` and `sharp` are "
            "**recomputed from scratch** (never read from the cache) and every image "
            "is read once to warm the OS cache first, so the numbers reflect "
            "compute cost rather than disk I/O."
        ),
        "",
        *_md_table(
            ["metric", "avg ms", "n", "total s"],
            ["l", "r", "r", "r"],
            rows,
        ),
        "",
    ]


def write_details_report(moth_utils, out_path: Path) -> None:
    """Compute and write the details-correlation Markdown report."""
    per_tax, overall, stats = collect_details(moth_utils)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_details_md(per_tax, overall, stats), encoding="utf-8")
    print(
        f"Wrote {out_path} - {overall['n']} rated image(s) across "
        f"{overall['tax_count']} tax_id(s)"
        + (f" ({stats['tax_skipped']} stale/missing caches skipped)."
           if stats["tax_skipped"] else ".")
    )


def main() -> int:
    args = parse_args()

    if not args.output_dir and not args.details_md:
        print(
            "Nothing to do: pass OUTPUT_DIR (histograms) and/or --details-md PATH.",
            file=sys.stderr,
        )
        return 2

    moth_utils = bootstrap_django()

    # The Markdown report needs no plotting deps, so do it first (and on its own
    # it lets the tool run without matplotlib installed).
    if args.details_md:
        write_details_report(moth_utils, args.details_md)

    if not args.output_dir:
        return 0

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print(
            "Histograms need matplotlib and numpy (pip install matplotlib numpy).",
            file=sys.stderr,
        )
        return 0 if args.details_md else 2

    groups = _make_groups(moth_utils)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets, stats = collect(moth_utils, groups)

    print(
        f"Read cached pose data for {stats['tax_used']} tax_id(s) "
        f"({stats['tax_skipped']} skipped: missing/stale cache); "
        f"{stats['images']} images matched a group."
    )
    if stats["tax_skipped"]:
        print("  (run tools/rebuild_poses.py to refresh stale caches.)")

    group_labels = {gkey: label for gkey, label, _pred in groups}
    written = 0
    for gkey in group_labels:
        for mkey, mlabel, xlabel, _extract, _scale, zoom in METRICS:
            slot = buckets[gkey][mkey]
            out_path = out_dir / f"{mkey}_{gkey}.png"
            if not slot["all"]:
                print(f"  skip {out_path.name}: no values")
                continue
            render_hist(
                plt, np, out_path,
                f"{mlabel} — {group_labels[gkey]}",
                xlabel, slot["all"], slot["starred"], args.bins,
            )
            written += 1
            print(
                f"  wrote {out_path.name} "
                f"(all={len(slot['all'])}, starred={len(slot['starred'])})"
            )

            if zoom is None:
                continue
            # Zoomed-in view: restrict to the window and renormalise so the
            # in-window shape (where most images cluster) reads in fine detail.
            lo, hi = zoom
            zoom_all = [v for v in slot["all"] if lo <= v <= hi]
            zoom_starred = [v for v in slot["starred"] if lo <= v <= hi]
            zoom_path = out_dir / f"{mkey}_{gkey}_zoom.png"
            if not zoom_all:
                print(f"  skip {zoom_path.name}: no values in [{lo}, {hi}]")
                continue
            render_hist(
                plt, np, zoom_path,
                f"{mlabel} — {group_labels[gkey]} (zoom [{lo}, {hi}])",
                xlabel, zoom_all, zoom_starred, args.bins, value_range=zoom,
            )
            written += 1
            print(
                f"  wrote {zoom_path.name} "
                f"(all={len(zoom_all)}, starred={len(zoom_starred)})"
            )

    print(f"\nWrote {written} histogram(s) to {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
