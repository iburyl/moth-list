#!/usr/bin/env python3

"""Rebuild the cached pose-data and summary JSON files for every tax_id.

The web app builds these lazily and, to stay fast, keeps them around across
edits (only touching the rows that change). After a schema/formula change — e.g.
a ``POSE_DATA_VERSION`` bump or a new scoring formula — the caches are stale
until each tax_id is visited (and rebuilt) in the browser. This standalone tool
forces that rebuild for the whole dataset up front, regenerating both::

    labels/{tax_id}_pose_data.json   (pose classes, metrics, scores, thumbnail)
    labels/{tax_id}_summary.json     (index-page names/counts/source-data)

After the per-tax pass it also rebuilds the single hierarchical aggregate::

    labels/tax_summary.json          (superfamily->...->species coverage counts)

from the names CSV plus every per-tax summary on disk (a complete rebuild,
independent of which tax_ids were refreshed above).

It reuses the app's own logic (``moths.utils``) so the results match exactly
what the site would compute. The dataset layout comes from the environment, just
like the web app (``MOTHS_IMAGE_DIR`` / ``MOTHS_LABEL_DIR`` / ...); every
``MOTHS_*`` path must be set or ``settings.py`` raises.

``build_pose_data`` keeps the normalized crop/thumbnail images when an image's
keypoints are unchanged, so this only regenerates the JSON (and any crops whose
keypoints actually moved), not every ``.norm.jpg`` on disk.

Pass ``--missing-only`` to rebuild just the tax_ids that have no
``labels/{tax_id}_summary.json`` yet (useful to fill in newly-added taxa without
recomputing everything).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import wraps
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-image steps to time inside build_pose_data, as (label, moths.utils name).
# These are the functions compute_pose_row / build_pose_data call per image; the
# re-entrancy guard in StepTimer attributes any nested wrapped call to the
# outermost step, so the labels below partition the time without overlap.
POSE_STEPS = [
    ("keypoints", "_pose_source_keypoints"),
    ("classify", "classify_pose"),
    ("size", "get_image_size"),
    ("symmetry", "pose_symmetry_metric"),
    ("pixels", "pose_pixel_span"),
    ("sharpness", "compute_sharpness"),
    ("flags", "get_image_flags"),
    ("crop", "touch_normalized"),
    ("crop", "clear_normalized"),
    ("thumb", "refresh_tax_thumbnail"),
    ("write", "_write_pose_data"),
]

# Column order for the one-line per-tax report ("scan" and "summary" are timed
# at the tool level; the rest come from POSE_STEPS).
STEP_ORDER = [
    "scan",
    "keypoints",
    "classify",
    "size",
    "symmetry",
    "pixels",
    "sharpness",
    "flags",
    "crop",
    "thumb",
    "write",
    "summary",
]


class StepTimer:
    """Accumulates wall time per named step, counting only the outermost call.

    A single depth counter is shared across every wrapped function, so when one
    timed step calls another (e.g. ``symmetry`` internally reads ``size``), the
    inner call runs untimed and its cost stays attributed to the outer step.
    This keeps the reported steps a non-overlapping partition of the total.
    """

    def __init__(self):
        self.totals: dict[str, float] = {}
        self._depth = 0

    def wrap(self, label, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self._depth:
                return func(*args, **kwargs)
            self._depth += 1
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self._depth -= 1
                self.add(label, time.perf_counter() - start)

        return wrapper

    def add(self, label, seconds):
        self.totals[label] = self.totals.get(label, 0.0) + seconds

    def reset(self):
        self.totals = {}


def install_timer(moth_utils, timer: StepTimer) -> None:
    """Monkey-patch the per-image pose functions with timing wrappers.

    ``compute_pose_row`` refers to these as module globals, so replacing the
    module attributes makes it call the wrappers without any app-code change.
    """
    for label, attr in POSE_STEPS:
        setattr(moth_utils, attr, timer.wrap(label, getattr(moth_utils, attr)))


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
            "Rebuild labels/{tax_id}_pose_data.json and "
            "labels/{tax_id}_summary.json for every tax_id (or the ones given), "
            "using the Django dataset configured via the environment."
        )
    )
    parser.add_argument(
        "tax_ids",
        nargs="*",
        help="Optional tax_id(s) to rebuild; defaults to all under MOTHS_IMAGE_DIR.",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help=(
            "Only rebuild tax_ids that have no summary JSON yet "
            "(labels/{tax_id}_summary.json); skip any that already have one."
        ),
    )
    return parser.parse_args()


def _timing_line(index, total, tax_id, n_images, timer, wall):
    """Format the one-line per-tax report: counts + cumulative step timings."""
    parts = [f"[{index} of {total}] {tax_id} imgs={n_images} |"]
    for label in STEP_ORDER:
        parts.append(f"{label}={timer.totals.get(label, 0.0):.3f}")
    parts.append(f"| total={wall:.3f}s")
    return " ".join(parts)


def main() -> int:
    args = parse_args()
    moth_utils = bootstrap_django()

    timer = StepTimer()
    install_timer(moth_utils, timer)

    tax_ids = args.tax_ids or moth_utils.list_tax_ids()

    if args.missing_only:
        # Keep only tax_ids without a summary file on disk (existence, not
        # version — a present-but-stale summary is left untouched here).
        before = len(tax_ids)
        tax_ids = [
            tax_id
            for tax_id in tax_ids
            if not moth_utils.get_summary_path(tax_id).is_file()
        ]
        print(
            f"--missing-only: {len(tax_ids)} of {before} tax_id(s) lack a "
            f"summary; skipping the {before - len(tax_ids)} that already have one."
        )

    total = len(tax_ids)
    if not total:
        # Nothing to rebuild per-tax (e.g. --missing-only with no gaps), but the
        # hierarchical aggregate is still refreshed from the CSV + existing
        # summaries below.
        print("No tax_ids to rebuild.")

    failures = []
    for index, tax_id in enumerate(tax_ids, start=1):
        timer.reset()
        wall_start = time.perf_counter()
        try:
            scan_start = time.perf_counter()
            images = moth_utils.scan_tax_images(tax_id)
            timer.add("scan", time.perf_counter() - scan_start)

            # Pose data first: recomputes classes/metrics/scores and refreshes the
            # tax thumbnail; summary then rebuilds counts/source-data around it.
            moth_utils.build_pose_data(tax_id, images)

            summary_start = time.perf_counter()
            # Skip the per-tax soft update of tax_summary.json; we do one full
            # rebuild of that aggregate after the loop instead.
            moth_utils.build_summary(tax_id, images, update_tree=False)
            timer.add("summary", time.perf_counter() - summary_start)
        except Exception as exc:  # keep going; report at the end
            failures.append((tax_id, exc))
            print(f"[{index} of {total}] {tax_id} FAILED ({exc})")
        else:
            wall = time.perf_counter() - wall_start
            print(_timing_line(index, total, tax_id, len(images), timer, wall))

    # Complete rebuild of the hierarchical taxonomy aggregate from the names CSV
    # + all per-tax summaries (regardless of which tax_ids were rebuilt above).
    tree_start = time.perf_counter()
    tree = moth_utils.build_tax_summary()
    print(
        f"\nRebuilt labels/tax_summary.json: {tree['species_want']} species "
        f"across {tree['want']} superfamilies "
        f"({tree['species_have_image']} with images, "
        f"{tree['species_have_zero']} with a summary but no images) "
        f"in {time.perf_counter() - tree_start:.3f}s."
    )

    done = total - len(failures)
    print(f"\nRebuilt {done} of {total} tax_id(s).")
    if failures:
        print(f"{len(failures)} failed:")
        for tax_id, exc in failures:
            print(f"  {tax_id}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
