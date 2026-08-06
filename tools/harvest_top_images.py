#!/usr/bin/env python3

"""Harvest the best top-down and side-view photos for a taxon from iNaturalist.

For a given iNaturalist taxon this walks research-grade observations
(newest first) and, for each one whose main photo carries a Creative Commons
(``cc*``) licence, branches on the observation's Life Stage annotation:

* **Adult or unannotated** – the pose candidate path: download the main photo
  and run the exact same three-model prediction scheme as
  ``ultralytics-predict.py`` (shared code): a classification model picks the
  viewpoint, then the general or side-view pose model places keypoints (with
  the same confidence-independent visibility rules). Only images that come out
  as ``top-down`` (pinned or not) or ``side view`` are kept:

  - top-down: scored on symmetry / pixel-span / sharpness (via ``moths.utils``);
  - side view: scored on pixel-span / sharpness only (symmetry is undefined for
    a single wing);

  and kept only when the *minimum* of the relevant scaled 0..1 sub-scores
  exceeds ``--score-threshold`` (default ``0.5``). Kept images are labelled
  ``Adult``.
* **Any other stage** (larva/pupa/egg) – the sample path, routed purely by the
  iNaturalist Life Stage flag (never the model): no pose prediction; keep up to
  ``--stage-samples`` images per stage (default 5) and label each with the
  matching Django stage. Stages with no moth class (nymph, juvenile, subimago)
  are skipped.

It keeps paging back through observations until it has collected ``--target``
**top-down** images OR ``--target`` **side-view** images (default 20), or
iNaturalist runs out. The search stops as soon as *either* target is met
(stage-sample quotas may end up partially filled). Non-qualifying downloads
(wrong pose or low score) are deleted so the dataset only gains images that pass.

The data directories come from Django, exactly like the other tools in this
folder (``ultralytics-predict.py`` etc.): the environment must set every
``MOTHS_*`` path (no defaults). ``MOTHS_IMAGE_DIR`` holds the photos plus
``<tax>_observations.json``, ``MOTHS_PREDICTION_DIR`` the predicted keypoint
``.txt`` files, ``MOTHS_LABEL_DIR`` the ``<tax>_pose_data.json``
(pose/metrics/scores), ``MOTHS_THUMBNAIL_DIR`` the normalized-crop cache and
``MOTHS_CLASS_DIR`` the ``<name>.class`` stage labels. Keypoints live only in
the prediction dir and pose data only in the labels dir;
``<tax>_observations.json`` stays pure observation metadata.

Two input modes, both handled in a single process so Django and the three YOLO
models are loaded once and amortized across every taxon:

* a bare iNaturalist **taxon id** harvests that one taxon;
* a **CSV path** plus ``--column NAME`` walks that column (each value treated as
  an iNaturalist taxon id) and harvests each in turn.

In CSV mode, press SPACE to stop cleanly: the current taxon finishes and no
later taxon is started. The per-taxon audit CSV still gates already-finished
taxa cheaply (before the models are loaded), so re-running a list only tops up
what is missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import requests

import utils_prediction as pred


# --- Repo / Django bootstrap -------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = Path(__file__).resolve().parent


def bootstrap_django():
    """Set up Django and return the ``moths.utils`` module.

    The metric math lives in the Django app, so we import it rather than
    re-implement it (keeping this script's scores identical to the site's).
    """
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from moths import utils as moth_utils  # noqa: E402  (after django.setup)

    return moth_utils


def import_inat():
    """Import the shared iNaturalist download helpers from the sibling script."""
    sys.path.insert(0, str(TOOLS_DIR))
    import get_inats_images as inat  # noqa: E402

    return inat


# --- Clean-stop watcher ------------------------------------------------------


class SpaceStopWatcher:
    """Background watcher that trips a flag when SPACE is pressed.

    Lets the operator ask for a *clean stop* while walking a CSV of taxa: the
    taxon currently being harvested is never interrupted (it finishes normally),
    and the caller checks :attr:`requested` between taxa to stop before starting
    the next one.

    Cross-platform and best-effort: it uses ``msvcrt`` on Windows and
    ``termios``/``select`` on POSIX. When stdin is not an interactive terminal
    (e.g. output is redirected) it stays disabled and the batch simply runs to
    completion.
    """

    def __init__(self) -> None:
        self._requested = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._restore = None
        self.enabled = False

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    def start(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():
            return
        try:
            import msvcrt  # noqa: F401  (Windows only)

            target = self._run_windows
        except ImportError:
            if not self._setup_posix():
                return
            target = self._run_posix
        self.enabled = True
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def _trigger(self) -> None:
        if not self._requested.is_set():
            self._requested.set()
            print(
                "\n[stop] SPACE pressed — will stop cleanly after the current "
                "taxon finishes.",
                flush=True,
            )

    def _run_windows(self) -> None:
        import msvcrt

        while not self._stop.is_set():
            while msvcrt.kbhit():
                if msvcrt.getwch() == " ":
                    self._trigger()
            time.sleep(0.05)

    def _setup_posix(self) -> bool:
        try:
            import termios
            import tty
        except ImportError:
            return False
        self._fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(self._fd)
        except termios.error:
            return False
        tty.setcbreak(self._fd)
        self._restore = lambda: termios.tcsetattr(
            self._fd, termios.TCSADRAIN, old
        )
        return True

    def _run_posix(self) -> None:
        import select

        while not self._stop.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready and sys.stdin.read(1) == " ":
                self._trigger()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.3)
        if self._restore is not None:
            try:
                self._restore()
            except Exception:
                pass
            self._restore = None


# --- Licence filter ----------------------------------------------------------


def is_cc_license(code: str | None) -> bool:
    """True for any Creative Commons licence code (``cc0``, ``cc-by``, ...).

    iNaturalist uses ``None`` for all-rights-reserved, so anything that starts
    with ``cc`` is a CC licence we may redistribute.
    """
    return bool(code) and str(code).lower().startswith("cc")


# --- iNaturalist life-stage annotation ---------------------------------------

# iNaturalist "Life Stage" controlled term (term_id 1) and its value ids.
LIFE_STAGE_TERM_ID = 1

# Values that mean an adult moth (winged): routed to the pose path and labelled
# "Adult". Teneral is a freshly emerged adult, so it belongs here too.
LIFE_STAGE_ADULT_VALUES = {2, 3}  # Adult, Teneral

# Non-adult life-stage values that map onto a Django stage class. Nymph (5),
# Juvenile (8) and Subimago (16) have no moth stage class, so they are skipped.
LIFE_STAGE_TO_STAGE = {
    4: "Pupa",
    6: "Larva",
    7: "Egg",
}

# Human-readable names for the Life Stage value ids (for the CSV log).
LIFE_STAGE_NAME = {
    2: "Adult",
    3: "Teneral",
    4: "Pupa",
    5: "Nymph",
    6: "Larva",
    7: "Egg",
    8: "Juvenile",
    16: "Subimago",
}

# Columns of the <tax>_observations_list.csv audit log. Kept minimal: enough to
# find and re-check an observation later without the noise of derived fields.
# ``status`` is first so the terminal marker rows read clearly at the bottom.
CSV_FIELDS = [
    "status",
    "observation_id",
    "created_at",
    "life_stage",
]

# Terminal ``status`` markers appended once the tax is fully handled. Their
# presence means the run finished (or was flagged) — a re-run should not touch
# the tax again. ``done`` = target met; ``no_more_observations`` = ran out
# before the target; ``reached_scan_limit`` = hit --max-observations before the
# target; ``corrupted`` = a previous run was interrupted.
TERMINATION_STATUSES = {
    "done",
    "no_more_observations",
    "reached_scan_limit",
    "corrupted",
}


def load_csv_index(path: Path) -> dict[str, dict[str, str]]:
    """Load an existing observations-list CSV as ``{observation_id: row}``.

    Lets repeat runs accumulate into one growing audit log rather than
    overwriting earlier results. Terminal marker rows (empty observation_id) are
    skipped. Missing/unreadable files yield an empty index.
    """
    index: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return index
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                obs_id = (row.get("observation_id") or "").strip()
                if obs_id:
                    index[obs_id] = row
    except OSError:
        pass
    return index


def csv_termination_status(path: Path) -> str | None:
    """Return the terminal marker found in the CSV, or ``None`` if none present.

    Used at startup to decide whether a tax has already been fully processed.
    """
    if not path.is_file():
        return None
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                status = (row.get("status") or "").strip()
                if status in TERMINATION_STATUSES:
                    return status
    except OSError:
        pass
    return None


def append_termination(path: Path, status: str) -> None:
    """Append a single terminal marker row (only ``status`` set) to the CSV."""
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
        ).writerow({"status": status})


def life_stage_value(observation: dict[str, Any]) -> int | None:
    """Return the observation's Life Stage annotation value id, or ``None``.

    Reads the ``annotations`` already present in the observation payload (no
    extra request). If several people annotated different values, the one with
    the highest ``vote_score`` wins (ties resolved by first seen). This is the
    value used to *route* an observation (adult pose vs stage sample); the audit
    log records the raw documented value(s) instead (see
    :func:`life_stage_documented`).
    """
    best_value: int | None = None
    best_score = None
    for annotation in observation.get("annotations") or []:
        if annotation.get("controlled_attribute_id") != LIFE_STAGE_TERM_ID:
            continue
        value = annotation.get("controlled_value_id")
        if value is None:
            continue
        score = annotation.get("vote_score", 0) or 0
        if best_score is None or score > best_score:
            best_value, best_score = int(value), score
    return best_value


def life_stage_documented(observation: dict[str, Any]) -> str:
    """Return the Life Stage value(s) exactly as documented on the observation.

    Lists every distinct Life Stage annotation value present (mapped to its
    term name; unknown ids kept as their number), joined by ``/`` and preserving
    document order. No vote-based deduction is applied, so the audit log mirrors
    what iNaturalist shows. Empty string when the stage isn't annotated.
    """
    names: list[str] = []
    for annotation in observation.get("annotations") or []:
        if annotation.get("controlled_attribute_id") != LIFE_STAGE_TERM_ID:
            continue
        value = annotation.get("controlled_value_id")
        if value is None:
            continue
        name = LIFE_STAGE_NAME.get(int(value), str(value))
        if name not in names:
            names.append(name)
    return "/".join(names)


# --- Observation paging ------------------------------------------------------


def iter_observations(
    inat,
    session: requests.Session,
    taxon_id: int,
    per_page: int,
    exclude_ids: set[Any],
) -> Iterator[dict[str, Any]]:
    """Yield research-grade observations for a taxon, newest first, lazily.

    Pages through the API on demand so the caller can stop as soon as it has
    collected enough qualifying images.
    """
    page = 1
    while True:
        params: dict[str, Any] = {
            "taxon_id": taxon_id,
            "photos": "true",
            "quality_grade": "research",
            "order_by": "created_at",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }
        payload = inat.request_json(session, inat.API_URL, params=params)
        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            return

        for observation in results:
            if observation.get("id") in exclude_ids:
                continue
            yield observation

        if len(results) < per_page:
            return
        page += 1
        time.sleep(0.25)


# --- Scoring helpers ---------------------------------------------------------


def _fmt(score: float | None) -> str:
    """Format an optional 0..1 sub-score for the progress line."""
    return "n/a" if score is None else f"{score:.3f}"


def side_pixel_span(moth_utils, filename: str) -> float | None:
    """Largest pixel span among the visible keypoints of a side pose.

    A side view has only F, B and one wing visible, so the app's
    ``pose_pixel_span`` (which needs all four keypoints) returns ``None``. Here
    the span is the biggest pairwise distance among the visible keypoints, in
    original-image pixels — the side-view analogue of the top-down pixel span.
    Returns ``None`` when it can't be computed.
    """
    annotations, _source = moth_utils.load_pose_source(filename)
    if not annotations:
        return None
    keypoints = annotations[0].keypoints
    if len(keypoints) < 4:
        return None
    size = moth_utils.get_image_size(filename)
    if size is None:
        return None
    width, height = size
    pts = [
        (kp.x * width, kp.y * height)
        for kp in keypoints[:4]
        if kp.visibility > 0
    ]
    if len(pts) < 2:
        return None
    span = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            span = max(span, math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))
    return span


# --- Cleanup -----------------------------------------------------------------


def discard(moth_utils, image_path: Path, filename: str) -> None:
    """Remove a rejected download and its derived files from the dataset."""
    for path in (
        image_path,
        moth_utils.get_prediction_path(filename),
        moth_utils.get_prediction_class_path(filename),
    ):
        try:
            path.unlink()
        except OSError:
            pass
    try:
        moth_utils.clear_normalized(filename)
    except Exception:
        pass


# --- Metadata ----------------------------------------------------------------


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    return data if isinstance(data, list) else []


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download, predict and score iNaturalist photos for a taxon, "
            "keeping only the best top-down and side-view images (via the same "
            "3-model prediction scheme as ultralytics-predict.py)."
        )
    )
    parser.add_argument(
        "taxon",
        help=(
            "iNaturalist taxon id to harvest, OR a path to a CSV file whose "
            "--column holds taxon ids (each is harvested in turn)."
        ),
    )
    parser.add_argument(
        "--column",
        default=None,
        help=(
            "When the positional argument is a CSV file, the name of the "
            "column holding iNaturalist taxon ids."
        ),
    )
    parser.add_argument(
        "--classification-model",
        dest="cls_model",
        type=Path,
        required=True,
        help="Box-only viewpoint/stage classification model (.pt).",
    )
    parser.add_argument(
        "--pose-model",
        dest="pose_model",
        type=Path,
        required=True,
        help="General F/L/R/B pose model (.pt).",
    )
    parser.add_argument(
        "--side-model",
        dest="side_model",
        type=Path,
        required=True,
        help="Side-view pose model (F, B, wing; class encodes L/R) (.pt).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=20,
        help=(
            "Stop once this many top-down OR this many side-view images are "
            "kept (default: 20 each; whichever is reached first)."
        ),
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=100,
        help="Observations fetched per API page (default: 100).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum scaled sub-score to keep an image: top-down needs "
            "min(symmetry, pixels, sharpness) to exceed it; side view needs "
            "min(pixels, sharpness) (symmetry undefined). Default: 0.5."
        ),
    )
    parser.add_argument(
        "--stage-samples",
        type=int,
        default=5,
        help=(
            "Max images to keep per non-adult life stage (larva/pupa/egg) "
            "encountered while searching for pose images (default: 5)."
        ),
    )
    parser.add_argument(
        "--image-size",
        choices=("original", "large", "medium"),
        default="original",
        help="Requested iNaturalist image rendition (default: original).",
    )
    parser.add_argument("--imgsz", type=int, default=768, help="Inference image size.")
    parser.add_argument(
        "--conf", type=float, default=0.10, help="Minimum detection (box) confidence."
    )
    parser.add_argument(
        "--device", default="0", help="Inference device: 0, 1, cpu, etc."
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=250,
        help=(
            "Stop searching after examining this many observations/images "
            "(default: 250; 0 = no cap)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run a tax that was already processed. Images already kept "
            "(CSV 'taken' rows) are preserved and counted toward the quotas; "
            "the search then starts over for every other observation (skipping "
            "the already-taken ones). The audit CSV is rewritten keeping only "
            "the 'taken' rows, with fresh outcomes appended by the re-run."
        ),
    )
    return parser.parse_args()


def make_model_loader(args):
    """Return a lazy loader that builds the three YOLO models once, on demand.

    Importing ultralytics pulls in torch and loading each model costs several
    seconds, so the load is deferred until a taxon actually reaches the
    download/predict stage. In a CSV batch, taxa already finished (skipped on
    their audit-CSV startup gate) never trigger it; the first taxon that needs
    the models pays the cost and every later one reuses the same instances.
    """
    cache: dict[str, Any] = {}

    def load():
        if not cache:
            from ultralytics import YOLO

            print(f"Loading classification model: {args.cls_model}")
            print(f"Loading pose model:           {args.pose_model}")
            print(f"Loading side-view pose model: {args.side_model}")
            cache["models"] = {
                "cls": YOLO(str(args.cls_model)),
                "pose": YOLO(str(args.pose_model)),
                "side": YOLO(str(args.side_model)),
            }
        return cache["models"]

    return load


def iter_taxon_ids(target: str, column: str | None) -> Iterator[str]:
    """Yield taxon-id strings from either a single id or a CSV column.

    ``target`` is either a path to an existing CSV file — then ``column`` names
    the taxon-id column and every non-empty value is yielded, in file order — or
    a single taxon id, yielded on its own.
    """
    path = Path(target)
    if path.is_file():
        if not column:
            raise SystemExit(
                "A CSV file was given; use --column NAME to pick the taxon-id "
                "column."
            )
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column not in reader.fieldnames:
                available = ", ".join(reader.fieldnames or [])
                raise SystemExit(
                    f"Column {column!r} not found. Available columns: {available}"
                )
            for row in reader:
                value = (row.get(column) or "").strip()
                if value:
                    yield value
    else:
        value = target.strip()
        if value:
            yield value


def run_taxon(taxon_id, args, moth_utils, inat, session, load_models) -> None:
    """Harvest a single taxon: download, predict, score, log and cache.

    Uses the already-bootstrapped Django ``moth_utils``, the shared iNaturalist
    ``session`` and the lazy ``load_models`` loader (so the models are only built
    the first time any taxon reaches the prediction stage). The audit-CSV
    startup gate can return early (already finished / corrupted) before the
    models are ever loaded.
    """
    tax = str(taxon_id)

    csv_path = moth_utils.get_image_dir() / f"{tax}_observations_list.csv"

    # ``taken_ids`` are observations kept by a previous run (force only); the
    # CSV rewrite in ``finally`` starts from these preserved rows. Empty for a
    # fresh tax or a plain (non-force) run.
    taken_ids: set[str] = set()
    csv_base_index: dict[str, dict[str, str]] = {}

    # Startup gate on the audit CSV. If the tax was already finished (any
    # terminal marker present) we skip it entirely. If the CSV exists without a
    # terminal marker, a previous run was interrupted (Ctrl-C): flag it
    # ``corrupted`` and skip, leaving it for manual inspection.
    #
    # With ``--force`` an already-processed tax is instead re-run: the CSV's
    # ``taken`` rows are preserved (every other row and the terminal marker are
    # dropped) and their images are kept and counted toward the quotas, while
    # the search starts over for all other observations.
    if csv_path.is_file():
        if not args.force:
            termination = csv_termination_status(csv_path)
            if termination is not None:
                print(f"Tax {tax} already processed ({termination}); skipping.")
                return
            append_termination(csv_path, "corrupted")
            print(
                f"Tax {tax} has an unfinished previous run; marked 'corrupted' "
                f"and skipping."
            )
            return
        for obs_id, row in load_csv_index(csv_path).items():
            if (row.get("status") or "").strip() == "taken":
                taken_ids.add(obs_id)
                csv_base_index[obs_id] = row
        print(
            f"Force re-run for tax {tax}: preserving {len(taken_ids)} taken "
            f"image(s); re-examining all other observations."
        )

    image_directory = moth_utils.get_image_dir() / tax
    image_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = moth_utils.get_observations_path(tax)

    existing_items = load_json_list(metadata_path)
    existing_ids = {
        item.get("observation_id")
        for item in existing_items
        if item.get("observation_id") is not None
    }

    # Heavy initialization is deferred until AFTER the CSV startup gate above:
    # importing ultralytics pulls in torch and loading the three YOLO models
    # costs several seconds each. Skipped (already-finished) taxa never reach
    # here, so the models load only when there is real work to do; ``load_models``
    # caches them so every later taxon in a CSV batch reuses the same instances.
    models = load_models()

    top_down_items: list[dict[str, Any]] = []       # adult top-down pose images
    side_items: list[dict[str, Any]] = []           # adult side-view pose images
    stage_items: list[dict[str, Any]] = []          # non-adult stage samples
    stage_counts = {stage: 0 for stage in LIFE_STAGE_TO_STAGE.values()}

    # Force re-run: never touch images already on disk. EVERY observation that
    # already has an image is excluded from the search, so it is neither
    # re-downloaded nor ever passed to ``discard`` — guaranteeing no existing
    # image is deleted or overwritten. Those images also count toward the quotas
    # (``base_top_down`` / ``base_side`` for adult poses, ``stage_counts`` for
    # stage samples) so the re-run only tops them up. Taken CSV rows are added
    # too, defensively, in case an image is missing on disk.
    base_top_down = 0
    base_side = 0
    if args.force:
        for obs_id in taken_ids:
            try:
                existing_ids.add(int(obs_id))
            except ValueError:
                existing_ids.add(obs_id)
        for image in moth_utils.scan_tax_images(tax):
            try:
                existing_ids.add(int(image.obs_id))
            except (TypeError, ValueError):
                existing_ids.add(image.obs_id)
            stage = moth_utils.get_image_class(image.filename)
            if stage in stage_counts:
                stage_counts[stage] += 1
            elif stage == "Adult":
                pose = moth_utils.classify_pose(image.filename)
                if pose == moth_utils.POSE_TOP_DOWN:
                    base_top_down += 1
                elif pose == moth_utils.POSE_SIDE:
                    base_side += 1
        print(
            f"  preserved on disk: top-down={base_top_down}, side={base_side}, "
            + ", ".join(f"{s}={c}" for s, c in stage_counts.items())
        )

    examined = 0
    downloaded = 0
    rejected_license = 0
    rejected_pose = 0
    rejected_score = 0
    no_prediction = 0
    rejected_stage_full = 0   # non-adult but its category quota is already full
    rejected_stage_other = 0  # non-adult stage with no matching moth class

    def targets_met() -> bool:
        return (
            base_top_down + len(top_down_items) >= args.target
            or base_side + len(side_items) >= args.target
        )

    print(
        f"Harvesting up to {args.target} top-down OR {args.target} side-view "
        f"images for taxon {taxon_id} (kept when the minimum relevant sub-score "
        f"> {args.score_threshold}); up to {args.stage_samples} samples per "
        f"non-adult stage"
    )

    # A single dropped image prints one dot (no newline) so progress is visible
    # without a line per rejection. ``progress_open`` tracks whether a run of
    # dots is on the current line so full messages break to a fresh line first.
    progress_open = False

    def dropped(mark: str = ".") -> None:
        # Progress marks by reason: c=non-CC licence, q=non-adult quota full,
        # -=no pose detected, /=wrong pose (not top-down/side), b=failed score,
        # .=other.
        nonlocal progress_open
        print(mark, end="", flush=True)
        progress_open = True

    def newline_if_needed() -> None:
        nonlocal progress_open
        if progress_open:
            print()
            progress_open = False

    # Set when the search stopped because of the --max-observations cap (rather
    # than reaching the target or running out of observations).
    hit_scan_limit = False

    # Audit log: one row per observation checked this run, recording the outcome
    # (``taken`` or the rejection reason) so images can be revisited/added later.
    records: list[dict[str, str]] = []

    def log_status(observation: dict[str, Any], status: str) -> None:
        obs_id = observation.get("id")
        records.append(
            {
                "observation_id": "" if obs_id is None else str(obs_id),
                "created_at": observation.get("created_at") or "",
                "life_stage": life_stage_documented(observation),
                "status": status,
            }
        )

    try:
        for observation in iter_observations(
            inat, session, taxon_id, args.per_page, existing_ids
        ):
            if targets_met():
                break
            if args.max_observations and examined >= args.max_observations:
                newline_if_needed()
                print("Reached --max-observations cap.")
                hit_scan_limit = True
                break
            examined += 1

            selected = inat.select_first_photos([observation], args.image_size)
            if not selected:
                log_status(observation, "no_photo")
                continue
            item = selected[0]

            if not is_cc_license(item.get("license_code")):
                rejected_license += 1
                log_status(observation, "rejected_non_cc")
                dropped("c")
                continue

            # Read the observation's life stage before doing any pose work. Only
            # adult (or unannotated) specimens go through pose prediction; other
            # stages are collected as capped per-category samples instead.
            stage_value = life_stage_value(observation)
            if stage_value is not None and stage_value not in LIFE_STAGE_ADULT_VALUES:
                stage = LIFE_STAGE_TO_STAGE.get(stage_value)
                if stage is None:
                    # e.g. nymph/juvenile/subimago: no moth stage class to set.
                    rejected_stage_other += 1
                    log_status(observation, "rejected_stage_no_class")
                    dropped(".")
                    continue
                if stage_counts[stage] >= args.stage_samples:
                    rejected_stage_full += 1
                    log_status(observation, "rejected_stage_quota_full")
                    dropped("q")
                    continue

                try:
                    dest = inat.download_photo(
                        session=session,
                        item=item,
                        image_directory=image_directory,
                        taxon_id=taxon_id,
                    )
                except requests.RequestException as error:
                    newline_if_needed()
                    print(
                        f"  observation {item['observation_id']} download "
                        f"failed: {error}"
                    )
                    log_status(observation, "download_failed")
                    continue
                downloaded += 1
                moth_utils.set_image_class(dest.name, stage)
                stage_counts[stage] += 1
                stage_items.append(item)
                log_status(observation, "taken")
                newline_if_needed()
                print(
                    f"  sample [{stage} {stage_counts[stage]}/{args.stage_samples}]"
                    f" obs {item['observation_id']} ({dest.name})"
                )
                continue

            try:
                dest = inat.download_photo(
                    session=session,
                    item=item,
                    image_directory=image_directory,
                    taxon_id=taxon_id,
                )
            except requests.RequestException as error:
                newline_if_needed()
                print(f"  observation {item['observation_id']} download failed: {error}")
                log_status(observation, "download_failed")
                continue
            downloaded += 1
            filename = dest.name

            # --- Three-model prediction scheme (shared: utils_prediction).
            # Classify the viewpoint first; only top-down (pinned or not) and
            # side-view specimens are harvest targets, so non-targets are
            # rejected before running any (expensive) pose model.
            classification = pred.classify_top(models, dest, args)
            if classification is None:
                no_prediction += 1
                discard(moth_utils, dest, filename)
                log_status(observation, "rejected_no_pose")
                dropped("-")
                continue
            cls_id = classification.cls_id

            if cls_id in pred.CLS_TOP_DOWN_ANY:
                category = "top_down"
            elif cls_id == pred.CLS_SIDE_VIEW:
                category = "side"
            else:
                # bottom-up / unclear / macro / larva: not a harvest target.
                rejected_pose += 1
                discard(moth_utils, dest, filename)
                log_status(observation, "rejected_not_top_down")
                dropped("/")
                continue

            # Run the matching pose model(s) via the shared pipeline step, then
            # write the label + .class sidecars exactly as ultralytics-predict.
            status, box, keypoints = pred.predict_pose_for_class(
                models, dest, args, cls_id, moth_utils
            )
            if keypoints is None:
                no_prediction += 1
                discard(moth_utils, dest, filename)
                log_status(observation, "rejected_no_pose")
                dropped("-")
                continue

            prediction = pred.Prediction(
                status=status,
                classification=classification,
                box=box,
                keypoints=keypoints,
            )
            pred.write_prediction(
                prediction,
                moth_utils.get_prediction_path(filename),
                moth_utils.get_prediction_class_path(filename),
                moth_utils,
            )

            # The visibility rules may downgrade a geometry that disagrees with
            # the classification to ``unclear``; confirm the pose really is the
            # target category before scoring.
            actual_pose = moth_utils.classify_pose(filename)
            row = moth_utils.compute_pose_row(filename)

            if category == "top_down":
                if actual_pose != moth_utils.POSE_TOP_DOWN:
                    rejected_pose += 1
                    discard(moth_utils, dest, filename)
                    log_status(observation, "rejected_not_top_down")
                    dropped("/")
                    continue
                # Top-down uses all three sub-scores.
                s_sym, s_pixels, s_sharp = moth_utils.score_components(
                    row["symmetry"], row["pixel_span"], row["sharpness"]
                )
                sub_scores = [s_sym, s_pixels, s_sharp]
                metric_text = (
                    f"sym={_fmt(s_sym)} pix={_fmt(s_pixels)} sharp={_fmt(s_sharp)}"
                )
            else:  # side
                if actual_pose != moth_utils.POSE_SIDE:
                    rejected_pose += 1
                    discard(moth_utils, dest, filename)
                    log_status(observation, "rejected_not_top_down")
                    dropped("/")
                    continue
                # Side view has one wing, so symmetry is undefined; score on the
                # pixel span (of the visible keypoints) and sharpness only.
                _s_sym, s_pixels, s_sharp = moth_utils.score_components(
                    None, side_pixel_span(moth_utils, filename), row["sharpness"]
                )
                sub_scores = [s_pixels, s_sharp]
                metric_text = f"pix={_fmt(s_pixels)} sharp={_fmt(s_sharp)}"

            if any(score is None for score in sub_scores) or (
                min(sub_scores) <= args.score_threshold
            ):
                rejected_score += 1
                discard(moth_utils, dest, filename)
                log_status(observation, "rejected_low_score")
                dropped("b")
                continue

            newline_if_needed()
            # A kept top-down/side specimen is an adult; label it accordingly.
            moth_utils.set_image_class(filename, "Adult")
            # observations.json stays pure observation metadata; the pose/
            # metrics/keypoints live in the prediction .txt (test dir) and
            # {tax}_pose_data.json (labels dir), matching the Django layout.
            if category == "top_down":
                top_down_items.append(item)
                progress = (
                    f"top-down {base_top_down + len(top_down_items):02d}/"
                    f"{args.target}"
                )
            else:
                side_items.append(item)
                progress = f"side {base_side + len(side_items):02d}/{args.target}"
            log_status(observation, "taken")
            print(
                f"  [{progress}] kept obs {item['observation_id']} "
                f"({filename}) {metric_text}"
            )
    finally:
        newline_if_needed()
        # Persist kept observation metadata (merged, newest first) so the site
        # has licence/quality info for the harvested images — both the adult
        # pose images and the non-adult stage samples.
        new_items = top_down_items + side_items + stage_items
        if new_items:
            merged = existing_items + new_items
            merged.sort(key=inat.sort_key, reverse=True)
            metadata_path.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Persist the per-observation audit log. A fresh run starts empty; a
        # force re-run starts from the preserved ``taken`` rows only (all other
        # prior rows and the terminal marker are dropped). Both then overlay the
        # outcomes recorded this run. Always written (even with no rows) so the
        # file's presence signals the tax_id has at least been looked at.
        index = dict(csv_base_index)
        for record in records:
            index[record["observation_id"]] = record
        rows = sorted(
            index.values(),
            key=lambda r: r.get("created_at") or "",
            reverse=True,
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)

    # Build the pose-data cache for the images now on disk. This writes
    # <labels-dir>/<tax>_pose_data.json (version + per-image rows) and the
    # normalized crops in the cache dir, in the exact format Django expects.
    # Stage samples have no prediction, so they simply record pose "none".
    if top_down_items or side_items or stage_items:
        moth_utils.build_pose_data(tax, moth_utils.scan_tax_images(tax))

    # Effective totals include images preserved from a previous run (force).
    total_top_down = base_top_down + len(top_down_items)
    total_side = base_side + len(side_items)

    # Terminal marker — the very last thing written, only reached on a clean
    # finish. Its absence is how a re-run detects an interrupted (Ctrl-C) run.
    if total_top_down >= args.target or total_side >= args.target:
        termination = "done"
    elif hit_scan_limit:
        termination = "reached_scan_limit"
    else:
        termination = "no_more_observations"
    append_termination(csv_path, termination)

    # On a force re-run, note the preserved counts alongside the new ones.
    def _kept(total: int, new: int) -> str:
        return f"{total}" + (f" (+{new} new)" if base_top_down or base_side else "")

    print()
    print(f"Examined observations: {examined}")
    print(f"Downloaded:            {downloaded}")
    print(f"Rejected (licence):    {rejected_license}")
    print(f"Rejected (no pose):    {no_prediction}")
    print(f"Rejected (wrong pose): {rejected_pose}")
    print(f"Rejected (low score):  {rejected_score}")
    print(f"Rejected (stage full): {rejected_stage_full}")
    print(f"Rejected (stage n/a):  {rejected_stage_other}")
    print(f"Kept (top-down):       {_kept(total_top_down, len(top_down_items))}")
    print(f"Kept (side view):      {_kept(total_side, len(side_items))}")
    stage_summary = ", ".join(
        f"{stage}={count}" for stage, count in stage_counts.items()
    )
    print(f"Kept (stage samples):  {stage_summary}")
    if total_top_down < args.target and total_side < args.target:
        print("Ran out of observations before reaching either pose target.")


def main() -> None:
    args = parse_args()

    for label, model_path in (
        ("classification", args.cls_model),
        ("pose", args.pose_model),
        ("side", args.side_model),
    ):
        if not model_path.exists():
            raise SystemExit(f"{label} model does not exist: {model_path}")

    # Directories come from the environment via Django settings (every MOTHS_*
    # path must be set, no defaults), exactly like the other tools here.
    moth_utils = bootstrap_django()
    inat = import_inat()
    session = inat.create_session()
    load_models = make_model_loader(args)

    # Clean-stop watcher: pressing SPACE lets the current taxon finish, then
    # stops before the next one. Only meaningful when walking a CSV of many
    # taxa; harmless for a single taxon.
    watcher = SpaceStopWatcher()
    watcher.start()
    if watcher.enabled:
        print(
            "Press SPACE at any time to stop cleanly after the current taxon "
            "finishes."
        )

    # Materialize the list up front so each caption can show "N of M".
    taxon_ids = list(iter_taxon_ids(args.taxon, args.column))
    total = len(taxon_ids)

    processed = 0
    stopped_early = False
    try:
        for index, raw_id in enumerate(taxon_ids, start=1):
            # Honor a SPACE press before starting the next taxon.
            if watcher.requested:
                stopped_early = True
                print(
                    f"[stop] Clean stop requested — not starting taxon "
                    f"{raw_id} or any later ones."
                )
                break
            try:
                taxon_id = int(raw_id)
            except ValueError:
                print(f"Skipping non-numeric taxon id: {raw_id!r}")
                continue
            print()
            print(f"=== [{index} of {total}]: tax {taxon_id} ===")
            run_taxon(taxon_id, args, moth_utils, inat, session, load_models)
            processed += 1
    finally:
        watcher.close()

    print()
    print(f"Taxa processed: {processed}")
    if stopped_early:
        print("Stopped early on user request (SPACE).")


if __name__ == "__main__":
    main()
