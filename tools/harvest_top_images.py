#!/usr/bin/env python3

"""Harvest the best top-down photos for a taxon from iNaturalist.

For a given iNaturalist taxon this walks research-grade observations
(newest first) and, for each one whose main photo carries a Creative Commons
(``cc*``) licence, branches on the observation's Life Stage annotation:

* **Adult or unannotated** – the pose candidate path: download the main photo,
  run the YOLO pose model, keep only ``top-down`` poses, compute the same
  symmetry / pixel-span / sharpness metrics the Django app uses (via
  ``moths.utils``) and keep the image only when *every* scaled 0..1 sub-score
  exceeds a threshold (default ``0.5``). Kept images are labelled ``Adult``.
* **Any other stage** (larva/pupa/egg) – the sample path: no pose prediction;
  just keep up to ``--stage-samples`` images per stage (default 5) and label
  each with the matching Django stage. Stages with no moth class (nymph,
  juvenile, subimago) are skipped.

It keeps going, paging further back through observations, until it has
collected ``--target`` adult pose images (default 20) or iNaturalist has no more
observations. The search stops as soon as the pose target is met (stage-sample
quotas may end up partially filled). Non-qualifying pose downloads (wrong pose
or low score) are deleted so the dataset only gains images that pass.

The data directories are passed explicitly (never read from ``settings.py``):
``--images-dir`` (photos + ``<tax>_observations.json``), ``--test-dir``
(predicted keypoint ``.txt`` files), ``--labels-dir`` (``<tax>_pose_data.json``
with pose/metrics/scores), ``--cache-dir`` (normalized-crop cache) and
``--classes-dir`` (``<name>.class`` stage labels). They mirror the Django layout
exactly, so pointing the app's ``MOTHS_IMAGE_DIR`` / ``MOTHS_PREDICTION_DIR`` /
``MOTHS_LABEL_DIR`` / ``MOTHS_THUMBNAIL_DIR`` / ``MOTHS_CLASS_DIR`` at these same
folders makes the harvest directly browsable.
Keypoints are stored only in the test dir and pose data only in the labels dir;
``<tax>_observations.json`` stays pure observation metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import requests


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


# --- YOLO prediction ---------------------------------------------------------


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def predict_pose_line(
    model,
    image_path: Path,
    imgsz: int,
    conf: float,
    device: str,
    keypoint_conf: float,
) -> str | None:
    """Run the model on one image and return a single YOLO-pose label line.

    Returns ``None`` when nothing is detected. Only the highest-confidence
    detection is used (a photo is expected to contain one moth).
    """
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        max_det=1,
        device=device,
        batch=1,
        stream=False,
        verbose=False,
    )
    result = results[0]
    if (
        result.boxes is None
        or len(result.boxes) == 0
        or result.keypoints is None
        or len(result.keypoints) == 0
    ):
        return None

    index = int(result.boxes.conf.argmax().item())
    class_id = int(result.boxes.cls[index].item())
    cx, cy, width, height = result.boxes.xywhn[index].detach().cpu().tolist()

    values = [
        str(class_id),
        f"{clamp01(float(cx)):.6f}",
        f"{clamp01(float(cy)):.6f}",
        f"{clamp01(float(width)):.6f}",
        f"{clamp01(float(height)):.6f}",
    ]

    keypoints_xy = result.keypoints.xyn[index].detach().cpu().tolist()
    kp_conf_values = None
    if result.keypoints.conf is not None:
        kp_conf_values = result.keypoints.conf[index].detach().cpu().tolist()

    for kp_index, (x, y) in enumerate(keypoints_xy):
        if kp_conf_values is None:
            visibility = 2
        else:
            visibility = 2 if float(kp_conf_values[kp_index]) >= keypoint_conf else 0
        if visibility == 0:
            x = y = 0.0
        values.extend(
            [f"{clamp01(float(x)):.6f}", f"{clamp01(float(y)):.6f}", str(visibility)]
        )

    return " ".join(values)


# --- Cleanup -----------------------------------------------------------------


def discard(moth_utils, image_path: Path, filename: str) -> None:
    """Remove a rejected download and its derived files from the dataset."""
    prediction_path = moth_utils.get_prediction_path(filename)
    for path in (image_path, prediction_path):
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
            "keeping only the best top-down images."
        )
    )
    parser.add_argument("taxon_id", type=int, help="iNaturalist taxon ID")
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the trained YOLO pose model (e.g. .../weights/best.pt)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=20,
        help="Stop once this many qualifying images are kept (default: 20).",
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
            "Minimum for EACH scaled sub-score (symmetry, pixels, sharpness); "
            "an image is kept only when all three exceed it (default: 0.5)."
        ),
    )
    # Data directories are passed explicitly (never taken from settings.py) so
    # a harvest can target its own tree. They mirror the Django layout exactly,
    # so pointing the app's settings at the same four dirs makes the result
    # directly browsable.
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help=(
            "Images root (photos in <images-dir>/<taxon_id>/ and "
            "<taxon_id>_observations.json at the root). Maps to MOTHS_IMAGE_DIR."
        ),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help=(
            "Prediction (tested keypoints) root: one YOLO-pose <name>.txt per "
            "image under <test-dir>/<taxon_id>/. Maps to MOTHS_PREDICTION_DIR."
        ),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help=(
            "Labels root holding <taxon_id>_pose_data.json (pose class, metrics "
            "and scores). Maps to MOTHS_LABEL_DIR."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help=(
            "Thumbnail/normalized-crop cache root (needed to score sharpness). "
            "Maps to MOTHS_THUMBNAIL_DIR."
        ),
    )
    parser.add_argument(
        "--classes-dir",
        type=Path,
        required=True,
        help=(
            "Stage-classification root: one <name>.class per image under "
            "<classes-dir>/<taxon_id>/. Maps to MOTHS_CLASS_DIR."
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
        "--conf", type=float, default=0.10, help="Minimum detection confidence."
    )
    parser.add_argument(
        "--keypoint-conf",
        type=float,
        default=0.0,
        help="Keypoints below this confidence are written with visibility 0.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model does not exist: {args.model}")

    # The Django settings module requires every data path via the environment
    # (no defaults). Feed it the explicitly-passed args before bootstrapping so
    # the whole flow (download, prediction, scoring) works on this same tree.
    os.environ["MOTHS_IMAGE_DIR"] = str(args.images_dir)
    os.environ["MOTHS_PREDICTION_DIR"] = str(args.test_dir)
    # No human labels exist in a harvest, so this (label-free) labels dir doubles
    # as the pose source and load_pose_source falls back to predictions.
    os.environ["MOTHS_LABEL_DIR"] = str(args.labels_dir)
    os.environ["MOTHS_THUMBNAIL_DIR"] = str(args.cache_dir)
    os.environ["MOTHS_CLASS_DIR"] = str(args.classes_dir)
    # Not used by the harvest, but the settings module still requires it;
    # point at a (typically absent) file under the labels dir so imports succeed.
    os.environ.setdefault("TAX_CSV", str(args.labels_dir / "names.csv"))

    moth_utils = bootstrap_django()
    inat = import_inat()

    taxon_id = args.taxon_id
    tax = str(taxon_id)

    csv_path = moth_utils.get_image_dir() / f"{tax}_observations_list.csv"

    # Startup gate on the audit CSV. If the tax was already finished (any
    # terminal marker present) we skip it entirely. If the CSV exists without a
    # terminal marker, a previous run was interrupted (Ctrl-C): flag it
    # ``corrupted`` and skip, leaving it for manual inspection.
    if csv_path.is_file():
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

    from ultralytics import YOLO

    image_directory = moth_utils.get_image_dir() / tax
    image_directory.mkdir(parents=True, exist_ok=True)
    metadata_path = moth_utils.get_observations_path(tax)

    existing_items = load_json_list(metadata_path)
    existing_ids = {
        item.get("observation_id")
        for item in existing_items
        if item.get("observation_id") is not None
    }

    session = inat.create_session()
    print(f"Loading model: {args.model}")
    model = YOLO(str(args.model))

    kept_items: list[dict[str, Any]] = []          # adult top-down pose images
    stage_items: list[dict[str, Any]] = []          # non-adult stage samples
    stage_counts = {stage: 0 for stage in LIFE_STAGE_TO_STAGE.values()}
    examined = 0
    downloaded = 0
    rejected_license = 0
    rejected_pose = 0
    rejected_score = 0
    no_prediction = 0
    rejected_stage_full = 0   # non-adult but its category quota is already full
    rejected_stage_other = 0  # non-adult stage with no matching moth class

    print(
        f"Harvesting up to {args.target} top-down images for taxon {taxon_id} "
        f"(each sub-score > {args.score_threshold}); up to {args.stage_samples} "
        f"samples per non-adult stage"
    )

    # A single dropped image prints one dot (no newline) so progress is visible
    # without a line per rejection. ``progress_open`` tracks whether a run of
    # dots is on the current line so full messages break to a fresh line first.
    progress_open = False

    def dropped(mark: str = ".") -> None:
        # Progress marks by reason: c=non-CC licence, q=non-adult quota full,
        # -=no pose detected, /=not top-down, b=failed score, .=other.
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
            if len(kept_items) >= args.target:
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

            line = predict_pose_line(
                model=model,
                image_path=dest,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                keypoint_conf=args.keypoint_conf,
            )
            if line is None:
                no_prediction += 1
                discard(moth_utils, dest, filename)
                log_status(observation, "rejected_no_pose")
                dropped("-")
                continue

            prediction_path = moth_utils.get_prediction_path(filename)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(line + "\n", encoding="utf-8")

            if moth_utils.classify_pose(filename) != moth_utils.POSE_TOP_DOWN:
                rejected_pose += 1
                discard(moth_utils, dest, filename)
                log_status(observation, "rejected_not_top_down")
                dropped("/")
                continue

            row = moth_utils.compute_pose_row(filename)
            s_sym, s_pixels, s_sharp = moth_utils.score_components(
                row["symmetry"], row["pixel_span"], row["sharpness"]
            )
            passed = (
                s_sym is not None
                and s_pixels is not None
                and s_sharp is not None
                and s_sym > args.score_threshold
                and s_pixels > args.score_threshold
                and s_sharp > args.score_threshold
            )
            if not passed:
                rejected_score += 1
                discard(moth_utils, dest, filename)
                log_status(observation, "rejected_low_score")
                dropped("b")
                continue

            newline_if_needed()
            # A predicted top-down specimen is an adult; label it accordingly.
            moth_utils.set_image_class(filename, "Adult")
            # observations.json stays pure observation metadata; the pose/
            # metrics/keypoints live in the prediction .txt (test dir) and
            # {tax}_pose_data.json (labels dir), matching the Django layout.
            kept_items.append(item)
            log_status(observation, "taken")
            print(
                f"  [{len(kept_items):02d}/{args.target}] kept obs "
                f"{item['observation_id']} ({filename}) "
                f"sym={s_sym:.3f} pix={s_pixels:.3f} sharp={s_sharp:.3f} "
                f"score={row['score']:.3f}"
            )
    finally:
        newline_if_needed()
        # Persist kept observation metadata (merged, newest first) so the site
        # has licence/quality info for the harvested images — both the adult
        # pose images and the non-adult stage samples.
        new_items = kept_items + stage_items
        if new_items:
            merged = existing_items + new_items
            merged.sort(key=inat.sort_key, reverse=True)
            metadata_path.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Persist the per-observation audit log, merging with any prior run so
        # the CSV grows into a complete record of every observation checked.
        # Always written (even with no rows) so the file's presence signals the
        # tax_id has at least been looked at.
        index = load_csv_index(csv_path)
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
    if kept_items or stage_items:
        moth_utils.build_pose_data(tax, moth_utils.scan_tax_images(tax))
        print(f"Pose data:             {moth_utils.get_pose_data_path(tax)}")

    # Terminal marker — the very last thing written, only reached on a clean
    # finish. Its absence is how a re-run detects an interrupted (Ctrl-C) run.
    if len(kept_items) >= args.target:
        termination = "done"
    elif hit_scan_limit:
        termination = "reached_scan_limit"
    else:
        termination = "no_more_observations"
    append_termination(csv_path, termination)

    print()
    print(f"Examined observations: {examined}")
    print(f"Downloaded:            {downloaded}")
    print(f"Rejected (licence):    {rejected_license}")
    print(f"Rejected (no pose):    {no_prediction}")
    print(f"Rejected (not top-down): {rejected_pose}")
    print(f"Rejected (low score):  {rejected_score}")
    print(f"Rejected (stage full): {rejected_stage_full}")
    print(f"Rejected (stage n/a):  {rejected_stage_other}")
    print(f"Kept (adult pose):     {len(kept_items)}")
    stage_summary = ", ".join(
        f"{stage}={count}" for stage, count in stage_counts.items()
    )
    print(f"Kept (stage samples):  {len(stage_items)} ({stage_summary})")
    if len(kept_items) < args.target:
        print("Ran out of observations before reaching the pose target.")
    print(f"Images directory:      {image_directory}")
    print(f"Classes directory:     {moth_utils.get_class_dir() / tax}")
    print(f"Observations log:      {csv_path}")
    print(f"Predictions directory: {moth_utils.get_prediction_dir() / tax}")
    print(f"Labels directory:      {moth_utils.get_label_dir()}")
    print(f"Cache directory:       {moth_utils.get_thumbnail_dir()}")


if __name__ == "__main__":
    main()
