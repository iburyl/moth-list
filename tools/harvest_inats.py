#!/usr/bin/env python3

"""Harvest iNaturalist taxon JSON (+ scientific synonyms) from the names CSV.

**Only this tool reads the original names CSV.** Everything downstream uses
``inats_list.json`` / ``inats_summary.json``.

Writes::

    <MOTHS_DATA_DIR>/<tax_id>/<tax_id>.inats
    <MOTHS_DATA_DIR>/<tax_id>/<tax_id>.synonyms
    <MOTHS_DATA_DIR>/parents/<parent_tax_id>.inats
    <MOTHS_DATA_DIR>/inats_list.json

``inats_list.json`` lists ``found`` ``{id, name}`` and ``not_found``
``{id, name}`` (CSV id + species name when iNat could not be resolved).

By default, skips taxa that already have a ``.inats`` hit, and skips
existing synonyms / parent files. Prior misses (empty ``<tax_id>/`` with no
``.inats``) are retried. Pass ``--force-rerun`` to re-fetch everything and
refresh data.

Taxa come from v2 (slim ``fields``, 200 per request), except the ``vision``
(computer-vision model) flag, which only v1 exposes and which is therefore
patched into ``.inats`` from the v1 passes.

Run ``tools/prepare_inats.py`` afterward to build ``inats_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import requests

# --- Repo / Django bootstrap -------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

_CONTACT = (
    os.environ.get("MOTHS_INATS_CONTACT")
    or os.environ.get("MOTHS_WIKI_CONTACT")
    or os.environ.get("WIKI_CONTACT")
    or "https://github.com/iburylov/moth-list"
)
USER_AGENT = (
    f"moth-list-harvest-inats/1.0 "
    f"(moth taxonomy research dataset; contact: {_CONTACT}) "
    f"python-requests/{requests.__version__}"
)

API_BASE_V2 = "https://api.inaturalist.org/v2"
API_BASE_V1 = "https://api.inaturalist.org/v1"

# v2 has no `vision` field and silently drops unknown names from `fields`,
# so the computer-vision flag comes from v1 (see `harvest_vision`).
TAXON_FIELDS = ",".join(
    [
        "id",
        "name",
        "rank",
        "rank_level",
        "parent_id",
        "ancestor_ids",
        "is_active",
        "extinct",
        "observations_count",
        "preferred_common_name",
        "wikipedia_url",
    ]
)
TAXON_BATCH_SIZE = 200
V1_BATCH_SIZE = 30
MAX_RETRY_AFTER_SECONDS = 300.0
MAX_RATE_LIMIT_RETRIES = 8
PARENTS_DIRNAME = "parents"
INATS_LIST_NAME = "inats_list.json"
INATS_LIST_VERSION = 1


# --- Clean-stop watcher (same idea as harvest_wiki.py) -----------------------


class SpaceStopWatcher:
    """Background watcher that trips a flag when SPACE is pressed."""

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

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._restore is not None:
            self._restore()
            self._restore = None

    def _run_windows(self) -> None:
        import msvcrt

        while not self._stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == " ":
                    self._requested.set()
                    return
            time.sleep(0.05)

    def _setup_posix(self) -> bool:
        try:
            import termios
            import tty
        except ImportError:
            return False
        try:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
        except (termios.error, ValueError):
            return False

        def restore() -> None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

        self._restore = restore
        tty.setcbreak(fd)
        return True

    def _run_posix(self) -> None:
        import select

        fd = sys.stdin.fileno()
        while not self._stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch == " ":
                self._requested.set()
                return


# --- Django ------------------------------------------------------------------


def bootstrap_django():
    """Set up Django and return ``(settings, load_names_csv)``."""
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from django.conf import settings
    from moths.utils.names import load_names_csv

    return settings, load_names_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harvest iNat taxa from the names CSV; write .inats/.synonyms, "
            "parents/, and inats_list.json. Skips existing harvests unless "
            "--force-rerun."
        )
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Re-fetch every species, synonym, and parent even when output "
            "files already exist."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to wait between API calls (default: 1.0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many newly processed species (0 = no limit).",
    )
    parser.add_argument(
        "--skip-parents",
        action="store_true",
        help="Do not fetch ancestor taxa under parents/.",
    )
    parser.add_argument(
        "--skip-synonyms",
        action="store_true",
        help="Do not harvest per-taxon .synonyms files.",
    )
    return parser.parse_args()


# --- HTTP / iNat -------------------------------------------------------------


def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
    raw = (response.headers.get("Retry-After") or "").strip()
    if raw:
        try:
            return min(max(float(raw), 0.0), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
        try:
            from email.utils import parsedate_to_datetime

            when = parsedate_to_datetime(raw)
            delay = when.timestamp() - time.time()
            return min(max(delay, 0.0), MAX_RETRY_AFTER_SECONDS)
        except (TypeError, ValueError, OSError, OverflowError):
            pass
    return min(2.0**attempt, MAX_RETRY_AFTER_SECONDS)


def _get_with_rate_limit(
    session: requests.Session,
    url: str,
    params: dict,
) -> requests.Response:
    last_response: requests.Response | None = None
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        response = session.get(url, params=params, timeout=60)
        if response.status_code != 429:
            return response
        last_response = response
        delay = _retry_after_seconds(response, attempt)
        retry_after = response.headers.get("Retry-After", "")
        print(
            f"  rate-limited (429), sleeping {delay:.1f}s "
            f"(Retry-After={retry_after!r}, attempt {attempt}/"
            f"{MAX_RATE_LIMIT_RETRIES})",
            file=sys.stderr,
        )
        time.sleep(delay)
    assert last_response is not None
    return last_response


def fetch_taxa_by_ids_v2(
    session: requests.Session,
    tax_ids: list[int | str],
) -> dict[str, dict[str, Any]]:
    """Return ``{id_str: slim taxon}`` from v2 (missing ids omitted).

    Uses ``GET /v2/taxa?id=…&fields=…`` so batches can exceed the path
    endpoint's 30-id cap.
    """
    if not tax_ids:
        return {}
    id_list = [str(i) for i in tax_ids]
    response = _get_with_rate_limit(
        session,
        f"{API_BASE_V2}/taxa",
        {
            "id": ",".join(id_list),
            "fields": TAXON_FIELDS,
            "per_page": len(id_list),
        },
    )
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    out: dict[str, dict[str, Any]] = {}
    for taxon in response.json().get("results") or []:
        tid = taxon.get("id")
        if tid is not None:
            out[str(tid)] = taxon
    return out


def search_species_taxon(
    session: requests.Session,
    scientific_name: str,
) -> dict[str, Any] | None:
    """Search v2 taxa by name; return the best species-rank match."""
    params = {
        "q": scientific_name,
        "rank": "species",
        "is_active": "true",
        "per_page": 30,
        "fields": TAXON_FIELDS,
    }
    response = _get_with_rate_limit(session, f"{API_BASE_V2}/taxa", params)
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None

    needle = scientific_name.strip().casefold()
    exact: list[dict[str, Any]] = []
    for taxon in results:
        if (taxon.get("rank") or "").casefold() != "species":
            continue
        name = (taxon.get("name") or "").strip().casefold()
        if name == needle:
            exact.append(taxon)
    if exact:
        return exact[0]

    for taxon in results:
        if (taxon.get("rank") or "").casefold() == "species":
            return taxon
    return None


def _is_scientific_name_entry(entry: dict[str, Any]) -> bool:
    locale = (entry.get("locale") or "").strip().casefold()
    lexicon = (entry.get("lexicon") or "").strip().casefold()
    if locale == "sci":
        return True
    return lexicon in {"scientific-names", "scientific names"}


def scientific_names_from_all_names(
    names: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Deduped scientific name rows: ``{name, is_valid}``."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in names or []:
        if not _is_scientific_name_entry(entry):
            continue
        name = (entry.get("name") or "").strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": name,
                "is_valid": bool(entry.get("is_valid", False)),
            }
        )
    return out


def fetch_taxa_by_ids_v1(
    session: requests.Session,
    tax_ids: list[str],
    *,
    all_names: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return ``{id_str: full v1 taxon}`` (missing ids omitted).

    v1 has no ``fields`` selector, so payloads are fat — but it is the only
    endpoint exposing ``vision`` (and ``names`` for synonyms).
    """
    if not tax_ids:
        return {}
    id_path = ",".join(tax_ids)
    url = f"{API_BASE_V1}/taxa/{id_path}"
    params = {"all_names": "true"} if all_names else {}
    response = _get_with_rate_limit(session, url, params)
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    out: dict[str, dict[str, Any]] = {}
    for taxon in response.json().get("results") or []:
        tid = taxon.get("id")
        if tid is not None:
            out[str(tid)] = taxon
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_inats(path: Path) -> dict[str, Any] | None:
    data = load_json(path)
    return data if isinstance(data, dict) else None


def ancestor_ids_from_taxon(taxon: dict[str, Any]) -> list[str]:
    """Higher-rank ancestor ids only (excludes the taxon itself).

    On ``GET /v2/taxa?id=…``, ``ancestor_ids`` includes ``id`` as the last
    entry — skip that so parents harvest does not re-fetch species.
    """
    self_id = taxon.get("id")
    self_key = str(self_id) if self_id is not None else None

    raw = taxon.get("ancestor_ids")
    if isinstance(raw, list) and raw:
        return [
            str(i)
            for i in raw
            if i is not None and (self_key is None or str(i) != self_key)
        ]

    ancestry = (taxon.get("ancestry") or "").strip()
    if ancestry:
        return [
            p
            for p in ancestry.split("/")
            if p and (self_key is None or p != self_key)
        ]

    parent_id = taxon.get("parent_id")
    if parent_id is not None and (self_key is None or str(parent_id) != self_key):
        return [str(parent_id)]
    return []


# --- Harvest -----------------------------------------------------------------


def write_species_taxon(
    data_dir: Path,
    tax_id: str,
    taxon: dict[str, Any],
) -> None:
    write_json(data_dir / tax_id / f"{tax_id}.inats", taxon)


def harvest_species_batched(
    session: requests.Session,
    data_dir: Path,
    names: dict[str, dict],
    tax_ids: list[str],
    *,
    force_rerun: bool,
    sleep: float,
    limit: int,
    watcher: SpaceStopWatcher,
) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    """Batch-fetch slim taxa for CSV ids; name-search only for misses.

    Returns ``(counts, harvested_taxa, found_rows, not_found_rows)``.
    """
    counts = {"wrote": 0, "missing": 0, "skip": 0, "error": 0}
    harvested: list[dict[str, Any]] = []
    found_rows: list[dict[str, str]] = []
    not_found_rows: list[dict[str, str]] = []
    found_ids: set[str] = set()
    processed = 0

    def _note_found(taxon: dict[str, Any]) -> None:
        tid = str(taxon.get("id") or "")
        if not tid or tid in found_ids:
            return
        found_ids.add(tid)
        found_rows.append(
            {
                "id": tid,
                "name": (taxon.get("name") or "").strip(),
            }
        )

    def _note_missing(csv_tax_id: str, species: str) -> None:
        not_found_rows.append(
            {
                "id": str(csv_tax_id),
                "name": (species or "").strip(),
            }
        )

    pending: list[str] = []
    for csv_tax_id in tax_ids:
        out_path = data_dir / csv_tax_id / f"{csv_tax_id}.inats"
        if not force_rerun and out_path.is_file():
            counts["skip"] += 1
            taxon = load_inats(out_path)
            if taxon is not None:
                harvested.append(taxon)
                _note_found(taxon)
            continue
        pending.append(csv_tax_id)

    for i in range(0, len(pending), TAXON_BATCH_SIZE):
        if watcher.requested:
            print("\nClean stop requested; not starting further species.")
            break
        if limit and processed >= limit:
            print(f"\nReached --limit {limit}.")
            break

        batch = pending[i : i + TAXON_BATCH_SIZE]
        if limit:
            batch = batch[: max(0, limit - processed)]
        if not batch:
            break

        try:
            found = fetch_taxa_by_ids_v2(session, batch)
        except (requests.RequestException, ValueError) as exc:
            print(f"  ERROR species batch {batch[0]}…: {exc}", file=sys.stderr)
            counts["error"] += len(batch)
            processed += len(batch)
            continue

        misses: list[str] = []
        for csv_tax_id in batch:
            taxon = found.get(csv_tax_id)
            species = (names[csv_tax_id].get("species") or "").strip()
            label = species or "(empty species)"
            if taxon is not None:
                write_species_taxon(data_dir, csv_tax_id, taxon)
                harvested.append(taxon)
                _note_found(taxon)
                counts["wrote"] += 1
                processed += 1
                print(f"OK      {csv_tax_id}  {label}  (by_id)")
            else:
                misses.append(csv_tax_id)

        if sleep > 0 and batch:
            time.sleep(sleep)

        for csv_tax_id in misses:
            if watcher.requested:
                break
            if limit and processed >= limit:
                break
            species = (names[csv_tax_id].get("species") or "").strip()
            label = species or "(empty species)"
            if not species:
                (data_dir / csv_tax_id).mkdir(parents=True, exist_ok=True)
                counts["missing"] += 1
                processed += 1
                _note_missing(csv_tax_id, species)
                print(f"MISSING {csv_tax_id}  {label}")
                continue
            try:
                taxon = search_species_taxon(session, species)
            except (requests.RequestException, ValueError) as exc:
                print(f"  ERROR search {species!r}: {exc}", file=sys.stderr)
                counts["error"] += 1
                processed += 1
                continue
            if taxon is None or taxon.get("id") is None:
                (data_dir / csv_tax_id).mkdir(parents=True, exist_ok=True)
                counts["missing"] += 1
                processed += 1
                _note_missing(csv_tax_id, species)
                print(f"MISSING {csv_tax_id}  {label}")
            else:
                resolved_id = str(taxon["id"])
                write_species_taxon(data_dir, resolved_id, taxon)
                harvested.append(taxon)
                _note_found(taxon)
                counts["wrote"] += 1
                processed += 1
                print(f"OK      {resolved_id}  {label}  (by_name)")
            if sleep > 0:
                time.sleep(sleep)

        if limit and processed >= limit:
            print(f"\nReached --limit {limit}.")
            break

    # Include any on-disk .inats not touched this run (e.g. prior harvest).
    for path in species_inats_paths(data_dir):
        taxon = load_inats(path)
        if taxon is not None:
            _note_found(taxon)

    return counts, harvested, found_rows, not_found_rows


def write_inats_list(
    data_dir: Path,
    found_rows: list[dict[str, str]],
    not_found_rows: list[dict[str, str]],
) -> Path:
    """Write ``inats_list.json`` (found / not_found id+name lists)."""

    def sort_key(row: dict[str, str]):
        tid = row.get("id") or ""
        return (0, int(tid)) if tid.isdigit() else (1, tid)

    # Dedup found by id (last name wins); not_found by (id, name).
    found_by_id: dict[str, dict[str, str]] = {}
    for row in found_rows:
        tid = (row.get("id") or "").strip()
        if tid:
            found_by_id[tid] = {"id": tid, "name": (row.get("name") or "").strip()}
    found = sorted(found_by_id.values(), key=sort_key)

    seen_nf: set[tuple[str, str]] = set()
    not_found: list[dict[str, str]] = []
    for row in not_found_rows:
        key = ((row.get("id") or "").strip(), (row.get("name") or "").strip())
        if key in seen_nf:
            continue
        seen_nf.add(key)
        not_found.append({"id": key[0], "name": key[1]})
    not_found.sort(key=sort_key)

    payload = {
        "version": INATS_LIST_VERSION,
        "found": found,
        "not_found": not_found,
    }
    out = data_dir / INATS_LIST_NAME
    write_json(out, payload)
    return out


def species_inats_paths(data_dir: Path) -> list[Path]:
    parents_root = data_dir / PARENTS_DIRNAME
    paths = []
    for path in data_dir.glob("*/*.inats"):
        if parents_root in path.parents:
            continue
        paths.append(path)
    return paths


def synonyms_path_for(inats_path: Path) -> Path:
    return inats_path.with_suffix(".synonyms")


def build_synonyms_record(
    tax_id: str,
    current_name: str,
    scientific_names: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "taxon_id": int(tax_id) if tax_id.isdigit() else tax_id,
        "current_name": current_name,
        "scientific_names": scientific_names,
    }


def harvest_synonyms(
    session: requests.Session,
    data_dir: Path,
    *,
    force_rerun: bool,
    sleep: float,
    watcher: SpaceStopWatcher,
) -> dict[str, int]:
    """Write ``{id}.synonyms`` for each species ``.inats``; return counts.

    The v1 payload also carries ``vision``, so patch it into ``.inats`` here
    and spare `harvest_vision` a second round of requests.
    """
    counts = {"wrote": 0, "skip": 0, "missing": 0, "error": 0}
    pending: list[tuple[str, Path, Path, str]] = []

    for inats_path in species_inats_paths(data_dir):
        tax_id = inats_path.stem
        syn_path = synonyms_path_for(inats_path)
        if not force_rerun and syn_path.is_file():
            counts["skip"] += 1
            continue
        taxon = load_inats(inats_path)
        current_name = ""
        if taxon:
            current_name = (taxon.get("name") or "").strip()
        pending.append((tax_id, inats_path, syn_path, current_name))

    for i in range(0, len(pending), V1_BATCH_SIZE):
        if watcher.requested:
            print("\nClean stop requested; not fetching further synonyms.")
            break
        batch = pending[i : i + V1_BATCH_SIZE]
        ids = [tax_id for tax_id, _, _, _ in batch]
        try:
            taxa = fetch_taxa_by_ids_v1(session, ids, all_names=True)
        except (requests.RequestException, ValueError) as exc:
            print(f"  ERROR synonyms batch {ids[0]}…: {exc}", file=sys.stderr)
            counts["error"] += len(batch)
            continue

        for tax_id, inats_path, syn_path, current_name in batch:
            v1_taxon = taxa.get(tax_id)
            if v1_taxon is not None:
                patch_species_vision(inats_path, v1_taxon)
            sci = (
                scientific_names_from_all_names(v1_taxon.get("names"))
                if v1_taxon is not None
                else None
            )
            if sci is None:
                sci = []
                if not current_name:
                    counts["missing"] += 1
                else:
                    sci = [{"name": current_name, "is_valid": True}]

            if not current_name:
                for entry in sci:
                    if entry.get("is_valid"):
                        current_name = entry["name"]
                        break
                if not current_name and sci:
                    current_name = sci[0]["name"]

            write_json(
                syn_path,
                build_synonyms_record(tax_id, current_name, sci),
            )
            syn_count = sum(1 for n in sci if not n.get("is_valid"))
            print(
                f"OK      synonyms {tax_id}  {current_name}  "
                f"({syn_count} synonyms)"
            )
            counts["wrote"] += 1

        if sleep > 0:
            time.sleep(sleep)

    return counts


def patch_species_vision(inats_path: Path, v1_taxon: dict[str, Any]) -> bool:
    """Copy ``vision`` from a v1 taxon into ``{id}.inats``; True if changed."""
    if "vision" not in v1_taxon:
        return False
    taxon = load_inats(inats_path)
    if taxon is None:
        return False
    vision = bool(v1_taxon.get("vision"))
    if taxon.get("vision") is vision:
        return False
    taxon["vision"] = vision
    write_json(inats_path, taxon)
    return True


def harvest_vision(
    session: requests.Session,
    data_dir: Path,
    *,
    force_rerun: bool,
    sleep: float,
    watcher: SpaceStopWatcher,
) -> dict[str, int]:
    """Fill the ``vision`` flag on species ``.inats`` files from v1.

    A no-op for taxa already carrying the flag (the synonyms pass sets it),
    so this only pays for taxa harvested with synonyms skipped.
    """
    counts = {"wrote": 0, "skip": 0, "missing": 0, "error": 0}
    pending: list[tuple[str, Path]] = []

    for inats_path in species_inats_paths(data_dir):
        taxon = load_inats(inats_path)
        if taxon is None:
            continue
        if not force_rerun and "vision" in taxon:
            counts["skip"] += 1
            continue
        pending.append((inats_path.stem, inats_path))

    if not pending:
        return counts

    print(f"\nFilling iNat vision flag for {len(pending)} species (v1)…")
    for i in range(0, len(pending), V1_BATCH_SIZE):
        if watcher.requested:
            print("\nClean stop requested; not fetching further vision flags.")
            break
        batch = pending[i : i + V1_BATCH_SIZE]
        ids = [tax_id for tax_id, _ in batch]
        try:
            taxa = fetch_taxa_by_ids_v1(session, ids)
        except (requests.RequestException, ValueError) as exc:
            print(f"  ERROR vision batch {ids[0]}…: {exc}", file=sys.stderr)
            counts["error"] += len(batch)
            continue

        for tax_id, inats_path in batch:
            v1_taxon = taxa.get(tax_id)
            if v1_taxon is None or "vision" not in v1_taxon:
                print(f"MISSING vision {tax_id}")
                counts["missing"] += 1
                continue
            patch_species_vision(inats_path, v1_taxon)
            counts["wrote"] += 1
            print(
                f"OK      vision {tax_id}  "
                f"{'yes' if v1_taxon.get('vision') else 'no'}"
            )

        if sleep > 0:
            time.sleep(sleep)

    return counts


def collect_parent_ids(data_dir: Path) -> set[str]:
    parents: set[str] = set()
    parents_root = data_dir / PARENTS_DIRNAME
    for path in data_dir.glob("*/*.inats"):
        if parents_root in path.parents:
            continue
        taxon = load_inats(path)
        if not taxon:
            continue
        parents.update(ancestor_ids_from_taxon(taxon))
    return parents


def harvest_parents(
    session: requests.Session,
    data_dir: Path,
    parent_ids: set[str],
    *,
    force_rerun: bool,
    sleep: float,
    watcher: SpaceStopWatcher,
) -> dict[str, int]:
    parents_dir = data_dir / PARENTS_DIRNAME
    parents_dir.mkdir(parents=True, exist_ok=True)

    counts = {"wrote": 0, "skip": 0, "missing": 0, "error": 0}
    pending: list[str] = []
    for pid in sorted(parent_ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        path = parents_dir / f"{pid}.inats"
        if not force_rerun and path.is_file():
            counts["skip"] += 1
            continue
        pending.append(pid)

    for i in range(0, len(pending), TAXON_BATCH_SIZE):
        if watcher.requested:
            print("\nClean stop requested; not fetching further parents.")
            break
        batch = pending[i : i + TAXON_BATCH_SIZE]
        try:
            found = fetch_taxa_by_ids_v2(session, batch)
        except (requests.RequestException, ValueError) as exc:
            print(f"  ERROR parents batch {batch[0]}…: {exc}", file=sys.stderr)
            counts["error"] += len(batch)
            continue

        for pid in batch:
            taxon = found.get(pid)
            if taxon is None:
                print(f"MISSING parent {pid}")
                counts["missing"] += 1
                continue
            write_json(parents_dir / f"{pid}.inats", taxon)
            print(f"OK      parent {pid}  {taxon.get('name') or ''}".rstrip())
            counts["wrote"] += 1

        if sleep > 0:
            time.sleep(sleep)

    return counts


def main() -> int:
    args = parse_args()
    settings, load_names_csv = bootstrap_django()
    data_dir = Path(settings.MOTHS_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    names = load_names_csv()
    if not names:
        print("Names CSV is empty or unreadable.", file=sys.stderr)
        return 1

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )

    def sort_key(tax_id: str):
        return (0, int(tax_id)) if tax_id.isdigit() else (1, tax_id)

    tax_ids = sorted(names.keys(), key=sort_key)

    watcher = SpaceStopWatcher()
    watcher.start()
    if watcher.enabled:
        print("Press SPACE for a clean stop between requests.")

    species_counts = {"wrote": 0, "missing": 0, "skip": 0, "error": 0}
    synonym_counts = {"wrote": 0, "skip": 0, "missing": 0, "error": 0}
    vision_counts = {"wrote": 0, "skip": 0, "missing": 0, "error": 0}
    parent_counts = {"wrote": 0, "skip": 0, "missing": 0, "error": 0}
    harvested_taxa: list[dict[str, Any]] = []
    found_rows: list[dict[str, str]] = []
    not_found_rows: list[dict[str, str]] = []

    try:
        (
            species_counts,
            harvested_taxa,
            found_rows,
            not_found_rows,
        ) = harvest_species_batched(
            session,
            data_dir,
            names,
            tax_ids,
            force_rerun=args.force_rerun,
            sleep=args.sleep,
            limit=args.limit,
            watcher=watcher,
        )

        if not args.skip_synonyms and not watcher.requested:
            print("\nHarvesting scientific synonyms (v1 all_names)…")
            synonym_counts = harvest_synonyms(
                session,
                data_dir,
                force_rerun=args.force_rerun,
                sleep=args.sleep,
                watcher=watcher,
            )

        if not watcher.requested:
            vision_counts = harvest_vision(
                session,
                data_dir,
                force_rerun=args.force_rerun,
                sleep=args.sleep,
                watcher=watcher,
            )

        if not args.skip_parents and not watcher.requested:
            parent_ids = collect_parent_ids(data_dir)
            for taxon in harvested_taxa:
                parent_ids.update(ancestor_ids_from_taxon(taxon))
            print(f"\nParents to consider: {len(parent_ids)}")
            parent_counts = harvest_parents(
                session,
                data_dir,
                parent_ids,
                force_rerun=args.force_rerun,
                sleep=args.sleep,
                watcher=watcher,
            )
    finally:
        watcher.stop()

    list_path = write_inats_list(data_dir, found_rows, not_found_rows)
    list_payload = load_json(list_path) or {}
    print(
        f"\nWrote {list_path}: "
        f"found={len(list_payload.get('found') or [])} "
        f"not_found={len(list_payload.get('not_found') or [])}"
    )

    print(
        f"\nSpecies: wrote={species_counts['wrote']} "
        f"missing={species_counts['missing']} skip={species_counts['skip']} "
        f"error={species_counts['error']}"
    )
    if not args.skip_synonyms:
        print(
            f"Synonyms: wrote={synonym_counts['wrote']} "
            f"missing={synonym_counts['missing']} "
            f"skip={synonym_counts['skip']} "
            f"error={synonym_counts['error']}"
        )
    print(
        f"Vision: wrote={vision_counts['wrote']} "
        f"missing={vision_counts['missing']} skip={vision_counts['skip']} "
        f"error={vision_counts['error']}"
    )
    if not args.skip_parents:
        print(
            f"Parents: wrote={parent_counts['wrote']} "
            f"missing={parent_counts['missing']} skip={parent_counts['skip']} "
            f"error={parent_counts['error']}"
        )
    print(f"→ {data_dir}")

    errors = (
        species_counts["error"]
        + (0 if args.skip_synonyms else synonym_counts["error"])
        + vision_counts["error"]
        + (0 if args.skip_parents else parent_counts["error"])
    )
    wrote = (
        species_counts["wrote"]
        + (0 if args.skip_synonyms else synonym_counts["wrote"])
        + vision_counts["wrote"]
        + (0 if args.skip_parents else parent_counts["wrote"])
    )
    return 1 if errors and not wrote else 0


if __name__ == "__main__":
    raise SystemExit(main())
