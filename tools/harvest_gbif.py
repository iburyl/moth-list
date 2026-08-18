#!/usr/bin/env python3

"""Harvest GBIF Catalogue of Life XR + occurrence summary for iNat species.

Reads ``inats_summary.json`` (found taxa only) and, for each tax_id, matches
a GBIF taxon by scientific name — first the current iNat ``name``, then each
synonym from ``inats_synonyms_summary.json``. Stops at the first usable match
(species-rank EXACT/FUZZY).

Writes::

    <MOTHS_DATA_DIR>/<tax_id>/<tax_id>.gbif
    <MOTHS_DATA_DIR>/gbif_list.json
    <MOTHS_DATA_DIR>/gbif_summary.json

Each ``.gbif`` record stores:

* ``usageKey`` — current Catalogue of Life XR identifier
* ``url`` — current GBIF ``/taxon/<usageKey>`` page
* ``accepted`` — whether the matched name is the accepted taxonomy name
* ``observations_count`` — GBIF PRESENT occurrence count
* ``last_observation_date`` — latest ``YYYY-MM-DD`` with records (or null)
* ``match`` — complete v2 match response for future parsing

``gbif_list.json`` lists ``found`` ``{id, queried, usageKey, url}`` and
``not_found`` ``{id, name}``. ``gbif_summary.json`` indexes full records by
iNat tax id.

Misses write a ``.gbif`` stub with ``found: false`` so later runs skip them
(tax folders usually already hold ``.inats`` / ``.wiki`` files). Pass
``--force-rerun`` to re-fetch. Press SPACE between taxa for a clean stop.
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

REPO_ROOT = Path(__file__).resolve().parent.parent

_CONTACT = (
    os.environ.get("MOTHS_GBIF_CONTACT")
    or os.environ.get("MOTHS_INATS_CONTACT")
    or os.environ.get("MOTHS_WIKI_CONTACT")
    or os.environ.get("WIKI_CONTACT")
    or "https://github.com/iburylov/moth-list"
)
USER_AGENT = (
    f"moth-list-harvest-gbif/1.0 "
    f"(moth taxonomy research dataset; contact: {_CONTACT}) "
    f"python-requests/{requests.__version__}"
)

API_BASE_V1 = "https://api.gbif.org/v1"
API_BASE_V2 = "https://api.gbif.org/v2"
COL_XR_CHECKLIST_KEY = "7ddf754f-d193-4cc9-b351-99906754a03b"
MAX_RETRY_AFTER_SECONDS = 300.0
MAX_RATE_LIMIT_RETRIES = 8

INATS_SUMMARY_NAME = "inats_summary.json"
INATS_SYNONYMS_SUMMARY_NAME = "inats_synonyms_summary.json"
GBIF_LIST_NAME = "gbif_list.json"
GBIF_SUMMARY_NAME = "gbif_summary.json"
GBIF_RECORD_VERSION = 2
GBIF_LIST_VERSION = 2
GBIF_SUMMARY_VERSION = 2

# Species-rank matches only; HIGHERRANK / NONE are treated as misses.
_OK_MATCH_TYPES = frozenset({"EXACT", "FUZZY"})
_OK_RANKS = frozenset({"SPECIES", "SUBSPECIES", "VARIETY", "FORM"})


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
            import msvcrt  # noqa: F401

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


def bootstrap_django():
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()
    from django.conf import settings

    return settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GBIF taxon + occurrence summary for found taxa in "
            "inats_summary.json (try current name, then synonyms)."
        )
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-fetch even when <tax_id>.gbif already exists (hit or miss).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Seconds to wait between taxa (default: 0.5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after this many newly processed taxa (0 = no limit).",
    )
    return parser.parse_args()


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
    params: dict[str, Any] | None = None,
) -> requests.Response:
    last_response: requests.Response | None = None
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        response = session.get(url, params=params or {}, timeout=60)
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


def _load_json(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def gbif_taxon_url(usage_key: str) -> str:
    return f"https://www.gbif.org/taxon/{usage_key}"


def match_species(
    session: requests.Session,
    scientific_name: str,
) -> dict[str, Any] | None:
    """Return a usable Catalogue of Life XR species-rank match."""
    name = (scientific_name or "").strip()
    if not name:
        return None
    response = _get_with_rate_limit(
        session,
        f"{API_BASE_V2}/species/match",
        {
            "scientificName": name,
            "kingdom": "Animalia",
            "checklistKey": COL_XR_CHECKLIST_KEY,
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    diagnostics = payload.get("diagnostics")
    if not isinstance(usage, dict) or not isinstance(diagnostics, dict):
        return None
    if (diagnostics.get("matchType") or "").upper() not in _OK_MATCH_TYPES:
        return None
    if (usage.get("rank") or "").upper() not in _OK_RANKS:
        return None
    if not str(usage.get("key") or "").strip():
        return None
    return payload


def occurrence_search(
    session: requests.Session,
    *,
    taxon_key: int | str,
    year: int | None = None,
    month: int | None = None,
    facet: str | None = None,
    facet_limit: int = 300,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "taxonKey": taxon_key,
        "checklistKey": COL_XR_CHECKLIST_KEY,
        "occurrenceStatus": "PRESENT",
        "limit": 0,
    }
    if year is not None:
        params["year"] = year
    if month is not None:
        params["month"] = month
    if facet:
        params["facet"] = facet
        params["facetLimit"] = facet_limit
    response = _get_with_rate_limit(
        session, f"{API_BASE_V1}/occurrence/search", params
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _facet_int_names(payload: dict[str, Any], field: str) -> list[int]:
    want = field.upper()
    for facet in payload.get("facets") or []:
        if not isinstance(facet, dict):
            continue
        if (facet.get("field") or "").upper() != want:
            continue
        out: list[int] = []
        for row in facet.get("counts") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if name.isdigit():
                out.append(int(name))
        return out
    return []


def last_observation_date(
    session: requests.Session,
    taxon_key: int | str,
    *,
    years_payload: dict[str, Any] | None = None,
) -> str | None:
    """Latest calendar day with PRESENT occurrences, as ``YYYY-MM-DD``."""
    years_payload = years_payload or occurrence_search(
        session, taxon_key=taxon_key, facet="year", facet_limit=400
    )
    years = _facet_int_names(years_payload, "year")
    if not years:
        return None
    max_year = max(years)

    months_payload = occurrence_search(
        session,
        taxon_key=taxon_key,
        year=max_year,
        facet="month",
        facet_limit=12,
    )
    months = _facet_int_names(months_payload, "month")
    if not months:
        return f"{max_year:04d}"
    max_month = max(months)

    days_payload = occurrence_search(
        session,
        taxon_key=taxon_key,
        year=max_year,
        month=max_month,
        facet="day",
        facet_limit=31,
    )
    days = _facet_int_names(days_payload, "day")
    if not days:
        return f"{max_year:04d}-{max_month:02d}"
    max_day = max(days)
    return f"{max_year:04d}-{max_month:02d}-{max_day:02d}"


def build_record(
    session: requests.Session,
    *,
    tax_id: str,
    queried: str,
    match: dict[str, Any],
) -> dict[str, Any]:
    usage = match.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("GBIF match has no usage")
    accepted_usage = match.get("acceptedUsage")
    if not isinstance(accepted_usage, dict):
        accepted_usage = None
    diagnostics = match.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}

    matched_key = str(usage.get("key") or "").strip()
    if not matched_key:
        raise ValueError("GBIF match usage has no key")
    accepted = not bool(match.get("synonym")) and (
        (usage.get("status") or "").strip().upper() == "ACCEPTED"
    )
    concept = accepted_usage or usage
    concept_key = str(concept.get("key") or matched_key).strip()
    status = (usage.get("status") or "").strip().upper() or None

    years_payload = occurrence_search(
        session, taxon_key=concept_key, facet="year", facet_limit=400
    )
    count = int(years_payload.get("count") or 0)
    last_date = None
    if count > 0:
        last_date = last_observation_date(
            session, concept_key, years_payload=years_payload
        )

    return {
        "version": GBIF_RECORD_VERSION,
        "id": str(tax_id),
        "queried": queried,
        # The public id and URL always identify the accepted CoL XR concept.
        "usageKey": concept_key,
        "matchedUsageKey": matched_key,
        "scientificName": usage.get("name") or None,
        "canonicalName": usage.get("canonicalName") or None,
        "acceptedScientificName": concept.get("name") or None,
        "acceptedCanonicalName": concept.get("canonicalName") or None,
        "authorship": usage.get("authorship") or None,
        "acceptedAuthorship": concept.get("authorship") or None,
        "rank": usage.get("rank") or None,
        "code": usage.get("code") or None,
        "nameType": usage.get("type") or None,
        "status": status,
        "matchType": diagnostics.get("matchType") or None,
        "confidence": diagnostics.get("confidence"),
        "accepted": accepted,
        "url": gbif_taxon_url(concept_key),
        "observations_count": count,
        "last_observation_date": last_date,
        "classification": (
            match.get("classification")
            if isinstance(match.get("classification"), list)
            else []
        ),
        # Preserve the complete upstream response in the raw harvest. The
        # downstream data_summary intentionally selects only display fields.
        "match": match,
    }


def load_found_taxa(data_dir: Path) -> list[tuple[str, str, list[str]]]:
    """Return ``[(tax_id, current_name, synonym_names), ...]`` for found taxa."""
    summary = _load_json(data_dir / INATS_SUMMARY_NAME)
    if not summary:
        return []
    by_tax = summary.get("by_tax_id")
    if not isinstance(by_tax, dict):
        return []

    syn_summary = _load_json(data_dir / INATS_SYNONYMS_SUMMARY_NAME) or {}
    syn_by_tax = syn_summary.get("by_tax_id")
    if not isinstance(syn_by_tax, dict):
        syn_by_tax = {}

    rows: list[tuple[str, str, list[str]]] = []
    for tax_id, entry in by_tax.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("found") is False:
            continue
        current = (entry.get("name") or "").strip()
        syn_entry = syn_by_tax.get(str(tax_id))
        synonyms: list[str] = []
        if isinstance(syn_entry, dict):
            if not current:
                current = (syn_entry.get("current_name") or "").strip()
            for name in syn_entry.get("synonyms") or []:
                name = (name or "").strip()
                if name and name.casefold() != current.casefold():
                    synonyms.append(name)
        rows.append((str(tax_id), current, synonyms))

    def sort_key(item: tuple[str, str, list[str]]):
        tid = item[0]
        return (0, int(tid)) if tid.isdigit() else (1, tid)

    rows.sort(key=sort_key)
    return rows


def name_candidates(current_name: str, synonyms: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in [current_name, *synonyms]:
        name = (name or "").strip()
        if not name:
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def miss_record(tax_id: str, name: str) -> dict[str, Any]:
    return {
        "version": GBIF_RECORD_VERSION,
        "id": str(tax_id),
        "found": False,
        "name": (name or "").strip(),
        "queried": None,
        "usageKey": None,
        "matchedUsageKey": None,
        "scientificName": None,
        "canonicalName": None,
        "acceptedScientificName": None,
        "acceptedCanonicalName": None,
        "authorship": None,
        "acceptedAuthorship": None,
        "rank": None,
        "code": None,
        "nameType": None,
        "status": None,
        "matchType": None,
        "confidence": None,
        "accepted": False,
        "url": None,
        "observations_count": 0,
        "last_observation_date": None,
        "classification": [],
        "match": None,
    }


def harvest_one(
    session: requests.Session,
    data_dir: Path,
    tax_id: str,
    names: list[str],
    *,
    force_rerun: bool,
    label: str,
) -> tuple[str, dict[str, Any] | None]:
    """Fetch one taxon. Returns ``(status, record_or_none)``."""
    tax_dir = data_dir / tax_id
    gbif_path = tax_dir / f"{tax_id}.gbif"

    if not force_rerun and gbif_path.is_file():
        existing = _load_json(gbif_path)
        if (
            isinstance(existing, dict)
            and existing.get("version") == GBIF_RECORD_VERSION
        ):
            return "skip", existing

    if not names:
        record = miss_record(tax_id, label)
        tax_dir.mkdir(parents=True, exist_ok=True)
        _write_json(gbif_path, record)
        return "missing", record

    last_error: Exception | None = None
    errors = 0
    for queried in names:
        try:
            match = match_species(session, queried)
            if match is None:
                continue
            record = build_record(
                session, tax_id=tax_id, queried=queried, match=match
            )
            record["found"] = True
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            last_error = exc
            errors += 1
            continue

        tax_dir.mkdir(parents=True, exist_ok=True)
        _write_json(gbif_path, record)
        return "wrote", record

    if names and errors == len(names):
        print(f"  ERROR {tax_id}: {last_error}", file=sys.stderr)
        return "error", None

    record = miss_record(tax_id, label)
    tax_dir.mkdir(parents=True, exist_ok=True)
    _write_json(gbif_path, record)
    return "missing", record


def load_existing_records(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Load on-disk hit ``.gbif`` files keyed by iNat tax id."""
    by_id: dict[str, dict[str, Any]] = {}
    for path in data_dir.glob("*/*.gbif"):
        if path.parent.name == "parents":
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        if data.get("version") != GBIF_RECORD_VERSION:
            continue
        if data.get("found") is False:
            continue
        if data.get("usageKey") is None:
            continue
        tid = str(data.get("id") or path.stem).strip()
        if tid:
            by_id[tid] = data
    return by_id


def write_gbif_list(
    data_dir: Path,
    found_records: dict[str, dict[str, Any]],
    not_found_rows: list[dict[str, str]],
) -> Path:
    def sort_key(row: dict[str, Any]):
        tid = str(row.get("id") or "")
        return (0, int(tid)) if tid.isdigit() else (1, tid)

    found = []
    for tid, record in found_records.items():
        found.append(
            {
                "id": str(tid),
                "queried": (record.get("queried") or "").strip(),
                "usageKey": record.get("usageKey"),
                "url": (record.get("url") or "").strip(),
            }
        )
    found.sort(key=sort_key)

    seen_nf: set[tuple[str, str]] = set()
    not_found: list[dict[str, str]] = []
    for row in not_found_rows:
        key = ((row.get("id") or "").strip(), (row.get("name") or "").strip())
        if key in seen_nf:
            continue
        seen_nf.add(key)
        not_found.append({"id": key[0], "name": key[1]})
    not_found.sort(key=sort_key)

    out = data_dir / GBIF_LIST_NAME
    _write_json(
        out,
        {
            "version": GBIF_LIST_VERSION,
            "found": found,
            "not_found": not_found,
        },
    )
    return out


def write_gbif_summary(
    data_dir: Path,
    found_records: dict[str, dict[str, Any]],
    not_found_rows: list[dict[str, str]],
) -> Path:
    def sort_key(tid: str):
        return (0, int(tid)) if tid.isdigit() else (1, tid)

    by_tax_id = {
        tid: found_records[tid]
        for tid in sorted(found_records.keys(), key=sort_key)
    }
    seen_nf: set[tuple[str, str]] = set()
    not_found: list[dict[str, str]] = []
    for row in not_found_rows:
        key = ((row.get("id") or "").strip(), (row.get("name") or "").strip())
        if key in seen_nf:
            continue
        seen_nf.add(key)
        not_found.append({"id": key[0], "name": key[1]})
    not_found.sort(
        key=lambda r: (
            (0, int(r["id"])) if r["id"].isdigit() else (1, r["id"]),
            r["name"],
        )
    )
    out = data_dir / GBIF_SUMMARY_NAME
    _write_json(
        out,
        {
            "version": GBIF_SUMMARY_VERSION,
            "by_tax_id": by_tax_id,
            "not_found": not_found,
        },
    )
    return out


def main() -> int:
    args = parse_args()
    settings = bootstrap_django()
    data_dir = Path(settings.MOTHS_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    taxa = load_found_taxa(data_dir)
    if not taxa:
        print(
            f"No found taxa in {data_dir / INATS_SUMMARY_NAME}. "
            "Run prepare_inats.py first.",
            file=sys.stderr,
        )
        return 1

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        }
    )

    found_records = load_existing_records(data_dir)
    not_found_rows: list[dict[str, str]] = []
    # Seed not_found from prior list so skips of empty miss folders stay listed.
    prior_list = _load_json(data_dir / GBIF_LIST_NAME) or {}
    for row in prior_list.get("not_found") or []:
        if isinstance(row, dict):
            not_found_rows.append(
                {
                    "id": str(row.get("id") or "").strip(),
                    "name": str(row.get("name") or "").strip(),
                }
            )

    watcher = SpaceStopWatcher()
    watcher.start()
    if watcher.enabled:
        print("Press SPACE for a clean stop between taxa.")

    counts = {"wrote": 0, "missing": 0, "skip": 0, "error": 0}
    processed = 0

    try:
        for tax_id, current_name, synonyms in taxa:
            if watcher.requested:
                print("\nClean stop requested; not starting further taxa.")
                break

            names = name_candidates(current_name, synonyms)
            label = current_name or "(no name)"
            status, record = harvest_one(
                session,
                data_dir,
                tax_id,
                names,
                force_rerun=args.force_rerun,
                label=label,
            )
            counts[status] = counts.get(status, 0) + 1

            if status == "skip":
                if (
                    record is not None
                    and record.get("found") is not False
                    and record.get("usageKey") is not None
                ):
                    found_records[tax_id] = record
                    not_found_rows = [
                        r for r in not_found_rows if r.get("id") != tax_id
                    ]
                elif record is not None and record.get("found") is False:
                    found_records.pop(tax_id, None)
                    not_found_rows.append(
                        {
                            "id": tax_id,
                            "name": (record.get("name") or label).strip(),
                        }
                    )
            elif status == "wrote" and record is not None:
                found_records[tax_id] = record
                not_found_rows = [
                    r for r in not_found_rows if r.get("id") != tax_id
                ]
                via = ""
                queried = record.get("queried") or ""
                if queried.casefold() != (current_name or "").casefold():
                    via = f" via {queried!r}"
                accepted = "accepted" if record.get("accepted") else "synonym"
                print(
                    f"OK      {tax_id}  {label}{via}  "
                    f"key={record.get('usageKey')}  {accepted}  "
                    f"obs={record.get('observations_count')}  "
                    f"last={record.get('last_observation_date') or '-'}"
                )
                processed += 1
            elif status == "missing":
                found_records.pop(tax_id, None)
                not_found_rows.append({"id": tax_id, "name": label})
                print(
                    f"MISSING {tax_id}  {label}  "
                    f"(tried {len(names)} names)"
                )
                processed += 1
            else:
                processed += 1

            if status != "skip" and args.sleep > 0:
                time.sleep(args.sleep)

            if args.limit and processed >= args.limit:
                print(f"\nReached --limit {args.limit}.")
                break
    finally:
        watcher.stop()

    list_path = write_gbif_list(data_dir, found_records, not_found_rows)
    summary_path = write_gbif_summary(data_dir, found_records, not_found_rows)
    print(
        f"\nWrote {list_path}: found={len(found_records)} "
        f"not_found={len({(r.get('id'), r.get('name')) for r in not_found_rows})}"
    )
    print(f"Wrote {summary_path}")
    print(
        f"Done. wrote={counts['wrote']} missing={counts['missing']} "
        f"skip={counts['skip']} error={counts['error']} "
        f"→ {data_dir}"
    )
    return 1 if counts["error"] and not counts["wrote"] and not counts["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
