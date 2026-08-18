#!/usr/bin/env python3

"""Harvest Wikipedia article source (wikitext) for found iNat species.

Reads ``inats_summary.json`` (found taxa only) and, for each tax_id, looks up
a Wikipedia page by scientific name — first the current iNat ``name``, then
each synonym from ``inats_synonyms_summary.json``. Ignores any
``wikipedia_url`` on the iNat record. Stops at the first hit.

Writes::

    <MOTHS_DATA_DIR>/<tax_id>/<tax_id>.wiki
    <MOTHS_DATA_DIR>/wiki_list.json

``wiki_list.json`` records each successful fetch as
``{id, queried, title, url}`` — iNat tax id, the name that hit, the
canonical page title after redirects, and the article URL.

If no page is found, creates an empty ``<tax_id>/`` folder (miss marker) and
leaves no ``.wiki`` file. Re-runs skip existing hits/misses unless ``--force-rerun``.

Press SPACE between taxa for a clean stop.
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
from urllib.parse import quote

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

_WIKI_CONTACT = (
    os.environ.get("MOTHS_WIKI_CONTACT")
    or os.environ.get("WIKI_CONTACT")
    or "https://github.com/iburylov/moth-list"
)
USER_AGENT = (
    f"moth-list-harvest-wiki/1.0 "
    f"(moth taxonomy research dataset; contact: {_WIKI_CONTACT}) "
    f"python-requests/{requests.__version__}"
)
API_PATH = "/w/api.php"
MAX_RETRY_AFTER_SECONDS = 300.0
MAX_RATE_LIMIT_RETRIES = 8

INATS_SUMMARY_NAME = "inats_summary.json"
INATS_SYNONYMS_SUMMARY_NAME = "inats_synonyms_summary.json"
WIKI_LIST_NAME = "wiki_list.json"
WIKI_LIST_VERSION = 1


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
            "Fetch Wikipedia wikitext for found taxa in inats_summary.json "
            "(try current name, then synonyms)."
        )
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Wikipedia language subdomain (default: en).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Re-fetch even when <tax_id>.wiki already exists or the empty "
            "folder marker is present."
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
        help="Stop after this many newly processed taxa (0 = no limit).",
    )
    return parser.parse_args()


def wiki_api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org{API_PATH}"


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
    api_url: str,
    params: dict,
) -> requests.Response:
    last_response: requests.Response | None = None
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        response = session.get(api_url, params=params, timeout=60)
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


def wikipedia_article_url(lang: str, title: str) -> str:
    """Build a Wikipedia article URL for ``title`` on ``lang`` wiki."""
    return f"https://{lang}.wikipedia.org/wiki/" + quote(
        title.replace(" ", "_"),
        safe="",
    )


def fetch_wikitext(
    session: requests.Session,
    api_url: str,
    title: str,
) -> tuple[str, str] | None:
    """Return ``(wikitext, resolved_title)``, or ``None`` if the page is missing.

    With ``redirects=1``, ``resolved_title`` is the canonical page title after
    any redirect from the queried ``title``.
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": title,
    }
    response = _get_with_rate_limit(session, api_url, params)
    response.raise_for_status()
    payload = response.json()

    pages = (payload.get("query") or {}).get("pages") or []
    if not pages:
        return None
    page = pages[0]
    if page.get("missing") or page.get("invalid"):
        return None
    revisions = page.get("revisions") or []
    if not revisions:
        return None
    slots = revisions[0].get("slots") or {}
    main = slots.get("main") or {}
    content = main.get("content")
    if content is None:
        content = revisions[0].get("content")
    if content is None:
        return None
    resolved = (page.get("title") or title or "").strip()
    return content, resolved or title


def already_done(tax_dir: Path, wiki_path: Path) -> bool:
    if wiki_path.is_file():
        return True
    return tax_dir.is_dir() and not any(tax_dir.iterdir())


def _load_json(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_wiki_list_by_id(data_dir: Path) -> dict[str, dict[str, str]]:
    """Load existing ``wiki_list.json`` found rows keyed by iNat tax id."""
    payload = _load_json(data_dir / WIKI_LIST_NAME) or {}
    found = payload.get("found")
    if not isinstance(found, list):
        return {}
    by_id: dict[str, dict[str, str]] = {}
    for row in found:
        if not isinstance(row, dict):
            continue
        tid = str(row.get("id") or "").strip()
        if not tid:
            continue
        by_id[tid] = {
            "id": tid,
            "queried": str(row.get("queried") or "").strip(),
            "title": str(row.get("title") or "").strip(),
            "url": str(row.get("url") or "").strip(),
        }
    return by_id


def write_wiki_list(data_dir: Path, by_id: dict[str, dict[str, str]]) -> Path:
    """Write ``wiki_list.json`` with found ``{id, queried, title, url}`` rows."""

    def sort_key(row: dict[str, str]):
        tid = row.get("id") or ""
        return (0, int(tid)) if tid.isdigit() else (1, tid)

    found = sorted(by_id.values(), key=sort_key)
    out = data_dir / WIKI_LIST_NAME
    _write_json(
        out,
        {
            "version": WIKI_LIST_VERSION,
            "found": found,
        },
    )
    return out


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


def title_candidates(current_name: str, synonyms: list[str]) -> list[str]:
    """Ordered unique titles: current name first, then synonyms."""
    out: list[str] = []
    seen: set[str] = set()
    for title in [current_name, *synonyms]:
        title = (title or "").strip()
        if not title:
            continue
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(title)
    return out


def harvest_one(
    session: requests.Session,
    api_url: str,
    lang: str,
    data_dir: Path,
    tax_id: str,
    titles: list[str],
    *,
    force_rerun: bool,
) -> tuple[str, dict[str, str] | None]:
    """Fetch one taxon. Returns ``(status, list_row_or_none)``."""
    tax_dir = data_dir / tax_id
    wiki_path = tax_dir / f"{tax_id}.wiki"

    if not force_rerun and already_done(tax_dir, wiki_path):
        return "skip", None

    if not titles:
        tax_dir.mkdir(parents=True, exist_ok=True)
        if wiki_path.exists():
            wiki_path.unlink()
        return "missing", None

    last_error: Exception | None = None
    errors = 0
    for queried in titles:
        try:
            fetched = fetch_wikitext(session, api_url, queried)
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            errors += 1
            continue
        if fetched is None:
            continue
        wikitext, resolved_title = fetched
        tax_dir.mkdir(parents=True, exist_ok=True)
        wiki_path.write_text(wikitext, encoding="utf-8")
        row = {
            "id": str(tax_id),
            "queried": queried,
            "title": resolved_title,
            "url": wikipedia_article_url(lang, resolved_title),
        }
        return "wrote", row

    tax_dir.mkdir(parents=True, exist_ok=True)
    if wiki_path.exists():
        wiki_path.unlink()
    if titles and errors == len(titles):
        print(f"  ERROR {tax_id}: {last_error}", file=sys.stderr)
        return "error", None
    return "missing", None


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

    api_url = wiki_api_url(args.lang)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    wiki_by_id = load_wiki_list_by_id(data_dir)

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

            titles = title_candidates(current_name, synonyms)
            label = current_name or "(no name)"
            status, row = harvest_one(
                session,
                api_url,
                args.lang,
                data_dir,
                tax_id,
                titles,
                force_rerun=args.force_rerun,
            )
            counts[status] = counts.get(status, 0) + 1

            if status == "skip":
                pass
            elif status == "wrote" and row is not None:
                wiki_by_id[tax_id] = row
                queried = row["queried"]
                title = row["title"]
                via = ""
                if queried.casefold() != (current_name or "").casefold():
                    via = f" via queried {queried!r}"
                if title.casefold() != queried.casefold():
                    via += f" → {title!r}"
                print(f"OK      {tax_id}  {label}{via}")
                processed += 1
            elif status == "missing":
                wiki_by_id.pop(tax_id, None)
                print(f"MISSING {tax_id}  {label}  (tried {len(titles)} titles)")
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

    list_path = write_wiki_list(data_dir, wiki_by_id)
    print(
        f"\nWrote {list_path}: found={len(wiki_by_id)}"
    )
    print(
        f"Done. wrote={counts['wrote']} missing={counts['missing']} "
        f"skip={counts['skip']} error={counts['error']} "
        f"→ {data_dir}"
    )
    return 1 if counts["error"] and not counts["wrote"] and not counts["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())