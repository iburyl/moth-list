#!/usr/bin/env python3

"""Harvest Wikipedia article source (wikitext) for every species in the names CSV.

Reads the taxonomy names CSV configured via Django (``TAX_CSV`` /
``MOTHS_NAMES_CSV``) and, for each ``tax_id``, looks up the Wikipedia page
whose title is the CSV ``species`` value (scientific name). The raw wikitext
is written to::

    <MOTHS_DATA_DIR>/<tax_id>/<tax_id>.wiki

If no matching page exists (or the species cell is empty), the script still
creates an empty ``<tax_id>/`` folder and leaves no ``.wiki`` file — that empty
folder is the durable "not found" marker.

Directories come from Django, exactly like the other tools in this folder: the
environment must set every ``MOTHS_*`` path (including the new
``MOTHS_DATA_DIR``). Re-runs skip tax_ids that already have a ``.wiki`` file, or
an empty folder (previously missing), unless ``--force`` is passed.

Press SPACE between taxa for a clean stop (current request finishes; no later
taxon is started).
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import requests

# --- Repo / Django bootstrap -------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Wikimedia requires a descriptive User-Agent that identifies the client and a
# way to contact the operator
# (https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy).
# Override the contact with MOTHS_WIKI_CONTACT (email or URL).
_WIKI_CONTACT = (
    os.environ.get("MOTHS_WIKI_CONTACT")
    or os.environ.get("WIKI_CONTACT")
    or "https://github.com/iburyl/moth-list"
)
USER_AGENT = (
    f"moth-list-harvest-wiki/1.0 "
    f"(moth taxonomy research dataset; contact: {_WIKI_CONTACT}) "
    f"python-requests/{requests.__version__}"
)
API_PATH = "/w/api.php"
# Cap how long we wait on a single 429, and how many times we retry one title.
MAX_RETRY_AFTER_SECONDS = 300.0
MAX_RATE_LIMIT_RETRIES = 8


# --- Clean-stop watcher (same idea as harvest_top_images.py) -----------------


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


# --- Django -----------------------------------------------------------------


def bootstrap_django():
    """Set up Django and return ``(settings, load_names)``."""
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from django.conf import settings
    from moths.utils.names import load_names

    return settings, load_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Wikipedia wikitext for every species in the names CSV and "
            "store it under MOTHS_DATA_DIR/<tax_id>/<tax_id>.wiki."
        )
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Wikipedia language subdomain (default: en).",
    )
    parser.add_argument(
        "--force",
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


# --- Wikipedia --------------------------------------------------------------


def wiki_api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org{API_PATH}"


def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
    """Seconds to sleep after a 429, preferring the Retry-After header."""
    raw = (response.headers.get("Retry-After") or "").strip()
    if raw:
        try:
            # Integer / float seconds (the common Wikimedia form).
            return min(max(float(raw), 0.0), MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
        try:
            # HTTP-date form.
            from email.utils import parsedate_to_datetime

            when = parsedate_to_datetime(raw)
            delay = when.timestamp() - time.time()
            return min(max(delay, 0.0), MAX_RETRY_AFTER_SECONDS)
        except (TypeError, ValueError, OSError, OverflowError):
            pass
    # No usable header: exponential backoff, capped.
    return min(2.0**attempt, MAX_RETRY_AFTER_SECONDS)


def _get_with_rate_limit(
    session: requests.Session,
    api_url: str,
    params: dict,
) -> requests.Response:
    """GET that honors 429 Retry-After before giving up."""
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


def fetch_wikitext(
    session: requests.Session,
    api_url: str,
    title: str,
) -> str | None:
    """Return raw wikitext for ``title``, or ``None`` if the page is missing.

    Follows redirects so a species synonym title still yields the target
    article source. Raises ``requests.HTTPError`` on transport/HTTP failures
    (including a 429 that still fails after retries).
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
        # Older API shape without slots.
        content = revisions[0].get("content")
    if content is None:
        return None
    return content


def already_done(tax_dir: Path, wiki_path: Path) -> bool:
    """True when a previous run already recorded a hit or a miss for this tax."""
    if wiki_path.is_file():
        return True
    # Empty folder = previous miss marker.
    return tax_dir.is_dir() and not any(tax_dir.iterdir())


def harvest_one(
    session: requests.Session,
    api_url: str,
    data_dir: Path,
    tax_id: str,
    species: str,
    *,
    force: bool,
) -> str:
    """Fetch one taxon. Returns a short status word: wrote / missing / skip / error."""
    tax_dir = data_dir / tax_id
    wiki_path = tax_dir / f"{tax_id}.wiki"

    if not force and already_done(tax_dir, wiki_path):
        return "skip"

    if not species:
        tax_dir.mkdir(parents=True, exist_ok=True)
        if wiki_path.exists():
            wiki_path.unlink()
        return "missing"

    try:
        wikitext = fetch_wikitext(session, api_url, species)
    except (requests.RequestException, ValueError) as exc:
        print(f"  ERROR {tax_id} ({species}): {exc}", file=sys.stderr)
        return "error"

    tax_dir.mkdir(parents=True, exist_ok=True)
    if wikitext is None:
        if wiki_path.exists():
            wiki_path.unlink()
        return "missing"

    wiki_path.write_text(wikitext, encoding="utf-8")
    return "wrote"


def main() -> int:
    args = parse_args()
    settings, load_names = bootstrap_django()
    data_dir = Path(settings.MOTHS_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    names = load_names()
    if not names:
        print("Names CSV is empty or unreadable.", file=sys.stderr)
        return 1

    api_url = wiki_api_url(args.lang)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    # Stable order: numeric tax_id when possible, else lexical.
    def sort_key(tax_id: str):
        return (0, int(tax_id)) if tax_id.isdigit() else (1, tax_id)

    tax_ids = sorted(names.keys(), key=sort_key)

    watcher = SpaceStopWatcher()
    watcher.start()
    if watcher.enabled:
        print("Press SPACE for a clean stop between taxa.")

    counts = {"wrote": 0, "missing": 0, "skip": 0, "error": 0}
    processed = 0

    try:
        for tax_id in tax_ids:
            if watcher.requested:
                print("\nClean stop requested; not starting further taxa.")
                break

            info = names[tax_id]
            species = (info.get("species") or "").strip()
            label = species or "(empty species)"
            status = harvest_one(
                session,
                api_url,
                data_dir,
                tax_id,
                species,
                force=args.force,
            )
            counts[status] = counts.get(status, 0) + 1

            if status == "skip":
                # Quiet on the common re-run path.
                pass
            elif status == "wrote":
                print(f"OK      {tax_id}  {label}")
                processed += 1
            elif status == "missing":
                print(f"MISSING {tax_id}  {label}")
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

    print(
        f"\nDone. wrote={counts['wrote']} missing={counts['missing']} "
        f"skip={counts['skip']} error={counts['error']} "
        f"→ {data_dir}"
    )
    return 1 if counts["error"] and not counts["wrote"] and not counts["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
