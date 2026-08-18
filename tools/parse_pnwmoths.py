#!/usr/bin/env python3

"""Parse the Pacific Northwest Moths checklist dump into ``pnwmoths_summary.json``.

Reads ``MOTHS_DATA_DIR/pnwmoths.dump.html`` (download once from
https://moths.pnwinsects.org/checklist/) and writes every species link found::

    MOTHS_DATA_DIR/pnwmoths_summary.json

Each row stores ``name`` (scientific), ``common_name``, and ``url``.
Not scoped to iNat. Run ``tools/parse_data_summary.py`` afterward to match
rows onto the ``inats_summary`` universe by scientific name.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

REPO_ROOT = Path(__file__).resolve().parent.parent

DUMP_NAME = "pnwmoths.dump.html"
PNW_SUMMARY_NAME = "pnwmoths_summary.json"
PNW_BASE_URL = "https://moths.pnwinsects.org/"
PNW_SUMMARY_VERSION = 1

# <li data-slug="…"><a href="/species/…/"><em>Sci name</em> — Common</a></li>
_LI_RE = re.compile(
    r'<li\b[^>]*\bdata-slug=["\'][^"\']+["\'][^>]*>\s*'
    r'<a\b[^>]*\bhref=(["\'])(?P<href>.*?)\1[^>]*>\s*'
    r"(?P<body>.*?)\s*</a>\s*</li>",
    re.IGNORECASE | re.DOTALL,
)
_EM_RE = re.compile(r"<em>\s*(.*?)\s*</em>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Common-name separator on the live site is an em dash (—); tolerate hyphen too.
_SEP_RE = re.compile(r"\s+[—–-]\s+")


def bootstrap_django():
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()
    from django.conf import settings

    return settings


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description=(
            "Parse pnwmoths.dump.html into MOTHS_DATA_DIR/pnwmoths_summary.json "
            "(all species in the dump)."
        )
    ).parse_args()


def _strip_href(raw: str) -> str:
    return raw.strip().strip("\"'")


def _abs_pnw_url(href: str | None) -> str | None:
    if not href:
        return None
    href = _strip_href(href)
    if not href:
        return None
    return urljoin(PNW_BASE_URL, href)


def _plain(text: str) -> str:
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def parse_pnwmoths_species(html: str) -> list[dict]:
    """Return one dict per checklist species ``<li data-slug>`` row."""
    species: list[dict] = []
    seen_urls: set[str] = set()

    for match in _LI_RE.finditer(html):
        href = match.group("href")
        body = match.group("body") or ""
        url = _abs_pnw_url(href)
        if not url or url in seen_urls:
            continue

        em = _EM_RE.search(body)
        if em:
            name = _plain(em.group(1))
            rest = body[em.end() :]
        else:
            plain_body = _plain(body)
            parts = _SEP_RE.split(plain_body, maxsplit=1)
            name = (parts[0] or "").strip()
            rest = parts[1] if len(parts) > 1 else ""

        if not name:
            continue

        common = _plain(rest)
        common = _SEP_RE.sub("", common, count=1).strip(" \t\r\n—–-")

        seen_urls.add(url)
        species.append(
            {
                "name": name,
                "common_name": common,
                "url": url,
            }
        )

    return species


def main() -> int:
    parse_args()
    settings = bootstrap_django()
    data_dir = Path(settings.MOTHS_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    dump_path = data_dir / DUMP_NAME
    if not dump_path.is_file():
        print(
            f"Missing {dump_path}. Download it first, e.g.:\n"
            f"  curl -L -A 'Mozilla/5.0' -o {dump_path} "
            f"https://moths.pnwinsects.org/checklist/",
            file=sys.stderr,
        )
        return 1

    print(f"Parsing {dump_path} …")
    html = dump_path.read_text(encoding="utf-8", errors="replace")
    species = parse_pnwmoths_species(html)
    with_common = sum(1 for s in species if s.get("common_name"))
    with_url = sum(1 for s in species if s.get("url"))

    payload = {
        "version": PNW_SUMMARY_VERSION,
        "species": species,
    }
    out = data_dir / PNW_SUMMARY_NAME
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {out}: species={len(species)} "
        f"with_common_name={with_common} with_url={with_url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
