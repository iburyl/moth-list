#!/usr/bin/env python3

"""Parse the Butterflies of America dump into a single ``boa_summary.json``.

Reads ``MOTHS_DATA_DIR/butterfliesofamerica.dump.html`` (download once with
curl) and writes **every** species row found in the dump to::

    MOTHS_DATA_DIR/boa_summary.json

Not scoped to iNat / CSV. Run ``tools/parse_data_summary.py`` afterward to
match BOA rows onto the ``inats_summary`` universe.
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

DUMP_NAME = "butterfliesofamerica.dump.html"
BOA_SUMMARY_NAME = "boa_summary.json"
BOA_BASE_URL = "https://butterfliesofamerica.com/L/"
BOA_SUMMARY_VERSION = 1

_BLOCK_RE = re.compile(r"<p\s+id=([a-z])>(.*?)</p>", re.IGNORECASE | re.DOTALL)
_NAME_RE = re.compile(r"<i>\s*([^<]+?)\s*</i>", re.IGNORECASE)
_TAXON_PAGE_RE = re.compile(
    r'<a\s+[^>]*title="taxon page"[^>]*href=([^\s>]+)',
    re.IGNORECASE,
)
_THUMB_RE = re.compile(
    r'<a\s+[^>]*title="species thumbnails"[^>]*href=([^\s>]+)',
    re.IGNORECASE,
)
_INAT_RE = re.compile(
    r"inaturalist\.org/observations\?taxon_id=(\d+)",
    re.IGNORECASE,
)
_LOC_RE = re.compile(r"<b\s+id=l>(.*?)</b>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


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
            "Parse butterfliesofamerica.dump.html into "
            "MOTHS_DATA_DIR/boa_summary.json (all species in the dump)."
        )
    ).parse_args()


def _strip_href(raw: str) -> str:
    return raw.strip().strip("\"'")


def _abs_boa_url(href: str | None) -> str | None:
    if not href:
        return None
    href = _strip_href(href)
    if not href:
        return None
    return urljoin(BOA_BASE_URL, href)


def _plain_location(raw: str | None) -> str:
    if not raw:
        return ""
    text = re.sub(r"<a\b[^>]*>.*?</a>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    text = _TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    return _WS_RE.sub(" ", text).strip(" \t\r\n\xa0;,")


def parse_boa_species(html: str) -> list[dict]:
    """Return one dict per ``<p id=s>`` species row in the dump."""
    blocks = list(_BLOCK_RE.finditer(html))
    species: list[dict] = []

    i = 0
    while i < len(blocks):
        kind, body = blocks[i].group(1).lower(), blocks[i].group(2)
        if kind != "s":
            i += 1
            continue

        name_match = _NAME_RE.search(body)
        name = (name_match.group(1) if name_match else "").strip()
        name = _WS_RE.sub(" ", html_lib.unescape(name)).strip()

        page = _TAXON_PAGE_RE.search(body)
        thumb = _THUMB_RE.search(body)
        url = _abs_boa_url(page.group(1) if page else None)
        if url is None:
            url = _abs_boa_url(thumb.group(1) if thumb else None)

        loc_match = _LOC_RE.search(body)
        location = _plain_location(loc_match.group(1) if loc_match else None)

        inat = _INAT_RE.search(body)
        tax_id = inat.group(1) if inat else None

        ssp = 0
        j = i + 1
        while j < len(blocks) and blocks[j].group(1).lower() == "b":
            ssp += 1
            j += 1

        species.append(
            {
                "tax_id": tax_id,
                "name": name,
                "url": url,
                "subspecies": ssp,
                "location": location,
            }
        )
        i = j

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
            f"https://butterfliesofamerica.com/L/All.htm",
            file=sys.stderr,
        )
        return 1

    print(f"Parsing {dump_path} …")
    html = dump_path.read_text(encoding="utf-8", errors="replace")
    species = parse_boa_species(html)
    with_tax = sum(1 for s in species if s.get("tax_id"))
    with_url = sum(1 for s in species if s.get("url"))

    payload = {
        "version": BOA_SUMMARY_VERSION,
        "species": species,
    }
    out = data_dir / BOA_SUMMARY_NAME
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {out}: species={len(species)} "
        f"with_tax_id={with_tax} with_url={with_url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
