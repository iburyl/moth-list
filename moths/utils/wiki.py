"""Load reference-data summaries for browse / poses decoration.

``tools/parse_wiki.py`` / ``parse_boa.py`` write raw harvest parses
(``wiki_summary.json`` / ``boa_summary.json``). ``tools/parse_data_summary.py``
joins those with the names CSV into Django-oriented files:

* ``data_summary.json`` — flat ``{tax_id: {wiki, boa}}`` (per-species lookups)
* ``tax_summary.json`` — hierarchical roll-up counts (browse columns)

The labels tree remains the listing authority; these helpers only decorate.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from django.conf import settings


DATA_SUMMARY_VERSION = 1
TAX_TREE_VERSION = 1

_DATA_SUMMARY_CACHE: dict = {"path": None, "mtime": None, "data": None}
_TAX_TREE_CACHE: dict = {"path": None, "mtime": None, "data": None}


def get_data_dir() -> Path:
    """Return the configured moth data directory."""
    return Path(settings.MOTHS_DATA_DIR)


def get_data_summary_path() -> Path:
    return get_data_dir() / "data_summary.json"


def get_wiki_tax_summary_path() -> Path:
    """Path to the hierarchical reference-data roll-up (browse counts)."""
    return get_data_dir() / "tax_summary.json"


# Back-compat aliases used by older imports / comments.
get_wiki_summary_path = get_data_summary_path


def get_wiki_article_path(tax_id: str) -> Path:
    """Path to the harvested wikitext file for a tax_id (may be missing)."""
    return get_data_dir() / str(tax_id) / f"{tax_id}.wiki"


def _load_versioned(path: Path, version: int, cache: dict) -> dict | None:
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    if cache["path"] == key and cache["mtime"] == mtime and cache["data"] is not None:
        return cache["data"]

    data = None
    if mtime is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raw = None
        if isinstance(raw, dict) and raw.get("version") == version:
            data = raw

    cache.update(path=key, mtime=mtime, data=data)
    return data


def load_data_summary() -> dict | None:
    """Return the flat CSV-scoped data summary, or ``None`` if missing/stale."""
    return _load_versioned(
        get_data_summary_path(), DATA_SUMMARY_VERSION, _DATA_SUMMARY_CACHE
    )


def load_wiki_tax_summary() -> dict | None:
    """Return the hierarchical browse roll-up, or ``None`` if missing/stale."""
    return _load_versioned(
        get_wiki_tax_summary_path(), TAX_TREE_VERSION, _TAX_TREE_CACHE
    )


def load_wiki_summary(tax_id: str) -> dict | None:
    """Return a per-tax wiki-shaped dict for display helpers, or ``None``.

    Adapts ``data_summary.json``'s ``wiki`` object into the shape expected by
    :func:`wiki_row_from_summary` (adds ``have_wiki`` / ``species``).
    """
    data = load_data_summary()
    if not data:
        return None
    species = data.get("species")
    if not isinstance(species, dict):
        return None
    row = species.get(str(tax_id))
    if not isinstance(row, dict):
        return None
    wiki = row.get("wiki")
    if not isinstance(wiki, dict):
        return None
    return {
        "have_wiki": True,
        "have_speciesbox": bool(wiki.get("have_speciesbox")),
        "scientific_name": wiki.get("scientific_name") or "",
        "authority": wiki.get("authority") or "",
        "species": wiki.get("scientific_name") or "",
        "iucn": wiki.get("iucn") if isinstance(wiki.get("iucn"), dict) else None,
        "tnc": wiki.get("tnc") if isinstance(wiki.get("tnc"), dict) else None,
        "boa": row.get("boa") if isinstance(row.get("boa"), dict) else None,
    }


def load_species_data(tax_id: str) -> dict:
    """Return ``{wiki, boa}`` for a tax_id (either side may be ``None``)."""
    data = load_data_summary()
    blank = {"wiki": None, "boa": None}
    if not data:
        return blank
    species = data.get("species")
    if not isinstance(species, dict):
        return blank
    row = species.get(str(tax_id))
    if not isinstance(row, dict):
        return blank
    return {
        "wiki": row.get("wiki") if isinstance(row.get("wiki"), dict) else None,
        "boa": row.get("boa") if isinstance(row.get("boa"), dict) else None,
    }


def wikipedia_url(title: str | None) -> str | None:
    """Build an en.wikipedia.org article URL from a page title / binomial."""
    name = (title or "").strip()
    if not name:
        return None
    return "https://en.wikipedia.org/wiki/" + quote(name.replace(" ", "_"))


def wiki_row_from_summary(summary: dict | None) -> dict:
    """Flatten a per-tax wiki summary into browse / poses display fields."""
    summary = summary or {}
    have_wiki = bool(summary.get("have_wiki"))
    iucn = summary.get("iucn") if isinstance(summary.get("iucn"), dict) else None
    tnc = summary.get("tnc") if isinstance(summary.get("tnc"), dict) else None
    sci = (summary.get("scientific_name") or summary.get("species") or "").strip()
    return {
        "have_wiki": have_wiki,
        "articles": 1 if have_wiki else 0,
        "scientific_name": sci,
        "authority": (summary.get("authority") or "").strip(),
        "article_url": wikipedia_url(sci) if have_wiki and sci else None,
        "iucn_status": (iucn or {}).get("status") or "",
        "iucn_url": (iucn or {}).get("url") or None,
        "has_iucn": iucn is not None,
        "tnc_status": (tnc or {}).get("status") or "",
        "tnc_url": (tnc or {}).get("url") or None,
        "has_tnc": tnc is not None,
    }


def wiki_node_counts(node: dict | None) -> dict:
    """Return ``{articles, iucn, tnc}`` counts for a tax_summary tree node."""
    node = node or {}
    return {
        "articles": int(node.get("total") or 0),
        "iucn": int(node.get("with_iucn") or 0),
        "tnc": int(node.get("with_tnc") or 0),
    }


def boa_node_counts(node: dict | None) -> dict:
    """Return ``{species}`` = count of species with a BOA page under ``node``."""
    node = node or {}
    boa = node.get("boa") if isinstance(node.get("boa"), dict) else {}
    return {"species": int(boa.get("total") or 0)}


def boa_row_from_data(boa: dict | None) -> dict:
    """Flatten a per-tax ``boa`` object for genus-table / poses display."""
    boa = boa if isinstance(boa, dict) else None
    url = (boa or {}).get("url") or None
    return {
        "has_page": bool(url),
        "url": url,
        "subspecies": int((boa or {}).get("subspecies") or 0) if boa else 0,
        "location": ((boa or {}).get("location") or "").strip(),
    }


def descend_wiki_node(tree: dict | None, keys: list[str]) -> dict | None:
    """Walk ``tree`` along ``keys`` (same lineage as the labels browse tree)."""
    if tree is None:
        return None
    node = tree
    for key in keys:
        container = node.get("superfamilies") if node is tree else node.get("children")
        if not isinstance(container, dict):
            return None
        child = container.get(key)
        if child is None:
            return None
        node = child
    return node
