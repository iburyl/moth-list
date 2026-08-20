"""Load reference-data summaries for browse / species decoration.

``tools/parse_data_summary.py`` joins harvest parses with ``inats_summary.json``
into Django-oriented files:

* ``data_summary.json`` — flat ``{tax_id: {lineage..., sources: {...}}}``
* ``tax_summary.json`` — hierarchical ``sources`` roll-up counts (browse columns)

Display layout for ``sources`` fields is driven by ``integration.json``; this
module only loads the data and builds flat-taxonomy row structure.
"""
from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings

from .integration import build_integration_groups


DATA_SUMMARY_VERSION = 10
TAX_TREE_VERSION = 6

_DATA_SUMMARY_CACHE: dict = {"path": None, "mtime": None, "data": None}
_TAX_TREE_CACHE: dict = {"path": None, "mtime": None, "data": None}


def get_data_dir() -> Path:
    """Return the configured moth data directory."""
    return Path(settings.MOTHS_DATA_DIR)


def get_data_summary_path() -> Path:
    return get_data_dir() / "data_summary.json"


def get_data_tax_summary_path() -> Path:
    """Path to the hierarchical reference-data roll-up (browse counts)."""
    return get_data_dir() / "tax_summary.json"


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


def load_data_tax_summary() -> dict | None:
    """Return the hierarchical browse roll-up, or ``None`` if missing/stale."""
    return _load_versioned(
        get_data_tax_summary_path(), TAX_TREE_VERSION, _TAX_TREE_CACHE
    )


def load_species_data(tax_id: str) -> dict:
    """Return the opaque ``sources`` map for one species."""
    data = load_data_summary()
    if not data:
        return {}
    species = data.get("species")
    if not isinstance(species, dict):
        return {}
    row = species.get(str(tax_id))
    if not isinstance(row, dict):
        return {}
    sources = row.get("sources")
    if not isinstance(sources, dict):
        return {}
    return sources


def flat_taxonomy_rows(data_summary: dict | None) -> list[dict]:
    """Build flat-taxonomy table rows (rank headers + species source cells)."""
    if not data_summary:
        return []
    species_map = data_summary.get("species")
    if not isinstance(species_map, dict):
        return []

    items: list[tuple[str, dict]] = []
    for tax_id, row in species_map.items():
        if not isinstance(row, dict):
            continue
        items.append((str(tax_id), row))

    def sort_key(item: tuple[str, dict]):
        tax_id, row = item
        return (
            (row.get("superfamily") or "").casefold(),
            (row.get("family") or "").casefold(),
            (row.get("subfamily") or "").casefold(),
            (row.get("species") or "").casefold(),
            (0, int(tax_id)) if tax_id.isdigit() else (1, tax_id),
        )

    items.sort(key=sort_key)

    rows: list[dict] = []
    prev_sf = prev_fam = prev_subf = object()
    for tax_id, row in items:
        sf = (row.get("superfamily") or "").strip() or "(unknown)"
        fam = (row.get("family") or "").strip() or "(unknown)"
        subf = (row.get("subfamily") or "").strip() or "-"
        if sf != prev_sf:
            rows.append({"kind": "header", "label": f"Superfamily: {sf}"})
            prev_sf = sf
            prev_fam = object()
            prev_subf = object()
        if fam != prev_fam:
            rows.append({"kind": "header", "label": f"Family: {fam}"})
            prev_fam = fam
            prev_subf = object()
        if subf != prev_subf:
            rows.append({"kind": "header", "label": f"Subfamily: {subf}"})
            prev_subf = subf

        sources = row.get("sources") if isinstance(row.get("sources"), dict) else {}
        inats = sources.get("inats") if isinstance(sources.get("inats"), dict) else None
        inat_name = ((inats or {}).get("name") or "").strip()
        species_name = (row.get("species") or "").strip() or inat_name

        rows.append(
            {
                "kind": "species",
                "tax_id": tax_id,
                "species": species_name,
                "integrations": build_integration_groups(
                    "flat",
                    sources,
                    context={"tax_id": tax_id},
                    reference_name=species_name,
                ),
            }
        )
    return rows
