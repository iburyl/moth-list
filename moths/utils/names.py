#!/usr/bin/env python3
"""Taxonomy-name lookups and iNaturalist observation metadata.

Runtime name info (``load_names`` / ``get_name_info``) comes from
``data_summary.json`` (built by ``tools/parse_data_summary.py`` from
``inats_summary.json``).

``load_names_csv`` remains only for ``tools/harvest_inats.py`` — the sole
consumer of the original names CSV.
"""
from __future__ import annotations

import csv
import json

from pathlib import Path

from django.conf import settings

from .paths import (
    get_image_dir,
    image_basename,
    parse_filename,
)


# --- Taxonomy names (tax_id -> family / species / name) -----------------------

_EMPTY_NAME_INFO = {
    "superfamily": "",
    "family": "",
    "subfamily": "",
    "species": "",
    "name": "",
    "obs": "",
}

# Cache of the parsed names CSV (tools only), invalidated by mtime.
_NAMES_CSV_CACHE: dict = {"path": None, "mtime": None, "data": {}}
# Cache of lineage extracted from data_summary.json.
_NAMES_CACHE: dict = {"path": None, "mtime": None, "data": {}}


def get_names_csv_path() -> Path:
    """Return the configured path to the taxonomy names CSV."""
    return Path(settings.MOTHS_NAMES_CSV)


def load_names_csv() -> dict[str, dict]:
    """Return ``{tax_id: lineage}`` parsed from the names CSV.

    **Only** ``tools/harvest_inats.py`` should call this. Everything else uses
    :func:`load_names` / ``data_summary.json``.
    """
    path = get_names_csv_path()
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    if _NAMES_CSV_CACHE["path"] == key and _NAMES_CSV_CACHE["mtime"] == mtime:
        return _NAMES_CSV_CACHE["data"]

    data: dict[str, dict] = {}
    if mtime is not None:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    tax_id = (row.get("id") or "").strip()
                    if not tax_id:
                        continue
                    data[tax_id] = {
                        "superfamily": (row.get("superfamily") or "").strip(),
                        "family": (row.get("family") or "").strip(),
                        "subfamily": (row.get("subfamily") or "").strip(),
                        "species": (row.get("species") or "").strip(),
                        "name": (row.get("name") or "").strip(),
                        "obs": (row.get("obs") or "").strip(),
                    }
        except OSError:
            data = {}

    _NAMES_CSV_CACHE.update(path=key, mtime=mtime, data=data)
    return data


def load_names() -> dict[str, dict]:
    """Return ``{tax_id: lineage}`` from ``data_summary.json``.

    Each value has ``superfamily``, ``family``, ``subfamily``, ``species``,
    ``name``, and ``obs`` (iNat observation count when harvested). Missing or
    stale summary → empty mapping.
    """
    # Local import avoids a cycle at module load (wiki does not import names).
    from .wiki import get_data_summary_path, load_data_summary

    path = get_data_summary_path()
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    if _NAMES_CACHE["path"] == key and _NAMES_CACHE["mtime"] == mtime:
        return _NAMES_CACHE["data"]

    data: dict[str, dict] = {}
    summary = load_data_summary()
    species = summary.get("species") if isinstance(summary, dict) else None
    if isinstance(species, dict):
        for tax_id, row in species.items():
            if not isinstance(row, dict):
                continue
            data[str(tax_id)] = {
                "superfamily": (row.get("superfamily") or "").strip(),
                "family": (row.get("family") or "").strip(),
                "subfamily": (row.get("subfamily") or "").strip(),
                "species": (row.get("species") or "").strip(),
                "name": (row.get("name") or "").strip(),
                "obs": str(row.get("obs") or "").strip(),
            }

    _NAMES_CACHE.update(path=key, mtime=mtime, data=data)
    return data


def get_name_info(tax_id) -> dict:
    """Return lineage dict for a tax_id (empty strings if unknown)."""
    return load_names().get(str(tax_id), _EMPTY_NAME_INFO)


# --- iNaturalist observation metadata ----------------------------------------

# Single-slot cache of the last-loaded observations file, keyed by path+mtime.
_OBSERVATIONS_CACHE: dict = {"path": None, "mtime": None, "data": {}}


def get_observations_path(tax_id: str) -> Path:
    """Path to a tax_id's downloaded observation metadata JSON."""
    return get_image_dir() / f"{tax_id}_observations.json"


def load_observations(tax_id: str) -> dict[str, dict]:
    """Return ``{observation_id: item}`` from ``{tax_id}_observations.json``.

    Cached and re-read only when the file changes. Missing/unreadable → ``{}``.
    """
    path = get_observations_path(tax_id)
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    if _OBSERVATIONS_CACHE["path"] == key and _OBSERVATIONS_CACHE["mtime"] == mtime:
        return _OBSERVATIONS_CACHE["data"]

    data: dict[str, dict] = {}
    if mtime is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raw = None
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("observation_id") is not None:
                    data[str(item["observation_id"])] = item

    _OBSERVATIONS_CACHE.update(path=key, mtime=mtime, data=data)
    return data


def get_observation_info(image_filename: str) -> dict | None:
    """Return the observation metadata for an image, or ``None`` if unavailable."""
    parsed = parse_filename(image_basename(image_filename))
    if parsed is None:
        return None
    return load_observations(parsed.tax_id).get(str(parsed.obs_id))
