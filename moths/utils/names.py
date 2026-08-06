"""Taxonomy-name CSV and iNaturalist observation metadata lookups."""
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

# Cache of the parsed names CSV, invalidated by the file's mtime.
_NAMES_CACHE: dict = {"path": None, "mtime": None, "data": {}}
_EMPTY_NAME_INFO = {
    "superfamily": "",
    "family": "",
    "subfamily": "",
    "species": "",
    "name": "",
    "obs": "",
}


def get_names_csv_path() -> Path:
    """Return the configured path to the taxonomy names CSV."""
    return Path(settings.MOTHS_NAMES_CSV)


def load_names() -> dict[str, dict]:
    """Return ``{tax_id: {family, species, name}}`` parsed from the names CSV.

    The result is cached and only re-read when the file changes. A missing or
    unreadable file yields an empty mapping (callers fall back to the raw id).
    """
    path = get_names_csv_path()
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    if _NAMES_CACHE["path"] == key and _NAMES_CACHE["mtime"] == mtime:
        return _NAMES_CACHE["data"]

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
                        # iNaturalist observation count; kept out of the summary
                        # cache so the CSV stays the single source of truth.
                        "obs": (row.get("obs") or "").strip(),
                    }
        except OSError:
            data = {}

    _NAMES_CACHE.update(path=key, mtime=mtime, data=data)
    return data


def get_name_info(tax_id) -> dict:
    """Return ``{family, species, name}`` for a tax_id (empty strings if unknown)."""
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
