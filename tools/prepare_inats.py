#!/usr/bin/env python3

"""Prepare flat iNaturalist summaries from harvest outputs.

Reads ``inats_list.json`` plus per-tax ``.inats`` / ``.synonyms`` / ``parents/``
(from ``harvest_inats.py``) and writes::

    MOTHS_DATA_DIR/inats_summary.json
        Ground-truth species list. ``by_tax_id`` holds found taxa (slim fields
        + ``ancestors``). ``not_found`` repeats the unresolved CSV id/name rows.

    MOTHS_DATA_DIR/inats_synonyms_summary.json
        ``by_tax_id`` / ``by_name`` indexes for scientific-name matching,
        plus ``collisions`` / ``collision_list``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

INATS_LIST_NAME = "inats_list.json"
INATS_SUMMARY_NAME = "inats_summary.json"
INATS_SYNONYMS_SUMMARY_NAME = "inats_synonyms_summary.json"
INATS_SUMMARY_VERSION = 2
INATS_SYNONYMS_SUMMARY_VERSION = 1
PARENTS_DIRNAME = "parents"

_ANCESTOR_FIELDS = ("id", "name", "rank", "rank_level")


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
            "Build inats_summary.json + inats_synonyms_summary.json from "
            "inats_list.json and harvested .inats / .synonyms files."
        )
    ).parse_args()


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def normalize_name(name: str) -> str:
    return " ".join(name.strip().casefold().split())


def _coerce_id(value: Any, fallback: str) -> Any:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if value is not None:
        return value
    return int(fallback) if fallback.isdigit() else fallback


def _taxon_lookup_entry(data: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    return {
        "id": _coerce_id(data.get("id"), fallback_id),
        "name": data.get("name"),
        "rank": data.get("rank"),
        "rank_level": data.get("rank_level"),
    }


def _ancestor_ids_for_species(taxon: dict[str, Any], tax_id: str) -> list[str]:
    self_key = str(taxon.get("id") if taxon.get("id") is not None else tax_id)
    raw = taxon.get("ancestor_ids")
    if isinstance(raw, list) and raw:
        return [str(i) for i in raw if i is not None and str(i) != self_key]
    parent_id = taxon.get("parent_id")
    if parent_id is not None and str(parent_id) != self_key:
        return [str(parent_id)]
    return []


def _ancestors_for_species(
    taxon: dict[str, Any],
    tax_id: str,
    lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ancestors: list[dict[str, Any]] = []
    for aid in _ancestor_ids_for_species(taxon, tax_id):
        parent = lookup.get(aid)
        if parent is not None:
            ancestors.append({field: parent.get(field) for field in _ANCESTOR_FIELDS})
        else:
            ancestors.append(
                {
                    "id": _coerce_id(aid, aid),
                    "name": None,
                    "rank": None,
                    "rank_level": None,
                }
            )
    return ancestors


def _load_parents_lookup(data_dir: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    parents_root = data_dir / PARENTS_DIRNAME
    if not parents_root.is_dir():
        return lookup
    for path in sorted(parents_root.glob("*.inats")):
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        tax_id = str(data.get("id") or path.stem)
        lookup[tax_id] = _taxon_lookup_entry(data, tax_id)
    return lookup


def build_inats_summary(data_dir: Path, inats_list: dict) -> dict[str, Any]:
    """Build species-only summary from found list + not_found list."""
    lookup = _load_parents_lookup(data_dir)
    by_tax_id: dict[str, dict[str, Any]] = {}

    for row in inats_list.get("found") or []:
        if not isinstance(row, dict):
            continue
        tax_id = str(row.get("id") or "").strip()
        if not tax_id:
            continue
        path = data_dir / tax_id / f"{tax_id}.inats"
        data = _load_json(path)
        if not isinstance(data, dict):
            # Listed found but file missing — still record a stub.
            by_tax_id[tax_id] = {
                "id": _coerce_id(tax_id, tax_id),
                "name": (row.get("name") or "").strip(),
                "found": True,
                "ancestors": [],
            }
            continue
        entry = dict(data)
        entry["found"] = True
        lookup.setdefault(tax_id, _taxon_lookup_entry(data, tax_id))
        entry["ancestors"] = _ancestors_for_species(data, tax_id, lookup)
        by_tax_id[tax_id] = entry

    not_found: list[dict[str, str]] = []
    for row in inats_list.get("not_found") or []:
        if not isinstance(row, dict):
            continue
        item = {
            "id": str(row.get("id") or "").strip(),
            "name": (row.get("name") or "").strip(),
        }
        not_found.append(item)

    return {
        "version": INATS_SUMMARY_VERSION,
        "by_tax_id": by_tax_id,
        "not_found": not_found,
    }


def collect_synonyms_summary(data_dir: Path, found_ids: list[str]) -> dict[str, Any]:
    """Build synonym indexes for found taxa only."""
    by_tax_id: dict[str, dict[str, Any]] = {}
    name_claims: dict[str, dict[str, bool]] = {}

    for tax_id in found_ids:
        syn_path = data_dir / tax_id / f"{tax_id}.synonyms"
        data = _load_json(syn_path)
        if not isinstance(data, dict):
            # Fall back to current name from .inats if synonyms file missing.
            inats = _load_json(data_dir / tax_id / f"{tax_id}.inats")
            current_name = ""
            if isinstance(inats, dict):
                current_name = (inats.get("name") or "").strip()
            if current_name:
                by_tax_id[tax_id] = {
                    "current_name": current_name,
                    "synonyms": [],
                }
                key = normalize_name(current_name)
                if key:
                    name_claims.setdefault(key, {})[tax_id] = True
            continue

        current_name = (data.get("current_name") or "").strip()
        current_key = normalize_name(current_name) if current_name else ""
        scientific = data.get("scientific_names") or []
        synonyms: list[str] = []

        for entry in scientific:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            if not entry.get("is_valid"):
                synonyms.append(name)
            key = normalize_name(name)
            if not key:
                continue
            claims = name_claims.setdefault(key, {})
            claims[tax_id] = claims.get(tax_id, False) or (key == current_key)

        by_tax_id[tax_id] = {
            "current_name": current_name,
            "synonyms": synonyms,
        }

    by_name: dict[str, str] = {}
    collision_list: list[dict[str, Any]] = []

    for key, claims in sorted(name_claims.items()):
        tax_ids = sorted(
            claims.keys(),
            key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
        )
        if len(tax_ids) == 1:
            by_name[key] = tax_ids[0]
            continue
        preferred = [tid for tid in tax_ids if claims.get(tid)]
        by_name[key] = preferred[0] if preferred else tax_ids[0]
        collision_list.append(
            {
                "name": key,
                "tax_ids": tax_ids,
                "chosen_tax_id": by_name[key],
            }
        )

    return {
        "version": INATS_SYNONYMS_SUMMARY_VERSION,
        "by_tax_id": by_tax_id,
        "by_name": by_name,
        "collisions": len(collision_list),
        "collision_list": collision_list,
    }


def main() -> int:
    parse_args()
    settings = bootstrap_django()
    data_dir = Path(settings.MOTHS_DATA_DIR)
    if not data_dir.is_dir():
        print(f"MOTHS_DATA_DIR missing: {data_dir}", file=sys.stderr)
        return 1

    list_path = data_dir / INATS_LIST_NAME
    inats_list = _load_json(list_path)
    if not isinstance(inats_list, dict):
        print(
            f"Missing or invalid {list_path}. Run harvest_inats.py first.",
            file=sys.stderr,
        )
        return 1

    summary = build_inats_summary(data_dir, inats_list)
    inats_out = data_dir / INATS_SUMMARY_NAME
    inats_out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {inats_out}: found={len(summary['by_tax_id'])} "
        f"not_found={len(summary['not_found'])}"
    )

    found_ids = sorted(
        summary["by_tax_id"].keys(),
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )
    synonyms = collect_synonyms_summary(data_dir, found_ids)
    syn_out = data_dir / INATS_SYNONYMS_SUMMARY_NAME
    syn_out.write_text(
        json.dumps(synonyms, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {syn_out}: taxa={len(synonyms['by_tax_id'])} "
        f"names={len(synonyms['by_name'])} "
        f"collisions={synonyms['collisions']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
