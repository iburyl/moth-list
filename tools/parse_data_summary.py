#!/usr/bin/env python3

"""Build Django-efficient data summaries from CSV + wiki/boa harvest parses.

Reads:

* names CSV (``TAX_CSV``) — taxonomy universe and lineage
* ``MOTHS_DATA_DIR/wiki_summary.json`` — optional (from ``parse_wiki.py``)
* ``MOTHS_DATA_DIR/boa_summary.json`` — optional (from ``parse_boa.py``)

Writes two files tuned for the web app (single load + dict / tree walk)::

    MOTHS_DATA_DIR/data_summary.json
        Flat ``{ tax_id: { wiki, boa } }`` for every CSV species. One file to
        cache in process for per-species poses / index decoration.

    MOTHS_DATA_DIR/tax_summary.json
        Hierarchical superfamily→…→species tree with roll-up counts
        (wiki articles / iucn / tnc, boa page links) for the browse view.

The labels tree remains the listing authority for which taxa appear in browse;
these files only decorate CSV taxa with harvested reference data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_SUMMARY_NAME = "data_summary.json"
TAX_SUMMARY_NAME = "tax_summary.json"
WIKI_SUMMARY_NAME = "wiki_summary.json"
BOA_SUMMARY_NAME = "boa_summary.json"

DATA_SUMMARY_VERSION = 1
TAX_TREE_VERSION = 1

_UNKNOWN_TAXON = "(unknown)"
_NO_SUBFAMILY = "-"
_WS_RE = re.compile(r"\s+")


def bootstrap_django():
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "moths_list.settings")

    import django

    django.setup()

    from django.conf import settings
    from moths.utils.names import load_names

    return settings, load_names


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(
        description=(
            "Join names CSV with wiki_summary.json / boa_summary.json into "
            "data_summary.json + tax_summary.json for Django."
        )
    ).parse_args()


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return raw if isinstance(raw, dict) else None


def _normalize_name(name: str) -> str:
    return _WS_RE.sub(" ", name).strip().lower()


def index_boa(boa_summary: dict | None) -> tuple[dict[str, dict], dict[str, dict]]:
    """Build ``(by_tax_id, by_name)`` lookup maps from boa_summary.json."""
    by_tax_id: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    if not boa_summary:
        return by_tax_id, by_name
    rows = boa_summary.get("species")
    if not isinstance(rows, list):
        return by_tax_id, by_name
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry = {
            "url": row.get("url"),
            "subspecies": int(row.get("subspecies") or 0),
            "location": row.get("location") or "",
            "name": row.get("name") or "",
        }
        tax_id = row.get("tax_id")
        if tax_id:
            by_tax_id[str(tax_id)] = entry
        name = entry["name"]
        if name:
            by_name.setdefault(_normalize_name(name), entry)
    return by_tax_id, by_name


def lookup_boa(
    tax_id: str,
    species: str,
    by_tax_id: dict[str, dict],
    by_name: dict[str, dict],
) -> dict | None:
    entry = by_tax_id.get(str(tax_id))
    if entry is not None:
        return entry
    key = _normalize_name(species)
    return by_name.get(key) if key else None


def lookup_wiki(tax_id: str, wiki_summary: dict | None) -> dict | None:
    if not wiki_summary:
        return None
    species_map = wiki_summary.get("species")
    if not isinstance(species_map, dict):
        return None
    entry = species_map.get(str(tax_id))
    return entry if isinstance(entry, dict) else None


def _wiki_payload(entry: dict | None) -> dict | None:
    """Slim wiki object for data_summary (None when no harvested article)."""
    if entry is None:
        return None
    return {
        "scientific_name": entry.get("scientific_name") or "",
        "authority": entry.get("authority") or "",
        "have_speciesbox": bool(entry.get("have_speciesbox")),
        "iucn": entry.get("iucn") if isinstance(entry.get("iucn"), dict) else None,
        "tnc": entry.get("tnc") if isinstance(entry.get("tnc"), dict) else None,
    }


def _boa_payload(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    return {
        "url": entry.get("url"),
        "subspecies": int(entry.get("subspecies") or 0),
        "location": entry.get("location") or "",
    }


def _tax_lineage_keys(info: dict) -> tuple[str, str, str, str]:
    species = (info.get("species") or "").strip()
    genus = species.split()[0] if species else ""
    return (
        info.get("superfamily") or _UNKNOWN_TAXON,
        info.get("family") or _UNKNOWN_TAXON,
        info.get("subfamily") or _NO_SUBFAMILY,
        genus or _UNKNOWN_TAXON,
    )


def _wiki_counts(leaves: list[dict]) -> dict[str, int]:
    return {
        "total": sum(1 for leaf in leaves if leaf.get("have_wiki")),
        "with_iucn": sum(1 for leaf in leaves if leaf.get("has_iucn")),
        "with_tnc": sum(1 for leaf in leaves if leaf.get("has_tnc")),
    }


def _boa_counts(leaves: list[dict]) -> dict[str, int]:
    return {
        "total": sum(1 for leaf in leaves if leaf.get("has_boa")),
    }


def _collect_leaves(node: dict) -> list[dict]:
    children = node.get("children")
    if not isinstance(children, dict) or not children:
        if "have_wiki" in node or "has_boa" in node:
            return [node]
        return []
    out: list[dict] = []
    for child in children.values():
        if (
            isinstance(child, dict)
            and "children" not in child
            and ("have_wiki" in child or "has_boa" in child)
        ):
            out.append(child)
        else:
            out.extend(_collect_leaves(child))
    return out


def _parent_node(children: dict) -> dict:
    leaves: list[dict] = []
    for child in children.values():
        leaves.extend(_collect_leaves(child))
    return {
        **_wiki_counts(leaves),
        "boa": _boa_counts(leaves),
        "children": children,
    }


def build_tax_tree(names: dict[str, dict], data_species: dict[str, dict]) -> dict:
    tree: dict = {}
    for tax_id, info in names.items():
        row = data_species.get(str(tax_id)) or {}
        wiki = row.get("wiki") if isinstance(row.get("wiki"), dict) else None
        boa = row.get("boa") if isinstance(row.get("boa"), dict) else None
        sf, fam, subf, genus = _tax_lineage_keys(info)
        leaf = {
            "species": info.get("species") or "",
            "name": info.get("name") or "",
            "have_wiki": wiki is not None,
            "have_speciesbox": bool((wiki or {}).get("have_speciesbox")),
            "has_iucn": isinstance((wiki or {}).get("iucn"), dict),
            "has_tnc": isinstance((wiki or {}).get("tnc"), dict),
            "has_boa": bool(boa and boa.get("url")),
            "boa_subspecies": int((boa or {}).get("subspecies") or 0),
        }
        (
            tree.setdefault(sf, {})
            .setdefault(fam, {})
            .setdefault(subf, {})
            .setdefault(genus, {})
        )[str(tax_id)] = leaf

    superfamilies: dict = {}
    for sf, fams in tree.items():
        fam_nodes: dict = {}
        for fam, subfs in fams.items():
            subf_nodes: dict = {}
            for subf, genera in subfs.items():
                genus_nodes: dict = {}
                for genus, species_map in genera.items():
                    leaves = list(species_map.values())
                    genus_nodes[genus] = {
                        **_wiki_counts(leaves),
                        "boa": _boa_counts(leaves),
                        "children": species_map,
                    }
                subf_nodes[subf] = _parent_node(genus_nodes)
            fam_nodes[fam] = _parent_node(subf_nodes)
        superfamilies[sf] = _parent_node(fam_nodes)

    root_leaves: list[dict] = []
    for node in superfamilies.values():
        root_leaves.extend(_collect_leaves(node))

    return {
        "version": TAX_TREE_VERSION,
        **_wiki_counts(root_leaves),
        "boa": _boa_counts(root_leaves),
        "superfamilies": superfamilies,
    }


def main() -> int:
    parse_args()
    settings, load_names = bootstrap_django()
    data_dir = Path(settings.MOTHS_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    names = load_names()
    if not names:
        print("Names CSV is empty or unreadable.", file=sys.stderr)
        return 1

    wiki_summary = _load_json(data_dir / WIKI_SUMMARY_NAME)
    boa_summary = _load_json(data_dir / BOA_SUMMARY_NAME)
    if wiki_summary is None:
        print(f"Note: {WIKI_SUMMARY_NAME} missing; wiki fields will be empty.")
    if boa_summary is None:
        print(f"Note: {BOA_SUMMARY_NAME} missing; boa fields will be empty.")

    boa_by_id, boa_by_name = index_boa(boa_summary)

    data_species: dict[str, dict] = {}
    counts = {"wiki": 0, "iucn": 0, "tnc": 0, "boa": 0, "boa_url": 0}

    for tax_id, info in names.items():
        species_name = (info.get("species") or "").strip()
        wiki = _wiki_payload(lookup_wiki(tax_id, wiki_summary))
        boa = _boa_payload(
            lookup_boa(tax_id, species_name, boa_by_id, boa_by_name)
        )
        data_species[str(tax_id)] = {"wiki": wiki, "boa": boa}
        if wiki is not None:
            counts["wiki"] += 1
            if wiki.get("iucn"):
                counts["iucn"] += 1
            if wiki.get("tnc"):
                counts["tnc"] += 1
        if boa is not None:
            counts["boa"] += 1
            if boa.get("url"):
                counts["boa_url"] += 1

    data_payload = {
        "version": DATA_SUMMARY_VERSION,
        "species": data_species,
    }
    data_path = data_dir / DATA_SUMMARY_NAME
    data_path.write_text(
        json.dumps(data_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tree = build_tax_tree(names, data_species)
    tree_path = data_dir / TAX_SUMMARY_NAME
    tree_path.write_text(
        json.dumps(tree, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"CSV species={len(names)}: "
        f"wiki={counts['wiki']} iucn={counts['iucn']} tnc={counts['tnc']} "
        f"boa={counts['boa']} boa_url={counts['boa_url']}"
    )
    print(f"Wrote {data_path}")
    print(
        f"Wrote {tree_path}: wiki.total={tree['total']} "
        f"boa.total={tree['boa']['total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
