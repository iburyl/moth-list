#!/usr/bin/env python3

"""Build Django-efficient data summaries from inats + wiki/boa/gbif/pnw parses.

Reads:

* ``MOTHS_DATA_DIR/inats_summary.json`` — ground-truth species list (found +
  not_found) from ``prepare_inats.py``
* ``MOTHS_DATA_DIR/wiki_summary.json`` — optional (from ``parse_wiki.py``)
* ``MOTHS_DATA_DIR/boa_summary.json`` — optional (from ``parse_boa.py``)
* ``MOTHS_DATA_DIR/gbif_summary.json`` — optional (from ``harvest_gbif.py``)
* ``MOTHS_DATA_DIR/pnwmoths_summary.json`` — optional (from ``parse_pnwmoths.py``)

Writes::

    MOTHS_DATA_DIR/data_summary.json
        Flat ``{ tax_id: { lineage..., sources: {wiki, boa, ...} } }`` for
        every species in ``inats_summary`` (found and not_found).

    MOTHS_DATA_DIR/tax_summary.json
        Hierarchical roll-up counts under each node's uniform ``sources`` map.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_SUMMARY_NAME = "data_summary.json"
TAX_SUMMARY_NAME = "tax_summary.json"
WIKI_SUMMARY_NAME = "wiki_summary.json"
WIKI_LIST_NAME = "wiki_list.json"
BOA_SUMMARY_NAME = "boa_summary.json"
GBIF_SUMMARY_NAME = "gbif_summary.json"
PNW_SUMMARY_NAME = "pnwmoths_summary.json"
INATS_SUMMARY_NAME = "inats_summary.json"

# This envelope is intentionally independent of the set of source keys.
# Adding another entry under ``sources`` is not a schema change.
SUMMARY_SCHEMA = 1
DATA_SUMMARY_VERSION = 10
TAX_TREE_VERSION = 6

_UNKNOWN_TAXON = "(unknown)"
_NO_SUBFAMILY = "-"
_WS_RE = re.compile(r"\s+")

_ARTICLE_KIND_KEYS = (
    "matches_name",
    "known_synonym",
    "unknown_synonym",
    "higher_rank",
    "unknown_type",
)

_INATS_FIELDS = (
    "id",
    "name",
    "rank",
    "rank_level",
    "parent_id",
    "is_active",
    "extinct",
    "observations_count",
    "preferred_common_name",
    "vision",
    "wikipedia_url",
)

_GBIF_FIELDS = (
    "queried",
    "usageKey",
    "matchedUsageKey",
    "scientificName",
    "canonicalName",
    "acceptedScientificName",
    "acceptedCanonicalName",
    "authorship",
    "acceptedAuthorship",
    "rank",
    "code",
    "nameType",
    "status",
    "matchType",
    "confidence",
    "accepted",
    "url",
    "observations_count",
    "last_observation_date",
    "classification",
)


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
            "Join inats_summary.json with wiki/boa/gbif/pnw into "
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
            "name": (row.get("name") or "").strip(),
            "subspecies": int(row.get("subspecies") or 0),
            "location": row.get("location") or "",
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


def index_pnw(pnw_summary: dict | None) -> dict[str, dict]:
    """Index PNW moths rows by normalized scientific name."""
    by_name: dict[str, dict] = {}
    if not pnw_summary:
        return by_name
    rows = pnw_summary.get("species")
    if not isinstance(rows, list):
        return by_name
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = (row.get("name") or "").strip()
        if not name:
            continue
        entry = {
            "name": name,
            "common_name": (row.get("common_name") or "").strip(),
            "url": row.get("url"),
        }
        by_name.setdefault(_normalize_name(name), entry)
    return by_name


def lookup_pnw(species: str, by_name: dict[str, dict]) -> dict | None:
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


def index_wiki_urls(wiki_list: dict | None) -> dict[str, str]:
    """Return downloaded Wikipedia article URLs keyed by iNaturalist tax id."""
    by_tax_id: dict[str, str] = {}
    if not wiki_list:
        return by_tax_id
    rows = wiki_list.get("found")
    if not isinstance(rows, list):
        return by_tax_id
    for row in rows:
        if not isinstance(row, dict):
            continue
        tax_id = str(row.get("id") or "").strip()
        url = (row.get("url") or "").strip()
        if tax_id and url:
            by_tax_id[tax_id] = url
    return by_tax_id


def lookup_gbif(tax_id: str, gbif_summary: dict | None) -> dict | None:
    if not gbif_summary:
        return None
    by_tax = gbif_summary.get("by_tax_id")
    if not isinstance(by_tax, dict):
        return None
    entry = by_tax.get(str(tax_id))
    if not isinstance(entry, dict):
        return None
    if entry.get("found") is False:
        return None
    if entry.get("usageKey") is None:
        return None
    return entry


def _wiki_payload(entry: dict | None, url: str | None = None) -> dict | None:
    if entry is None:
        return None
    kind = (entry.get("article_kind") or "").strip()
    if kind not in _ARTICLE_KIND_KEYS:
        if entry.get("have_speciesbox"):
            kind = "unknown_synonym"
        elif entry.get("have_automatic_taxobox"):
            kind = "higher_rank"
        else:
            kind = "unknown_type"
    matches = entry.get("name_matches_inats")
    if matches is not None:
        matches = bool(matches)
    scientific_name = (entry.get("scientific_name") or "").strip()
    article_url = (url or "").strip()
    if not article_url and scientific_name:
        article_url = (
            "https://en.wikipedia.org/wiki/"
            + quote(scientific_name.replace(" ", "_"))
        )
    return {
        "url": article_url or None,
        "scientific_name": scientific_name,
        "authority": entry.get("authority") or "",
        "have_speciesbox": bool(entry.get("have_speciesbox")),
        "have_automatic_taxobox": bool(entry.get("have_automatic_taxobox")),
        "name_matches_inats": matches,
        "article_kind": kind,
        "iucn": entry.get("iucn") if isinstance(entry.get("iucn"), dict) else None,
        "tnc": entry.get("tnc") if isinstance(entry.get("tnc"), dict) else None,
    }


def _boa_payload(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    return {
        "url": entry.get("url"),
        "name": (entry.get("name") or "").strip(),
        "subspecies": int(entry.get("subspecies") or 0),
        "location": entry.get("location") or "",
    }


def _pnw_payload(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    url = entry.get("url")
    if not url:
        return None
    return {
        "url": url,
        "name": (entry.get("name") or "").strip(),
        "common_name": (entry.get("common_name") or "").strip(),
    }


def _inats_payload(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    out = {field: entry.get(field) for field in _INATS_FIELDS}
    ancestors = entry.get("ancestors")
    out["ancestors"] = ancestors if isinstance(ancestors, list) else []
    out["found"] = bool(entry.get("found", True))
    tax_id = str(out.get("id") or "").strip()
    out["url"] = f"https://www.inaturalist.org/taxa/{tax_id}" if tax_id else None
    return out


def _split_name_authority(full: str, canonical: str, authorship: str) -> tuple[str, str]:
    """Return ``(canonical, authorship)``, filling authorship from a full name."""
    full = (full or "").strip()
    canonical = (canonical or "").strip()
    authorship = (authorship or "").strip()
    if not authorship and full and canonical and full.startswith(canonical):
        authorship = full[len(canonical) :].strip()
    if not canonical and full:
        canonical = full
        authorship = ""
    return canonical, authorship


def _gbif_payload(entry: dict | None) -> dict | None:
    if entry is None:
        return None
    out = {field: entry.get(field) for field in _GBIF_FIELDS}
    out["accepted"] = bool(entry.get("accepted"))
    try:
        out["observations_count"] = int(entry.get("observations_count") or 0)
    except (TypeError, ValueError):
        out["observations_count"] = 0
    canonical, authorship = _split_name_authority(
        out.get("scientificName") or "",
        out.get("canonicalName") or "",
        out.get("authorship") or "",
    )
    accepted_canonical, accepted_authorship = _split_name_authority(
        out.get("acceptedScientificName") or "",
        out.get("acceptedCanonicalName") or "",
        out.get("acceptedAuthorship") or "",
    )
    out["canonicalName"] = canonical
    out["authorship"] = authorship
    out["acceptedCanonicalName"] = accepted_canonical
    out["acceptedAuthorship"] = accepted_authorship
    # Flat taxonomy prefers the accepted name, but an accepted match has no
    # separate accepted-name fields. Resolve that fallback once in the parser.
    out["displayCanonicalName"] = accepted_canonical or canonical
    out["displayAuthorship"] = accepted_authorship or (
        authorship if not accepted_canonical else ""
    )
    return out


def _ancestor_rank_name(ancestors: list, rank: str) -> str:
    needle = rank.casefold()
    for item in ancestors:
        if not isinstance(item, dict):
            continue
        if (item.get("rank") or "").casefold() == needle:
            return (item.get("name") or "").strip()
    return ""


def _lineage_from_inats(entry: dict | None, fallback_name: str = "") -> dict:
    """Build CSV-shaped lineage from an inats_summary species entry."""
    entry = entry if isinstance(entry, dict) else {}
    ancestors = entry.get("ancestors") if isinstance(entry.get("ancestors"), list) else []
    species = (entry.get("name") or fallback_name or "").strip()
    common = (entry.get("preferred_common_name") or "").strip()
    obs = entry.get("observations_count")
    obs_s = str(obs) if obs is not None else ""
    return {
        "superfamily": _ancestor_rank_name(ancestors, "superfamily"),
        "family": _ancestor_rank_name(ancestors, "family"),
        "subfamily": _ancestor_rank_name(ancestors, "subfamily"),
        "species": species,
        "name": common,
        "obs": obs_s,
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


def _wiki_counts(source_rows: list[dict | None]) -> dict[str, int]:
    counts = {
        "total": sum(1 for source in source_rows if source is not None),
        "with_iucn": sum(
            1 for source in source_rows
            if isinstance(source, dict) and isinstance(source.get("iucn"), dict)
        ),
        "with_tnc": sum(
            1 for source in source_rows
            if isinstance(source, dict) and isinstance(source.get("tnc"), dict)
        ),
    }
    for key in _ARTICLE_KIND_KEYS:
        counts[key] = sum(
            1
            for source in source_rows
            if isinstance(source, dict) and source.get("article_kind") == key
        )
    return counts


def _boa_counts(source_rows: list[dict | None]) -> dict[str, int]:
    return {
        "total": sum(
            1 for source in source_rows
            if isinstance(source, dict) and bool(source.get("url"))
        )
    }


def _inats_counts(source_rows: list[dict | None]) -> dict[str, int]:
    return {
        "total": sum(1 for source in source_rows if source is not None),
        "observations": sum(
            int((source or {}).get("observations_count") or 0)
            for source in source_rows
        ),
    }


def _gbif_counts(source_rows: list[dict | None]) -> dict[str, int]:
    return {
        "total": sum(1 for source in source_rows if source is not None),
        "accepted": sum(
            1 for source in source_rows
            if isinstance(source, dict) and bool(source.get("accepted"))
        ),
        "synonym": sum(
            1
            for source in source_rows
            if source is not None and not bool((source or {}).get("accepted"))
        ),
        "observations": sum(
            int((source or {}).get("observations_count") or 0)
            for source in source_rows
        ),
    }


def _pnw_counts(source_rows: list[dict | None]) -> dict[str, int]:
    return {
        "total": sum(
            1 for source in source_rows
            if isinstance(source, dict) and bool(source.get("url"))
        )
    }


def _species_rollups(sources: dict) -> dict[str, dict[str, int]]:
    """Convert one data-summary source map to tax-summary count objects."""
    def source(source_id: str) -> dict | None:
        value = sources.get(source_id)
        return value if isinstance(value, dict) else None

    return {
        "wiki": _wiki_counts([source("wiki")]),
        "boa": _boa_counts([source("boa")]),
        "inats": _inats_counts([source("inats")]),
        "gbif": _gbif_counts([source("gbif")]),
        "pnw": _pnw_counts([source("pnw")]),
    }


def _rollup_sources(leaves: list[dict]) -> dict[str, dict[str, int]]:
    """Sum species count objects into the uniform roll-up for a parent node."""
    source_ids = ("wiki", "boa", "inats", "gbif", "pnw")
    out: dict[str, dict[str, int]] = {source_id: {} for source_id in source_ids}
    for leaf in leaves:
        sources = leaf.get("sources")
        if not isinstance(sources, dict):
            continue
        for source_id in source_ids:
            counts = sources.get(source_id)
            if not isinstance(counts, dict):
                continue
            target = out[source_id]
            for key, value in counts.items():
                if isinstance(value, (int, bool)):
                    target[key] = target.get(key, 0) + int(value)
    return out


def _collect_leaves(node: dict) -> list[dict]:
    children = node.get("children")
    if not isinstance(children, dict) or not children:
        if isinstance(node.get("sources"), dict):
            return [node]
        return []
    out: list[dict] = []
    for child in children.values():
        if (
            isinstance(child, dict)
            and "children" not in child
            and isinstance(child.get("sources"), dict)
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
        "sources": _rollup_sources(leaves),
        "children": children,
    }


def build_tax_tree(names: dict[str, dict], data_species: dict[str, dict]) -> dict:
    tree: dict = {}
    for tax_id, info in names.items():
        row = data_species.get(str(tax_id)) or {}
        sources = row.get("sources") if isinstance(row.get("sources"), dict) else {}
        sf, fam, subf, genus = _tax_lineage_keys(info)
        leaf = {
            "species": info.get("species") or "",
            "name": info.get("name") or "",
            "sources": _species_rollups(sources),
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
                        "sources": _rollup_sources(leaves),
                        "children": species_map,
                    }
                subf_nodes[subf] = _parent_node(genus_nodes)
            fam_nodes[fam] = _parent_node(subf_nodes)
        superfamilies[sf] = _parent_node(fam_nodes)

    root_leaves: list[dict] = []
    for node in superfamilies.values():
        root_leaves.extend(_collect_leaves(node))

    return {
        "schema": SUMMARY_SCHEMA,
        "version": TAX_TREE_VERSION,
        "sources": _rollup_sources(root_leaves),
        "superfamilies": superfamilies,
    }


def iter_inats_universe(inats_summary: dict) -> list[tuple[str, dict | None, str]]:
    """Yield ``(tax_id, found_entry_or_None, fallback_name)`` for all listed taxa."""
    rows: list[tuple[str, dict | None, str]] = []
    by_tax = inats_summary.get("by_tax_id")
    if isinstance(by_tax, dict):
        for tax_id, entry in by_tax.items():
            if not isinstance(entry, dict):
                continue
            rows.append((str(tax_id), entry, (entry.get("name") or "").strip()))
    for item in inats_summary.get("not_found") or []:
        if not isinstance(item, dict):
            continue
        tax_id = str(item.get("id") or "").strip()
        name = (item.get("name") or "").strip()
        if not tax_id and not name:
            continue
        # Prefer a stable key; fall back to name-only sentinel.
        key = tax_id or f"name:{_normalize_name(name)}"
        rows.append((key, None, name))

    def sort_key(row: tuple[str, dict | None, str]):
        tid = row[0]
        return (0, int(tid)) if tid.isdigit() else (1, tid)

    rows.sort(key=sort_key)
    return rows


def main() -> int:
    parse_args()
    settings = bootstrap_django()
    data_dir = Path(settings.MOTHS_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    inats_summary = _load_json(data_dir / INATS_SUMMARY_NAME)
    if not inats_summary:
        print(
            f"Missing {data_dir / INATS_SUMMARY_NAME}. Run prepare_inats.py first.",
            file=sys.stderr,
        )
        return 1

    wiki_summary = _load_json(data_dir / WIKI_SUMMARY_NAME)
    wiki_list = _load_json(data_dir / WIKI_LIST_NAME)
    boa_summary = _load_json(data_dir / BOA_SUMMARY_NAME)
    gbif_summary = _load_json(data_dir / GBIF_SUMMARY_NAME)
    pnw_summary = _load_json(data_dir / PNW_SUMMARY_NAME)
    if wiki_summary is None:
        print(f"Note: {WIKI_SUMMARY_NAME} missing; wiki fields will be empty.")
    if boa_summary is None:
        print(f"Note: {BOA_SUMMARY_NAME} missing; boa fields will be empty.")
    if gbif_summary is None:
        print(f"Note: {GBIF_SUMMARY_NAME} missing; gbif fields will be empty.")
    if pnw_summary is None:
        print(f"Note: {PNW_SUMMARY_NAME} missing; pnw fields will be empty.")

    boa_by_id, boa_by_name = index_boa(boa_summary)
    wiki_urls = index_wiki_urls(wiki_list)
    pnw_by_name = index_pnw(pnw_summary)
    universe = iter_inats_universe(inats_summary)
    if not universe:
        print("inats_summary.json has no species.", file=sys.stderr)
        return 1

    data_species: dict[str, dict] = {}
    names: dict[str, dict] = {}
    counts = {
        "wiki": 0,
        "iucn": 0,
        "tnc": 0,
        "boa": 0,
        "boa_url": 0,
        "inats": 0,
        "not_found": 0,
        "gbif": 0,
        "gbif_accepted": 0,
        "gbif_synonym": 0,
        "pnw": 0,
    }

    for tax_id, inats_entry, fallback_name in universe:
        lineage = _lineage_from_inats(inats_entry, fallback_name)
        species_name = lineage["species"]
        wiki = _wiki_payload(
            lookup_wiki(tax_id, wiki_summary), wiki_urls.get(str(tax_id))
        )
        boa = _boa_payload(
            lookup_boa(tax_id, species_name, boa_by_id, boa_by_name)
        )
        inats = _inats_payload(inats_entry)
        gbif = _gbif_payload(lookup_gbif(tax_id, gbif_summary))
        pnw = _pnw_payload(lookup_pnw(species_name, pnw_by_name))
        data_species[str(tax_id)] = {
            "tax_id": str(tax_id),
            **lineage,
            "sources": {
                "wiki": wiki,
                "boa": boa,
                "inats": inats,
                "gbif": gbif,
                "pnw": pnw,
            },
        }
        names[str(tax_id)] = lineage
        if inats is not None:
            counts["inats"] += 1
        else:
            counts["not_found"] += 1
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
        if gbif is not None:
            counts["gbif"] += 1
            if gbif.get("accepted"):
                counts["gbif_accepted"] += 1
            else:
                counts["gbif_synonym"] += 1
        if pnw is not None:
            counts["pnw"] += 1

    data_payload = {
        "schema": SUMMARY_SCHEMA,
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
        f"inats universe={len(universe)}: "
        f"found={counts['inats']} not_found={counts['not_found']} "
        f"wiki={counts['wiki']} iucn={counts['iucn']} tnc={counts['tnc']} "
        f"boa={counts['boa']} boa_url={counts['boa_url']} "
        f"gbif={counts['gbif']} "
        f"(accepted={counts['gbif_accepted']} synonym={counts['gbif_synonym']}) "
        f"pnw={counts['pnw']}"
    )
    print(f"Wrote {data_path}")
    tree_sources = tree.get("sources") if isinstance(tree.get("sources"), dict) else {}
    wiki_tree = tree_sources.get("wiki") if isinstance(tree_sources.get("wiki"), dict) else {}
    boa_tree = tree_sources.get("boa") if isinstance(tree_sources.get("boa"), dict) else {}
    inats_tree = tree_sources.get("inats") if isinstance(tree_sources.get("inats"), dict) else {}
    gbif_tree = tree_sources.get("gbif") if isinstance(tree_sources.get("gbif"), dict) else {}
    pnw_tree = tree_sources.get("pnw") if isinstance(tree_sources.get("pnw"), dict) else {}
    print(
        f"Wrote {tree_path}: wiki.total={wiki_tree.get('total', 0)} "
        f"boa.total={boa_tree.get('total', 0)} "
        f"inats.total={inats_tree.get('total', 0)} "
        f"inats.observations={inats_tree.get('observations', 0)} "
        f"gbif.total={gbif_tree.get('total', 0)} "
        f"gbif.observations={gbif_tree.get('observations', 0)} "
        f"pnw.total={pnw_tree.get('total', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
