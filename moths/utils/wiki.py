"""Load reference-data summaries for browse / poses decoration.

``tools/parse_wiki.py`` / ``parse_boa.py`` / ``prepare_inats.py`` /
``harvest_gbif.py`` / ``parse_pnwmoths.py`` write raw harvest parses.
``tools/parse_data_summary.py`` joins those with ``inats_summary.json`` into
Django-oriented files:

* ``data_summary.json`` — flat ``{tax_id: {wiki, boa, inats, gbif, pnw}}``
* ``tax_summary.json`` — hierarchical roll-up counts (browse columns)

The labels tree remains the listing authority; these helpers only decorate.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from django.conf import settings


DATA_SUMMARY_VERSION = 9
TAX_TREE_VERSION = 5

_DATA_SUMMARY_CACHE: dict = {"path": None, "mtime": None, "data": None}
_TAX_TREE_CACHE: dict = {"path": None, "mtime": None, "data": None}
_WIKI_LIST_CACHE: dict = {"path": None, "mtime": None, "by_id": None}

_ARTICLE_KIND_TOOLTIP_LINES = (
    ("matches_name", "articles matches name"),
    ("known_synonym", "articles shows known synonym"),
    ("unknown_synonym", "articles shows unknown synonym"),
    ("higher_rank", "articles shows higher taxonomy rank"),
    ("unknown_type", "unknown article type"),
)

_GBIF_STATUS_TOOLTIP_LINES = (
    ("accepted", "accepted names"),
    ("synonym", "synonym names"),
)


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
        "have_automatic_taxobox": bool(wiki.get("have_automatic_taxobox")),
        "name_matches_inats": wiki.get("name_matches_inats"),
        "article_kind": wiki.get("article_kind") or "",
        "scientific_name": wiki.get("scientific_name") or "",
        "authority": wiki.get("authority") or "",
        "species": wiki.get("scientific_name") or "",
        "iucn": wiki.get("iucn") if isinstance(wiki.get("iucn"), dict) else None,
        "tnc": wiki.get("tnc") if isinstance(wiki.get("tnc"), dict) else None,
        "boa": row.get("boa") if isinstance(row.get("boa"), dict) else None,
    }


def load_species_data(tax_id: str) -> dict:
    """Return ``{wiki, boa, inats, gbif, pnw}`` for a tax_id (any may be ``None``)."""
    data = load_data_summary()
    blank = {"wiki": None, "boa": None, "inats": None, "gbif": None, "pnw": None}
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
        "inats": row.get("inats") if isinstance(row.get("inats"), dict) else None,
        "gbif": row.get("gbif") if isinstance(row.get("gbif"), dict) else None,
        "pnw": row.get("pnw") if isinstance(row.get("pnw"), dict) else None,
    }


def wikipedia_url(title: str | None) -> str | None:
    """Build an en.wikipedia.org article URL from a page title / binomial."""
    name = (title or "").strip()
    if not name:
        return None
    return "https://en.wikipedia.org/wiki/" + quote(name.replace(" ", "_"))


def _wiki_list_by_id() -> dict[str, dict]:
    """Cached ``{tax_id: row}`` from ``wiki_list.json`` found entries."""
    path = get_data_dir() / "wiki_list.json"
    key = str(path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    cache = _WIKI_LIST_CACHE
    if cache["path"] == key and cache["mtime"] == mtime and cache["by_id"] is not None:
        return cache["by_id"]
    by_id: dict[str, dict] = {}
    if mtime is not None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            raw = None
        if isinstance(raw, dict):
            found = raw.get("found")
            if isinstance(found, list):
                for row in found:
                    if not isinstance(row, dict):
                        continue
                    tid = str(row.get("id") or "").strip()
                    if tid:
                        by_id[tid] = row
    cache.update(path=key, mtime=mtime, by_id=by_id)
    return by_id


def _wiki_list_entry(tax_id: str) -> dict | None:
    """Return the ``wiki_list.json`` found-row for ``tax_id``, if any."""
    return _wiki_list_by_id().get(str(tax_id))


def wiki_row_from_summary(summary: dict | None, tax_id: str | None = None) -> dict:
    """Flatten a per-tax wiki summary into browse / poses display fields."""
    summary = summary or {}
    have_wiki = bool(summary.get("have_wiki"))
    iucn = summary.get("iucn") if isinstance(summary.get("iucn"), dict) else None
    tnc = summary.get("tnc") if isinstance(summary.get("tnc"), dict) else None
    sci = (summary.get("scientific_name") or summary.get("species") or "").strip()
    matches = summary.get("name_matches_inats")
    if matches is not None:
        matches = bool(matches)
    article_url = None
    if have_wiki and tax_id is not None:
        listed = _wiki_list_entry(tax_id)
        if listed:
            article_url = (listed.get("url") or "").strip() or None
            if not sci:
                sci = (listed.get("title") or "").strip()
    if have_wiki and not article_url:
        article_url = wikipedia_url(sci)
    iucn_status = ((iucn or {}).get("status") or "").strip()
    tnc_status = ((tnc or {}).get("status") or "").strip()
    return {
        "have_wiki": have_wiki,
        "articles": 1 if have_wiki else 0,
        "scientific_name": sci,
        "authority": (summary.get("authority") or "").strip(),
        "article_url": article_url,
        "have_speciesbox": bool(summary.get("have_speciesbox")),
        "have_automatic_taxobox": bool(summary.get("have_automatic_taxobox")),
        "name_matches_inats": matches,
        "article_kind": (summary.get("article_kind") or "").strip(),
        "iucn_status": iucn_status,
        "iucn_url": (iucn or {}).get("url") or None,
        "has_iucn": bool(iucn_status),
        "tnc_status": tnc_status,
        "tnc_url": (tnc or {}).get("url") or None,
        "has_tnc": bool(tnc_status),
    }


def wiki_articles_tooltip(node: dict | None) -> str:
    """Build the browse Wikipedia-column tooltip from tax_summary kind counts."""
    node = node or {}
    total = int(node.get("total") or 0)
    if total <= 0:
        return ""
    lines = ["of which:"]
    for key, label in _ARTICLE_KIND_TOOLTIP_LINES:
        n = int(node.get(key) or 0)
        if n <= 0:
            continue
        pct = int(round(100.0 * n / total))
        lines.append(f"{n} ({pct}%) - {label}")
    return "\n".join(lines) if len(lines) > 1 else ""


def wiki_node_counts(node: dict | None) -> dict:
    """Return article / IUCN / TNC counts plus a Wikipedia-column tooltip."""
    node = node or {}
    return {
        "articles": int(node.get("total") or 0),
        "iucn": int(node.get("with_iucn") or 0),
        "tnc": int(node.get("with_tnc") or 0),
        "matches_name": int(node.get("matches_name") or 0),
        "known_synonym": int(node.get("known_synonym") or 0),
        "unknown_synonym": int(node.get("unknown_synonym") or 0),
        "higher_rank": int(node.get("higher_rank") or 0),
        "unknown_type": int(node.get("unknown_type") or 0),
        "articles_tooltip": wiki_articles_tooltip(node),
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
        "name": ((boa or {}).get("name") or "").strip(),
        "subspecies": int((boa or {}).get("subspecies") or 0) if boa else 0,
        "location": ((boa or {}).get("location") or "").strip(),
    }


def pnw_node_counts(node: dict | None) -> dict:
    """Return ``{species}`` = count of species with a PNW Moths page under ``node``."""
    node = node or {}
    pnw = node.get("pnw") if isinstance(node.get("pnw"), dict) else {}
    return {"species": int(pnw.get("total") or 0)}


def pnw_row_from_data(pnw: dict | None) -> dict:
    """Flatten a per-tax ``pnw`` object for genus-table / species display."""
    pnw = pnw if isinstance(pnw, dict) else None
    url = ((pnw or {}).get("url") or "").strip() or None
    return {
        "has_page": bool(url),
        "url": url,
        "name": ((pnw or {}).get("name") or "").strip(),
        "common_name": ((pnw or {}).get("common_name") or "").strip(),
    }


def inats_node_counts(node: dict | None) -> dict:
    """Return ``{species, observations}`` roll-ups for a tax_summary node."""
    node = node or {}
    inats = node.get("inats") if isinstance(node.get("inats"), dict) else {}
    return {
        "species": int(inats.get("total") or 0),
        "observations": int(inats.get("observations") or 0),
    }


def inats_row_from_data(inats: dict | None) -> dict:
    """Flatten a per-tax ``inats`` object for genus-table / poses display."""
    inats = inats if isinstance(inats, dict) else None
    obs = (inats or {}).get("observations_count")
    try:
        obs_n = int(obs) if obs is not None else 0
    except (TypeError, ValueError):
        obs_n = 0
    return {
        "has_taxon": inats is not None,
        "observations_count": obs_n,
        "preferred_common_name": ((inats or {}).get("preferred_common_name") or "").strip(),
        "is_active": bool((inats or {}).get("is_active")) if inats else False,
        "extinct": bool((inats or {}).get("extinct")) if inats else False,
        "vision": bool((inats or {}).get("vision")) if inats else False,
        "wikipedia_url": (inats or {}).get("wikipedia_url") or None,
        "name": ((inats or {}).get("name") or "").strip(),
    }


def gbif_species_tooltip(node: dict | None) -> str:
    """Build the browse GBIF-species-column tooltip from accepted/synonym counts."""
    node = node or {}
    total = int(node.get("total") or 0)
    if total <= 0:
        return ""
    lines = ["of which:"]
    for key, label in _GBIF_STATUS_TOOLTIP_LINES:
        n = int(node.get(key) or 0)
        if n <= 0:
            continue
        pct = int(round(100.0 * n / total))
        lines.append(f"{n} ({pct}%) - {label}")
    return "\n".join(lines) if len(lines) > 1 else ""


def gbif_node_counts(node: dict | None) -> dict:
    """Return species / obs roll-ups plus a GBIF-species-column tooltip."""
    node = node or {}
    gbif = node.get("gbif") if isinstance(node.get("gbif"), dict) else {}
    return {
        "species": int(gbif.get("total") or 0),
        "accepted": int(gbif.get("accepted") or 0),
        "synonym": int(gbif.get("synonym") or 0),
        "observations": int(gbif.get("observations") or 0),
        "species_tooltip": gbif_species_tooltip(gbif),
    }


def gbif_row_from_data(gbif: dict | None) -> dict:
    """Flatten a per-tax ``gbif`` object for genus-table / species display."""
    gbif = gbif if isinstance(gbif, dict) else None
    usage_key = (gbif or {}).get("usageKey")
    url = ((gbif or {}).get("url") or "").strip() or None
    obs = (gbif or {}).get("observations_count")
    try:
        obs_n = int(obs) if obs is not None else 0
    except (TypeError, ValueError):
        obs_n = 0

    def _pair(full: str, canonical: str, authorship: str) -> tuple[str, str]:
        full = (full or "").strip()
        canonical = (canonical or "").strip()
        authorship = (authorship or "").strip()
        if not authorship and full and canonical and full.startswith(canonical):
            authorship = full[len(canonical) :].strip()
        if not canonical and full:
            canonical = full
            authorship = ""
        return canonical, authorship

    canonical, authorship = _pair(
        (gbif or {}).get("scientificName") or "",
        (gbif or {}).get("canonicalName") or "",
        (gbif or {}).get("authorship") or "",
    )
    accepted_canonical, accepted_authorship = _pair(
        (gbif or {}).get("acceptedScientificName") or "",
        (gbif or {}).get("acceptedCanonicalName") or "",
        (gbif or {}).get("acceptedAuthorship") or "",
    )
    return {
        "has_taxon": gbif is not None and usage_key is not None,
        "usageKey": usage_key,
        "matchedUsageKey": (gbif or {}).get("matchedUsageKey"),
        "url": url,
        "accepted": bool((gbif or {}).get("accepted")) if gbif else False,
        "status": ((gbif or {}).get("status") or "").strip(),
        "scientificName": ((gbif or {}).get("scientificName") or "").strip(),
        "canonicalName": canonical,
        "authorship": authorship,
        "acceptedScientificName": (
            ((gbif or {}).get("acceptedScientificName") or "").strip()
        ),
        "acceptedCanonicalName": accepted_canonical,
        "acceptedAuthorship": accepted_authorship,
        "observations_count": obs_n,
        "last_observation_date": (
            ((gbif or {}).get("last_observation_date") or "").strip() or None
        ),
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


def _normalize_taxon_name(name: str) -> str:
    return " ".join((name or "").casefold().split())


def _source_name_cell(
    *,
    name: str,
    authority: str = "",
    url: str | None = None,
    ref_name: str = "",
) -> dict:
    """Build one flat-taxonomy source cell (name link / same / dash)."""
    name = (name or "").strip()
    authority = (authority or "").strip()
    url = (url or "").strip() or None
    if not name:
        return {"kind": "unknown", "text": "—", "url": None, "tooltip": ""}
    tooltip = f"{name} {authority}".strip() if authority else name
    if ref_name and _normalize_taxon_name(name) == _normalize_taxon_name(ref_name):
        return {
            "kind": "same",
            "text": "same",
            "url": url,
            "tooltip": tooltip,
        }
    return {
        "kind": "name",
        "text": name,
        "url": url,
        "tooltip": tooltip,
    }


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

        inats = row.get("inats") if isinstance(row.get("inats"), dict) else None
        wiki = row.get("wiki") if isinstance(row.get("wiki"), dict) else None
        boa = row.get("boa") if isinstance(row.get("boa"), dict) else None
        pnw = row.get("pnw") if isinstance(row.get("pnw"), dict) else None
        gbif = row.get("gbif") if isinstance(row.get("gbif"), dict) else None

        inat_name = ((inats or {}).get("name") or "").strip()
        species_name = (row.get("species") or "").strip() or inat_name
        inat_url = (
            f"https://www.inaturalist.org/taxa/{tax_id}" if inats is not None else None
        )

        wiki_name = ((wiki or {}).get("scientific_name") or "").strip()
        wiki_auth = ((wiki or {}).get("authority") or "").strip()
        wiki_url = None
        if wiki is not None:
            listed = _wiki_list_entry(tax_id)
            if listed:
                wiki_url = (listed.get("url") or "").strip() or None
            if not wiki_url:
                wiki_url = wikipedia_url(wiki_name)

        boa_name = ((boa or {}).get("name") or "").strip()
        boa_url = ((boa or {}).get("url") or "").strip() or None

        pnw_name = ((pnw or {}).get("name") or "").strip()
        pnw_common = ((pnw or {}).get("common_name") or "").strip()
        pnw_url = ((pnw or {}).get("url") or "").strip() or None

        gbif_name = (
            ((gbif or {}).get("acceptedCanonicalName") or "").strip()
            or ((gbif or {}).get("canonicalName") or "").strip()
        )
        gbif_auth = (
            ((gbif or {}).get("acceptedAuthorship") or "").strip()
            or ((gbif or {}).get("authorship") or "").strip()
        )
        if not gbif_auth:
            full = (
                ((gbif or {}).get("acceptedScientificName") or "").strip()
                or ((gbif or {}).get("scientificName") or "").strip()
            )
            if gbif_name and full.startswith(gbif_name):
                gbif_auth = full[len(gbif_name) :].strip()
        gbif_url = ((gbif or {}).get("url") or "").strip() or None

        rows.append(
            {
                "kind": "species",
                "tax_id": tax_id,
                "species": species_name,
                "inats": _source_name_cell(
                    name=inat_name if inats is not None else "",
                    url=inat_url,
                    ref_name=species_name,
                ),
                "wiki": _source_name_cell(
                    name=wiki_name if wiki is not None else "",
                    authority=wiki_auth,
                    url=wiki_url,
                    ref_name=species_name,
                ),
                "boa": _source_name_cell(
                    name=boa_name if boa is not None else "",
                    url=boa_url,
                    ref_name=species_name,
                ),
                "pnw": _source_name_cell(
                    name=pnw_name if pnw is not None else "",
                    authority=pnw_common,
                    url=pnw_url,
                    ref_name=species_name,
                ),
                "gbif": _source_name_cell(
                    name=gbif_name if gbif is not None else "",
                    authority=gbif_auth,
                    url=gbif_url,
                    ref_name=species_name,
                ),
            }
        )
    return rows
