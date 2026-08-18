#!/usr/bin/env python3

"""Parse harvested Wikipedia articles into a single ``wiki_summary.json``.

Reads every ``MOTHS_DATA_DIR/<tax_id>/<tax_id>.wiki`` file (from
``harvest_wiki.py``) — only taxa that were harvested — and writes::

    MOTHS_DATA_DIR/wiki_summary.json

Each entry holds taxobox fields plus an ``article_kind`` classification
against the iNat main name / synonyms (from ``inats_summary.json`` and
``inats_synonyms_summary.json``):

* ``matches_name`` — Speciesbox taxon matches iNat main name
* ``known_synonym`` — Speciesbox taxon matches a known iNat synonym
* ``unknown_synonym`` — Speciesbox present, name matches neither
* ``higher_rank`` — no Speciesbox, but ``{{Automatic taxobox`` is present
* ``unknown_type`` — harvested article with neither template

Also stores ``have_speciesbox``, ``have_automatic_taxobox``,
``name_matches_inats``, scientific name, authority, IUCN, and TNC.

Run ``tools/parse_data_summary.py`` afterward to fold this into the
inats-scoped Django lookup files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent

WIKI_SUMMARY_VERSION = 2
WIKI_SUMMARY_NAME = "wiki_summary.json"
INATS_SUMMARY_NAME = "inats_summary.json"
INATS_SYNONYMS_SUMMARY_NAME = "inats_synonyms_summary.json"

ARTICLE_KIND_MATCHES_NAME = "matches_name"
ARTICLE_KIND_KNOWN_SYNONYM = "known_synonym"
ARTICLE_KIND_UNKNOWN_SYNONYM = "unknown_synonym"
ARTICLE_KIND_HIGHER_RANK = "higher_rank"
ARTICLE_KIND_UNKNOWN_TYPE = "unknown_type"

_WIKILINK_RE = re.compile(r"\[\[([^\]|]*)\|([^\]]+)\]\]|\[\[([^\]]+)\]\]")
_REF_RE = re.compile(r"<ref\b[^>]*/\s*>|<ref\b[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_ITALIC_RE = re.compile(r"'{2,}")
_WS_RE = re.compile(r"\s+")
_IUCN_ID_RE = re.compile(
    r"(?:e\.)?T(?P<taxon>\d+)A(?P<assessment>\d+)",
    re.IGNORECASE,
)
_IUCN_URL_RE = re.compile(
    r"iucnredlist\.org/species/(?P<taxon>\d+)(?:/(?P<assessment>\d+))?",
    re.IGNORECASE,
)
_IUCN_DOI_RE = re.compile(
    r"RLTS\.T(?P<taxon>\d+)A(?P<assessment>\d+)",
    re.IGNORECASE,
)
_NATURESERVE_ID_RE = re.compile(
    r"ELEMENT_GLOBAL\.(?P<id>\d+\.\d+)",
    re.IGNORECASE,
)
_NATURESERVE_ID_PARAM_RE = re.compile(
    r"\|\s*id\s*=\s*(?P<id>\d+\.\d+)",
    re.IGNORECASE,
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
            "Parse all harvested *.wiki articles into "
            "MOTHS_DATA_DIR/wiki_summary.json."
        )
    ).parse_args()


def normalize_name(name: str) -> str:
    return _WS_RE.sub(" ", (name or "").strip().casefold())


# --- Speciesbox / taxobox parsing --------------------------------------------


def _find_matching_braces(text: str, open_idx: int) -> int | None:
    depth = 0
    i = open_idx
    n = len(text)
    while i < n - 1:
        pair = text[i : i + 2]
        if pair == "{{":
            depth += 1
            i += 2
            continue
        if pair == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                return i
            continue
        i += 1
    return None


def find_template(wikitext: str, *names: str) -> str | None:
    """Return the first matching ``{{Name ...}}`` template, or ``None``."""
    for name in names:
        pattern = re.compile(rf"\{{\{{\s*({re.escape(name)})\b", re.IGNORECASE)
        match = pattern.search(wikitext)
        if not match:
            continue
        end = _find_matching_braces(wikitext, match.start())
        if end is None:
            continue
        return wikitext[match.start() : end]
    return None


def parse_template_params(template: str) -> dict[str, str]:
    inner = template.strip()
    if inner.startswith("{{"):
        inner = inner[2:]
    if inner.endswith("}}"):
        inner = inner[:-2]
    pipe = inner.find("|")
    body = inner[pipe + 1 :] if pipe >= 0 else ""

    chunks: list[str] = []
    buf: list[str] = []
    brace_depth = 0
    link_depth = 0
    i = 0
    n = len(body)
    while i < n:
        if body[i : i + 2] == "{{":
            brace_depth += 1
            buf.append("{{")
            i += 2
            continue
        if body[i : i + 2] == "}}":
            brace_depth = max(brace_depth - 1, 0)
            buf.append("}}")
            i += 2
            continue
        if body[i : i + 2] == "[[":
            link_depth += 1
            buf.append("[[")
            i += 2
            continue
        if body[i : i + 2] == "]]":
            link_depth = max(link_depth - 1, 0)
            buf.append("]]")
            i += 2
            continue
        if body[i] == "|" and brace_depth == 0 and link_depth == 0:
            chunks.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(body[i])
        i += 1
    if buf or chunks:
        chunks.append("".join(buf))

    params: dict[str, str] = {}
    for chunk in chunks:
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip().lower()
        if key:
            params[key] = value.strip()
    return params


def short_authority(raw: str | None) -> str:
    if not raw:
        return ""
    text = _HTML_COMMENT_RE.sub("", raw)
    text = _REF_RE.sub("", text)
    text = _WIKILINK_RE.sub(
        lambda m: (m.group(2) or m.group(3) or "").strip(),
        text,
    )
    text = _ITALIC_RE.sub("", text)
    text = text.replace("&nbsp;", " ")
    return _WS_RE.sub(" ", text).strip(" ,;")


def scientific_name_from_params(params: dict[str, str]) -> str:
    for key in ("taxon", "tax"):
        value = (params.get(key) or "").strip()
        if value:
            return _WS_RE.sub(" ", _ITALIC_RE.sub("", value)).strip()
    genus = (params.get("genus") or "").strip()
    species = (params.get("species") or "").strip()
    if genus and species:
        parts = [genus, species]
        subsp = (
            params.get("subspecies")
            or params.get("subsp")
            or params.get("variety")
            or ""
        ).strip()
        if subsp:
            parts.append(subsp)
        return _WS_RE.sub(" ", _ITALIC_RE.sub("", " ".join(parts))).strip()
    return ""


def _inner_templates(text: str) -> list[str]:
    out: list[str] = []
    i = 0
    while True:
        start = text.find("{{", i)
        if start < 0:
            break
        end = _find_matching_braces(text, start)
        if end is None:
            break
        out.append(text[start:end])
        i = end
    return out


def _parse_iucn_ids(ref: str) -> tuple[str | None, str | None]:
    if not ref:
        return None, None
    for pattern in (_IUCN_URL_RE, _IUCN_ID_RE, _IUCN_DOI_RE):
        match = pattern.search(ref)
        if match:
            return match.group("taxon"), match.groupdict().get("assessment")
    return None, None


def _parse_natureserve_id(ref: str) -> str | None:
    if not ref:
        return None
    match = _NATURESERVE_ID_RE.search(ref)
    if match:
        return match.group("id")
    match = _NATURESERVE_ID_PARAM_RE.search(ref)
    if match:
        return match.group("id")
    for tpl in _inner_templates(ref):
        if re.match(r"\{\{\s*Cite\s*NatureServe\b", tpl, re.IGNORECASE):
            params = parse_template_params(tpl)
            nid = (params.get("id") or "").strip()
            if re.fullmatch(r"\d+\.\d+", nid):
                return nid
    return None


def _iucn_url(taxon_id: str | None, assessment_id: str | None, sci_name: str) -> str:
    if taxon_id and assessment_id:
        return f"https://www.iucnredlist.org/species/{taxon_id}/{assessment_id}"
    if taxon_id:
        return f"https://www.iucnredlist.org/species/{taxon_id}"
    if sci_name:
        return f"https://www.iucnredlist.org/search?query={quote(sci_name)}"
    return "https://www.iucnredlist.org/"


def _natureserve_url(ns_id: str | None, sci_name: str) -> str:
    if ns_id:
        slug = sci_name.replace(" ", "_") if sci_name else ""
        base = f"https://explorer.natureserve.org/Taxon/ELEMENT_GLOBAL.{ns_id}"
        return f"{base}/{slug}" if slug else base
    if sci_name:
        return "https://explorer.natureserve.org/Search#q=" + quote(sci_name)
    return "https://explorer.natureserve.org/"


def _is_iucn_system(system: str) -> bool:
    return system.upper().startswith("IUCN")


def _is_tnc_system(system: str) -> bool:
    return system.upper() in {"TNC", "NATURESERVE"}


def extract_status_block(
    params: dict[str, str],
    *,
    status_key: str,
    system_key: str,
    ref_key: str,
    sci_name: str,
) -> dict[str, Any] | None:
    status = (params.get(status_key) or "").strip()
    system = (params.get(system_key) or "").strip()
    ref = (params.get(ref_key) or "").strip()
    if not status and not system:
        return None

    if _is_iucn_system(system) or (
        status
        and not system
        and status.upper()
        in {
            "EX",
            "EW",
            "CR",
            "EN",
            "VU",
            "NT",
            "LC",
            "DD",
            "NE",
            "PE",
            "PEW",
            "LR/CD",
            "LR/NT",
            "LR/LC",
        }
    ):
        taxon_id, assessment_id = _parse_iucn_ids(ref)
        return {
            "kind": "iucn",
            "status": status,
            "system": system or "IUCN",
            "taxon_id": taxon_id,
            "assessment_id": assessment_id,
            "url": _iucn_url(taxon_id, assessment_id, sci_name),
        }

    if _is_tnc_system(system) or (
        status
        and not system
        and re.fullmatch(r"[GT][XH1-5U][A-Z0-9]*", status.upper())
    ):
        ns_id = _parse_natureserve_id(ref)
        return {
            "kind": "tnc",
            "status": status,
            "system": system or "TNC",
            "id": ns_id,
            "url": _natureserve_url(ns_id, sci_name),
        }

    return {
        "kind": "other",
        "status": status,
        "system": system,
        "url": None,
    }


def extract_status_fields(
    params: dict[str, str], sci_name: str
) -> tuple[dict | None, dict | None]:
    iucn = None
    tnc = None
    for status_key, system_key, ref_key in (
        ("status", "status_system", "status_ref"),
        ("status2", "status2_system", "status2_ref"),
    ):
        block = extract_status_block(
            params,
            status_key=status_key,
            system_key=system_key,
            ref_key=ref_key,
            sci_name=sci_name,
        )
        if block is None:
            continue
        if block["kind"] == "iucn" and iucn is None:
            iucn = {
                "status": block["status"],
                "system": block["system"],
                "taxon_id": block["taxon_id"],
                "assessment_id": block["assessment_id"],
                "url": block["url"],
            }
        elif block["kind"] == "tnc" and tnc is None:
            tnc = {
                "status": block["status"],
                "system": block["system"],
                "id": block["id"],
                "url": block["url"],
            }
    return iucn, tnc


def parse_speciesbox(wikitext: str) -> dict[str, Any] | None:
    template = find_template(wikitext, "Speciesbox", "Species box")
    if template is None:
        return None
    params = parse_template_params(template)
    sci = scientific_name_from_params(params)
    authority = short_authority(params.get("authority"))
    iucn, tnc = extract_status_fields(params, sci)
    return {
        "scientific_name": sci,
        "authority": authority,
        "have_speciesbox": True,
        "iucn": iucn,
        "tnc": tnc,
    }


def has_automatic_taxobox(wikitext: str) -> bool:
    return find_template(wikitext, "Automatic taxobox") is not None


def load_inat_name_index(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Return ``{tax_id: {main, synonyms: set[str]}}`` (normalized names)."""
    out: dict[str, dict[str, Any]] = {}
    summary = None
    syn_summary = None
    try:
        summary = json.loads((data_dir / INATS_SUMMARY_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        summary = None
    try:
        syn_summary = json.loads(
            (data_dir / INATS_SYNONYMS_SUMMARY_NAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError):
        syn_summary = None

    by_tax = summary.get("by_tax_id") if isinstance(summary, dict) else None
    if isinstance(by_tax, dict):
        for tax_id, entry in by_tax.items():
            if not isinstance(entry, dict):
                continue
            main = normalize_name(entry.get("name") or "")
            out[str(tax_id)] = {"main": main, "synonyms": set()}

    syn_by_tax = syn_summary.get("by_tax_id") if isinstance(syn_summary, dict) else None
    if isinstance(syn_by_tax, dict):
        for tax_id, entry in syn_by_tax.items():
            if not isinstance(entry, dict):
                continue
            bucket = out.setdefault(str(tax_id), {"main": "", "synonyms": set()})
            current = normalize_name(entry.get("current_name") or "")
            if current and not bucket["main"]:
                bucket["main"] = current
            for name in entry.get("synonyms") or []:
                key = normalize_name(name)
                if key and key != bucket["main"]:
                    bucket["synonyms"].add(key)
    return out


def classify_article(
    *,
    have_speciesbox: bool,
    have_automatic_taxobox: bool,
    scientific_name: str,
    inat_main: str,
    inat_synonyms: set[str],
) -> tuple[str, bool | None]:
    """Return ``(article_kind, name_matches_inats)``."""
    if have_speciesbox:
        sci_key = normalize_name(scientific_name)
        if sci_key and inat_main and sci_key == inat_main:
            return ARTICLE_KIND_MATCHES_NAME, True
        if sci_key and sci_key in inat_synonyms:
            return ARTICLE_KIND_KNOWN_SYNONYM, False
        return ARTICLE_KIND_UNKNOWN_SYNONYM, False if sci_key else None
    if have_automatic_taxobox:
        return ARTICLE_KIND_HIGHER_RANK, None
    return ARTICLE_KIND_UNKNOWN_TYPE, None


def parse_wiki_article(
    wikitext: str,
    *,
    inat_main: str = "",
    inat_synonyms: set[str] | None = None,
) -> dict[str, Any]:
    """Parse one harvested article into a wiki_summary species entry."""
    synonyms = inat_synonyms or set()
    if not wikitext.strip():
        kind, matches = classify_article(
            have_speciesbox=False,
            have_automatic_taxobox=False,
            scientific_name="",
            inat_main=inat_main,
            inat_synonyms=synonyms,
        )
        return {
            "scientific_name": "",
            "authority": "",
            "have_speciesbox": False,
            "have_automatic_taxobox": False,
            "name_matches_inats": matches,
            "article_kind": kind,
            "iucn": None,
            "tnc": None,
        }

    speciesbox = parse_speciesbox(wikitext)
    have_speciesbox = speciesbox is not None
    have_auto = False if have_speciesbox else has_automatic_taxobox(wikitext)
    sci = (speciesbox or {}).get("scientific_name") or ""
    authority = (speciesbox or {}).get("authority") or ""
    iucn = (speciesbox or {}).get("iucn")
    tnc = (speciesbox or {}).get("tnc")
    kind, matches = classify_article(
        have_speciesbox=have_speciesbox,
        have_automatic_taxobox=have_auto,
        scientific_name=sci,
        inat_main=inat_main,
        inat_synonyms=synonyms,
    )
    return {
        "scientific_name": sci,
        "authority": authority,
        "have_speciesbox": have_speciesbox,
        "have_automatic_taxobox": have_auto,
        "name_matches_inats": matches,
        "article_kind": kind,
        "iucn": iucn,
        "tnc": tnc,
    }


def iter_wiki_files(data_dir: Path):
    """Yield ``(tax_id, path)`` for every harvested ``<tax>/<tax>.wiki`` file."""
    if not data_dir.is_dir():
        return
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        tax_id = child.name
        path = child / f"{tax_id}.wiki"
        if path.is_file():
            yield tax_id, path


def main() -> int:
    parse_args()
    settings = bootstrap_django()
    data_dir = Path(settings.MOTHS_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    name_index = load_inat_name_index(data_dir)
    wiki_files = list(iter_wiki_files(data_dir))
    total = len(wiki_files)
    species: dict[str, dict] = {}
    counts = {
        "articles": 0,
        "speciesbox": 0,
        "automatic_taxobox": 0,
        "matches_name": 0,
        "known_synonym": 0,
        "unknown_synonym": 0,
        "higher_rank": 0,
        "unknown_type": 0,
        "iucn": 0,
        "tnc": 0,
    }

    if total == 0:
        print("No harvested *.wiki files found.")
    else:
        print(f"Parsing {total} Wikipedia articles…")

    for index, (tax_id, path) in enumerate(wiki_files, start=1):
        print(f"\r[{index} of {total}] {tax_id} ", end="", flush=True)
        try:
            wikitext = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"\n  ERROR {tax_id}: {exc}", file=sys.stderr)
            continue
        counts["articles"] += 1
        names = name_index.get(tax_id) or {"main": "", "synonyms": set()}
        entry = parse_wiki_article(
            wikitext,
            inat_main=names["main"],
            inat_synonyms=names["synonyms"],
        )
        species[tax_id] = entry
        kind = entry["article_kind"]
        counts[kind] = counts.get(kind, 0) + 1
        if entry.get("have_speciesbox"):
            counts["speciesbox"] += 1
        if entry.get("have_automatic_taxobox"):
            counts["automatic_taxobox"] += 1
        if entry.get("iucn"):
            counts["iucn"] += 1
        if entry.get("tnc"):
            counts["tnc"] += 1

    if total:
        print()

    payload = {
        "version": WIKI_SUMMARY_VERSION,
        "species": species,
    }
    out = data_dir / WIKI_SUMMARY_NAME
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {out}: articles={counts['articles']} "
        f"speciesbox={counts['speciesbox']} "
        f"automatic_taxobox={counts['automatic_taxobox']} "
        f"matches_name={counts['matches_name']} "
        f"known_synonym={counts['known_synonym']} "
        f"unknown_synonym={counts['unknown_synonym']} "
        f"higher_rank={counts['higher_rank']} "
        f"unknown_type={counts['unknown_type']} "
        f"iucn={counts['iucn']} tnc={counts['tnc']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
