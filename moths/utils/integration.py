"""Load and render the human-edited external-data integration layout.

``MOTHS_DATA_DIR/integration.json`` controls which fields from
``data_summary.json`` and ``tax_summary.json`` appear in the four reference-data
surfaces. Paths in the file are relative to a ``sources`` map; ``tax_id`` is
supplied as row context. The renderer deliberately supports only a small,
non-executable vocabulary.
"""
from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings


INTEGRATION_SCHEMA = 1
_CACHE: dict = {"path": None, "mtime": None, "data": None}


def get_integration_path() -> Path:
    """Return ``MOTHS_DATA_DIR/integration.json`` (the only valid location)."""
    return Path(settings.MOTHS_DATA_DIR) / "integration.json"


def load_integration() -> dict | None:
    """Load schema-1 integration config, cached by path and mtime."""
    path = get_integration_path()
    if not path.is_file():
        _CACHE.update(path=None, mtime=None, data=None)
        return None
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    key = str(path)
    if _CACHE["path"] == key and _CACHE["mtime"] == mtime:
        return _CACHE["data"]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = None
    data = (
        raw
        if isinstance(raw, dict) and raw.get("schema") == INTEGRATION_SCHEMA
        else None
    )
    _CACHE.update(path=key, mtime=mtime, data=data)
    return data


def integration_layout(view_name: str) -> list[dict]:
    """Return safe group metadata used to build table headers."""
    config = load_integration() or {}
    groups = config.get(view_name)
    if not isinstance(groups, list):
        return []
    out = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        items = group.get("items")
        if not isinstance(items, list):
            # ``flat`` entries are single cells rather than grouped items.
            items = [group] if view_name == "flat" else []
        safe_items = [
            {"name": str(item.get("name") or "")}
            for item in items
            if isinstance(item, dict)
        ]
        if safe_items:
            out.append(
                {
                    "id": str(group.get("id") or ""),
                    "name": str(group.get("name") or ""),
                    "items": safe_items,
                }
            )
    return out


def _value(sources: dict, path, context: dict):
    if not isinstance(path, list) or not path:
        return None
    first = path[0]
    if first in context:
        value = context.get(first)
        remaining = path[1:]
    else:
        value = sources
        remaining = path
    for key in remaining:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _condition_matches(spec: dict, sources: dict, context: dict) -> bool:
    show_if = spec.get("show_if")
    if show_if is not None and not bool(_value(sources, show_if, context)):
        return False
    show_when = spec.get("show_when")
    if isinstance(show_when, dict):
        actual = _value(sources, show_when.get("data"), context)
        expected = show_when.get("equals")
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.casefold() == expected.casefold()
        return actual == expected
    return True


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def _render_item(
    item: dict,
    sources: dict,
    context: dict,
    reference_name: str = "",
    preserve_hidden: bool = False,
) -> dict | None:
    if not _condition_matches(item, sources, context):
        if not preserve_hidden:
            return None
        return {
            "name": str(item.get("name") or ""),
            "kind": str(item.get("kind") or "text"),
            "text": "",
            "missing": True,
        }
    kind = str(item.get("kind") or "text")
    cell = {"name": str(item.get("name") or ""), "kind": kind}

    if kind == "link":
        href = _text(_value(sources, item.get("href"), context))
        if "text_data" in item:
            text = _text(_value(sources, item.get("text_data"), context))
        else:
            text = _text(item.get("text"))
        cell.update(
            href=href or None,
            text=text,
            missing=not bool(text and (href or "text_data" in item)),
        )
        return cell

    value = _value(sources, item.get("data"), context)
    if kind == "count":
        try:
            number = int(value) if value is not None else None
        except (TypeError, ValueError):
            number = None
        cell.update(value=number, text="" if number is None else str(number), missing=number is None)
        return cell

    if kind == "yesno":
        cell.update(value=bool(value), text="yes" if value else "no", missing=False)
        return cell

    if kind == "name_compare":
        name = _text(value)
        href = _text(_value(sources, item.get("href"), context))
        same = bool(
            name
            and reference_name
            and " ".join(name.casefold().split())
            == " ".join(reference_name.casefold().split())
        )
        cell.update(
            text="same" if same else name,
            href=href or None,
            same=same,
            missing=not bool(name),
        )
        return cell

    text = _text(value)
    cell.update(value=value, text=text, missing=not bool(text))
    return cell


def build_integration_groups(
    view_name: str,
    sources: dict | None,
    *,
    context: dict | None = None,
    reference_name: str = "",
) -> list[dict]:
    """Build template-ready groups/cells for one configured view."""
    sources = sources if isinstance(sources, dict) else {}
    context = context if isinstance(context, dict) else {}
    config = load_integration() or {}
    groups = config.get(view_name)
    if not isinstance(groups, list):
        return []
    rendered = []
    preserve_hidden = view_name in ("browse", "species-list")
    for group in groups:
        if not isinstance(group, dict) or not _condition_matches(group, sources, context):
            continue
        specs = group.get("items")
        if not isinstance(specs, list):
            specs = [group] if view_name == "flat" else []
        items = []
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            cell = _render_item(
                spec, sources, context, reference_name, preserve_hidden
            )
            if cell is not None:
                items.append(cell)
        if items:
            rendered.append(
                {
                    "id": str(group.get("id") or ""),
                    "name": str(group.get("name") or ""),
                    "items": items,
                }
            )
    return rendered
