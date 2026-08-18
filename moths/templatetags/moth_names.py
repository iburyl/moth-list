"""Template helpers for showing tax_ids as friendly species names.

``get_name_info`` reads lineage from ``data_summary.json``. Wherever a tax_id
is shown we display the species text, with the full ``{name} ({id})`` available
on hover.
"""

from django import template
from django.urls import reverse
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from ..utils import _NO_SUBFAMILY, _UNKNOWN_TAXON, _tax_lineage_keys, get_name_info

register = template.Library()


def _label(tax_id) -> str:
    """Species text for a tax_id, falling back to the raw id when unknown."""
    info = get_name_info(tax_id)
    return info["species"] or str(tax_id)


def _title(tax_id) -> str:
    """Hover text: ``{name} ({id})`` (or just the id when the name is unknown)."""
    info = get_name_info(tax_id)
    return f"{info['name']} ({tax_id})" if info["name"] else str(tax_id)


@register.simple_tag
def tax_species(tax_id) -> str:
    """Return the species label for a tax_id (id fallback)."""
    return _label(tax_id)


@register.simple_tag
def tax_family(tax_id) -> str:
    """Return the family for a tax_id (empty when unknown)."""
    return get_name_info(tax_id)["family"]


@register.simple_tag
def tax_title(tax_id) -> str:
    """Return the ``{name} ({id})`` hover text for a tax_id."""
    return _title(tax_id)


@register.simple_tag
def tax_name(tax_id):
    """Render a tax_id as ``<span title="{name} ({id})">species</span>``."""
    return format_html(
        '<span class="tax-name" title="{}">{}</span>',
        _title(tax_id),
        _label(tax_id),
    )


def _browse_seg(key: str) -> str:
    """Tree key -> browse URL segment (bucket keys travel as a bare "_")."""
    return "_" if key in (_UNKNOWN_TAXON, _NO_SUBFAMILY) else key


@register.simple_tag
def tax_breadcrumb(tax_id):
    """Render a linked taxonomy breadcrumb for a tax_id.

    ``All / {superfamily} / {family} [/ {subfamily}] / {Genus} epithet`` where
    "All" and each rank link to their browse page (subfamily is shown only when
    known) and the trailing epithet (species name minus its genus) is plain
    italic text.
    """
    info = get_name_info(tax_id)
    sf, fam, subf, genus = _tax_lineage_keys(info)
    segs = [_browse_seg(sf), _browse_seg(fam), _browse_seg(subf), _browse_seg(genus)]

    def burl(depth):
        return reverse("moths:browse", args=["/".join(segs[:depth])])

    parts = [
        format_html('<a href="{}">All</a>', reverse("moths:browse")),
        format_html('<a href="{}">{}</a>', burl(1), sf),
        format_html('<a href="{}">{}</a>', burl(2), fam),
    ]
    if subf != _NO_SUBFAMILY:
        parts.append(format_html('<a href="{}">{}</a>', burl(3), subf))

    species = (info.get("species") or "").strip()
    epithet = " ".join(species.split()[1:]) if len(species.split()) > 1 else ""
    genus_link = format_html('<a class="sci" href="{}" title="{}">{}</a>', burl(4), _title(tax_id), genus)
    if epithet:
        parts.append(format_html('{} <span class="sci">{}</span>', genus_link, epithet))
    else:
        parts.append(genus_link)

    return mark_safe(" <span class=\"crumb-sep\">/</span> ".join(parts))
