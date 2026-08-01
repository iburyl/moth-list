"""Template helpers for showing tax_ids as friendly species names.

``load_names`` maps each tax_id to its family/species/name (from the configured
names CSV). Wherever a tax_id is shown we display the species text, with the
full ``{name} ({id})`` available on hover.
"""

from django import template
from django.utils.html import format_html

from ..utils import get_name_info

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
