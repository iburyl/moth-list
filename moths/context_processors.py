"""Template context processors for the moths app."""

from .permissions import user_can_edit


def edit_permissions(request):
    """Expose ``can_edit`` so templates can hide edit controls from viewers."""
    return {"can_edit": user_can_edit(getattr(request, "user", None))}
