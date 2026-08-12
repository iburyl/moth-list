"""Edit-permission policy for the public-view / login-to-edit split.

Viewing is public; every mutating endpoint is gated by :func:`user_can_edit`,
and the templates hide their edit controls behind the same predicate (exposed
as ``can_edit`` by :mod:`moths.context_processors`).

Right now "may edit" means simply "is an authenticated Django user" (log in via
the ``/rule/`` admin, or ``/accounts/login/``). This single function is the one
place to broaden the policy later — e.g. an iNaturalist-OAuth allowlist — without
touching the views or templates.
"""

from functools import wraps

from django.http import JsonResponse


def user_can_edit(user) -> bool:
    """Return whether ``user`` is allowed to perform editing/mutating actions."""
    return bool(user is not None and user.is_authenticated)


def editor_required(view):
    """Gate a mutating view, returning **403 JSON** unless the user may edit.

    A JSON 403 (rather than Django's usual login *redirect*) is what the
    fetch-based editing endpoints expect: a clean, non-redirect failure the
    client can surface without following a 302 to an HTML login page.
    """

    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not user_can_edit(getattr(request, "user", None)):
            return JsonResponse({"error": "authentication required"}, status=403)
        return view(request, *args, **kwargs)

    return wrapped
