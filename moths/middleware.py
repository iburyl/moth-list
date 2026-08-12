"""Temporary site-wide login gate.

While the project isn't public-ready, every page redirects anonymous users to
the login screen, so web crawlers (and casual visitors) can't see in-progress
interface decisions. This is deliberately coarse and meant to be removed later:
it is controlled by ``settings.REQUIRE_LOGIN`` (env ``DJANGO_REQUIRE_LOGIN``),
and when that is off the middleware disables itself entirely, leaving the normal
public-view / login-to-edit split (Phase B) in force.
"""

from django.conf import settings
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import MiddlewareNotUsed
from django.shortcuts import resolve_url
from django.urls import NoReverseMatch, reverse


class RequireLoginMiddleware:
    """Redirect all anonymous requests to the login page (except login/static).

    Disables itself (``MiddlewareNotUsed``) when ``REQUIRE_LOGIN`` is falsey so
    the gate can be lifted with a single env var once the site is public.
    """

    def __init__(self, get_response):
        if not getattr(settings, "REQUIRE_LOGIN", False):
            raise MiddlewareNotUsed
        self.get_response = get_response
        # Reachable while logged out: the login/logout endpoints (so you can
        # actually authenticate) and static assets.
        exempt = []
        for name in ("login", "logout"):
            try:
                exempt.append(reverse(name))
            except NoReverseMatch:
                pass
        static_url = settings.STATIC_URL or "/static/"
        if not static_url.startswith("/"):
            static_url = "/" + static_url
        exempt.append(static_url)
        self._exempt = tuple(exempt)

    def __call__(self, request):
        if not request.user.is_authenticated and not self._is_exempt(request.path):
            return redirect_to_login(
                request.get_full_path(), resolve_url(settings.LOGIN_URL)
            )
        return self.get_response(request)

    def _is_exempt(self, path):
        return any(path == p or path.startswith(p) for p in self._exempt)
