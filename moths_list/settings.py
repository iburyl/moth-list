"""
Django settings for moths_list project.

Generated for Django 5.x. For more information, see
https://docs.djangoproject.com/en/5.1/topics/settings/
"""

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

# Build paths inside the project like this: BASE_DIR / "subdir".
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-change-me-in-production",
)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() == "true"

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(",") if os.environ.get("DJANGO_ALLOWED_HOSTS") else []

# Comma-separated https origins to trust for CSRF (needed for the admin login
# POST when running behind a cloud HTTPS load balancer), e.g.
# "https://moths.example.com". Empty in local dev.
CSRF_TRUSTED_ORIGINS = [
    o.strip()
    for o in os.environ.get("DJANGO_CSRF_TRUSTED_ORIGINS", "").split(",")
    if o.strip()
]

# When the container sits behind an HTTPS-terminating proxy, trust the
# forwarded-proto header so Django recognises the original scheme as HTTPS.
if os.environ.get("DJANGO_BEHIND_PROXY", "").lower() in ("1", "true", "yes"):
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "moths",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    # Temporary site-wide login gate (see REQUIRE_LOGIN below); disables itself
    # when the setting is off.
    "moths.middleware.RequireLoginMiddleware",
]

ROOT_URLCONF = "moths_list.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "moths.context_processors.edit_permissions",
            ],
        },
    },
]

WSGI_APPLICATION = "moths_list.wsgi.application"

# Auth redirects for the login-to-edit flow. LOGIN_URL is where @login-gated
# access sends anonymous users; the edit endpoints themselves return 403 JSON
# (see moths.permissions.editor_required) rather than redirecting.
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/"

# TEMPORARY: while the project isn't public-ready, gate the whole site behind
# login (moths.middleware.RequireLoginMiddleware) so crawlers can't index
# in-progress interface decisions. Set DJANGO_REQUIRE_LOGIN=0 to lift the gate
# and fall back to the normal public-view / login-to-edit behaviour.
REQUIRE_LOGIN = os.environ.get("DJANGO_REQUIRE_LOGIN", "True").lower() in (
    "1",
    "true",
    "yes",
)


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        # Override with DJANGO_DB_PATH to keep the sqlite file on a writable
        # volume (the app code dir is read-only in the container image). Only
        # Django's built-in tables (auth/sessions/admin) live here; the moth
        # data itself is read from the MOTHS_* directories.
        "NAME": os.environ.get("DJANGO_DB_PATH") or (BASE_DIR / "db.sqlite3"),
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = "static/"

# Where ``collectstatic`` gathers files for production serving (the admin's CSS
# / JS; the moth pages use inline styles and serve images through views).
STATIC_ROOT = os.environ.get("DJANGO_STATIC_ROOT") or (BASE_DIR / "staticfiles")

# Serve those collected files straight from the WSGI process with WhiteNoise
# when it is installed — it is in the Docker/server image (requirements-server.txt);
# plain dev environments that only install requirements.txt are left untouched.
try:
    import whitenoise  # noqa: F401
except ImportError:
    pass
else:
    _WHITENOISE_MW = "whitenoise.middleware.WhiteNoiseMiddleware"
    if _WHITENOISE_MW not in MIDDLEWARE:
        MIDDLEWARE.insert(
            MIDDLEWARE.index("django.middleware.security.SecurityMiddleware") + 1,
            _WHITENOISE_MW,
        )
    STORAGES = {
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        # Non-manifest storage avoids hashed-name lookups that would otherwise
        # require collectstatic to have run (keeps the admin working in dev too).
        "staticfiles": {"BACKEND": "whitenoise.storage.CompressedStaticFilesStorage"},
    }

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# --- Moth dataset locations --------------------------------------------------
# Every path below must be provided explicitly via environment variables; there
# are no built-in defaults. A minimal, sufficient set is::
#
#     set TAX_CSV=...\names.csv
#     set MOTHS_IMAGE_DIR=...\images
#     set MOTHS_PREDICTION_DIR=...\test
#     set MOTHS_LABEL_DIR=...\labels
#     set MOTHS_THUMBNAIL_DIR=...\cache
#     set MOTHS_DATA_DIR=...\data


def _required_env(name: str) -> str:
    """Return an environment variable's value or fail with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise ImproperlyConfigured(
            f"Environment variable {name} must be set "
            f"(the moth dataset paths have no built-in defaults)."
        )
    return value


# Directory that holds the moth training images (per-tax_id subfolders).
MOTHS_IMAGE_DIR = _required_env("MOTHS_IMAGE_DIR")

# Directory holding the YOLO-pose label files (``<name>.txt``) and the per-image
# stage/flag classification sidecars (``<name>.class``). These editable hand
# labels are also the source used for pose classification.
MOTHS_LABEL_DIR = _required_env("MOTHS_LABEL_DIR")

# Directory holding read-only model prediction files (YOLO-pose format,
# ``<name>.txt``). Shown as a reference overlay in the image edit view and
# never modified through the app.
MOTHS_PREDICTION_DIR = _required_env("MOTHS_PREDICTION_DIR")

# Directory where generated thumbnails / normalized crops are cached.
MOTHS_THUMBNAIL_DIR = _required_env("MOTHS_THUMBNAIL_DIR")

# CSV mapping tax_id -> taxonomy names. Relevant columns: id, family, species,
# name. Used to display friendly species labels (with a "{name} ({id})" title)
# wherever a tax_id appears.
MOTHS_NAMES_CSV = _required_env("TAX_CSV")

# Per-taxon harvested reference data (e.g. Wikipedia wikitext under
# ``<tax_id>/<tax_id>.wiki``). Written by tools/harvest_wiki.py; empty
# ``<tax_id>/`` folders mean no matching wiki page was found.
MOTHS_DATA_DIR = _required_env("MOTHS_DATA_DIR")

# Max thumbnail size (width, height) in pixels; aspect ratio is preserved.
MOTHS_THUMBNAIL_SIZE = (400, 400)
