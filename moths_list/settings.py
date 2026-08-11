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
            ],
        },
    },
]

WSGI_APPLICATION = "moths_list.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
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

# Max thumbnail size (width, height) in pixels; aspect ratio is preserved.
MOTHS_THUMBNAIL_SIZE = (400, 400)
