# syntax=docker/dockerfile:1
#
# Django web server for moths_list on Ubuntu 24.04 LTS.
#
# Scope: the Django app ONLY — Django + Pillow + numpy, served by gunicorn with
# WhiteNoise for static files (see requirements-server.txt). The tools/
# prediction pipeline (torch, ultralytics, requests, ...) is intentionally NOT
# installed in this image.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DJANGO_SETTINGS_MODULE=moths_list.settings \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Runtime deps: Python 3.12 (Ubuntu 24.04 default) + venv. Pillow and numpy ship
# self-contained manylinux wheels, so no image libraries or C toolchain needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv "$VIRTUAL_ENV"

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt requirements-server.txt ./
RUN pip install --upgrade pip && pip install -r requirements-server.txt

# App code only (tools/ and data are excluded via .dockerignore).
COPY manage.py ./
COPY moths_list ./moths_list
COPY moths ./moths

# Collect static assets (the admin CSS/JS) into STATIC_ROOT. collectstatic does
# not read the MOTHS_* dataset dirs, so feed throwaway values just for this
# build step; they are NOT persisted as image env, so a misconfigured run still
# fails loudly instead of silently pointing at bogus directories.
RUN MOTHS_IMAGE_DIR=/nonexistent MOTHS_LABEL_DIR=/nonexistent \
    MOTHS_PREDICTION_DIR=/nonexistent MOTHS_THUMBNAIL_DIR=/nonexistent \
    MOTHS_DATA_DIR=/nonexistent \
    TAX_CSV=/nonexistent/names.csv \
    python manage.py collectstatic --noinput

# Entrypoint (migrate + gunicorn).
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Writable dir for the sqlite DB (mount a volume here to persist admin
# users/sessions across restarts), then drop root.
RUN mkdir -p /data \
    && useradd --system --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /data
USER appuser

# Runtime defaults (override at deploy time as needed).
ENV DJANGO_DB_PATH=/data/db.sqlite3 \
    DJANGO_STATIC_ROOT=/app/staticfiles \
    GUNICORN_WORKERS=3 \
    GUNICORN_TIMEOUT=120 \
    PORT=8000

EXPOSE 8000

# Lightweight liveness probe: the login page needs only the DB, not the moth
# dataset volumes, and stays public even behind the site-wide login gate, so it
# works before data is mounted and regardless of REQUIRE_LOGIN.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/accounts/login/').status==200 else 1)" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
