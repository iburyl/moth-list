#!/usr/bin/env sh
# Container entrypoint for the moths_list Django web server.
#
# Applies the built-in Django migrations (auth / sessions / admin — the moth
# data itself lives in the MOTHS_* directories, not the DB) and then hands off
# to gunicorn. The real MOTHS_* env vars and a writable DJANGO_DB_PATH must be
# provided at run time. Passing a command runs that instead of the server.
set -eu

# An explicit command (e.g. `docker compose run --rm web python manage.py
# migrate`) replaces the server: run it and exit instead of falling through to
# gunicorn, which would otherwise keep the one-off container in the foreground.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

python manage.py migrate --noinput

# Access logging is off on purpose: every request (including the container
# HEALTHCHECK) would otherwise flood stderr with the Docker-network peer IP.
# External access / client-IP / crawler monitoring lives on Caddy instead.
# Django LOGGING + gunicorn's error log still surface 500s and worker failures.
exec gunicorn moths_list.wsgi:application \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers "${GUNICORN_WORKERS:-3}" \
    --timeout "${GUNICORN_TIMEOUT:-120}" \
    --access-logfile /dev/null \
    --error-logfile -
