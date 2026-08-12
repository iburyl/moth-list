#!/usr/bin/env sh
# Container entrypoint for the moths_list Django web server.
#
# Applies the built-in Django migrations (auth / sessions / admin — the moth
# data itself lives in the MOTHS_* directories, not the DB) and then hands off
# to gunicorn. The real MOTHS_* env vars and a writable DJANGO_DB_PATH must be
# provided at run time.
set -eu

python manage.py migrate --noinput

exec gunicorn moths_list.wsgi:application \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers "${GUNICORN_WORKERS:-3}" \
    --timeout "${GUNICORN_TIMEOUT:-120}" \
    --access-logfile - \
    --error-logfile -
