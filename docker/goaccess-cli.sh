#!/usr/bin/env sh
# Interactive GoAccess dashboard over Caddy access logs (host-side).
#
# Prerequisites: goaccess with built-in CADDY format (1.8+), e.g.
#   sudo apt install goaccess
#
# Usage (from the moth-list checkout on the server):
#   ./docker/goaccess-cli.sh
#
# Or point at a log dir explicitly:
#   LOG_DIR=/srv/caddy/log ./docker/goaccess-cli.sh
#
# Reads the active access.log plus any rotated *.gz siblings. Requires Caddy
# logging with ``format json`` (see docker/Caddyfile).

set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)

if [ -z "${LOG_DIR:-}" ]; then
    CADDY_HOST=""
    if [ -f "$REPO_ROOT/.env" ]; then
        # shellcheck disable=SC1091
        CADDY_HOST=$(
            grep -E '^[[:space:]]*MOTHS_CADDY_DIR_HOST=' "$REPO_ROOT/.env" \
                | tail -n1 \
                | cut -d= -f2- \
                | tr -d '[:space:]' \
                | tr -d '"' \
                | tr -d "'"
        )
    fi
    if [ -z "$CADDY_HOST" ]; then
        echo "Set MOTHS_CADDY_DIR_HOST in .env or pass LOG_DIR=..." >&2
        exit 1
    fi
    LOG_DIR="$CADDY_HOST/log"
fi

if ! command -v goaccess >/dev/null 2>&1; then
    echo "goaccess not found. Install it first, e.g.: sudo apt install goaccess" >&2
    exit 1
fi

if [ ! -d "$LOG_DIR" ]; then
    echo "Log directory missing: $LOG_DIR" >&2
    exit 1
fi

# Prefer the live file; include rotated gzipped siblings when present.
# zcat -f passes uncompressed files through unchanged.
set -- "$LOG_DIR"/access.log
for f in "$LOG_DIR"/access.log.*.gz "$LOG_DIR"/access.log.*.zst; do
    [ -e "$f" ] || continue
    set -- "$@" "$f"
done

if [ ! -f "$LOG_DIR/access.log" ] && [ "$#" -eq 1 ]; then
    echo "No access.log (or rotations) under $LOG_DIR yet." >&2
    exit 1
fi

echo "GoAccess over: $*" >&2
# stdin mode → interactive TUI; q to quit.
exec zcat -f -- "$@" | goaccess --log-format=CADDY --no-global-config -
