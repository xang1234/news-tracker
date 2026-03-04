#!/usr/bin/env bash
set -euo pipefail

LOCK_DIR=".docker-build.lock"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another docker build is already running (lock: $LOCK_DIR)." >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR"' EXIT

docker compose build "$@"
