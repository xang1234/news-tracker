#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-default}"
CDP_URL="${XUI_CDP_URL:-http://127.0.0.1:9222}"
CONFIG_DIR="${XUI_CONFIG_DIR:-$ROOT_DIR/runtime/xui}"
BASE_CONFIG="$CONFIG_DIR/config.toml"
AUTH_CONFIG="$CONFIG_DIR/config.auth.toml"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "Missing base config: $BASE_CONFIG" >&2
  echo "Run xui config init first (or set XUI_CONFIG_DIR)." >&2
  exit 1
fi

mkdir -p "$CONFIG_DIR"

awk -v cdp_url="$CDP_URL" '
  BEGIN { in_browser = 0; inserted = 0 }

  /^\[browser\]/ {
    in_browser = 1
    print
    next
  }

  /^\[/ && $0 !~ /^\[browser\]/ {
    if (in_browser && !inserted) {
      print "cdp_url = \"" cdp_url "\""
      inserted = 1
    }
    in_browser = 0
    print
    next
  }

  {
    if (in_browser && /^headless = /) {
      print "headless = false"
      next
    }
    if (in_browser && /^block_resources = /) {
      print "block_resources = false"
      next
    }
    if (in_browser && /^cdp_url = /) {
      print "cdp_url = \"" cdp_url "\""
      inserted = 1
      next
    }

    print
  }

  END {
    if (in_browser && !inserted) {
      print "cdp_url = \"" cdp_url "\""
    }
  }
' "$BASE_CONFIG" > "$AUTH_CONFIG"

if ! curl -fsS "$CDP_URL/json/version" >/dev/null 2>&1; then
  echo "Chrome DevTools endpoint is not reachable at $CDP_URL" >&2
  echo "Start Google Chrome with --remote-debugging-port=9222 first, then rerun." >&2
  exit 2
fi

echo "Using auth config: $AUTH_CONFIG"
uvx --from 'git+https://github.com/xang1234/xui.git' --with 'rich>=13.0' --with 'typer>=0.12' \
  xui auth login --profile "$PROFILE" --path "$AUTH_CONFIG"

echo

echo "Login complete. Storage state is saved under: $CONFIG_DIR/profiles/$PROFILE/session/storage_state.json"

echo "Ingestion containers will continue using: $BASE_CONFIG"
