#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[smoke] verifying deliverables exist"
required=(
  "docs/gamifyed-prd.md"
  "README.md"
  "package.json"
  "public/index.html"
  "src/db.js"
  "src/learning.js"
  "src/server.js"
  "test/learning.test.js"
)

for f in "${required[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[smoke][fail] missing required file: $f"
    exit 1
  fi
  echo "[smoke][ok] $f"
done

echo "[smoke] syntax + unit checks"
node --check src/server.js
node --check src/db.js
node --check src/learning.js
node --test test/learning.test.js

echo "[smoke] attempting dependency install and runtime health check"
if npm install --no-audit --no-fund >/tmp/gamifyed_npm_install.log 2>&1; then
  node src/server.js >/tmp/gamifyed_server.log 2>&1 &
  PID=$!
  cleanup() {
    kill "$PID" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT

  for _ in {1..20}; do
    if curl -fsS http://127.0.0.1:3000/api/health >/tmp/gamifyed_health.json 2>/dev/null; then
      break
    fi
    sleep 0.5
  done

  if [[ ! -s /tmp/gamifyed_health.json ]]; then
    echo "[smoke][fail] server health endpoint did not respond"
    echo "--- server log ---"
    cat /tmp/gamifyed_server.log
    exit 1
  fi

  echo "[smoke][ok] runtime health: $(cat /tmp/gamifyed_health.json)"
else
  echo "[smoke][warn] npm install failed in this environment; skipping runtime health check"
  echo "--- npm install log ---"
  cat /tmp/gamifyed_npm_install.log
fi

echo "[smoke] completed"
