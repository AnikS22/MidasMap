#!/usr/bin/env bash
# Run MidasMap Gradio on localhost (default http://127.0.0.1:7860).
#
# Usage:
#   ./scripts/run_local.sh
#   ./scripts/run_local.sh --server-name 0.0.0.0    # LAN / other devices on your network
#   ./scripts/run_local.sh --share                  # Gradio public tunnel (if blocked)

set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

exec python app.py "$@"
