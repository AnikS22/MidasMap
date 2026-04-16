#!/usr/bin/env bash
# Create a local venv, install deps, verify UI builds, run tests.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip -q
python -m pip install -r requirements-dev.txt -q
python -c "from app import build_app; build_app(); print('app.build_app: OK')"
python -m pytest tests -q
echo "All checks passed. Run: source .venv/bin/activate && python app.py --checkpoint checkpoints/final/final_model.pth"
