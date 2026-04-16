#!/usr/bin/env sh
# If no checkpoint is mounted, download public weights from the Hub (Docker / local compose).
set -e
CKPT="checkpoints/final/final_model.pth"
if [ ! -f "$CKPT" ]; then
  echo "No $CKPT — downloading from Hub model repo (CPU image, first start may take a minute)..."
  mkdir -p checkpoints/final
  python - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import os
import shutil

repo = os.environ.get("MIDASMAP_HF_WEIGHTS_REPO", "AnikS22/MidasMap").strip()
fn = os.environ.get("MIDASMAP_HF_WEIGHTS_FILE", "checkpoints/final/final_model.pth").strip()
cached = hf_hub_download(repo_id=repo, filename=fn, repo_type="model")
dest = Path("checkpoints/final/final_model.pth")
shutil.copy(cached, dest)
print("Weights cached at", dest)
PY
fi

PORT="${PORT:-7860}"
exec python app.py --server-name 0.0.0.0 --port "$PORT" "$@"
