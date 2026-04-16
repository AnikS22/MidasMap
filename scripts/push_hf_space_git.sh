#!/usr/bin/env bash
# Deploy to a Hugging Face Space using **git + push** (often works when huggingface-cli upload fails).
#
# Prerequisites:
#   export HF_TOKEN=hf_...   # write token
#   git and git-lfs installed:  brew install git-lfs && git lfs install
#   Space must exist (Gradio): https://huggingface.co/new-space
#
# Usage:
#   ./scripts/push_hf_space_git.sh
#
# Env:
#   HF_SPACE_REPO   default AnikS22/MidasMap
#   HF_MODEL_REPO   default AnikS22/MidasMap
#   HF_SPACE_SKIP_CHECKPOINT  default 1 (no .pth in repo; app downloads at runtime)

set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/.hf_space_staging"
MODEL_REPO="${HF_MODEL_REPO:-AnikS22/MidasMap}"
SPACE_REPO="${HF_SPACE_REPO:-AnikS22/MidasMap}"
SKIP_CKPT="${HF_SPACE_SKIP_CHECKPOINT:-1}"

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN (write access) for git HTTPS push."
  exit 1
fi
TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"

if [[ -f "$ROOT/.venv/bin/huggingface-cli" ]]; then
  HF=( "$ROOT/.venv/bin/huggingface-cli" )
else
  HF=( huggingface-cli )
fi

TOKEN_ARGS=( --token "$TOKEN" )
HF_USER="${HF_USER:-}"
if [[ -z "$HF_USER" ]]; then
  HF_USER="$("${HF[@]}" whoami "${TOKEN_ARGS[@]}" 2>/dev/null | head -1 | tr -d '\r')"
fi
if [[ -z "$HF_USER" ]] || [[ "$HF_USER" == *"Not logged"* ]]; then
  echo "Could not infer Hugging Face username. Set:  export HF_USER=YourUsername"
  exit 1
fi

TOKEN_ARGS=( --token "$TOKEN" )

# shellcheck source=hf_space_stage.inc.sh
source "$ROOT/scripts/hf_space_stage.inc.sh"
hf_space_stage

TMP=$(mktemp -d)
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

CLONE_URL="https://${HF_USER}:${TOKEN}@huggingface.co/spaces/${SPACE_REPO}"
echo "Cloning spaces/${SPACE_REPO} into temp dir ..."
git clone "$CLONE_URL" "$TMP/space"
cd "$TMP/space"
git lfs install 2>/dev/null || true

# Replace working tree with staged content (keep .git)
find "$TMP/space" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

rsync -a "$STAGE/" "$TMP/space/"

if [[ "$SKIP_CKPT" == "1" ]]; then
  MSG="Deploy via git: code only; weights from model repo at runtime"
else
  MSG="Deploy via git: includes checkpoints/final/final_model.pth"
  git lfs track "*.pth" 2>/dev/null || true
fi

git config user.email "${GIT_AUTHOR_EMAIL:-deploy@users.noreply.huggingface.co}"
git config user.name "${GIT_AUTHOR_NAME:-hf-deploy}"

git add -A
if git diff --staged --quiet; then
  echo "Nothing to commit (already up to date)."
else
  git commit -m "$MSG"
  echo "Pushing to Hugging Face ..."
  git push origin "$(git branch --show-current)"
fi

echo "Done: https://huggingface.co/spaces/${SPACE_REPO}"
