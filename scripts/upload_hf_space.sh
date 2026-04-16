#!/usr/bin/env bash
# Upload MidasMap to a Hugging Face Space via huggingface-cli.
# If this fails (LFS, timeouts), try: ./scripts/push_hf_space_git.sh
#
# See also: docs/DEPLOY.md

set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/.hf_space_staging"
MODEL_REPO="${HF_MODEL_REPO:-AnikS22/MidasMap}"
SPACE_REPO="${HF_SPACE_REPO:-AnikS22/MidasMap}"
SKIP_CKPT="${HF_SPACE_SKIP_CHECKPOINT:-1}"

if [[ -f "$ROOT/.venv/bin/huggingface-cli" ]]; then
  HF=( "$ROOT/.venv/bin/huggingface-cli" )
else
  HF=( huggingface-cli )
fi

TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARGS=( --token "$HF_TOKEN" )
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  TOKEN_ARGS=( --token "$HUGGING_FACE_HUB_TOKEN" )
fi

WHO=$("${HF[@]}" whoami ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"} 2>&1 || true)
if [[ ${#TOKEN_ARGS[@]} -eq 0 ]] && [[ "$WHO" == *"Not logged in"* ]]; then
  echo "Not authenticated. Do one of:"
  echo "  export HF_TOKEN=hf_...   # then re-run this script"
  echo "  huggingface-cli login"
  exit 1
fi

# shellcheck source=hf_space_stage.inc.sh
source "$ROOT/scripts/hf_space_stage.inc.sh"
hf_space_stage

if [[ "$SKIP_CKPT" == "1" ]]; then
  COMMIT_MSG="Deploy MidasMap Gradio app; weights downloaded from model repo at runtime"
else
  COMMIT_MSG="Deploy MidasMap Gradio app with bundled checkpoints/final/final_model.pth"
fi

echo "Uploading to Space spaces/$SPACE_REPO (set HF_SPACE_REPO to override) ..."
"${HF[@]}" upload "$SPACE_REPO" "$STAGE" . \
  --repo-type space \
  --commit-message "$COMMIT_MSG" \
  --exclude '**/__pycache__/**' \
  --exclude '**/*.pyc' \
  --exclude '.DS_Store' \
  ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}

echo "Done: https://huggingface.co/spaces/$SPACE_REPO"
