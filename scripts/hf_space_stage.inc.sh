# shellcheck shell=bash
# Shared staging for Hugging Face Space deploys. Source from repo root after setting:
#   ROOT STAGE MODEL_REPO SKIP_CKPT HF (array) TOKEN_ARGS (array)

hf_space_stage() {
  echo "Staging files in $STAGE ..."
  rm -rf "$STAGE"
  mkdir -p "$STAGE/checkpoints/final"
  cp "$ROOT/app.py" "$STAGE/"
  cp "$ROOT/huggingface-space/README.md" "$STAGE/README.md"
  cp "$ROOT/huggingface-space/requirements-space.txt" "$STAGE/requirements.txt"
  rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '.mypy_cache' "$ROOT/src" "$STAGE/"
  find "$STAGE/src" -depth -type d -name '__pycache__' -exec rm -rf '{}' \; 2>/dev/null || true
  find "$STAGE/src" -name '.DS_Store' -delete 2>/dev/null || true

  if [[ "$SKIP_CKPT" == "1" ]]; then
    echo "HF_SPACE_SKIP_CHECKPOINT=1 — omitting .pth from Space (app downloads from model repo at runtime)."
    echo "  Model repo: $MODEL_REPO  file: checkpoints/final/final_model.pth"
    rm -rf "$STAGE/checkpoints"
  else
    echo "Downloading checkpoint from model repo $MODEL_REPO (--repo-type model) ..."
    "${HF[@]}" download "$MODEL_REPO" checkpoints/final/final_model.pth \
      --repo-type model \
      --local-dir "$STAGE" ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}
  fi
}
