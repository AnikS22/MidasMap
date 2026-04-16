# Deploy MidasMap (local + Hugging Face + Docker)

## Localhost (simplest)

From the repo root, with a venv that has dependencies installed:

```bash
chmod +x scripts/run_local.sh
./scripts/run_local.sh
```

Open **http://127.0.0.1:7860**. You need `checkpoints/final/final_model.pth` on disk (or train / download it first).

- **LAN access:** `./scripts/run_local.sh --server-name 0.0.0.0`
- **Gradio share link:** `./scripts/run_local.sh --share`

## Localhost with Docker

```bash
docker compose up --build
```

Open **http://127.0.0.1:7860**. The image is **CPU** PyTorch. On first run, if you did not mount weights, the container downloads `checkpoints/final/final_model.pth` from the public Hub model repo.

**Optional — use your local checkpoint (no download):**

```yaml
# In docker-compose.yml, uncomment:
volumes:
  - ./checkpoints/final:/app/checkpoints/final:ro
```

(The host directory must contain `final_model.pth`.)

## Hugging Face Space — if `huggingface-cli upload` keeps failing

Try **git push** instead (often more reliable for LFS / large trees):

1. Install **git-lfs**: `brew install git-lfs` then `git lfs install`
2. Create a **Gradio** Space on the Hub (same name as your repo if you like).
3. Run:

```bash
export HF_TOKEN=hf_...          # write token
export HF_USER=YourHfUsername   # if whoami fails
export HF_SPACE_SKIP_CHECKPOINT=1
chmod +x scripts/push_hf_space_git.sh
./scripts/push_hf_space_git.sh
```

Ensure the **model** repo has `checkpoints/final/final_model.pth` so the Space can download it at startup (see `app.py` → `_resolve_checkpoint`).

**Other hosts (no HF):** push the same Docker image to **Fly.io**, **Railway**, **Google Cloud Run**, or any VM — use the `Dockerfile` and set `PORT` if the platform injects it.

## Environment variables (Space / Docker)

| Variable | Purpose |
|----------|---------|
| `MIDASMAP_HF_WEIGHTS_REPO` | Hub model repo id for weight download (default `AnikS22/MidasMap`) |
| `MIDASMAP_HF_WEIGHTS_FILE` | Path inside that repo (default `checkpoints/final/final_model.pth`) |
| `PORT` / `GRADIO_SERVER_PORT` | Listen port (Docker / platforms) |
