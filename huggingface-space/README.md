---
title: MidasMap
emoji: 🔬
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# MidasMap Space

Gradio demo for **[MidasMap](https://github.com/AnikS22/MidasMap)** (immunogold particle detection in TEM synapse images).

## Deploy from your laptop

From the **MidasMap** repo root:

```bash
export HF_TOKEN=hf_...   # write token
# Recommended: do not upload the ~100MB checkpoint into the Space (avoids LFS / size issues).
export HF_SPACE_SKIP_CHECKPOINT=1
./scripts/upload_hf_space.sh
```

If **`upload_hf_space.sh` fails**, use **git + LFS** instead (often more reliable):

```bash
brew install git-lfs && git lfs install   # once
export HF_TOKEN=hf_...
./scripts/push_hf_space_git.sh
```

Full options: [docs/DEPLOY.md](../docs/DEPLOY.md) in the main repo.

Create the Space once if needed (Gradio SDK required for auto-create):

```bash
huggingface-cli repo create MidasMap --type space --space_sdk gradio -y
```

Weights are loaded from the **model** repo `AnikS22/MidasMap` at `checkpoints/final/final_model.pth` when the file is not in the Space. Override with Space secrets / env: `MIDASMAP_HF_WEIGHTS_REPO`, `MIDASMAP_HF_WEIGHTS_FILE`.

To bundle the checkpoint in the Space instead (larger upload):

```bash
export HF_SPACE_SKIP_CHECKPOINT=0
./scripts/upload_hf_space.sh
```

## Troubleshooting uploads

| Symptom | What to do |
|--------|----------------|
| **401 / not logged in** | `export HF_TOKEN=hf_...` with a token that has **write** access, or `huggingface-cli login`. |
| **LFS / authorization / upload stuck** | Use `HF_SPACE_SKIP_CHECKPOINT=1` so only code uploads; ensure the **model** repo (not the Space) contains `checkpoints/final/final_model.pth`. |
| **Space does not exist** | Create it in the HF web UI (**New Space** → **Gradio**) or run `huggingface-cli repo create ... --type space --space_sdk gradio`. |
| **“No space_sdk provided”** | The Space repo must be created as **Gradio** (or pass `--space_sdk gradio` when using `repo create`). |
| **Model not found on Space** | First boot downloads weights from the Hub; public repos need no token. Private model repo: add `HF_TOKEN` as a Space **secret** (read). |
| **Still failing** | Try `pip install hf_transfer` and `export HF_HUB_ENABLE_HF_TRANSFER=1` before upload. Or use **git** + **git lfs** clone of the Space, copy files, commit, push. |

## Vercel embed

`https://yoursite.vercel.app/?embed=https://huggingface.co/spaces/YOUR_USER/YOUR_SPACE`
