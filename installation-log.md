# ComfyUI Installation Log

**Date:** 2026-06-25
**Hardware:** Apple M5 Pro, 48 GB unified memory, macOS 26.5.1

---

## Environment

| Item | Value |
|---|---|
| Python | 3.12.11 (via uv venv — no brew Python dependency) |
| uv | 0.7.2 |
| PyTorch | 2.12.1 |
| ComfyUI | 0.26.0 |
| MPS available | `True` |
| Device | `mps` |
| Shared VRAM | 49 GB |

---

## Step 1 — Prerequisites

All present, nothing installed:

- `git` 2.50.1 (Apple Git-155)
- Homebrew 6.0.2
- `ffmpeg` 7.1.1
- `uv` 0.7.2 (Homebrew)

---

## Step 2 — Clone and install ComfyUI

```bash
git clone https://github.com/comfyanonymous/ComfyUI /Users/claudiogomes/Desktop/Github/ComfyUI
cd /Users/claudiogomes/Desktop/Github/ComfyUI
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

81 packages installed cleanly, including:
- `torch==2.12.1`
- `torchvision==0.27.1`
- `numpy==2.5.0`
- `pillow==12.2.0`
- `huggingface-hub==1.20.1`
- `safetensors==0.8.0`
- `transformers==5.12.1`

**MPS check:**
```
python -c "import torch; print(torch.backends.mps.is_available())"
# True
```

---

## Step 3 — Install ComfyUI Manager

```bash
git clone https://github.com/ltdrdata/ComfyUI-Manager \
  /Users/claudiogomes/Desktop/Github/ComfyUI/custom_nodes/ComfyUI-Manager
uv pip install --python .venv/bin/python GitPython PyGithub matrix-nio toml chardet
```

24 additional packages installed. ComfyUI Manager auto-detected `uv` as its pip backend (no system pip needed).

---

## Step 4 — Python client libraries

```bash
uv pip install --python .venv/bin/python websocket-client
```

Already present from requirements.txt: `requests`, `pillow`, `huggingface-hub`.

---

## Step 5 — Model downloads

All models downloaded via `huggingface_hub` Python SDK with HF token.
Flux.1-schnell and Flux.1-Fill-dev required HF license acceptance at:
- https://huggingface.co/black-forest-labs/FLUX.1-schnell
- https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev

CodeFormer downloaded from GitHub release (HF repo was 404):
```bash
curl -L https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth \
  -o models/facerestore_models/codeformer.pth
```

| Model | Destination | Size |
|---|---|---|
| `flux1-schnell.safetensors` | `models/unet/` | 22 GB |
| `flux1-fill-dev.safetensors` | `models/unet/` | 22 GB |
| `ae.safetensors` (VAE) | `models/vae/` | 320 MB |
| `clip_l.safetensors` | `models/clip/` | 235 MB |
| `t5xxl_fp8_e4m3fn.safetensors` | `models/clip/` | 4.6 GB |
| `codeformer.pth` | `models/facerestore_models/` | 359 MB |

---

## Step 6 — Launch and REST API verification

```bash
cd /Users/claudiogomes/Desktop/Github/ComfyUI
.venv/bin/python main.py --listen 127.0.0.1 --port 8188
```

```bash
curl http://localhost:8188/system_stats
```

Response confirmed device `mps`, 49 GB VRAM, ComfyUI 0.26.0.

---

## Steps pending (manual, via browser UI)

- **Step 6 from handover prompt:** Install node packs via ComfyUI Manager UI at `http://localhost:8188`:
  - ComfyUI-Impact-Pack
  - ComfyUI-Advanced-ControlNet

---

## Notes

- `huggingface-cli` is deprecated in the installed version of huggingface-hub — use Python SDK (`hf_hub_download`) or the new `hf` CLI instead.
- `comfy-aimdo` logs a warning on macOS ("only supports Windows and Linux") — harmless, does not affect MPS inference.
- `triton` backend unavailable on macOS — expected, uses `eager` backend instead.
- ComfyUI Manager auto-falls-back to `uv` for pip operations, consistent with the uv-only setup.

---

## Launch command (future sessions)

```bash
cd /Users/claudiogomes/Desktop/Github/ComfyUI
.venv/bin/python main.py --listen 127.0.0.1 --port 8188
```

UI: `http://localhost:8188`
API: `http://localhost:8188/system_stats`
