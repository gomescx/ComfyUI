# ComfyUI Project

## Python Environment

The ComfyUI venv lives at `.venv/`. Always run scripts with `uv`:

```bash
uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python script_name.py
```

Never invoke `.venv/bin/python` directly.

## Starting ComfyUI

```bash
uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python main.py \
  --listen 127.0.0.1 --port 8188 --use-split-cross-attention
```

`--use-split-cross-attention` is required on this machine — ComfyUI itself recommends it in the startup log for MPS memory/speed.  
The process runs in the foreground (by design). Ctrl+C or closing the terminal stops the server.

## Hardware

Apple M5 Pro, 49 GB unified memory, macOS 26.5.1.  
ComfyUI reports device `mps`, vram state `SHARED` — GPU and CPU share the same pool. There is no hard VRAM wall; exceeding available memory causes macOS to page to SSD, which freezes the machine rather than producing a clean OOM error.

## Interactive Launcher

**Do not run `faceswap_run.py` directly.** Use the interactive launcher instead:

```bash
./run.sh
```

`run.sh` calls `run_menu.py`, which:
1. Checks ComfyUI is up at `http://127.0.0.1:8188`
2. Shows hardcoded input/output paths
3. Asks for free memory (GB) and uses that to gate which modes are available
4. Recommends a safe queue depth for the chosen mode
5. Confirms before running anything

## Face Swap Pipeline

`faceswap_run.py` drives frame-by-frame face (+ optional hair) transfer via the ComfyUI HTTP API.

### Hardcoded paths (top of file)

| Variable | Value |
|---|---|
| `VIDEO_IN` | `/Users/claudiogomes/Downloads/Lana.mp4` |
| `FACE_IN` | `/Users/claudiogomes/Downloads/Kate_2013.png` |
| `VIDEO_OUT` | `/Users/claudiogomes/Downloads/Lana_faceswap.mp4` |

### Modes

| Menu | Flag(s) | Free RAM needed | What it does |
|---|---|---|---|
| 1 | _(none)_ | ~1 GB | Face swap only (ReActor, fast) |
| 2 | `--restore` | ~2 GB | Face swap + CodeFormer face sharpening |
| 3 | `--hair` | ~38 GB | Face swap + Flux hair transfer (fp16) |
| 4 | `--hair --restore` | ~38 GB | Face swap + hair + CodeFormer |

**fp8 is not available on this machine.** There used to be a mode 5 (`--fp8`) that loaded `flux1-fill-dev.safetensors` at `fp8_e4m3fn` to halve RAM. fp8 is a CUDA-only dtype — **Apple's MPS backend has no fp8 support**, so the model loads but the KSampler crashes on the first forward pass (`TypeError: Trying to convert Float8_e4m3fn to the MPS backend`). The `--fp8` flag and menu mode 5 have been removed. Hair transfer requires fp16 (modes 3/4) and ~38 GB free. The only way to run fp8 weights would be to launch the whole ComfyUI server with `--cpu`, which makes Flux take minutes per frame — unusable for video. (The `t5xxl_fp8_e4m3fn` *text encoder* is a separate pre-quantized file that runs on CPU and is unaffected.)

`--queue N` sets pipeline depth (frames submitted to ComfyUI ahead of collection). Default 8. The menu recommends a safe value based on available RAM. With hair modes, use 1–2 to avoid memory pressure.

### Hair workflow: 19 ComfyUI nodes

```
1–3   LoadImage         frame, Kate photo, hair mask
4     ReActorFaceSwap   face swap (inswapper_128.onnx)
5     UNETLoader        flux1-fill-dev (weight_dtype: default — fp8 unsupported on MPS)
6     VAELoader         ae.safetensors
7     DualCLIPLoader    clip_l + t5xxl_fp8_e4m3fn (type: flux)
8     CLIPVisionLoader  sigclip_vision_patch14_384
9     StyleModelLoader  flux1-redux-dev
10–11 CLIPTextEncode    empty prompts (Redux drives style, not text)
12    CLIPVisionEncode  Kate photo → vision embedding  [crop: "center" required]
13    StyleModelApply   clip_vision_output, strength=1.0
14    FluxGuidance      guidance=3.5
15    ImageToMask       hair mask PNG → mask tensor
16    InpaintModelConditioning  face-swapped image + mask  [noise_mask: True required]
17    KSampler          euler/simple, 20 steps, cfg=1.0, denoise=0.75
18    VAEDecode
19    SaveImage
```

### Hair mask generation

OpenCV Haar cascade (`haarcascade_frontalface_default.xml`, already in cv2).  
Detects face → soft ellipse above forehead → Gaussian blur for feathering.  
Falls back to blank mask (no change) if face not detected in a frame.

## Models installed

| Model | Path | Size | Purpose |
|---|---|---|---|
| `inswapper_128.onnx` | `models/insightface/` | ~500 MB | Face swap |
| `codeformer.pth` | `models/facerestore_models/` | 359 MB | Face restoration (`--restore`) |
| `flux1-fill-dev.safetensors` | `models/unet/` | 22 GB | Inpainting UNET (hair) |
| `flux1-redux-dev.safetensors` | `models/style_models/` | 3.3 GB | Style conditioning |
| `ae.safetensors` | `models/vae/` | 320 MB | Flux VAE |
| `clip_l.safetensors` | `models/clip/` | 235 MB | Flux text encoder |
| `t5xxl_fp8_e4m3fn.safetensors` | `models/clip/` | 4.6 GB | Flux T5 encoder |
| `sigclip_vision_patch14_384.safetensors` | `models/clip_vision/` | 878 MB | Vision encoder for Redux |
| `buffalo_l` (ONNX set) | `models/insightface/models/` | — | Face detection |

`flux1-schnell.safetensors` (22 GB) is also on disk but **not used** by this pipeline.

## Memory budget (hair modes)

| Mode | Model RAM | + OS + working | Total needed |
|---|---|---|---|
| Face only | ~1 GB | — | ~2 GB free |
| Hair fp16 (mode 3/4) | ~31 GB | ~7 GB | ~38 GB free |

The machine has 49 GB total. With other apps running, 30 GB free is typical — which is **not enough** for hair transfer (fp16 needs ~38 GB free, and fp8 cannot run on MPS at all). To use modes 3/4, close other apps to free ~38 GB, or wait for the planned RAM upgrade. Modes 1/2 (face only / + restore) run comfortably in the meantime.

**Memory cleanup:** the ComfyUI *server* (`main.py`) keeps Flux/CLIP models resident after a prompt — even a failed one — so ~25 GB can appear "stuck" (attributed to VS Code if launched from its terminal). `faceswap_run.py` now POSTs to `/free` (unload models + free memory) on clean finish, on error, and on force-quit, so this is reclaimed automatically. If models ever linger, `curl -X POST http://127.0.0.1:8188/free -d '{"unload_models":true,"free_memory":true}'` clears them without restarting the server.

## Credentials

HuggingFace token stored in `.env` as `HUGGING_FACE`. Required for gated model downloads (Flux models). Use the `huggingface_hub` Python SDK with `token=` — the `huggingface-cli` is deprecated in the installed version.
