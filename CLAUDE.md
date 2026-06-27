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

## Face swap pipeline

The face swap / hair transfer client lives in a sibling repo:
`/Users/claudiogomes/Desktop/Github/fswap-comfyui`

See that repo's `CLAUDE.md` for pipeline docs, modes, models, and memory budget.
