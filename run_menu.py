#!/usr/bin/env python3
"""
Interactive launcher for faceswap_run.py.
Invoked by run.sh — do not run directly.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────
R  = "\033[0m"
B  = "\033[1m"
G  = "\033[32m"
Y  = "\033[33m"
RD = "\033[31m"
C  = "\033[36m"
D  = "\033[2m"

# ── Constants ─────────────────────────────────────────────
SCRIPT     = Path(__file__).parent / "faceswap_run.py"
COMFY_URL  = "http://127.0.0.1:8188"

VIDEO_IN   = "/Users/claudiogomes/Downloads/Lana.mp4"
FACE_IN    = "/Users/claudiogomes/Downloads/Kate_2013.png"
VIDEO_OUT  = "/Users/claudiogomes/Downloads/Lana_faceswap.mp4"

HAIR_MODELS_GB     = 31   # flux-fill-dev (fp16) + CLIP + VAE + Redux
HAIR_MINIMUM_GB    = 38   # models + working tensors + OS buffer

GGUF_Q6_GB         = 10   # flux1-fill-dev-Q6_K.gguf model RAM
GGUF_Q8_GB         = 13   # flux1-fill-dev-Q8_0.gguf model RAM
GGUF_OVERHEAD_GB   = 7    # CLIP + VAE + Redux + working tensors + OS buffer

MODELS_DIR = Path(__file__).parent / "models" / "unet"
GGUF_Q6_FILE = "flux1-fill-dev-Q6_K.gguf"
GGUF_Q8_FILE = "flux1-fill-dev-Q8_0.gguf"


# ── Helpers ───────────────────────────────────────────────
def hr():
    print("  " + "─" * 56)


def ask(prompt, default=None, valid=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"\n  {prompt}{suffix}: ").strip()
        val = raw or (str(default) if default is not None else "")
        if valid and val not in valid:
            opts = " / ".join(valid)
            print(f"  {RD}Enter one of: {opts}{R}")
            continue
        return val


def comfy_up():
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ── Main ──────────────────────────────────────────────────
def main():
    print()
    print(f"  {B}{'='*58}{R}")
    print(f"  {B}  ComfyUI Face Swap Runner{R}")
    print(f"  {B}{'='*58}{R}")
    print()

    # 1 ── ComfyUI health check
    if comfy_up():
        print(f"  {G}✓{R} ComfyUI is up at {COMFY_URL}")
    else:
        print(f"  {RD}✗ ComfyUI is not running at {COMFY_URL}{R}")
        print()
        print(f"  Start it first:")
        print(f"  {D}  uv run --python .venv/bin/python main.py \\")
        print(f"      --listen 127.0.0.1 --port 8188 \\")
        print(f"      --use-split-cross-attention{R}")
        print()
        sys.exit(1)

    # 2 ── Show hardcoded paths
    print()
    print(f"  {D}Input video : {VIDEO_IN}{R}")
    print(f"  {D}Face photo  : {FACE_IN}{R}")
    print(f"  {D}Output      : {VIDEO_OUT}{R}")
    print()
    hr()

    # 3 ── Available memory
    mem_gb = None
    while mem_gb is None:
        raw = ask("Free memory available right now (GB)")
        try:
            mem_gb = float(raw)
        except ValueError:
            print(f"  {RD}Enter a number, e.g.  30{R}")

    hair_ok  = mem_gb >= HAIR_MINIMUM_GB
    q6_exists = (MODELS_DIR / GGUF_Q6_FILE).exists()
    q8_exists = (MODELS_DIR / GGUF_Q8_FILE).exists()
    q6_min_gb = GGUF_Q6_GB + GGUF_OVERHEAD_GB   # ~17 GB
    q8_min_gb = GGUF_Q8_GB + GGUF_OVERHEAD_GB   # ~20 GB
    q6_ok     = q6_exists and mem_gb >= q6_min_gb
    q8_ok     = q8_exists and mem_gb >= q8_min_gb

    # 4 ── Mode selection
    print()
    hr()
    print()
    print(f"  {B}Select a mode:{R}")
    print()
    print(f"  [1]  Face swap only                      (~1 GB)    {G}available{R}")
    print(f"  [2]  Face swap + CodeFormer restore       (~2 GB)    {G}available{R}")

    def hair_status(min_gb):
        if mem_gb >= min_gb:
            return f"{G}available{R}"
        return f"{RD}need {min_gb - mem_gb:.0f} GB more{R}"

    def gguf_status(exists, min_gb, filename):
        if not exists:
            return f"{RD}not downloaded ({filename}){R}"
        if mem_gb < min_gb:
            return f"{RD}need {min_gb - mem_gb:.0f} GB more{R}"
        return f"{G}available{R}"

    print(f"  [3]  Face swap + hair                    (~38 GB)   {hair_status(HAIR_MINIMUM_GB)}")
    print(f"  [4]  Face swap + hair + restore          (~38 GB)   {hair_status(HAIR_MINIMUM_GB)}")
    print(f"  [5]  Face swap + hair GGUF Q6_K          (~17 GB)   {gguf_status(q6_exists, q6_min_gb, GGUF_Q6_FILE)}")
    print(f"  [6]  Face swap + hair GGUF Q8_0          (~20 GB)   {gguf_status(q8_exists, q8_min_gb, GGUF_Q8_FILE)}")
    print(f"  [7]  Face swap + hair GGUF Q6_K + restore (~17 GB)  {gguf_status(q6_exists, q6_min_gb, GGUF_Q6_FILE)}")
    print(f"  [8]  Face swap + hair GGUF Q8_0 + restore (~20 GB)  {gguf_status(q8_exists, q8_min_gb, GGUF_Q8_FILE)}")

    valid_modes = ["1", "2"]
    if hair_ok:
        valid_modes += ["3", "4"]
    if q6_ok:
        valid_modes += ["5", "7"]
    if q8_ok:
        valid_modes += ["6", "8"]

    mode = ask("Mode", default="1", valid=valid_modes)
    hair    = mode in ("3", "4", "5", "6", "7", "8")
    restore = mode in ("2", "4", "7", "8")
    gguf    = "q6" if mode in ("5", "7") else "q8" if mode in ("6", "8") else None

    # 5 ── Queue depth
    print()
    hr()
    print()
    if hair and gguf == "q6":
        headroom  = max(0, mem_gb - GGUF_Q6_GB - GGUF_OVERHEAD_GB)
        rec_queue = max(1, min(8, int(headroom / 3)))
    elif hair and gguf == "q8":
        headroom  = max(0, mem_gb - GGUF_Q8_GB - GGUF_OVERHEAD_GB)
        rec_queue = max(1, min(8, int(headroom / 3)))
    elif hair:
        headroom  = max(0, mem_gb - HAIR_MODELS_GB - 7)
        rec_queue = max(1, min(8, int(headroom / 3)))
    else:
        rec_queue = 8

    print(f"  {B}Queue depth{R}  (frames processed in parallel)")
    print(f"  {D}Higher = faster but uses more memory.  Recommended for your setup: {rec_queue}{R}")
    q_raw = ask("Queue depth", default=rec_queue)
    try:
        queue = max(1, int(q_raw))
    except ValueError:
        queue = rec_queue

    # 6 ── Confirm
    print()
    hr()
    print()
    mode_label = {
        "1": "Face swap only",
        "2": "Face swap + CodeFormer restore",
        "3": "Face swap + hair transfer (fp16)",
        "4": "Face swap + hair + restore (fp16)",
        "5": "Face swap + hair GGUF Q6_K",
        "6": "Face swap + hair GGUF Q8_0",
        "7": "Face swap + hair GGUF Q6_K + restore",
        "8": "Face swap + hair GGUF Q8_0 + restore",
    }[mode]

    print(f"  {B}Ready to run{R}")
    print()
    print(f"  Mode    : {C}{mode_label}{R}")
    if gguf:
        print(f"  GGUF    : {gguf.upper()}")
    print(f"  Queue   : {queue}")
    print(f"  Input   : {D}{VIDEO_IN}{R}")
    print(f"  Output  : {D}{VIDEO_OUT}{R}")

    confirm = ask("Proceed?", default="y", valid=["y", "Y", "n", "N"])
    if confirm.lower() != "y":
        print()
        print("  Aborted.")
        print()
        sys.exit(0)

    # 7 ── Launch
    print()
    hr()
    print()

    cmd = [sys.executable, str(SCRIPT), "--queue", str(queue)]
    if hair:
        cmd.append("--hair")
    if restore:
        cmd.append("--restore")
    if gguf:
        cmd += ["--gguf", gguf]

    print(f"  {D}$ {' '.join(cmd)}{R}")
    print()
    subprocess.run(cmd)
    print()


if __name__ == "__main__":
    main()
