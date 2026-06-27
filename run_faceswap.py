#!/usr/bin/env python3
"""
Face swap a video using ComfyUI ReActor, frame by frame via ffmpeg.
Pipelines N frames into the ComfyUI queue for maximum throughput.

Essential to ensure CompfyUI is set for sfw

/Users/claudiogomes/Desktop/Github/ComfyUI/custom_nodes/comfyui-reactor/scripts/reactor_sfw.py
def nsfw_image(img_data, model_path: str):
    # Local install: NSFW detector false-positives on ordinary footage
    # (score ~0.996 on Lana.mp4), blanking ~98% of frames to black. Disabled.
    return False

Usage:
  uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python run_faceswap.py
  uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python run_faceswap.py --restore  # enable CodeFormer (slower, better quality)
  uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python run_faceswap.py --queue 16 # change pipeline depth
"""

import argparse
import json
import subprocess
import sys
import time
import shutil
import signal
import urllib.request
from collections import deque
from pathlib import Path

# --- Config ---
VIDEO_IN  = "/Users/claudiogomes/Downloads/Lana.mp4"
FACE_IN   = "/Users/claudiogomes/Downloads/Kate_2013.png"
VIDEO_OUT = "/Users/claudiogomes/Downloads/Lana_faceswap.mp4"
COMFY_URL = "http://127.0.0.1:8188"
COMFY_IN  = "/Users/claudiogomes/Desktop/Github/ComfyUI/input"
COMFY_OUT = "/Users/claudiogomes/Desktop/Github/ComfyUI/output"

# --- Abort handling ---

_aborted = False

def _handle_sigint(sig, frame):
    global _aborted
    if not _aborted:
        _aborted = True
        print("\n[!] Abort requested — draining queue then stopping cleanly...")

signal.signal(signal.SIGINT, _handle_sigint)

# --- HTTP helpers ---

def post_json(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{COMFY_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get_json(path):
    with urllib.request.urlopen(f"{COMFY_URL}{path}") as resp:
        return json.loads(resp.read())


def delete_json(path):
    req = urllib.request.Request(f"{COMFY_URL}{path}", method="DELETE")
    with urllib.request.urlopen(req) as resp:
        return resp.read()

# --- ffmpeg helpers ---

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\nERROR: {cmd}\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def get_video_info():
    out = run(
        f'ffprobe -v error -select_streams v:0 '
        f'-show_entries stream=r_frame_rate,duration,nb_frames '
        f'-of csv=p=0 "{VIDEO_IN}"'
    )
    parts = out.split(",")
    fps_num, fps_den = map(int, parts[0].split("/"))
    fps = fps_num / fps_den
    duration = float(parts[1])
    nb_frames = int(parts[2])
    return fps, duration, nb_frames


def extract_frames(frames_dir: Path, fps: float):
    frames_dir.mkdir(parents=True, exist_ok=True)
    run(
        f'ffmpeg -y -i "{VIDEO_IN}" '
        f'-vf "fps={fps}" '
        f'"{frames_dir}/%06d.png"'
    )


def reassemble(swapped_dir: Path, fps: float):
    run(
        f'ffmpeg -y -framerate {fps} '
        f'-pattern_type glob -i "{swapped_dir}/*.png" '
        f'-i "{VIDEO_IN}" '
        f'-map 0:v -map 1:a '
        f'-c:v libx264 -preset fast -crf 18 '
        f'-c:a aac -shortest '
        f'"{VIDEO_OUT}"'
    )

# --- ComfyUI helpers ---

def make_workflow(frame_filename: str, face_filename: str, restore: bool) -> dict:
    return {
        "1": {
            "inputs": {"image": frame_filename, "upload": "image"},
            "class_type": "LoadImage",
        },
        "2": {
            "inputs": {"image": face_filename, "upload": "image"},
            "class_type": "LoadImage",
        },
        "3": {
            "inputs": {
                "enabled": True,
                "input_image": ["1", 0],
                "source_image": ["2", 0],
                "swap_model": "inswapper_128.onnx",
                "facedetection": "retinaface_resnet50",
                "face_restore_model": "codeformer.pth" if restore else "none",
                "face_restore_visibility": 1.0,
                "codeformer_weight": 0.5,
                "detect_gender_input": "no",
                "detect_gender_source": "no",
                "input_faces_index": "0",
                "source_faces_index": "0",
                "console_log_level": 0,
            },
            "class_type": "ReActorFaceSwap",
        },
        "4": {
            "inputs": {
                "filename_prefix": f"swap/{frame_filename.replace('.png', '')}",
                "images": ["3", 0],
            },
            "class_type": "SaveImage",
        },
    }


def submit(frame_filename: str, face_filename: str, restore: bool) -> str:
    resp = post_json("/prompt", {"prompt": make_workflow(frame_filename, face_filename, restore)})
    return resp["prompt_id"]


def poll_done(prompt_id: str):
    """Return result dict if done, None if still running, raises on error."""
    history = get_json(f"/history/{prompt_id}")
    if prompt_id not in history:
        return None
    result = history[prompt_id]
    if "outputs" in result:
        return result
    if result.get("status", {}).get("status_str") == "execution_error":
        raise RuntimeError(f"Execution error for prompt {prompt_id}")
    return None


def collect_output(result: dict, frame_name: str, swapped_dir: Path):
    node_out = result["outputs"].get("4", {})
    saved = node_out.get("images", [{}])[0]
    out_filename = saved.get("filename", "")
    out_subfolder = saved.get("subfolder", "swap")
    src = Path(COMFY_OUT) / out_subfolder / out_filename
    shutil.copy(src, swapped_dir / frame_name)

# --- Progress ---

def print_progress(completed: int, total: int, in_flight: int, elapsed: float):
    pct = completed / total
    bar_len = 28
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    rate = completed / elapsed if elapsed > 0 else 0
    eta = (total - completed) / rate if rate > 0 else 0
    eta_str = f"{int(eta // 60)}m{int(eta % 60):02d}s" if eta > 0 else "--"
    print(
        f"\r  [{bar}] {completed}/{total} ({pct*100:.1f}%)  "
        f"{rate:.1f} fr/s  queue={in_flight}  ETA {eta_str}   ",
        end="", flush=True,
    )

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true", help="Enable CodeFormer face restoration (slower)")
    parser.add_argument("--queue", type=int, default=8, help="Pipeline depth: frames submitted ahead (default 8)")
    args = parser.parse_args()

    print("=" * 60)
    print("Face Swap Runner (pipelined, memory-safe)")
    print(f"  restore={args.restore}  queue_depth={args.queue}")
    print("  Ctrl+C to abort cleanly")
    print("=" * 60)

    face_filename = Path(FACE_IN).name
    shutil.copy(FACE_IN, Path(COMFY_IN) / face_filename)
    print(f"[✓] Source face ready: {face_filename}")

    fps, duration, nb_frames = get_video_info()
    print(f"[✓] Video: {nb_frames} frames @ {fps:.1f}fps ({duration:.1f}s)")

    frames_dir = Path("/tmp/faceswap_frames")
    swapped_dir = Path("/tmp/faceswap_swapped")
    swapped_dir.mkdir(parents=True, exist_ok=True)
    (Path(COMFY_OUT) / "swap").mkdir(parents=True, exist_ok=True)

    print(f"[…] Extracting frames...")
    extract_frames(frames_dir, fps)
    frame_files = sorted(frames_dir.glob("*.png"))
    total = len(frame_files)
    print(f"[✓] Extracted {total} frames")

    # Pipeline: submit up to queue_depth frames ahead, collect as they finish
    # in_flight: deque of (prompt_id, frame_name)
    in_flight: deque = deque()
    submit_idx = 0
    completed = 0
    start_time = time.time()

    print(f"[…] Swapping faces...")

    while completed < total and not _aborted:
        # Fill the pipeline
        while submit_idx < total and len(in_flight) < args.queue and not _aborted:
            frame_path = frame_files[submit_idx]
            shutil.copy(frame_path, Path(COMFY_IN) / frame_path.name)
            pid = submit(frame_path.name, face_filename, args.restore)
            in_flight.append((pid, frame_path.name))
            submit_idx += 1

        if not in_flight:
            break

        # Check the oldest in-flight prompt
        pid, frame_name = in_flight[0]
        try:
            result = poll_done(pid)
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)

        if result is not None:
            collect_output(result, frame_name, swapped_dir)
            in_flight.popleft()
            completed += 1
            print_progress(completed, total, len(in_flight), time.time() - start_time)
        else:
            time.sleep(0.1)

    # If aborted, drain remaining in-flight before reassembling
    if _aborted and in_flight:
        print(f"\n[!] Draining {len(in_flight)} in-flight frames...")
        while in_flight:
            pid, frame_name = in_flight[0]
            while True:
                try:
                    result = poll_done(pid)
                except RuntimeError:
                    result = {}
                    break
                if result is not None:
                    break
                time.sleep(0.2)
            if result and "outputs" in result:
                collect_output(result, frame_name, swapped_dir)
                completed += 1
            in_flight.popleft()

    print()

    swapped_frames = sorted(swapped_dir.glob("*.png"))
    if not swapped_frames:
        print("[!] No frames to reassemble.")
        sys.exit(0)

    status = "partial" if _aborted else "complete"
    print(f"[✓] {len(swapped_frames)} frames swapped ({status}). Reassembling...")
    reassemble(swapped_dir, fps)
    print(f"[✓] Done! Output: {VIDEO_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
