#!/usr/bin/env python3
"""
Face swap (+ optional hairstyle transfer) a video via ComfyUI, frame by frame.
Pipelines N frames into the ComfyUI queue for maximum throughput.

NSFW detector disabled — see:
  custom_nodes/comfyui-reactor/scripts/reactor_sfw.py  (returns False)

Usage:
  uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python faceswap_run.py
  uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python faceswap_run.py --hair
  uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python faceswap_run.py --hair --restore
  uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python faceswap_run.py --queue 16

--hair requires two models in models/ (see CLAUDE.md for download links):
  style_models/flux1-redux-dev.safetensors      (3.3 GB)
  clip_vision/sigclip_vision_patch14_384.safetensors  (878 MB)
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
VIDEO_IN      = "/Users/claudiogomes/Downloads/Lana.mp4"
FACE_IN       = "/Users/claudiogomes/Downloads/Kate_2013.png"
VIDEO_OUT     = "/Users/claudiogomes/Downloads/Lana_faceswap.mp4"
COMFY_URL     = "http://127.0.0.1:8188"
COMFY_IN      = "/Users/claudiogomes/Desktop/Github/ComfyUI/input"
COMFY_OUT     = "/Users/claudiogomes/Desktop/Github/ComfyUI/output"
COMFY_MODELS  = "/Users/claudiogomes/Desktop/Github/ComfyUI/models"

HAIR_MODELS = {
    "style_models/flux1-redux-dev.safetensors": "123 MB",
    "clip_vision/sigclip_vision_patch14_384.safetensors": "817 MB",
}

# --- Abort handling ---

_aborted = False

def _handle_sigint(sig, frame):
    global _aborted
    if not _aborted:
        _aborted = True
        print("\n[!] Abort requested — interrupting current frame, "
              "draining queue, stopping cleanly...")
        print("    (press Ctrl+C again to force-quit immediately)")
        # Stop the in-progress render now instead of waiting up to ~46s for it.
        interrupt()
    else:
        # Second Ctrl+C: the user wants out NOW.
        print("\n[!] Force quit.")
        free_models()
        raise SystemExit(130)

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
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        print(f"\n[!] ComfyUI HTTP {e.code}: {detail[:400]}")
        raise


def get_json(path):
    with urllib.request.urlopen(f"{COMFY_URL}{path}") as resp:
        return json.loads(resp.read())


def delete_json(path):
    req = urllib.request.Request(f"{COMFY_URL}{path}", method="DELETE")
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def free_models(unload=True):
    """Ask ComfyUI to unload models + free memory. Best-effort; never raises."""
    try:
        post_json("/free", {"unload_models": unload, "free_memory": True})
    except Exception:
        pass


def interrupt():
    """Interrupt the currently-running prompt on the server. Best-effort."""
    try:
        post_json("/interrupt", {})
    except Exception:
        pass

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

# --- Hair model check ---

def check_hair_models():
    missing = [(p, s) for p, s in HAIR_MODELS.items()
               if not (Path(COMFY_MODELS) / p).exists()]
    if not missing:
        return
    print("\n[!] Missing models required for --hair mode:")
    print("    Download and place them in the models/ directory:\n")
    for rel_path, size in missing:
        print(f"  {rel_path}  ({size})")
    print()
    print("  flux1-redux-dev.safetensors:")
    print("    https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev")
    print("  sigclip_vision_patch14_384.safetensors:")
    print("    https://huggingface.co/Comfy-Org/sigclip_vision_patch14_384")
    print()
    sys.exit(1)

# --- Hair mask generation (uses OpenCV Haar cascade, no extra packages needed) ---

_face_cascade = None

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def generate_hair_mask(frame_path: Path, mask_dir: Path) -> Path:
    """
    Detect the face bounding box and create a soft white ellipse covering
    the hair region (above the forehead).  Falls back to a blank mask if
    no face is found so the frame still passes through unchanged.
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFilter

    img = cv2.imread(str(frame_path))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    mask = Image.new("RGB", (w, h), (0, 0, 0))

    if len(faces) > 0:
        # Pick the largest detected face
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        draw = ImageDraw.Draw(mask)
        # Hair crown: extends ~85% of face height above the forehead,
        # and ~30% wider on each side
        draw.ellipse([
            max(0,  fx - int(fw * 0.30)),
            max(0,  fy - int(fh * 0.85)),
            min(w,  fx + fw + int(fw * 0.30)),
            min(h,  fy + int(fh * 0.05)),
        ], fill=(255, 255, 255))

    # Soft edges blend the inpainting naturally
    mask = mask.filter(ImageFilter.GaussianBlur(radius=18))

    mask_path = mask_dir / frame_path.name
    mask.save(str(mask_path))
    return mask_path

# --- ComfyUI workflow builders ---

def make_workflow(frame_filename: str, face_filename: str, restore: bool) -> dict:
    """Face-swap only (4 nodes)."""
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


def make_hair_workflow(
    frame_filename: str,
    face_filename: str,
    mask_filename: str,
    restore: bool,
    seed: int,
) -> dict:
    """
    Face-swap (ReActor) then hairstyle inpainting (Flux Redux + Fill Dev).

    Node map:
      1  LoadImage  — video frame
      2  LoadImage  — Kate's photo  (used by both ReActor and Redux)
      3  LoadImage  — hair mask PNG
      4  ReActorFaceSwap
      5  UNETLoader — flux1-fill-dev
      6  VAELoader  — ae.safetensors
      7  DualCLIPLoader — clip_l + t5xxl (flux)
      8  CLIPVisionLoader — sigclip_vision_patch14_384
      9  StyleModelLoader — flux1-redux-dev
      10 CLIPTextEncode  positive (empty; Redux drives style)
      11 CLIPTextEncode  negative (empty)
      12 CLIPVisionEncode — encode Kate's image for Redux
      13 StyleModelApply  — merge Redux into positive conditioning
      14 FluxGuidance     — guidance=3.5
      15 ImageToMask      — hair mask image → mask tensor
      16 InpaintModelConditioning — face-swapped image + hair mask
      17 KSampler         — denoise=0.75, 20 steps
      18 VAEDecode
      19 SaveImage
    """
    return {
        # --- Input images ---
        "1": {"class_type": "LoadImage",
              "inputs": {"image": frame_filename, "upload": "image"}},
        "2": {"class_type": "LoadImage",
              "inputs": {"image": face_filename, "upload": "image"}},
        "3": {"class_type": "LoadImage",
              "inputs": {"image": mask_filename, "upload": "image"}},

        # --- Face swap ---
        "4": {
            "class_type": "ReActorFaceSwap",
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
        },

        # --- Model loaders ---
        # weight_dtype is always "default" (bf16/fp16). fp8_e4m3fn is NOT
        # supported by Apple's MPS backend — it loads but crashes the moment
        # the KSampler runs a forward pass. fp8 is a CUDA-only feature.
        "5": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "flux1-fill-dev.safetensors",
                         "weight_dtype": "default"}},
        "6": {"class_type": "VAELoader",
              "inputs": {"vae_name": "ae.safetensors"}},
        "7": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                         "type": "flux"}},
        "8": {"class_type": "CLIPVisionLoader",
              "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}},
        "9": {"class_type": "StyleModelLoader",
              "inputs": {"style_model_name": "flux1-redux-dev.safetensors"}},

        # --- Text conditioning (empty; Redux carries the style) ---
        "10": {"class_type": "CLIPTextEncode",
               "inputs": {"text": "", "clip": ["7", 0]}},
        "11": {"class_type": "CLIPTextEncode",
               "inputs": {"text": "", "clip": ["7", 0]}},

        # --- Redux image conditioning (Kate's photo → style reference) ---
        "12": {"class_type": "CLIPVisionEncode",
               "inputs": {"clip_vision": ["8", 0], "image": ["2", 0], "crop": "center"}},
        "13": {"class_type": "StyleModelApply",
               "inputs": {"conditioning": ["10", 0],
                          "style_model": ["9", 0],
                          "clip_vision_output": ["12", 0],
                          "strength": 1.0,
                          "strength_type": "multiply"}},

        # --- Flux guidance ---
        "14": {"class_type": "FluxGuidance",
               "inputs": {"conditioning": ["13", 0], "guidance": 3.5}},

        # --- Convert mask image to mask tensor ---
        "15": {"class_type": "ImageToMask",
               "inputs": {"image": ["3", 0], "channel": "red"}},

        # --- Inpainting setup: face-swapped image + hair mask ---
        "16": {"class_type": "InpaintModelConditioning",
               "inputs": {
                   "positive": ["14", 0],
                   "negative": ["11", 0],
                   "vae": ["6", 0],
                   "pixels": ["4", 0],
                   "mask": ["15", 0],
                   "noise_mask": True,
               }},

        # --- Sample ---
        "17": {"class_type": "KSampler",
               "inputs": {
                   "model": ["5", 0],
                   "positive": ["16", 0],
                   "negative": ["16", 1],
                   "latent_image": ["16", 2],
                   "seed": seed,
                   "steps": 20,
                   "cfg": 1.0,
                   "sampler_name": "euler",
                   "scheduler": "simple",
                   "denoise": 0.75,
               }},

        # --- Decode and save ---
        "18": {"class_type": "VAEDecode",
               "inputs": {"samples": ["17", 0], "vae": ["6", 0]}},
        "19": {"class_type": "SaveImage",
               "inputs": {
                   "filename_prefix": f"swap/{frame_filename.replace('.png', '')}",
                   "images": ["18", 0],
               }},
    }


def submit(
    frame_filename: str,
    face_filename: str,
    restore: bool,
    mask_filename: str | None = None,
    seed: int = 42,
) -> tuple[str, str]:
    if mask_filename:
        workflow = make_hair_workflow(frame_filename, face_filename,
                                     mask_filename, restore, seed)
        save_node = "19"
    else:
        workflow = make_workflow(frame_filename, face_filename, restore)
        save_node = "4"
    resp = post_json("/prompt", {"prompt": workflow})
    return resp["prompt_id"], save_node


def poll_done(prompt_id: str):
    """Return result dict if done, None if still running, raises on error."""
    history = get_json(f"/history/{prompt_id}")
    if prompt_id not in history:
        return None
    result = history[prompt_id]
    # Check for failure FIRST. ComfyUI writes a history entry with an (often
    # empty) "outputs" key even when the prompt errored, so testing "outputs"
    # before status would mask the real error and return a result with no
    # images — which then crashes collect_output. Surface the server-side
    # error messages so the cause is visible.
    status = result.get("status", {})
    if status.get("status_str") == "execution_error":
        msgs = status.get("messages", [])
        detail = ""
        for kind, info in msgs:
            if kind == "execution_error" and isinstance(info, dict):
                detail = info.get("exception_message", "") or detail
        raise RuntimeError(
            f"ComfyUI execution error for prompt {prompt_id}: {detail or 'see server log'}"
        )
    if "outputs" in result:
        return result
    return None


def collect_output(result: dict, frame_name: str, swapped_dir: Path, save_node: str = "4"):
    node_out = result["outputs"].get(save_node, {})
    images = node_out.get("images", [])
    if not images:
        raise RuntimeError(
            f"Prompt finished but produced no image for {frame_name} "
            f"(save node {save_node}). The frame likely failed on the server."
        )
    saved = images[0]
    out_filename = saved.get("filename", "")
    out_subfolder = saved.get("subfolder", "swap")
    if not out_filename:
        raise RuntimeError(f"Empty output filename for {frame_name}.")
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
    parser.add_argument("--restore", action="store_true",
                        help="Enable CodeFormer face restoration (slower)")
    parser.add_argument("--hair", action="store_true",
                        help="Transfer hairstyle from source photo via Flux Redux/Fill")
    parser.add_argument("--queue", type=int, default=8,
                        help="Pipeline depth: frames submitted ahead (default 8)")
    args = parser.parse_args()

    if args.hair:
        check_hair_models()

    print("=" * 60)
    print("Face Swap Runner (pipelined, memory-safe)")
    print(f"  restore={args.restore}  hair={args.hair}  queue_depth={args.queue}")
    print("  Ctrl+C to abort cleanly")
    print("=" * 60)

    face_filename = Path(FACE_IN).name
    shutil.copy(FACE_IN, Path(COMFY_IN) / face_filename)
    print(f"[✓] Source face ready: {face_filename}")

    fps, duration, nb_frames = get_video_info()
    print(f"[✓] Video: {nb_frames} frames @ {fps:.1f}fps ({duration:.1f}s)")

    frames_dir  = Path("/tmp/faceswap_frames")
    swapped_dir = Path("/tmp/faceswap_swapped")
    masks_dir   = Path("/tmp/faceswap_masks")

    # Always wipe frame dirs so stale files from previous runs don't bleed in
    for d in (frames_dir, swapped_dir, masks_dir):
        if d.exists():
            shutil.rmtree(d)
    frames_dir.mkdir(parents=True)
    swapped_dir.mkdir(parents=True)
    (Path(COMFY_OUT) / "swap").mkdir(parents=True, exist_ok=True)

    if args.hair:
        masks_dir.mkdir(parents=True)
        # Warm up the face cascade once (cheap)
        _get_face_cascade()
        print("[✓] Hair mask generator ready")

    print(f"[…] Extracting frames...")
    extract_frames(frames_dir, fps)
    frame_files = sorted(frames_dir.glob("*.png"))
    total = len(frame_files)
    print(f"[✓] Extracted {total} frames")

    # Pipeline: submit up to queue_depth frames ahead, collect as they finish
    # in_flight: deque of (prompt_id, frame_name, save_node)
    in_flight: deque = deque()
    submit_idx = 0
    completed  = 0
    start_time = time.time()

    print(f"[…] {'Swapping faces + transferring hair' if args.hair else 'Swapping faces'}...")

    while completed < total and not _aborted:
        # Fill the pipeline
        while submit_idx < total and len(in_flight) < args.queue and not _aborted:
            frame_path = frame_files[submit_idx]
            shutil.copy(frame_path, Path(COMFY_IN) / frame_path.name)

            mask_filename = None
            if args.hair:
                mask_path = generate_hair_mask(frame_path, masks_dir)
                mask_dest = f"hairmask_{frame_path.name}"
                shutil.copy(mask_path, Path(COMFY_IN) / mask_dest)
                mask_filename = mask_dest

            pid, save_node = submit(
                frame_path.name, face_filename, args.restore,
                mask_filename=mask_filename, seed=42,
            )
            in_flight.append((pid, frame_path.name, save_node))
            submit_idx += 1

        if not in_flight:
            break

        # Check the oldest in-flight prompt
        pid, frame_name, save_node = in_flight[0]
        try:
            result = poll_done(pid)
            if result is not None:
                collect_output(result, frame_name, swapped_dir, save_node)
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            print("[…] Unloading models from ComfyUI to free memory...")
            interrupt()
            free_models()
            sys.exit(1)

        if result is not None:
            in_flight.popleft()
            completed += 1
            # After the first frame completes, print a whole-job estimate so
            # the user knows up front how long the run will take.
            if completed == 1:
                per_frame = time.time() - start_time
                est_total = per_frame * total
                print(f"\n  First frame: {per_frame:.1f}s  →  "
                      f"estimated total ~{int(est_total // 60)}m{int(est_total % 60):02d}s "
                      f"for {total} frames (refines as it runs)\n")
            print_progress(completed, total, len(in_flight), time.time() - start_time)
        else:
            time.sleep(0.1)

    # If aborted, drain remaining in-flight before reassembling
    if _aborted and in_flight:
        print(f"\n[!] Draining {len(in_flight)} in-flight frames...")
        while in_flight:
            pid, frame_name, save_node = in_flight[0]
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
                collect_output(result, frame_name, swapped_dir, save_node)
                completed += 1
            in_flight.popleft()

    print()

    swapped_frames = sorted(swapped_dir.glob("*.png"))
    if not swapped_frames:
        print("[!] No frames to reassemble.")
        sys.exit(0)

    # Release the Flux/CLIP models the server is still holding so they don't
    # linger in (MPS) memory after we're done.
    if args.hair:
        print("[…] Unloading models from ComfyUI to free memory...")
        free_models()

    status = "partial" if _aborted else "complete"
    print(f"[✓] {len(swapped_frames)} frames processed ({status}). Reassembling...")
    reassemble(swapped_dir, fps)
    print(f"[✓] Done! Output: {VIDEO_OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
