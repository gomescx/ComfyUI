# Project Initiation Document — MLX (mflux) Hair-Transfer Pipeline

**Status:** Proposed / de-risking phase
**Owner:** Claudio
**Created:** 2026-06-27
**Audience:** A separate Claude Code chat that will build this in parallel with the existing ComfyUI work.

---

## 1. One-line summary

Port the **hair-transfer (Flux Fill + Redux inpainting)** stage of the existing
ComfyUI faceswap pipeline to **native Apple-Silicon MLX via [mflux](https://github.com/filipstrand/mflux)**,
to get ~1.5–2× faster per-frame render than the GGUF-on-MPS path, while fitting
comfortably in available unified memory.

---

## 2. Why this project exists

### Current state (the thing being replaced — for this stage only)
The repo already has a working frame-by-frame video faceswap + optional hair-transfer
pipeline driven entirely through the **ComfyUI HTTP API** ([faceswap_run.py](../faceswap_run.py)).
The hair stage is a 19-node ComfyUI graph:

```
ReActorFaceSwap (face)  →  Flux Fill (inpaint hair region)  conditioned by
Flux Redux (image-prompt embedding of the source photo)  →  KSampler (denoise 0.75)
```

The blocker: **Flux Fill is a 22 GB fp16 UNET.** On this Mac (unified memory) the full
hair pipeline needs ~38 GB free, which is rarely available.

### Two parallel mitigations are being pursued
- **Track A (separate, already in motion in the main chat): GGUF quantization** inside
  ComfyUI. ~3-line change, keeps the whole graph intact, fits 30 GB. Slower per step
  (dequant overhead) but low-effort and low-risk. **This is the safe baseline and will
  ship first.**
- **Track B (THIS document): native MLX via mflux.** Faster per step, but a standalone
  re-architecture. Higher reward, higher risk. **This project.**

These two tracks are independent. Track A is the fallback if Track B proves infeasible.

### Why MLX over GGUF
MLX runs native Metal kernels; GGUF on MPS dequantizes weights to fp16 and runs through
PyTorch-MPS, adding per-step overhead. For a multi-hundred-frame video, the per-frame
delta compounds. Memory footprint is roughly equal (both quantize weights) — **the win is
purely compute speed.**

---

## 3. Hardware / environment (critical constraints)

| Item | Value |
|---|---|
| Machine | Apple M5 Pro, **49 GB unified memory**, macOS 26.5.1 |
| Compute | MPS / Metal; vram state `SHARED` (CPU+GPU share one pool) |
| **Memory failure mode** | There is **no hard VRAM wall.** Exceeding available memory makes macOS page to SSD, which **freezes the machine** rather than throwing a clean OOM. Stay well under the free-memory budget. |
| Python | **Always `uv run`.** Never call a venv binary or system python directly. |
| Existing venv | `/Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python` |
| **fp8 on MPS** | **Does not work** — Apple's MPS backend has no fp8 tensor math. Don't propose fp8. (GGUF and MLX-quant are fine; they dequantize.) |

### uv invocation pattern (from project CLAUDE.md — obey it)
```bash
uv run --python /Users/claudiogomes/Desktop/Github/ComfyUI/.venv/bin/python script.py
```
mflux may warrant its **own** venv to avoid disturbing the ComfyUI environment — decide
during setup (see §7). If so, create it with uv and keep it separate from `.venv`.

---

## 4. Goal & success criteria

**Goal:** Reproduce the hair-transfer visual result of the ComfyUI graph using mflux,
faster, within the memory budget, as a scriptable frame loop.

**Done when:**
1. A single frame can be processed end-to-end: source-photo hair style transferred onto a
   target frame's masked hair region, output visually comparable to the ComfyUI Q6_K/fp16
   result.
2. Peak memory measured and documented; comfortably under ~30 GB free.
3. Per-frame wall-clock time measured and compared against the GGUF ComfyUI path.
4. A frame-loop runner exists that takes a video → frames → (faceswap) → hair → reassemble,
   mirroring [faceswap_run.py](../faceswap_run.py)'s structure.

**Explicitly out of scope (v1):** replacing the ReActor faceswap with an MLX equivalent
(none exists). Keep faceswap as-is (see §6).

---

## 5. THE key risk to de-risk FIRST (do this before anything else)

> **Can mflux fuse Fill + Redux in a single generation?**

The ComfyUI graph feeds a **Redux image embedding** (CLIP-Vision of the source photo via
`StyleModelApply`) into the **Fill inpaint** conditioning. In mflux these appear to be
**separate CLI tools** (`mflux-generate-fill` and `mflux-generate-redux`), and the combined
path is **not confirmed to exist**. This fusion is the whole point of the hair effect —
Redux carries the *style* (the text prompts are empty), Fill restricts it to the *masked
hair region*.

**First task for the new chat:** prove or disprove this fusion on ONE frame, via the mflux
**Python API** (not just CLI), before building any loop. Read mflux's source
(`src/mflux/.../flux_tools/`) to see whether Fill's generation accepts a Redux/image-prompt
conditioning input, or whether it must be patched in. If it can't be fused cleanly, escalate
to Claudio with options (patch mflux, approximate with img2img + Redux, or abandon Track B
and ship GGUF).

Do **not** build the frame loop until this is answered.

---

## 6. Target architecture (v1)

```
video.mp4
  └─(ffmpeg)→ frames/*.png
       └─ per frame:
            1. FACE SWAP   — keep existing ComfyUI ReActor call OR run insightface
                             inswapper_128.onnx directly in python (no MLX equivalent).
                             Simplest v1: reuse the ComfyUI face-only workflow already in
                             faceswap_run.py (make_workflow) to produce the swapped frame.
            2. HAIR MASK   — reuse generate_hair_mask() from faceswap_run.py verbatim
                             (OpenCV Haar cascade → soft ellipse → Gaussian blur). No change.
            3. HAIR FILL   — mflux: Flux Fill inpaint of the masked region, conditioned by
                             Redux embedding of the SOURCE photo. denoise≈0.75, 20 steps,
                             guidance≈3.5, cfg≈1.0, euler/simple (match ComfyUI params).
       └─(ffmpeg)→ reassemble with original audio
```

The faceswap and mask-generation logic already exist and are battle-tested in
[faceswap_run.py](../faceswap_run.py) — **lift them, don't rewrite them.**

### Parameters to match (from the ComfyUI graph, faceswap_run.py make_hair_workflow)
| Param | Value |
|---|---|
| Fill model | FLUX.1-Fill-dev |
| Redux | StyleModelApply strength 1.0, CLIP-Vision crop "center", source = the face photo |
| Text prompt | empty (Redux drives style) |
| FluxGuidance | 3.5 |
| KSampler | euler / simple, 20 steps, cfg 1.0, **denoise 0.75** |
| Mask | InpaintModelConditioning with `noise_mask: True` |

---

## 7. Setup checklist (for the new chat)

1. **Read** [faceswap_run.py](../faceswap_run.py) end-to-end and the repo
   [CLAUDE.md](../CLAUDE.md) — the existing pipeline, model inventory, and memory budget
   are documented there. Reuse `generate_hair_mask()` and the param values.
2. **Install mflux** (likely its own uv-managed venv to isolate from ComfyUI's `.venv`).
   Confirm it imports and MLX sees the GPU.
3. **Model weights:** mflux can download FLUX.1-Fill-dev itself (HuggingFace, **gated** —
   token is in `.env` as `HUGGING_FACE`, use the `huggingface_hub` SDK with `token=`). Use a
   **quantized** load (4-bit or 8-bit) to fit memory. The fp16 Fill safetensors already on
   disk at `models/unet/flux1-fill-dev.safetensors` is PyTorch-format; mflux wants its own
   format/quant — let mflux manage it, don't assume the existing file is reusable.
4. **De-risk §5** on one frame via Python API.
5. **Measure** peak RAM (e.g. sample `ps`/`vm_stat` during a run) and per-frame time.
6. **Then** build the frame-loop runner mirroring faceswap_run.py's pipelined structure.

### Test assets (already on this machine)
| Role | Path |
|---|---|
| Target video | `/Users/claudiogomes/Downloads/Lana.mp4` |
| Source face/hair photo | `/Users/claudiogomes/Downloads/Kate_2013.png` |
| Output | `/Users/claudiogomes/Downloads/Lana_faceswap.mp4` (pick a different name to avoid clobbering the ComfyUI output) |

---

## 8. Risks & open questions

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Fill+Redux fusion unsupported in mflux** (§5) | De-risk on one frame first; escalate before building loop. |
| 2 | Visual result diverges from ComfyUI (different sampler internals) | Side-by-side one-frame comparison vs the GGUF output before committing. |
| 3 | mflux quant quality on Fill inpainting | Try 8-bit first (closer to fp16), drop to 4-bit only if memory forces it. |
| 4 | No MLX faceswap | Keep ReActor/insightface for the face step; MLX only does the hair Fill. |
| 5 | Two venvs / dependency clashes (mlx vs torch) | Isolate mflux in its own uv venv. |
| 6 | macOS SSD-paging freeze if memory overshoots | Measure peak early; keep a hard headroom margin under free RAM. |

---

## 9. Relationship to the main (ComfyUI/GGUF) chat

- The **main chat** is shipping **Track A (GGUF Q6_K, possibly Q8_0)** inside ComfyUI now —
  that is the production path until this MLX track proves itself.
- This MLX track is **exploratory/parallel** and must not modify
  [faceswap_run.py](../faceswap_run.py) or the ComfyUI server config — build alongside, in
  new files (suggest a `mflux/` subdir or separate script).
- If Track B succeeds, we compare quality + speed and decide whether it supersedes Track A.

---

## 10. First message suggestion for the new chat

> "Read docs/mflux_hair_transfer_PID.md. Start with §5: install mflux in its own uv venv and
> prove on ONE frame whether mflux can do Flux Fill inpainting conditioned by a Flux Redux
> image embedding of a source photo, using the Python API. Use the test assets in §7. Don't
> build the frame loop yet — report whether the Fill+Redux fusion works first."
