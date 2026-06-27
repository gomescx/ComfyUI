# Hair Keyframe Propagation — Implementation Handover

## Objective

The current `faceswap_run.py` pipeline can face-swap a video frame by frame. When `--hair` is enabled, it also runs Flux Fill + Redux hair inpainting for every frame. This is too slow: at approximately 2 minutes per frame, a 10-second clip becomes a multi-hour render.

The new objective is to avoid running Flux hair generation on every frame.

Instead, implement a staged keyframe workflow:

```text
Face swap all frames
→ run hair generation only on selected keyframes
→ extract the generated hair region as an overlay
→ apply / track / interpolate that overlay across the rest of the frames
→ reassemble the final video
```

The first implementation should be deliberately simple. Do not try to solve perfect temporal consistency immediately. Build it in progressive steps, testing each step before moving to the next.

## Important constraints

Do not modify the existing resolution/FPS downscaling script. That has already been solved elsewhere.

Preserve the current modes:

```text
face swap only
face swap + restore
face swap + hair
face swap + hair + restore
face swap + hair GGUF q6/q8
face swap + hair GGUF q6/q8 + restore
```

Add the new behaviour as an additional mode or CLI option, not as a destructive replacement of the existing per-frame hair mode.

Suggested CLI options:

```bash
--hair-keyframes 24
--hair-propagation static
--hair-propagation tracked
--hair-test-frames 30
```

Suggested meaning:

```text
--hair-keyframes 24
    Run Flux hair generation every 24 frames only.

--hair-propagation static
    Apply the first generated hair overlay to all frames without tracking.
    This is the simplest proof mode.

--hair-propagation tracked
    Track the face/head position and move/scale the overlay per frame.

--hair-test-frames 30
    Process only the first 30 frames for testing.
```

If adding all flags at once is too much, start with only:

```bash
--hair-keyframes 24
```

and make it use static propagation initially.

## Current pipeline summary

The current `faceswap_run.py` does this:

```text
1. Copy source face image into ComfyUI input.
2. Read video metadata.
3. Extract all frames to /tmp/faceswap_frames.
4. For each frame:
   - copy frame into ComfyUI input
   - optionally generate a hair mask
   - submit a ComfyUI workflow
   - collect the output frame
5. Reassemble swapped frames into the final video.
```

The expensive path is inside `make_hair_workflow()`, where the workflow runs:

```text
ReActor face swap
→ Flux Fill / Redux conditioning
→ KSampler
→ VAE decode
→ SaveImage
```

The new keyframe mode should avoid calling `make_hair_workflow()` for every frame.

## Target architecture

The new keyframe mode should split the work into separate stages:

```text
Stage A — Extract frames
Stage B — Face swap all frames, no hair
Stage C — Select hair keyframes
Stage D — Run hair workflow only on selected keyframes
Stage E — Build hair overlays from keyframe outputs
Stage F — Apply overlays to non-keyframe frames
Stage G — Reassemble final video
```

In other words:

```text
Current expensive approach:
for every frame:
    face swap + Flux hair generation

New approach:
for every frame:
    face swap only

for selected keyframes only:
    Flux hair generation

for every frame:
    apply generated hair overlay using static or tracked propagation
```

## Phase 1 — Add a small frame-limit test option

### Change

Add a CLI option:

```bash
--limit-frames N
```

This should limit the number of frames processed after extraction.

Example:

```bash
python faceswap_run.py --limit-frames 30
```

Expected behaviour:

```text
- Extract all frames as usual, or extract normally then slice the list.
- Only submit/process the first N frames.
- Reassemble only the processed N frames.
```

This is useful for testing later phases quickly.

### Acceptance test

Run:

```bash
python faceswap_run.py --limit-frames 10
```

Expected:

```text
- Exactly 10 frames are processed.
- A short output video is created.
- Existing normal behaviour still works when --limit-frames is omitted.
```

Stop after this phase and report results.

## Phase 2 — Separate face-swap-only output from final output

### Change

Introduce a clear intermediate directory for face-swapped frames:

```text
/tmp/faceswap_face_only
```

In the new keyframe mode, every frame should first be processed using the existing face-swap-only workflow, without Flux hair.

Suggested directories:

```text
/tmp/faceswap_frames        original extracted frames
/tmp/faceswap_face_only     face-swapped frames before hair
/tmp/faceswap_hair_keys     Flux-generated keyframe hair outputs
/tmp/faceswap_final         final composited frames
/tmp/faceswap_masks         generated hair masks
```

Do not break the existing `swapped_dir` behaviour for the old modes. This separation is mainly for the new keyframe mode.

### Acceptance test

Run:

```bash
python faceswap_run.py --limit-frames 10
```

Expected:

```text
- Existing face-swap-only mode still works.
```

Then run the new experimental path, if already exposed:

```bash
python faceswap_run.py --hair-keyframes 24 --limit-frames 10
```

Expected at this phase:

```text
- The script creates face-swapped intermediate frames.
- It does not yet need to apply hair propagation.
- It should fail gracefully or stop with a clear message if later phases are not implemented yet.
```

Stop after this phase and report results.

## Phase 3 — Select hair keyframes

### Change

Add a function:

```python
def select_keyframes(frame_files: list[Path], interval: int) -> list[Path]:
    ...
```

Basic rule:

```text
Always include the first frame.
Then include every Nth frame.
Always include the last frame if it is not already included.
```

Example:

```text
frames: 1..100
interval: 24
selected: 1, 25, 49, 73, 97, 100
```

Use this only in the new keyframe hair mode.

### Acceptance test

Add a small internal test or temporary printout.

Run:

```bash
python faceswap_run.py --hair-keyframes 24 --limit-frames 60
```

Expected selected keyframes:

```text
frame 1
frame 25
frame 49
frame 60
```

Stop after this phase and report results.

## Phase 4 — Run Flux hair only on keyframes

### Change

In the new keyframe mode:

1. Generate face-swapped frames for all frames.
2. Select keyframes.
3. For selected keyframes only, submit the existing hair workflow.

Important: the hair workflow should use the face-swapped keyframe as its input image, not the original frame.

This may require copying the face-swapped keyframe into ComfyUI input before submitting `make_hair_workflow()`.

Expected result:

```text
/tmp/faceswap_face_only/000001.png
/tmp/faceswap_face_only/000002.png
...

/tmp/faceswap_hair_keys/000001.png
/tmp/faceswap_hair_keys/000025.png
...
```

Only the keyframes should exist in `/tmp/faceswap_hair_keys`.

### Acceptance test

Run:

```bash
python faceswap_run.py --hair --gguf q6 --hair-keyframes 24 --limit-frames 30
```

Expected:

```text
- 30 face-swapped frames are produced.
- Hair generation runs only on frame 1 and frame 25 and frame 30 if last-frame inclusion is implemented.
- It should not run Flux hair on all 30 frames.
```

The log should make this explicit, for example:

```text
[✓] Face-swapped 30 frames
[✓] Selected 3 hair keyframes: 000001, 000025, 000030
[…] Running Flux hair only on 3 keyframes
```

Stop after this phase and report results.

## Phase 5 — Build a hair overlay from one keyframe

### Change

Create a function that compares:

```text
face-only keyframe
hair-generated keyframe
hair mask
```

and outputs a transparent RGBA overlay containing only the generated hair region.

Suggested function:

```python
def make_hair_overlay(
    face_only_frame: Path,
    hair_frame: Path,
    mask_path: Path,
    overlay_path: Path,
) -> Path:
    ...
```

Suggested implementation:

```text
1. Load face-only frame as RGB.
2. Load hair-generated frame as RGB.
3. Load mask as grayscale.
4. Blur/soften the mask slightly if needed.
5. Create RGBA image:
   - RGB comes from hair-generated frame.
   - Alpha comes from mask.
6. Save overlay as PNG.
```

Do not overcomplicate this phase by calculating pixel differences. Use the mask as alpha first. Difference-based refinement can come later.

Output example:

```text
/tmp/faceswap_hair_overlays/000001.png
```

### Acceptance test

Run:

```bash
python faceswap_run.py --hair --gguf q6 --hair-keyframes 24 --limit-frames 10
```

Expected:

```text
- At least one RGBA overlay is created.
- Opening the overlay image should show only the hair region on transparency.
- The overlay dimensions match the video frame dimensions.
```

Stop after this phase and report results.

## Phase 6 — Static overlay propagation proof

### Change

Create a simple propagation mode:

```bash
--hair-propagation static
```

For the first implementation:

```text
- Use the first generated hair overlay.
- Composite it onto every face-swapped frame at the same position.
- Save the composited frames into /tmp/faceswap_final.
```

Suggested function:

```python
def composite_overlay_static(
    base_frame: Path,
    overlay_path: Path,
    output_path: Path,
) -> Path:
    ...
```

Implementation:

```text
1. Open base frame as RGBA.
2. Open overlay as RGBA.
3. Alpha composite overlay over base.
4. Convert back to RGB.
5. Save output PNG.
```

This phase intentionally ignores head movement. The purpose is only to prove that one generated hair result can be reused across many frames.

### Acceptance test

Run:

```bash
python faceswap_run.py --hair --gguf q6 --hair-keyframes 999 --hair-propagation static --limit-frames 30
```

Expected:

```text
- Flux hair generation runs once, or once plus last frame if last-frame inclusion remains active.
- All 30 final frames are produced.
- The output video is created.
- The result may not be perfect, but the hair overlay should appear on all frames.
```

If the output looks bad but the pipeline works, that is still a pass for this phase.

Stop after this phase and report results.

## Phase 7 — Track face position and move the overlay

### Change

Add a tracked propagation mode:

```bash
--hair-propagation tracked
```

Use the existing OpenCV Haar cascade as the first tracking implementation, because the current script already uses it for mask generation.

Suggested function:

```python
def detect_face_bbox(frame_path: Path) -> tuple[int, int, int, int] | None:
    ...
```

Suggested tracking data:

```python
@dataclass
class FaceTrack:
    frame_name: str
    bbox: tuple[int, int, int, int] | None
    center_x: float
    center_y: float
    scale: float
```

For each frame:

```text
1. Detect face bbox.
2. If detection fails, reuse previous successful bbox.
3. Compare current bbox to keyframe bbox.
4. Translate overlay by dx/dy.
5. Optionally scale overlay by current_face_width / keyframe_face_width.
6. Composite transformed overlay onto the current frame.
```

Keep this simple:

```text
- Translation is required.
- Scaling is useful but optional in the first pass.
- Rotation/warping is not required yet.
```

### Acceptance test

Run:

```bash
python faceswap_run.py --hair --gguf q6 --hair-keyframes 999 --hair-propagation tracked --limit-frames 30
```

Expected:

```text
- Flux hair generation runs once, or once plus last frame depending on keyframe logic.
- The overlay follows the head position better than static mode.
- If face detection fails on a frame, the script does not crash; it reuses the last known bbox.
```

Stop after this phase and report results.

## Phase 8 — Multiple keyframes with nearest-keyframe propagation

### Change

When multiple hair keyframes exist, apply the overlay from the nearest generated keyframe.

Example:

```text
Keyframes: 1, 25, 49
Frames 1-13 use keyframe 1 overlay.
Frames 14-37 use keyframe 25 overlay.
Frames 38-49 use keyframe 49 overlay.
```

Suggested function:

```python
def nearest_keyframe(frame_index: int, keyframe_indices: list[int]) -> int:
    ...
```

Use the corresponding overlay and corresponding keyframe face bbox as the reference for tracking.

### Acceptance test

Run:

```bash
python faceswap_run.py --hair --gguf q6 --hair-keyframes 24 --hair-propagation tracked --limit-frames 60
```

Expected:

```text
- Hair generation runs only on selected keyframes.
- Non-keyframe frames use the nearest overlay.
- Final video is produced.
- Runtime is much closer to number_of_keyframes × Flux time, not total_frames × Flux time.
```

Stop after this phase and report results.

## Phase 9 — Optional crossfade between keyframes

Only do this after Phase 8 works.

### Change

Instead of abruptly switching overlays at the midpoint between keyframes, blend between neighbouring overlays over a short transition window.

Suggested flag:

```bash
--hair-crossfade 4
```

Meaning:

```text
Blend overlays across 4 frames around the switch point.
```

This is optional. If implementation gets complicated, skip it.

### Acceptance test

Run:

```bash
python faceswap_run.py --hair --gguf q6 --hair-keyframes 24 --hair-propagation tracked --hair-crossfade 4 --limit-frames 60
```

Expected:

```text
- Final video is produced.
- There is less popping when switching from one generated hair keyframe to another.
```

Stop after this phase and report results.

## Phase 10 — Optional mask tracking upgrade

Only do this after the overlay approach works.

The current mask is generated per frame using a simple face detector and ellipse. That is acceptable for a first version.

Later, consider a better video mask tracker:

```text
- SAM2-based mask tracking
- MediaPipe face mesh
- optical flow
- manual keyframe masks
```

Do not implement this until the keyframe overlay pipeline is already working.

## Logging requirements

Add clear logs for the new keyframe mode.

Example:

```text
[✓] Extracted 240 frames
[…] Running face swap only on all frames
[✓] Face-swapped 240 frames
[✓] Selected 11 hair keyframes: 000001, 000025, 000049, ...
[…] Running Flux hair on 11 keyframes only
[✓] Built 11 hair overlays
[…] Propagating hair overlays using tracked mode
[✓] Final frames written: 240
[✓] Reassembled final video
```

The most important log line is the one that proves Flux is not running on every frame:

```text
Running Flux hair on 11 keyframes only
```

## Error handling requirements

The script should fail gracefully if:

```text
- --hair-keyframes is used without --hair
- no hair keyframes are generated
- a keyframe hair output is missing
- overlay dimensions do not match frame dimensions
- face detection fails on the first keyframe in tracked mode
```

For tracked mode, if face detection fails after the first valid frame:

```text
- Reuse the previous bbox.
- Log a warning.
- Continue.
```

## Suggested implementation order for the coding LLM

Do not implement everything at once.

Use this instruction pattern:

```text
Implement Phase 1 only. Stop after Phase 1. Show me the code changes and the test command.
```

Then:

```text
Implement Phase 2 only. Stop after Phase 2. Show me the code changes and the test command.
```

Continue one phase at a time.

## Definition of done

The new workflow is successful when this command:

```bash
python faceswap_run.py --hair --gguf q6 --hair-keyframes 24 --hair-propagation tracked
```

produces a final video where:

```text
- face swap still works
- Flux hair generation runs only on selected keyframes
- non-keyframe frames receive propagated hair overlays
- the script reassembles the final video with audio
- runtime scales primarily with number of keyframes, not total frame count
```

Visual quality can be improved later. The first goal is architectural: stop doing Flux hair generation on every single frame.
