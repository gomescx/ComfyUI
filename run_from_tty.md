# Running the hair transfer pipeline

Use this when you want to run hair mode (fp16), which needs ~38 GB free.
Must be done from native macOS Terminal — NOT from VS Code's terminal.

## Steps

1. Close VS Code and any other heavy apps (browser, etc.) to free RAM.

2. Open macOS Terminal. Check free memory — you need 38 GB+:
   ```
   python3 -c "import psutil; print(f'{psutil.virtual_memory().available / 1e9:.1f} GB free')"
   ```

3. Start the ComfyUI server in this terminal (leave it running):
   ```
   cd ~/Desktop/Github/ComfyUI
   uv run --python .venv/bin/python main.py --listen 127.0.0.1 --port 8188 --use-split-cross-attention
   ```

4. Open a second Terminal tab (Cmd+T) and run the launcher:
   ```
   cd ~/Desktop/Github/ComfyUI
   ./run.sh
   ```

5. In the menu:
   - Enter your free GB when asked
   - Pick mode **3** (face swap + hair, fp16)
   - Queue depth: **1**
   - Confirm and let it run

## Output

`/Users/claudiogomes/Downloads/Lana_faceswap.mp4`
