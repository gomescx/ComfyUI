#!/usr/bin/env bash
set -euo pipefail

INPUT="$1"

BASENAME="${INPUT%.*}"
OUTPUT="${BASENAME}_640x480_24fps.mp4"

ffmpeg -y -i "$INPUT" \
  -vf "fps=24,scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 \
  -preset medium \
  -crf 20 \
  -pix_fmt yuv420p \
  -c:a aac \
  -b:a 128k \
  "$OUTPUT"

echo "Created: $OUTPUT"