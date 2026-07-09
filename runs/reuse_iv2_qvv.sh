#!/bin/bash
# Wait for IV2 noASR query_vs_video, copy to ASR cache (identical queries+videos ⇒ identical).
src=runs/msrvtt_internvideo2_noASR/cache/query_vs_video.pt
dst=runs/msrvtt_internvideo2_ASR/cache/query_vs_video.pt
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
cd $REPO
for i in $(seq 1 240); do
  if [ -f "$src" ] && [ ! -f "$dst" ]; then
    cp -p "$src" "$dst" && echo "COPIED iv2 query_vs_video noASR->ASR at $(date +%T)"
    exit 0
  fi
  [ -f "$dst" ] && { echo "ASR qvv already present"; exit 0; }
  sleep 15
done
echo "TIMEOUT waiting for iv2 noASR query_vs_video"; exit 3
