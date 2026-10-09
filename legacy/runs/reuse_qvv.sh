#!/bin/bash
# Copy identical query_vs_video component from noASR cache to ASR cache to avoid recompute.
# enc = multiclip | internvideo2
enc=$1
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
src=$REPO/runs/msrvtt_${enc}_noASR/cache/query_vs_video.pt
dstdir=$REPO/runs/msrvtt_${enc}_ASR/cache
while [ ! -f "$src" ]; do sleep 20; done
mkdir -p "$dstdir"
# only copy if ASR run hasn't already produced/started it
if [ ! -f "$dstdir/query_vs_video.pt" ]; then
  cp "$src" "$dstdir/query_vs_video.pt"
  echo "[reuse] copied $enc query_vs_video noASR->ASR at $(date +%T)"
fi
