#!/bin/bash
# Full MultiVENT video fetch + frame-cache extraction (GPU-free, resumable, disk-safe).
# Runs alongside the ablation sweep; touches no GPU. Frame cache -> data/models/MultiCLIP/clip_frames/16.
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
cd $REPO
source .venv-eval/bin/activate
export CUDA_VISIBLE_DEVICES=""   # force CPU; never contend with the sweep on GPU1
RUN=$REPO/runs
LOG="$RUN/fetch_videos.log"
echo "=== fetch_videos started $(date -u +%FT%TZ) ===" | tee -a "$LOG"
python -u src/evaluation/fetch_all_videos.py >> "$LOG" 2>&1 \
  && echo "FETCH_VIDEOS_OK $(date -u +%FT%TZ)" | tee -a "$LOG" \
  || echo "FETCH_VIDEOS_FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"
touch "$RUN/FETCH_VIDEOS_DONE"
