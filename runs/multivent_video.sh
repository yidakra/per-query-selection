#!/bin/bash
# Stage 3: MultiVENT full-video reproduction. Waits for the ablation chain (Funiform-64 ->
# LLaMA-1B retry) to release GPU1, then encodes query_vs_video from the fetched frame caches
# and fuses it with the cached text components (Table-5 ablation, noASR+ASR, options a & b).
set -u
cd /home/ubuntu/q2e_repro
source .venv-eval/bin/activate
export HF_HOME=/home/ubuntu/q2e_repro/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export Q2E_MC_VIDEO_BS=8
RUN=/home/ubuntu/q2e_repro/runs
LOG="$RUN/multivent_video.log"

echo "=== multivent_video queued $(date -u +%FT%TZ), waiting for ablation chain (LLaMA-1B retry) ===" | tee -a "$LOG"
# Gate on the very last ablation marker so we never contend with the sweep or the retry for GPU1.
while [ ! -f "$RUN/LLAMA1B_RETRY_DONE" ]; do
  sleep 120
done
sleep 20
echo "=== encode query_vs_video $(date -u +%FT%TZ) ===" | tee -a "$LOG"
python -u src/evaluation/encode_multivent_video.py >> "$LOG" 2>&1 || { echo "ENCODE_FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"; touch "$RUN/MULTIVENT_VIDEO_DONE"; exit 1; }
echo "=== fuse full-video Table 5 $(date -u +%FT%TZ) ===" | tee -a "$LOG"
python -u src/evaluation/fuse_multivent_fullvideo.py >> "$LOG" 2>&1 && echo "MULTIVENT_VIDEO_OK $(date -u +%FT%TZ)" | tee -a "$LOG" || echo "FUSE_FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"
touch "$RUN/MULTIVENT_VIDEO_DONE"
