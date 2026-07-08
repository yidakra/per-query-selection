#!/bin/bash
# Finish the MultiVENT reproduction after the VPS restarts. Reprioritized:
#   1. MultiVENT full-video encode + fuse  (the downstream need; ~2h; independent of the 1B cell)
#   2. LLaMA-1B ablation cell               (best-effort; ENC_BS=64 -> fits 15 GB, ~2x faster than 32)
# Both on GPU1; whisper owns GPU0 (systemd). Each step resumes from its component cache, so a
# restart mid-run only loses the in-flight step, not the finished ones.
set -u
cd /home/ubuntu/q2e_repro
source .venv-eval/bin/activate
export HF_HOME=/home/ubuntu/q2e_repro/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=/home/ubuntu/q2e_repro/runs
LOG="$RUN/repro_finish.log"

echo "=== repro_finish started $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# --- Priority 1: MultiVENT full-video reproduction ---
export Q2E_MC_VIDEO_BS=8
echo "=== [1/2] video encode query_vs_video $(date -u +%FT%TZ) ===" | tee -a "$LOG"
if python -u src/evaluation/encode_multivent_video.py >> "$LOG" 2>&1; then
  echo "=== [1/2] fuse full-video Table 5 $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  python -u src/evaluation/fuse_multivent_fullvideo.py >> "$LOG" 2>&1 \
    && echo "VIDEO_OK $(date -u +%FT%TZ)" | tee -a "$LOG" \
    || echo "FUSE_FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"
else
  echo "ENCODE_FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"
fi
touch "$RUN/MULTIVENT_VIDEO_DONE"

# --- Priority 2: LLaMA-1B ablation cell (best-effort) ---
export Q2E_COLBERT_ENC_BS=64
export Q2E_COLBERT_SEARCH_BS=256
NAME=Q2E_MultiVENT_LLAMA_1B_InternVL_38B_Funiform_16_noASR
echo "=== [2/2] llama1b retry ENC_BS=64 $(date -u +%FT%TZ) ===" | tee -a "$LOG"
python -u src/evaluation/run_eval.py \
  --dataset_dir "data/MultiVENT/$NAME" \
  --t2v_encoder multiclip --no_video \
  --tag mv_llm_llama1b_noASR \
  --out "$RUN" >> "$LOG" 2>&1 \
  && echo "LLAMA1B_OK $(date -u +%FT%TZ)" | tee -a "$LOG" \
  || echo "LLAMA1B_FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"
touch "$RUN/LLAMA1B_RETRY_DONE"
echo "=== repro_finish done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
