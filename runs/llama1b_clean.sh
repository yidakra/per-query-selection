#!/usr/bin/env bash
# Clean LLaMA-1B ablation relaunch at the proven LLaMA-8B config (ENC_BS=128, SEARCH_BS=512).
# NO swap reliance (the previous halved-batch + 48G-swap config thrashed at 14422 s/it).
# run_eval.py resumes from the cached query_vs_captions.pt and computes prequel/during/sequel.
# GPU1 only (GPU0/whisper untouched). On OOM/error -> FAILED marker (no swap fallback).
set -uo pipefail
cd /home/ubuntu/q2e_repro
export HF_HOME=/home/ubuntu/q2e_repro/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export Q2E_COLBERT_ENC_BS=128      # proven 8B config
export Q2E_COLBERT_SEARCH_BS=512   # proven 8B config
LOG=/home/ubuntu/q2e_repro/runs/llama1b_clean.log
RUN=/home/ubuntu/q2e_repro/runs
rm -f "$RUN/LLAMA1B_CLEAN_DONE" "$RUN/LLAMA1B_CLEAN_FAILED"
echo "=== llama1b clean relaunch ENC_BS=128 SEARCH_BS=512 $(date -u +%FT%TZ) ===" | tee -a "$LOG"
source /home/ubuntu/q2e_repro/.venv-eval/bin/activate
python -u src/evaluation/run_eval.py \
  --dataset_dir data/MultiVENT/Q2E_MultiVENT_LLAMA_1B_InternVL_38B_Funiform_16_noASR \
  --t2v_encoder multiclip --no_video \
  --tag mv_llm_llama1b_noASR \
  --out "$RUN" >> "$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "LLAMA1B_CLEAN_OK $(date -u +%FT%TZ)" | tee -a "$LOG"
  touch "$RUN/LLAMA1B_CLEAN_DONE"
else
  echo "LLAMA1B_CLEAN_FAILED rc=$rc $(date -u +%FT%TZ) (do NOT retry with swap; park + document instead)" | tee -a "$LOG"
  touch "$RUN/LLAMA1B_CLEAN_FAILED"
fi
