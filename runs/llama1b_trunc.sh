#!/usr/bin/env bash
# LLaMA-1B ablation, TRUNCATION retry. Root cause (confirmed 3x): 1B emits degenerate events with up
# to 270 paraphrases each (vs 8B's max 35); get_many_to_many_score accumulates T*mx_queries ColBERT
# queries -> ~70k-query encode that OOMs/stalls. Fix: Q2E_EVENT_MAXPARAS=32 caps paraphrases/event
# -> peak memory <= the successful 8B run (max 35). Documented divergence. 8B batch config; no swap.
# Resumes from cached query_vs_captions.pt (query is NOT an event -> untouched -> faithful).
# GPU1 only; GPU0/whisper untouched. On error -> FAILED marker (no further auto-retry).
set -uo pipefail
cd /home/ubuntu/q2e_repro
export HF_HOME=/home/ubuntu/q2e_repro/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export Q2E_COLBERT_ENC_BS=128
export Q2E_COLBERT_SEARCH_BS=512
export Q2E_EVENT_MAXPARAS=32       # the divergence knob (cap paraphrases/event; 8B maxed at 35)
LOG=/home/ubuntu/q2e_repro/runs/llama1b_trunc.log
RUN=/home/ubuntu/q2e_repro/runs
rm -f "$RUN/LLAMA1B_TRUNC_DONE" "$RUN/LLAMA1B_TRUNC_FAILED"
echo "=== llama1b TRUNC retry EVENT_MAXPARAS=32 ENC_BS=128 SEARCH_BS=512 $(date -u +%FT%TZ) ===" | tee -a "$LOG"
source /home/ubuntu/q2e_repro/.venv-eval/bin/activate
python -u src/evaluation/run_eval.py \
  --dataset_dir data/MultiVENT/Q2E_MultiVENT_LLAMA_1B_InternVL_38B_Funiform_16_noASR \
  --t2v_encoder multiclip --no_video \
  --tag mv_llm_llama1b_noASR \
  --out "$RUN" >> "$LOG" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  echo "LLAMA1B_TRUNC_OK $(date -u +%FT%TZ)" | tee -a "$LOG"
  touch "$RUN/LLAMA1B_TRUNC_DONE"
else
  echo "LLAMA1B_TRUNC_FAILED rc=$rc $(date -u +%FT%TZ)" | tee -a "$LOG"
  touch "$RUN/LLAMA1B_TRUNC_FAILED"
fi
