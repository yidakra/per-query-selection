#!/bin/bash
# Retry the LLaMA-1B LLM-sweep config that OOM-crashed at prequel_vs_captions.
# query_vs_captions.pt is already cached and will be reused (run_eval resumes from cache);
# only prequel/during/sequel are recomputed, with a smaller ColBERT encode batch to fit 15 GB.
# Waits for the main sweep (Funiform-64) to release GPU1 before starting.
set -u
cd /home/ubuntu/q2e_repro
source .venv-eval/bin/activate
export HF_HOME=/home/ubuntu/q2e_repro/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export Q2E_COLBERT_ENC_BS=32       # was 128 in the sweep -> OOM on 1B's long event text
export Q2E_COLBERT_SEARCH_BS=256
RUN=/home/ubuntu/q2e_repro/runs
LOG="$RUN/llama1b_retry.log"
NAME=Q2E_MultiVENT_LLAMA_1B_InternVL_38B_Funiform_16_noASR
TAG=mv_llm_llama1b_noASR

echo "=== llama1b_retry queued $(date -u +%FT%TZ), waiting for main sweep to finish ===" | tee -a "$LOG"
# Wait until the whole main sweep is done (marker is touched after ALL configs, even on
# per-config failure) so we never contend with the last config (Funiform-64) for GPU1.
while [ ! -f "$RUN/ABLATION_SWEEP_DONE" ]; do
  sleep 120
done
sleep 20
echo "=== llama1b_retry starting $(date -u +%FT%TZ) ENC_BS=$Q2E_COLBERT_ENC_BS ===" | tee -a "$LOG"

python -u src/evaluation/run_eval.py \
  --dataset_dir "data/MultiVENT/$NAME" \
  --t2v_encoder multiclip --no_video \
  --tag "$TAG" \
  --out "$RUN" >> "$LOG" 2>&1 && echo "LLAMA1B_RETRY_OK $(date -u +%FT%TZ)" | tee -a "$LOG" \
  || echo "LLAMA1B_RETRY_FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"
touch "$RUN/LLAMA1B_RETRY_DONE"
