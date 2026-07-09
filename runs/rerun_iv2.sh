#!/bin/bash
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
cd $REPO
source .venv-eval/bin/activate
export HF_HOME=$REPO/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export Q2E_COLBERT_ENC_BS=128
export Q2E_COLBERT_SEARCH_BS=512
export Q2E_IV2_VIDEO_BS=8
export Q2E_IV2_TEXT_BS=32
RUN=$REPO/runs
for cfg in noASR ASR; do
  echo "[iv2rerun] === MSRVTT internvideo2 $cfg $(date +%T) ==="
  python -u src/evaluation/run_eval.py \
    --dataset_dir data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_${cfg} \
    --t2v_encoder internvideo2 \
    --tag msrvtt_internvideo2_${cfg} \
    --out "$RUN" && echo "[iv2rerun] $cfg OK" || echo "[iv2rerun] $cfg FAILED"
done
echo "IV2_RERUN_DONE $(date +%T)"
