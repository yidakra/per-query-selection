#!/bin/bash
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
cd $REPO
source .venv-eval/bin/activate
export HF_HOME=$REPO/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export Q2E_COLBERT_ENC_BS=128
export Q2E_COLBERT_SEARCH_BS=512
export Q2E_MC_VIDEO_BS=32
for cfg in noASR ASR; do
  echo "===== MSRVTT multiclip $cfg $(date +%T) ====="
  python -u src/evaluation/run_eval.py \
    --dataset_dir data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_${cfg} \
    --t2v_encoder multiclip --tag msrvtt_multiclip_${cfg} --out $REPO/runs
done
echo ALL_MSRVTT_MULTICLIP_DONE
