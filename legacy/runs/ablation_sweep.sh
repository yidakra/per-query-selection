#!/bin/bash
# Representative MultiVENT size/frame ablation sweep (noASR, text-only "Q2E - Video").
# 6 new configs: VLM {InternVL_1B, InternVL_8B}, LLM {LLaMA_1B, LLaMA_8B}, Frame {Funiform_2, Funiform_64}.
# The 38B/70B/Funiform_16 anchor point is already done (runs/multivent_textonly_noASR).
# Each config ~7h ColBERT on GPU1. Downloads are done first so availability fails fast.
set -u
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
cd $REPO
source .venv-eval/bin/activate
export HF_HOME=$REPO/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export Q2E_COLBERT_ENC_BS=128
export Q2E_COLBERT_SEARCH_BS=512
RUN=$REPO/runs
LOG="$RUN/ablation_sweep.log"

# repo-id-suffix  ->  local-dir-name  ->  run-tag
# columns: HF_REPO  LOCAL_NAME  TAG
CONFIGS=(
  "dipta007/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_1B_Funiform_16_noASR|Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_1B_Funiform_16_noASR|mv_vlm_internvl1b_noASR"
  "dipta007/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_8B_Funiform_16_noASR|Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_8B_Funiform_16_noASR|mv_vlm_internvl8b_noASR"
  "dipta007/Q2E_MultiVENT_LLAMA_1B_InternVL_38B_Funiform_16_noASR|Q2E_MultiVENT_LLAMA_1B_InternVL_38B_Funiform_16_noASR|mv_llm_llama1b_noASR"
  "dipta007/Q2E_MultiVENT_LLAMA_8B_InternVL_38B_Funiform_16_noASR|Q2E_MultiVENT_LLAMA_8B_InternVL_38B_Funiform_16_noASR|mv_llm_llama8b_noASR"
  "dipta007/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_2_noASR|Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_2_noASR|mv_frame_f2_noASR"
  "dipta007/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_64_noASR|Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_64_noASR|mv_frame_f64_noASR"
)

echo "=== ablation_sweep started $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# --- Phase 1: download all variants (save_to_disk format) ---
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r REPO NAME TAG <<< "$c"
  DEST="data/MultiVENT/$NAME"
  if [ -f "$DEST/dataset_info.json" ]; then
    echo "[dl] $NAME already present, skip" | tee -a "$LOG"; continue
  fi
  echo "[dl] $REPO -> $DEST  $(date +%T)" | tee -a "$LOG"
  python - "$REPO" "$DEST" <<'PY' 2>>"$LOG"
import sys
from datasets import load_dataset
repo, dest = sys.argv[1], sys.argv[2]
ds = load_dataset(repo)["train"]
ds.save_to_disk(dest)
print(f"    saved {len(ds)} rows to {dest}")
PY
done
echo "[dl] all downloads done $(date +%T)" | tee -a "$LOG"

# --- Phase 2: run text-only eval per config ---
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r REPO NAME TAG <<< "$c"
  if [ -f "$RUN/$TAG/metrics.json" ]; then
    echo "[run] $TAG already has metrics.json, skip" | tee -a "$LOG"; continue
  fi
  echo "[run] === $TAG $(date +%T) ===" | tee -a "$LOG"
  python -u src/evaluation/run_eval.py \
    --dataset_dir "data/MultiVENT/$NAME" \
    --t2v_encoder multiclip --no_video \
    --tag "$TAG" \
    --out "$RUN" >> "$LOG" 2>&1 || echo "[run] $TAG FAILED" | tee -a "$LOG"
done

echo "ABLATION_SWEEP_DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
touch "$RUN/ABLATION_SWEEP_DONE"
