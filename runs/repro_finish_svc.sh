#!/bin/bash
# systemd-managed, reboot-resilient version of the reproduction finisher.
# Idempotent + resumable: each step is skipped if already done, so re-running after a
# VPS restart only redoes the step that was in flight. Writes runs/REPRO_FINISH_COMPLETE
# once BOTH deliverables exist; the unit's ConditionPathExists on that file stops it from
# re-running on later boots. Always exits 0 (best-effort 1B must not wedge the service).
set -u
cd /home/ubuntu/q2e_repro
source .venv-eval/bin/activate
export HF_HOME=/home/ubuntu/q2e_repro/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=/home/ubuntu/q2e_repro/runs
LOG="$RUN/repro_finish.log"
TABLE5=/home/ubuntu/q2e_repro/results/ablations/table5_multivent_fullvideo.json
QV=/home/ubuntu/q2e_repro/runs/multivent_video_multiclip/cache/query_vs_video.pt
ONEB_CACHE=/home/ubuntu/q2e_repro/runs/mv_llm_llama1b_noASR/cache

echo "=== repro_finish_svc (systemd) start $(date -u +%FT%TZ) ===" | tee -a "$LOG"

# --- Priority 1: MultiVENT full-video reproduction (skip if already produced) ---
if [ -f "$TABLE5" ] && [ -f "$QV" ]; then
  echo "[1/2] video already complete -> skip" | tee -a "$LOG"
else
  export Q2E_MC_VIDEO_BS=8
  echo "=== [1/2] video encode $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  if python -u src/evaluation/encode_multivent_video.py >> "$LOG" 2>&1; then
    python -u src/evaluation/fuse_multivent_fullvideo.py >> "$LOG" 2>&1 \
      && echo "VIDEO_OK $(date -u +%FT%TZ)" | tee -a "$LOG" \
      || echo "FUSE_FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"
  else
    echo "ENCODE_FAILED $(date -u +%FT%TZ)" | tee -a "$LOG"
  fi
  touch "$RUN/MULTIVENT_VIDEO_DONE"
fi

# --- Priority 2: LLaMA-1B ablation cell (best-effort; run_eval resumes from component cache) ---
if [ "$(ls "$ONEB_CACHE"/{prequel,during,sequel}_vs_captions.pt 2>/dev/null | wc -l)" -eq 3 ]; then
  echo "[2/2] llama1b event caches complete -> skip recompute" | tee -a "$LOG"
else
  export Q2E_COLBERT_ENC_BS=64
  export Q2E_COLBERT_SEARCH_BS=128   # halved: keeps peak RAM ~15 GB in physical mem; 48G swap is backstop
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
fi

# --- Completion sentinel: both deliverables present -> service goes dormant on next boot ---
if [ -f "$TABLE5" ] && [ "$(ls "$ONEB_CACHE"/{prequel,during,sequel}_vs_captions.pt 2>/dev/null | wc -l)" -eq 3 ]; then
  touch "$RUN/REPRO_FINISH_COMPLETE"
  echo "=== REPRO COMPLETE $(date -u +%FT%TZ) (service will stay dormant) ===" | tee -a "$LOG"
else
  echo "=== repro_finish_svc pass done, still incomplete (will resume on next boot) $(date -u +%FT%TZ) ===" | tee -a "$LOG"
fi
exit 0
