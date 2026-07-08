#!/bin/bash
# Sequential GPU1 job queue. Waits for the MSR-VTT MultiCLIP run to finish, then runs
# MultiVENT text-only (both ASR settings) and MSR-VTT InternVideo2 (both ASR settings).
set -u
cd /home/ubuntu/q2e_repro
source .venv-eval/bin/activate
export HF_HOME=/home/ubuntu/q2e_repro/data/hf_cache
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export Q2E_COLBERT_ENC_BS=128
export Q2E_COLBERT_SEARCH_BS=512
export Q2E_MC_VIDEO_BS=32
export Q2E_IV2_VIDEO_BS=8
RUN=/home/ubuntu/q2e_repro/runs

# 1) wait for MSR-VTT multiclip to finish
echo "[queue] waiting for MSR-VTT multiclip to finish..."
while ! grep -q ALL_MSRVTT_MULTICLIP_DONE "$RUN/msrvtt_multiclip.log" 2>/dev/null; do sleep 30; done
echo "[queue] MSR-VTT multiclip done at $(date +%T)"

# 2) MultiVENT text-only (no videos available -> YouTube-scraped; skip query_vs_video)
for cfg in noASR ASR; do
  echo "[queue] === MultiVENT textonly $cfg $(date +%T) ==="
  python -u src/evaluation/run_eval.py \
    --dataset_dir data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_${cfg} \
    --t2v_encoder multiclip --no_video \
    --tag multivent_textonly_${cfg} \
    --out "$RUN" || echo "[queue] MultiVENT $cfg FAILED"
done

# 3) MSR-VTT InternVideo2 (needs checkpoint) — wait for it if still downloading
echo "[queue] waiting for InternVideo2 checkpoint..."
for i in $(seq 1 120); do
  if grep -q IV2_DOWNLOADED /home/ubuntu/q2e_repro/data/raw/iv2_dl.log 2>/dev/null; then break; fi
  sleep 20
done
for cfg in noASR ASR; do
  echo "[queue] === MSRVTT internvideo2 $cfg $(date +%T) ==="
  python -u src/evaluation/run_eval.py \
    --dataset_dir data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_${cfg} \
    --t2v_encoder internvideo2 \
    --tag msrvtt_internvideo2_${cfg} \
    --out "$RUN" || echo "[queue] MSRVTT iv2 $cfg FAILED"
done
echo "ALL_QUEUE_DONE $(date +%T)"
