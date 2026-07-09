#!/bin/bash
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
cd $REPO
while true; do
  done_marker=$(grep -c ALL_MSRVTT_MULTICLIP_DONE runs/msrvtt_multiclip.log 2>/dev/null)
  no=$([ -f runs/msrvtt_multiclip_noASR/metrics.json ] && echo yes || echo no)
  asr=$([ -f runs/msrvtt_multiclip_ASR/metrics.json ] && echo yes || echo no)
  err=$(grep -c "OutOfMemory\|Traceback" runs/msrvtt_multiclip.log 2>/dev/null)
  stage=$(tr '\r' '\n' < runs/msrvtt_multiclip.log | grep -aE "compute\]|cache\]|MSRVTT multiclip|many-to-many score: *(100|[0-9][0-9])%" | tail -1)
  echo "$(date +%T) done=$done_marker noASR=$no ASR=$asr err=$err | $stage"
  if [ "$done_marker" -ge 1 ] || [ "$err" -ge 1 ]; then echo "WATCH_EXIT"; break; fi
  sleep 60
done
