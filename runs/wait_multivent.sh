#!/bin/bash
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
RUN=$REPO/runs
target="$RUN/multivent_textonly_ASR/metrics.json"
for i in $(seq 1 600); do
  if [ -f "$target" ]; then echo "READY multivent_textonly_ASR at $(date +%T)"; exit 0; fi
  if ! pgrep -f queue_remaining.sh >/dev/null && ! pgrep -f "run_eval.py" >/dev/null; then
    echo "QUEUE_DIED at $(date +%T) (no queue_remaining.sh and no run_eval.py running)"; exit 2
  fi
  sleep 60
done
echo "TIMEOUT after ~10h"; exit 3
