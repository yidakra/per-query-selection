#!/bin/bash
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
cd $REPO
for i in $(seq 1 400); do
  if grep -q "IV2_RERUN_DONE" runs/rerun_iv2.log 2>/dev/null; then echo "IV2_RERUN_DONE seen at $(date +%T)"; exit 0; fi
  if grep -q "OutOfMemory\|Traceback" runs/rerun_iv2.log 2>/dev/null && ! pgrep -f rerun_iv2.sh >/dev/null; then echo "IV2 ERRORED at $(date +%T)"; exit 2; fi
  if ! pgrep -f rerun_iv2.sh >/dev/null && ! pgrep -f "run_eval.py.*internvideo2" >/dev/null; then echo "IV2 process gone at $(date +%T)"; exit 4; fi
  sleep 30
done
echo "TIMEOUT"; exit 3
