#!/bin/bash
# Shallow-evidence ablation from the research plan: rerun the score-only family with the top-k
# window it may read set to 5, 10, 20, 50, 100 and 1000, symmetric protocol, then per-row inference.
set -eo pipefail
cd "$(dirname "$0")/../.." && source .venv-eval/bin/activate
for K in 5 10 20 50 100 1000; do
  python src/multivent2/mv2_qpp_table.py --group-cv --nested-calibration --score-depth $K --tag "_depth${K}"
  python src/multivent2/mv2_row_inference.py --table "mv2_qpp_table_depth${K}.json" --no-external-learned \
    --out "results/ablations/mv2_row_inference_depth${K}.json" | tail -3
done
echo DEPTH-DONE
