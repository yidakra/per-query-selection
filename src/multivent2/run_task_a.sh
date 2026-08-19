#!/bin/bash
# Task A driver: build the per-formulation cells, then run the symmetric protocol and per-row
# inference for each formulation. Pause variant generation before running (needs the GPU).
set -e
cd "$(dirname "$0")/../.." && source .venv-eval/bin/activate

python src/multivent2/mv2_variant_task_a.py --sample 0

for F in original genqr genqr_ensemble mugi qa_expand query2doc query2e; do
  python src/multivent2/mv2_qpp_table.py --group-cv --nested-calibration \
    --cell "ASRvOCR-$F=mv2_cell_asr_v_ocr_$F.json" \
    --queries "data/multivent2/variant_queries_$F.csv" \
    --score-run "data/multivent2/asr_dense_bge-m3_var_$F.json" \
    --tag "_va_$F"
  python src/multivent2/mv2_row_inference.py \
    --table "mv2_qpp_table_va_$F.json" \
    --cell "ASRvOCR-$F=mv2_cell_asr_v_ocr_$F.json" \
    --no-external-learned \
    --out "results/ablations/mv2_row_inference_va_$F.json"
done
echo TASK-A-DONE
