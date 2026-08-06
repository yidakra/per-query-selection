"""Materialise each Table 1 cell's B run, so the nugget columns cost three judge passes and not thirty.

A predictor row in Table 1 executes, per query, either run A (visual alone) or run B (visual fused with
that cell's channel). Its nDCG is already a per-query mix of the two. Nugget coverage is per-query in
exactly the same way -- a report is written from a ranked list and scored against that query's gold
nuggets, so a policy's coverage is the mean of the coverage of whichever run it chose. Nothing about the
report depends on *which predictor* chose it.

So the whole table needs the coverage of run A and run B once per cell, and every row after that is
arithmetic. Run A is visual alone and is the same in all three cells, and it has already been judged as
the `visual` policy of the RAG arm. This writes the three B runs; `mv2_table1_nuggets.py` does the
mixing.

Weights come from each cell's own sweep and are checked the same way `mv2_recall_sidecar.py` checks
them: the rebuilt run must reproduce the nDCG already stored for the cell, or the run is not written.

  python src/multivent2/mv2_cell_runs.py
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run, load_qrels          # noqa: E402
from mv2_ab import per_query_ndcg                # noqa: E402
from mv2_channels import fuse                    # noqa: E402
from mv2_recall_sidecar import CELLS, VISUAL, TOL  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
RAG = os.path.join(ABL, "rag")

OUT = {"ASR-shipped": "cellB_asr_shipped.json",
       "ASR-dense": "cellB_asr_dense.json",
       "OCR": "cellB_ocr.json"}
KEEP = 20          # the generator reads the top 5; 20 leaves room without carrying 2.5 M postings


def main():
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    visual = load_run(os.path.join(DATA, VISUAL))

    # only the queries the RAG arm judged: the gold nuggets exist for those and nothing else
    gold_path = os.path.join(RAG, "gold_nuggets_n400.jsonl")
    keep = set()
    with open(gold_path) as f:
        for line in f:
            keep.add(json.loads(line)["qid"])
    print(f"{len(keep)} judged queries")

    for cell, (chan_fn, w, cell_fn) in CELLS.items():
        ref = json.load(open(os.path.join(ABL, cell_fn)))
        qids = list(ref["per_query"])
        runs = {"visual": visual, "chan": load_run(os.path.join(DATA, chan_fn))}
        runB = fuse(runs, {"visual": 1.0, "chan": w}, qids)

        nd = per_query_ndcg(qrels, runB)
        got = float(np.mean([nd[q] for q in qids if q in nd]))
        d = abs(got - ref["ndcgB"])
        print(f"{cell:<12} B {got:.4f} vs {ref['ndcgB']:.4f} (d {d:.6f})   "
              f"{'OK' if d < TOL else 'MISMATCH'}")
        if d >= TOL:
            print(f"  not written: {cell} does not reproduce its stored nDCG at w={w}")
            continue

        sub = {}
        for q in qids:
            if q not in keep:
                continue
            top = sorted(runB[q].items(), key=lambda kv: -kv[1])[:KEEP]
            sub[q] = {d_: s for d_, s in top}
        dest = os.path.join(DATA, OUT[cell])
        json.dump(sub, open(dest, "w"))
        print(f"  wrote {dest} ({len(sub)} queries)")


if __name__ == "__main__":
    main()
