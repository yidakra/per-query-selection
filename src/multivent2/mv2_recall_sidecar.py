"""Per-query Recall@100 for each cell's A and B runs, so Table 1 can carry their Recall@100 column.

Arabzadeh et al.'s Table 1 reports each selector under four metrics; ours reports two. This fills the
retrieval-side gap. The nugget columns need a judge and are handled elsewhere.

The cell files (`mv2_chan_visual_to_*.json`) store per-query nDCG@10 for the cheap run A and the fused
run B but not the runs themselves, so A and B are rebuilt here from the shipped ranked lists under the
same weighted RRF. That reconstruction is the risky step, so it is CHECKED: the rebuilt run's mean
nDCG@10 must match the `ndcgA` / `ndcgB` already stored in the cell file. A mismatch aborts rather than
writing a plausible wrong number, because a silently mis-weighted B run would put a wrong Recall column
in the paper's main table.

CPU-only.

  python src/multivent2/mv2_recall_sidecar.py
"""
import os
import sys
import json

import numpy as np
import ir_measures
from ir_measures import R

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run, load_qrels          # noqa: E402
from mv2_ab import per_query_ndcg               # noqa: E402
from mv2_channels import fuse                   # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

VISUAL = "10pyscene_clip.json"
# cell -> (channel run added to visual, its RRF weight, the cell file whose ndcgA/ndcgB we reproduce).
# B is the best WEIGHTED fusion from each cell's own sweep, not a unit-weight fusion -- unit weights
# miss by 0.020 / 0.004 / 0.002 nDCG, which the check below caught. Weights from mv2_channels*.json
# `best_weights`; the sweep fixes visual at 1.0.
CELLS = {
    "ASR-shipped": ("whisperASR_clip.json",  0.5, "mv2_chan_visual_to_asr.json"),
    "ASR-dense":   ("asr_dense_bge-m3.json", 0.5, "mv2_chan_visual_to_asr_dense_m3.json"),
    "OCR":         ("ocr_dense_bge-m3.json", 1.0, "mv2_chan_visual_to_ocr_dense_m3.json"),
}
TOL = 5e-4          # the check is on a mean over 2,546 queries; anything real is far bigger than this


def per_query_recall(qrels, run, k=100):
    return {m.query_id: m.value for m in ir_measures.iter_calc([R @ k], qrels, run)}


def main():
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    visual = load_run(os.path.join(DATA, VISUAL))
    out = {}
    for cell, (chan_fn, w, cell_fn) in CELLS.items():
        p = os.path.join(DATA, chan_fn)
        if not os.path.exists(p):
            print(f"{cell}: missing {chan_fn}, skipped")
            continue
        ref = json.load(open(os.path.join(ABL, cell_fn)))
        qids = list(ref["per_query"])
        runs = {"visual": visual, "chan": load_run(p)}
        runA = {q: visual[q] for q in qids if q in visual}
        runB = fuse(runs, {"visual": 1.0, "chan": w}, qids)

        ndA = per_query_ndcg(qrels, runA)
        ndB = per_query_ndcg(qrels, runB)
        gotA = float(np.mean([ndA[q] for q in qids if q in ndA]))
        gotB = float(np.mean([ndB[q] for q in qids if q in ndB]))
        dA, dB = abs(gotA - ref["ndcgA"]), abs(gotB - ref["ndcgB"])
        flag = "OK " if dA < TOL and dB < TOL else "MISMATCH"
        print(f"{cell:<12} A {gotA:.4f} vs {ref['ndcgA']:.4f} (d {dA:.5f})   "
              f"B {gotB:.4f} vs {ref['ndcgB']:.4f} (d {dB:.5f})   {flag}")
        if flag != "OK ":
            print(f"  {cell} reconstruction does not reproduce the stored nDCG. The fusion weights "
                  f"used for this cell are not {w}; recall not written.")
            continue

        rA, rB = per_query_recall(qrels, runA), per_query_recall(qrels, runB)
        common = [q for q in qids if q in rA and q in rB]
        out[cell] = {"n": len(common),
                     "recallA": float(np.mean([rA[q] for q in common])),
                     "recallB": float(np.mean([rB[q] for q in common])),
                     "per_query": {q: [rA[q], rB[q]] for q in common}}
        print(f"  R@100  A {out[cell]['recallA']:.4f}  B {out[cell]['recallB']:.4f}")

    dest = os.path.join(ABL, "mv2_recall_sidecar.json")
    json.dump(out, open(dest, "w"))
    print(f"\nwrote {dest} ({len(out)}/{len(CELLS)} cells verified)")


if __name__ == "__main__":
    main()
