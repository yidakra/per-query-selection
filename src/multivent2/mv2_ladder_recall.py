"""Recall along the channel-strength ladder: the metric the ladder section owed the reader.

The k-way selector optimises nDCG@10 and the binary cells showed that better nDCG selection costs
Recall@100. The ladder was reported nDCG-only. This rebuilds each rung's seven policy fusions,
scores Recall@100 per query per policy, reruns the deterministic nested selection to recover the
per-query picks, and reports routed recall next to the fixed policy's recall at every rung.

  python src/multivent2/mv2_ladder_recall.py
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels  # noqa: E402
from mv2_channels import fuse  # noqa: E402
from mv2_channel_select import load_cell, mk  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

RUNGS = [
    ("dense original", ["asr=asr_dense_bge-m3.json"]),
    ("translated", ["asr=asr_dense_bge-m3-mt.json"]),
    ("translated+reranked", ["asr=asr_rerank_v2m3_mt.json"]),
    ("plaidx tdist", ["asr=asr_plaidx_tdist_en.json"]),
    ("plaidx + ocr bge-m3", ["asr=asr_plaidx_tdist_en.json", "ocr=ocr_dense_bge-m3.json"]),
]


def per_query_recall100(qrels, run):
    import ir_measures
    from ir_measures import R
    return {m.query_id: m.value for m in ir_measures.iter_calc([R @ 100], qrels, run)}


def main():
    from sklearn.model_selection import GroupKFold
    from mv2_io import load_run

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    out = {}
    for label, overrides in RUNGS:
        channels, names, policies, qids, X, Y, _ = load_cell(overrides)
        grp = event_groups(qids, qrels)
        splits = list(GroupKFold(5).split(X, groups=grp))

        runs = {n: load_run(os.path.join(DATA, fn)) for n, fn in channels.items()}
        R = np.zeros_like(Y)
        for j, pol in enumerate(policies):
            w = {n: (1.0 if n in pol.split("+") else 0.0) for n in names}
            rec = per_query_recall100(qrels, fuse(runs, w, qids))
            R[:, j] = [rec.get(q, 0.0) for q in qids]

        routed_nd = np.zeros(len(qids)); routed_rec = np.zeros(len(qids))
        fixed_nd = np.zeros(len(qids)); fixed_rec = np.zeros(len(qids))
        for tr, te in splits:
            m = mk().fit(X[tr], Y[tr])
            sel = np.argmax(m.predict(X[te]), axis=1)
            fixed_j = int(np.argmax(Y[tr].mean(axis=0)))
            routed_nd[te] = Y[te, sel]; routed_rec[te] = R[te, sel]
            fixed_nd[te] = Y[te, fixed_j]; fixed_rec[te] = R[te, fixed_j]

        out[label] = {"fixed_ndcg": float(fixed_nd.mean()), "routed_ndcg": float(routed_nd.mean()),
                      "fixed_recall100": float(fixed_rec.mean()),
                      "routed_recall100": float(routed_rec.mean()),
                      "recall_delta": float(routed_rec.mean() - fixed_rec.mean())}
        print(f"{label:22s} nDCG {fixed_nd.mean():.4f}->{routed_nd.mean():.4f}  "
              f"R@100 {fixed_rec.mean():.4f}->{routed_rec.mean():.4f} "
              f"({routed_rec.mean() - fixed_rec.mean():+.4f})", flush=True)

    json.dump(out, open(os.path.join(ABL, "mv2_ladder_recall.json"), "w"), indent=2)
    print("wrote", os.path.join(ABL, "mv2_ladder_recall.json"))


if __name__ == "__main__":
    main()
