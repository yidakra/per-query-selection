"""Significance of the best single post-retrieval predictor on the formulation decision.

The main table reports the rewriting picked by the best score-based predictor (NQC_norm, 0.332)
without a test. This reads the stored per-query picks (`variant_picks_full.jsonl`, written by
`mv2_variant_selection.py --save-picks`) and the per-candidate nDCG@10 in
`mv2_variant_features_full.json`, and runs the same one-sided event-grouped sign-flip test against
the original query as every other cell, with Holm over the predictors whose picks are stored.

  python src/multivent2/mv2_variant_single_test.py
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from mv2_row_inference import group_stats, holm  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")


def main():
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    feats = json.load(open(os.path.join(ABL, "mv2_variant_features_full.json")))
    nd = {(r["qid"], r["label"]): r["ndcg10"] for r in feats["rows"]}
    picks = [json.loads(line) for line in open(os.path.join(DATA, "variant_picks_full.jsonl"))]
    policies = [k for k in picks[0] if k.startswith("sel_")]
    qids = [p["qid"] for p in picks]
    grp = event_groups(qids, qrels)
    orig = np.array([nd[(q, "original")] for q in qids])

    out = {"n_queries": len(qids), "baseline": float(orig.mean()), "policies": {}}
    for pol in policies:
        sel = np.array([nd[(p["qid"], p[pol]["pick"])] for p in picks])
        p, obs, lo, hi = group_stats(sel - orig, grp)
        out["policies"][pol] = {"achieved": float(sel.mean()), "gain": obs,
                                "p_signflip_group": p, "ci95": [lo, hi]}
    keys = list(out["policies"])
    for k, p in zip(keys, holm([out["policies"][k]["p_signflip_group"] for k in keys])):
        out["policies"][k]["p_holm_stored"] = float(p)
        r = out["policies"][k]
        print(f"{k}: {r['achieved']:.4f} (gain {100 * r['gain']:+.2f}), "
              f"p={r['p_signflip_group']:.4f}, holm={p:.4f}", flush=True)
    path = os.path.join(ABL, "mv2_variant_single_test.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
