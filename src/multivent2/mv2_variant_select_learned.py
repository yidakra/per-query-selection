"""The positive control for the variant axis: can a trained selector pick the query variant?

The third cell of the three-axis table. Channel selection has its ridge (+7.59) and language
selection has its ridge (+2.16); the variant axis so far has only the analytic families. This
scores each candidate pointwise from its own confidence features plus its generating method, trains
across queries with event-grouped folds, and takes the argmax within each query, which is the same
decision rule the analytic predictors use and the same protocol the other two axes use.

Pointwise rather than a fixed slot layout: candidates are exchangeable within a method, and a
pointwise model stays valid if the pool composition changes.

  python src/multivent2/mv2_variant_select_learned.py
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

from mv2_io import load_qrels  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(ABL, "mv2_variant_features_full.json"))
    ap.add_argument("--center", action="store_true",
                    help="centre features and target within each query before fitting. Raw "
                         "pointwise regression spends its capacity on between-query difficulty, "
                         "which is why an unnormalised fit loses to a normalised analytic "
                         "predictor; centring makes the model learn within-query contrasts, the "
                         "quantity the argmax actually needs")
    ap.add_argument("--no-method-feature", action="store_true",
                    help="drop the generating-method indicators, leaving only score statistics")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_variant_select_learned.json"))
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    from mv2_channel_select import mk, group_signflip_p

    d = json.load(open(a.features))
    rows = d["rows"]
    by_q = collections.defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    qids = sorted(by_q)
    methods = sorted({r["method"] for r in rows})
    print(f"{len(qids)} queries, {len(rows)} candidates, {len(methods)} methods", flush=True)

    X, y, qidx = [], [], []
    for i, q in enumerate(qids):
        for r in sorted(by_q[q], key=lambda z: z["label"]):
            feats = list(r["x"])
            if not a.no_method_feature:
                feats += [1.0 if r["method"] == m else 0.0 for m in methods]
            X.append(feats); y.append(r["ndcg10"]); qidx.append(i)
    X = np.asarray(X); y = np.asarray(y); qidx = np.asarray(qidx)
    y_raw = y.copy()
    if a.center:
        for i in range(len(qids)):
            m = qidx == i
            X[m] = X[m] - X[m].mean(axis=0)
            y[m] = y[m] - y[m].mean()
    print(f"{X.shape[1]} features per candidate"
          f"{', centred within query' if a.center else ''}", flush=True)

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    grp_q = np.asarray(event_groups(qids, qrels))
    grp_row = grp_q[qidx]

    orig = np.array([next(r["ndcg10"] for r in by_q[q] if r["label"] == "original") for q in qids])
    best = np.array([max(r["ndcg10"] for r in by_q[q]) for q in qids])

    sel = np.zeros(len(qids))
    for tr_q, te_q in GroupKFold(5).split(np.arange(len(qids)), groups=grp_q):
        tr_mask = np.isin(qidx, tr_q)
        m = mk().fit(X[tr_mask], y[tr_mask])
        pred = m.predict(X[~tr_mask])
        pred = pred.ravel() if pred.ndim > 1 else pred
        rows_te = np.where(~tr_mask)[0]
        for qi in te_q:
            mask = qidx[rows_te] == qi
            local = rows_te[mask]
            sel[qi] = y_raw[local[int(np.argmax(pred[mask]))]]

    p, _ = group_signflip_p(sel - orig, grp_q)
    out = {"n_queries": len(qids), "n_candidates": len(rows),
           "n_features": int(X.shape[1]), "method_feature": not a.no_method_feature,
           "centred": bool(a.center),
           "original": float(orig.mean()), "selected": float(sel.mean()),
           "oracle": float(best.mean()),
           "gap_vs_original": float(100 * (sel.mean() - orig.mean())),
           "p_vs_original": float(p),
           "fraction_of_oracle_headroom": float(
               (sel.mean() - orig.mean()) / (best.mean() - orig.mean()))}
    print(f"original {orig.mean():.4f} -> learned {sel.mean():.4f} "
          f"({out['gap_vs_original']:+.2f}, p={p:.4f}); oracle {best.mean():.4f}, "
          f"captures {100*out['fraction_of_oracle_headroom']:.0f}% of the headroom")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
