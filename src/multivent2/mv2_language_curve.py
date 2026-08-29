"""How much supervision does the language selector need?

The channel selector's learning curve is in the cost paragraph (roughly 100 judged queries buys
+5.1 of its +7.59). The language axis is newer and a deployer would ask the same question before
committing to label anything, so this is the same measurement: within each outer fold, subsample the
training queries by event group, fit, and score the untouched test fold against always-English.

  python src/multivent2/mv2_language_curve.py
"""
import os
import sys
import json
import argparse
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from mv2_language_select_learned import LANGS, SUFFIX, overlap  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.05,0.1,0.2,0.4,0.7,1.0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_language_curve.json"))
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    from mv2_channel_select import mk

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    runs = {L: load_run(os.path.join(DATA, f"asr_dense_bge-m3{SUFFIX[L]}.json")) for L in LANGS}
    qids = sorted(set(qrels) & set.intersection(*[set(r) for r in runs.values()]))
    cols = []
    for L in LANGS:
        pq = per_query_ndcg(qrels, {q: runs[L][q] for q in qids})
        cols.append(np.array([pq.get(q, 0.0) for q in qids]))
    Y = np.stack(cols, axis=1)
    default = Y[:, LANGS.index("en")]

    X = []
    for q in qids:
        row, tops = [], {}
        for L in LANGS:
            r = runs[L][q]
            tops[L] = sorted(r, key=r.get, reverse=True)[:100]
            cf = conf_features(list(r.values()))
            row += [cf[k] for k in FEATURE_ORDER]
        for L1, L2 in itertools.combinations(LANGS, 2):
            row += [overlap(tops[L1], tops[L2], 10), overlap(tops[L1], tops[L2], 100)]
        X.append(row)
    X = np.asarray(X)

    grp = np.asarray(event_groups(qids, qrels))
    splits = list(GroupKFold(5).split(X, groups=grp))
    out = {"n_queries": len(qids), "default": float(default.mean()),
           "oracle": float(Y.max(axis=1).mean()), "fractions": {}}
    for frac in [float(x) for x in a.fractions.split(",")]:
        gaps, ntrain = [], []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            fold_gaps = []
            for tr, te in splits:
                groups = np.unique(grp[tr])
                take = max(2, int(round(frac * len(groups))))
                keep = set(rng.choice(groups, size=take, replace=False))
                sub = np.array([i for i in tr if grp[i] in keep])
                m = mk().fit(X[sub], Y[sub])
                sel = np.argmax(m.predict(X[te]), axis=1)
                fold_gaps.append(100 * (Y[te, sel].mean() - default[te].mean()))
                ntrain.append(len(sub))
            gaps.append(float(np.mean(fold_gaps)))
            if frac == 1.0:
                break
        out["fractions"][f"{frac}"] = {"gap_mean": float(np.mean(gaps)),
                                       "gap_sd_over_seeds": float(np.std(gaps)),
                                       "mean_train_queries": float(np.mean(ntrain))}
        print(f"frac {frac:>4}: ~{np.mean(ntrain):5.0f} judged queries -> "
              f"{np.mean(gaps):+5.2f} (sd {np.std(gaps):.2f})", flush=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
