"""How much supervision does the selector need? The learning curve behind the cost paragraph.

The selector that converts the routing headroom trains on judged queries, and the honest version of
"it needs supervision" is a number. Within each outer fold of the standard event-grouped split, the
training fold is subsampled by event group to a fraction of its queries, the selector is fit on the
subsample, and the nested gap against best-fixed-on-train (chosen on the same subsample) is scored
on the untouched test fold. Five outer folds per fraction, three subsample seeds averaged.

  python src/multivent2/mv2_label_curve.py --channel asr=asr_dense_bge-m3.json
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels  # noqa: E402
from mv2_channel_select import load_cell, mk  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--fractions", default="0.05,0.1,0.2,0.4,0.7,1.0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_label_curve.json"))
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold

    channels, names, policies, qids, X, Y, _ = load_cell(a.channel)
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    grp = np.asarray(event_groups(qids, qrels))
    splits = list(GroupKFold(5).split(X, groups=grp))
    print(f"cell: {names}, {len(qids)} queries, {len(set(grp))} groups", flush=True)

    out = {"channels": channels, "fractions": {}}
    for frac in [float(x) for x in a.fractions.split(",")]:
        gaps, n_train = [], []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            fold_gaps = []
            for tr, te in splits:
                tr_groups = np.unique(grp[tr])
                take = max(2, int(round(frac * len(tr_groups))))
                chosen = set(rng.choice(tr_groups, size=take, replace=False))
                sub = np.array([i for i in tr if grp[i] in chosen])
                m = mk().fit(X[sub], Y[sub])
                sel = np.argmax(m.predict(X[te]), axis=1)
                fixed_j = int(np.argmax(Y[sub].mean(axis=0)))
                fold_gaps.append(100 * (Y[te, sel].mean() - Y[te, fixed_j].mean()))
                n_train.append(len(sub))
            gaps.append(float(np.mean(fold_gaps)))
            if frac == 1.0:
                break  # no subsampling variance at 100%
        out["fractions"][f"{frac}"] = {
            "nested_gap_mean": float(np.mean(gaps)),
            "nested_gap_sd_over_seeds": float(np.std(gaps)),
            "mean_train_queries": float(np.mean(n_train))}
        print(f"frac {frac:>4}: ~{np.mean(n_train):5.0f} judged queries -> "
              f"gap {np.mean(gaps):+5.2f} (sd {np.std(gaps):.2f} over seeds)", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
