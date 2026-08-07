"""Baselines and model ablations for the K-way channel selector, on the same cells as
mv2_channel_select.py. Two review questions get answered here before a reviewer asks them.

First, what would a practitioner build in an afternoon? Heuristic pickers that choose the
single channel whose own confidence feature is highest (z1, top-5 softmax mass, max softmax
prob, lowest entropy), plus a uniform-random policy pick. These need no training, so they are
evaluated directly.

Second, is the ridge leaving accuracy on the table? Same protocol, same features, three models:
the RidgeCV used everywhere, gradient-boosted trees, and a small MLP. Each gets the out-of-fold
selected mean and the nested gap (best fixed policy chosen on train).

CPU-only.

  python src/multivent2/mv2_select_baselines.py --channel asr=asr_dense_bge-m3.json --cell-tag _dense_m3
"""
import os
import sys
import json
import argparse
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_channel_select import load_cell, nested_selection, mk  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
ABL = os.path.join(_ROOT, "results", "ablations")

MODELS = {
    "ridge": mk,
    "gbdt": lambda: MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=200, max_depth=4, random_state=0)),
    "mlp": lambda: Pipeline([("sc", StandardScaler()),
                             ("m", MLPRegressor(hidden_layer_sizes=(64,), max_iter=1500,
                                                random_state=0))]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--cell-tag", default="")
    a = ap.parse_args()
    out_path = os.path.join(ABL, f"mv2_select_baselines{a.cell_tag}.json")

    channels, names, policies, qids, X, Y, feat_order = load_cell(a.channel)
    n = len(qids)
    best_fixed_j = int(np.argmax(Y.mean(axis=0)))
    print(f"cell: {names}, n={n}, best fixed {policies[best_fixed_j]} {Y[:, best_fixed_j].mean():.5f}")

    single_j = [policies.index(nm) for nm in names]
    fidx = {lbl: i for i, lbl in enumerate(feat_order)}

    heur = {"random_policy": float(Y.mean())}
    for feat, sign in [("z1", 1), ("top5mass", 1), ("maxp", 1), ("entropy", -1)]:
        F = np.stack([sign * X[:, fidx[f"{nm}:{feat}"]] for nm in names], axis=1)
        pick = F.argmax(axis=1)
        heur[f"pick_by_{feat}"] = float(Y[np.arange(n), [single_j[c] for c in pick]].mean())
    print("heuristics: " + "  ".join(f"{k}={v:.5f}" for k, v in heur.items()))

    models = {}
    for name, factory in MODELS.items():
        Yhat = cross_val_predict(factory(), X, Y, cv=KFold(5, shuffle=True, random_state=0))
        sel = np.argmax(Yhat, axis=1)
        oof = float(Y[np.arange(n), sel].mean())
        ng, ngs, *_ = nested_selection(X, Y, model=factory)
        models[name] = {"selected_oof": oof, "nested_gap": ng, "nested_sem": ngs}
        print(f"{name:<6} selected {oof:.5f}  nested {ng:+.2f} +/- {ngs:.2f}")

    json.dump({"channels": channels, "policies": policies, "n_queries": n,
               "best_fixed": policies[best_fixed_j],
               "best_fixed_ndcg": float(Y[:, best_fixed_j].mean()),
               "heuristics": heur, "models": models},
              open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
