#!/usr/bin/env python
"""K-way selection on the ORIGINAL cells: per query, pick which Q2E component subset to trust.

The MultiVENT 2.0 channel-selection result (mv2_channel_select.py, +6.2/+8.1 nested) currently
stands on one benchmark. This runs the identical protocol on the original MultiVENT and MSR-VTT
cells, with Q2E's own components as the evidence sources and the paper's tiers/ablations as the
policy set: A_visual, B_noevents, C_full, -Query, -Video.

One deliberate difference from the 2.0 cells. There every channel was cheap, so features could
read all of them. Here three policies contain event components that cost ~30 LLM calls, so the
selector stays CASCADE-LEGAL: features come only from the two cheap components
(query_vs_video, query_vs_captions), which exist before any spend decision. The selector chooses
whether to buy the expensive components without ever seeing their scores.

Given the B->Full unpredictability result (tau ~ +0.002) the expectation is that selection here
wins little and mostly toggles between A and B. Either outcome is informative: a gap confirms
the heterogeneity law on a second dataset; no gap bounds the K-way claim the same way B->Full
bounded the pairwise one.

CPU-only, cached component tensors.

  CUDA_VISIBLE_DEVICES="" python src/evaluation/router_kway_select.py
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "multivent2"))
from tracking import track  # noqa: E402
from oracle_router_headroom import fuse, per_query, LADDER, ABLATION  # noqa: E402
from router_hetero import comps_multivent, comps_msrvtt  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
RNG = np.random.default_rng(0)

POLICIES = {"A_visual": LADDER["A_visual"], "B_noevents": LADDER["B_noevents"],
            "C_full": LADDER["C_full"], "minus_query": ABLATION["-Query"],
            "minus_video": ABLATION["-Video"]}
CHEAP = ["query_vs_video", "query_vs_captions"]


def mk():
    return Pipeline([("sc", StandardScaler()),
                     ("m", RidgeCV(alphas=np.logspace(-2, 3, 12), alpha_per_target=True))])


def cheap_features(comps):
    """Per-query features from the two cheap components only: conf shape of each, plus top-K
    agreement between their rankings."""
    qv, qc = comps["query_vs_video"], comps["query_vs_captions"]
    rows = []
    for i in range(qv.shape[0]):
        a, b = qv[i].numpy(), qc[i].numpy()
        row = []
        for s in (a, b):
            f = conf_features(s)
            row += [f[k] for k in FEATURE_ORDER]
        ta10, tb10 = np.argsort(-a)[:10], np.argsort(-b)[:10]
        ta100, tb100 = np.argsort(-a)[:100], np.argsort(-b)[:100]
        row.append(len(set(ta10) & set(tb10)) / 10)
        row.append(len(set(ta100) & set(tb100)) / min(100, len(a)))
        rows.append(row)
    return np.array(rows)


def nested_selection(X, Y):
    outer = KFold(5, shuffle=True, random_state=1)
    gaps = []
    for tr, te in outer.split(X):
        m = mk().fit(X[tr], Y[tr])
        sel = np.argmax(m.predict(X[te]), axis=1)
        fixed_j = int(np.argmax(Y[tr].mean(axis=0)))
        gaps.append(100 * (Y[te, sel].mean() - Y[te, fixed_j].mean()))
    return float(np.mean(gaps)), float(np.std(gaps, ddof=1) / np.sqrt(len(gaps)))


def cell(name, target, comps):
    pols = list(POLICIES)
    Y = np.stack([per_query(fuse(comps, POLICIES[p]), target)[0].numpy() for p in pols], axis=1)
    X = cheap_features(comps)
    n = len(Y)

    Yhat = cross_val_predict(mk(), X, Y, cv=KFold(5, shuffle=True, random_state=0))
    sel = np.argmax(Yhat, axis=1)
    achieved = Y[np.arange(n), sel]
    best_j = int(np.argmax(Y.mean(axis=0)))
    perm = np.array([Y[np.arange(n), sel[RNG.permutation(n)]].mean() for _ in range(2000)])
    p_perm = float((1 + (perm >= achieved.mean()).sum()) / 2001)
    ng, ngs = nested_selection(X, Y)
    picks = {pols[j]: int((sel == j).sum()) for j in range(len(pols))}

    out = {"n": n, "policy_means": {p: float(Y[:, j].mean()) for j, p in enumerate(pols)},
           "best_fixed": pols[best_j], "selected_oof": float(achieved.mean()),
           "p_perm": p_perm, "nested_gap": ng, "nested_sem": ngs, "picks": picks}
    print(f"[{name}] n={n} best fixed {pols[best_j]} {100*Y[:, best_j].mean():.2f} | "
          f"selected {100*achieved.mean():.2f} (p={p_perm:.4f}) | "
          f"nested {ng:+.2f} +/- {ngs:.2f} | picks " +
          " ".join(f"{k}={v}" for k, v in sorted(picks.items(), key=lambda x: -x[1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{_ROOT}/results/ablations/router_kway_select.json")
    a = ap.parse_args()

    with track("router-kway-select", gpu_ids=[], tags=["router", "kway"], config={}) as tr:
        out = {}
        for setting in ["noASR", "ASR"]:
            _, target, comps = comps_multivent(setting)
            out[f"multivent_{setting}"] = cell(f"multivent {setting}", target, comps)
        for enc in ["multiclip", "internvideo2"]:
            for setting in ["noASR", "ASR"]:
                _, target, comps = comps_msrvtt(enc, setting)
                out[f"msrvtt_{enc}_{setting}"] = cell(f"msrvtt {enc} {setting}", target, comps)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
        tr.summary({f"{k}_nested_gap": v["nested_gap"] for k, v in out.items()})


if __name__ == "__main__":
    main()
