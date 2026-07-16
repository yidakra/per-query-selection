#!/usr/bin/env python
"""Does the expected-gain router GENERALIZE across cells, or is it refit per cell?

`router_hetero.py` fits and scores the A->B gain router WITHIN each of the six cells (nested CV).
The reviewer question is transfer: train the ridge on one cell, apply it UNCHANGED to another, and
see whether it still orders queries by true A->B gain. Off-diagonal transfer is inherently honest
(disjoint data, no CV needed); the diagonal is within-cell OOF so it is comparable, not inflated.

Three regimes, increasingly hard:
  - same encoder, across ASR/noASR setting  (mildest shift: same gallery, same scale)
  - across encoder, same dataset            (MSR-VTT multiclip <-> internvideo2: score scale shifts)
  - across dataset                          (MultiVENT <-> MSR-VTT: query length, gold multiplicity)

Headline is LEAVE-ONE-CELL-OUT: train on the pooled other five, deploy on the held-out cell. That
is the "train once, route anywhere" number a deployment would actually see.

Two feature sets: the full A_COLS (confidence + query char/word length) and CONFIDENCE-ONLY (drop
the two query-text surface features, which are dataset-specific and were shown useless in isolation,
router_diag.py). If transfer improves when the text features are dropped, they were overfitting the
source cell.

Metric: Spearman/Pearson rho(predicted gain, true gain) on the target cell (scale-free, no operating
point to choose), plus the realized frontier gap at a fixed f=0.5 (mid-plateau) in NDCG points.
CPU-only; reads cached component tensors.
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_hetero import comps_multivent, comps_msrvtt, mk, gap_at, A_COLS, FRACS  # noqa: E402
from oracle_router_headroom import fuse, per_query, LADDER  # noqa: E402
from router_cascade_exp import conf_feats  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402

RNG = np.random.default_rng(0)
FIXED_F = 0.5                          # mid-plateau operating point, common across cells
CONF_COLS = ["A_margin12", "A_margin13", "A_z1", "A_entropy", "A_maxp", "A_top5mass", "A_std", "A_top1"]
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{_ROOT}/results/ablations/router_transfer.json"
FIG = f"{_ROOT}/reports/figures"

CELLS = [("MV/mCLIP/noASR", lambda: comps_multivent("noASR")),
         ("MV/mCLIP/ASR", lambda: comps_multivent("ASR")),
         ("MSR/mCLIP/noASR", lambda: comps_msrvtt("multiclip", "noASR")),
         ("MSR/mCLIP/ASR", lambda: comps_msrvtt("multiclip", "ASR")),
         ("MSR/IV2/noASR", lambda: comps_msrvtt("internvideo2", "noASR")),
         ("MSR/IV2/ASR", lambda: comps_msrvtt("internvideo2", "ASR"))]
NAMES = [c[0] for c in CELLS]


def build_features(queries, target, comps):
    """(X dataframe with A_COLS, true A->B gain g) for one cell -- mirrors router_hetero.cell()."""
    fA = fuse(comps, LADDER["A_visual"]); fB = fuse(comps, LADDER["B_noevents"])
    ndA = per_query(fA, target)[0].numpy(); ndB = per_query(fB, target)[0].numpy()
    rows = []
    for i, q in enumerate(queries):
        r = {"charlen": len(q), "wordlen": len(q.split())}
        for k, v in conf_feats(fA[i]).items():
            r[f"A_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows), (ndB - ndA)


def group_of(i, j):
    """Which transfer regime does train i -> test j fall in?"""
    ni, nj = NAMES[i].split("/"), NAMES[j].split("/")
    di, ei = ni[0], ni[1]
    dj, ej = nj[0], nj[1]
    if di != dj:
        return "cross-dataset"
    if ei != ej:
        return "cross-encoder"
    return "same-enc/cross-setting"      # same dataset+encoder, differ only in ASR/noASR


def main():
    print("[build] loading 6 cells ...", flush=True)
    data = {}
    for name, fn in CELLS:
        df, g = build_features(*fn())
        data[name] = (df, g)
        print(f"  {name:18s} n={len(g)}  mean gain={g.mean()*100:+.2f}", flush=True)

    out = {"fixed_f": FIXED_F, "cells": NAMES, "feature_sets": {}}

    for tag, cols in (("full_A_COLS", A_COLS), ("confidence_only", CONF_COLS)):
        R = np.full((6, 6), np.nan)      # rho(pred, true) on target
        G = np.full((6, 6), np.nan)      # frontier gap @ f=0.5, NDCG points
        for i, tr in enumerate(NAMES):
            Xtr, gtr = data[tr][0][cols].values, data[tr][1]
            for j, te in enumerate(NAMES):
                Xte, gte = data[te][0][cols].values, data[te][1]
                if i == j:
                    pred = cross_val_predict(mk(), Xte, gte, cv=KFold(5, shuffle=True, random_state=0))
                else:
                    pred = mk().fit(Xtr, gtr).predict(Xte)
                R[i, j] = np.corrcoef(pred, gte)[0, 1]
                G[i, j] = gap_at(FIXED_F, pred, gte) * 100

        # leave-one-cell-out: pooled other five -> held-out
        loco = []
        for j, te in enumerate(NAMES):
            Xtr = np.vstack([data[NAMES[k]][0][cols].values for k in range(6) if k != j])
            gtr = np.concatenate([data[NAMES[k]][1] for k in range(6) if k != j])
            Xte, gte = data[te][0][cols].values, data[te][1]
            model = mk().fit(Xtr, gtr)
            pred = model.predict(Xte)
            rho = float(np.corrcoef(pred, gte)[0, 1])
            perm = np.array([np.corrcoef(RNG.permutation(pred), gte)[0, 1] for _ in range(2000)])
            p = float((1 + (perm >= rho).sum()) / (1 + len(perm)))
            gap = float(gap_at(FIXED_F, pred, gte) * 100)
            withincell = float(R[j, j])
            loco.append(dict(cell=te, rho=rho, p=p, gap=gap, within_cell_rho=withincell))

        # regime summaries (off-diagonal only)
        regimes = {}
        for i in range(6):
            for j in range(6):
                if i == j:
                    continue
                regimes.setdefault(group_of(i, j), []).append(R[i, j])
        regime_mean = {k: float(np.mean(v)) for k, v in regimes.items()}

        out["feature_sets"][tag] = {
            "rho_matrix": R.tolist(), "gap_matrix": G.tolist(),
            "diagonal_within_cell_rho": [float(R[i, i]) for i in range(6)],
            "offdiag_mean_rho": float(np.nanmean(R[~np.eye(6, dtype=bool)])),
            "regime_mean_rho": regime_mean,
            "loco": loco,
        }
        print(f"\n=== feature set: {tag} ===")
        print(f"  within-cell rho (diag): {[round(R[i,i],3) for i in range(6)]}")
        print(f"  off-diagonal mean rho : {np.nanmean(R[~np.eye(6,dtype=bool)]):+.3f}")
        for k, v in regime_mean.items():
            print(f"    {k:24s}: mean transfer rho {v:+.3f}")
        print("  LEAVE-ONE-CELL-OUT (train pooled 5 -> held-out):")
        for L in loco:
            star = "*" if L["p"] < 0.05 else " "
            print(f"    {L['cell']:18s} rho={L['rho']:+.3f} (within {L['within_cell_rho']:+.3f}) "
                  f"gap@0.5={L['gap']:+.2f} p={L['p']:.4f}{star}")

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")

    # ---- heatmap of the confidence-only transfer matrix ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        R = np.array(out["feature_sets"]["confidence_only"]["rho_matrix"])
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        im = ax.imshow(R, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(NAMES, fontsize=8)
        ax.set_xlabel("TEST cell (deployed on)"); ax.set_ylabel("TRAIN cell (fit on)")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{R[i,j]:+.2f}", ha="center", va="center", fontsize=8,
                        color="k" if abs(R[i, j]) < 0.2 else "w",
                        fontweight="bold" if i == j else "normal")
        ax.set_title("Router transfer: rho(predicted gain, true A->B gain)\n"
                     "diagonal = within-cell OOF; off-diagonal = pure transfer (confidence-only features)",
                     fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson rho")
        fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(f"{FIG}/router_transfer.{e}", dpi=160, bbox_inches="tight")
        print(f"wrote {FIG}/router_transfer.{{pdf,png}}")
    except Exception as e:
        print(f"[fig skipped] {e}")


if __name__ == "__main__":
    main()
