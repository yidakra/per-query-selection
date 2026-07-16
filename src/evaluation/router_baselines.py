#!/usr/bin/env python
"""Baselines the A->B gain router must beat, per the JHU discussion:

(1) QPP-PREDICTOR BASELINES. Our router regresses the A->B gain from a battery of cheap-tier
    confidence features. The reviewer question: does that learned multi-feature model actually beat
    routing on a SINGLE standard Query Performance Prediction predictor? We take five canonical QPP
    predictors -- max score, score SD, NQC (Shtok 2012), WIG (Zhou & Croft 2007), Clarity/peakedness
    (Cronen-Townsend 2002) -- computed from the tier-A similarity scores over the gallery, adapted to
    cosine-similarity retrieval following iQPP (Poesina et al., 2023). Each is run through the SAME
    nested pipeline as a single input feature, so it gets OOF-calibrated direction and scale; a
    single-feature ridge is exactly oriented thresholding on that predictor. Beating it is therefore
    a strong claim: the multi-feature router earns its keep over the best classical predictor.

(2) MODEL-CLASS ABLATION. "The type of classifier matters." We hold the full A_COLS feature set fixed
    and swap the estimator: RidgeCV (our reference -- a REGRESSION on the continuous gain, not a
    classifier), Logistic on sign(gain) (the classification framing), GradientBoosting, RandomForest,
    and SVR (iQPP's supervised meta-regressor). If a nonlinear model captured more of the gain signal,
    ridge would be leaving accuracy on the table; showing it does not is the point.

Metric per (cell, method): out-of-fold Kendall tau AND Pearson rho of the routing score vs true A->B
gain, and the realized frontier gap at f=0.5 (NDCG points). Reported next to the full router. The
Ridge/A_COLS row reproduces router_hetero.py's within-cell tau exactly (same estimator, same folds),
a built-in cross-check. CPU-only; reads cached component tensors.
"""
import os, sys, json, numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_hetero import comps_multivent, comps_msrvtt, mk, gap_at, A_COLS  # noqa: E402
from oracle_router_headroom import fuse, per_query, LADDER  # noqa: E402
from router_cascade_exp import conf_feats  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

FIXED_F = 0.5
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{_ROOT}/results/ablations/router_baselines.json"

CELLS = [("MultiVENT/mCLIP/noASR", lambda: comps_multivent("noASR")),
         ("MultiVENT/mCLIP/ASR", lambda: comps_multivent("ASR")),
         ("MSR-VTT/mCLIP/noASR", lambda: comps_msrvtt("multiclip", "noASR")),
         ("MSR-VTT/mCLIP/ASR", lambda: comps_msrvtt("multiclip", "ASR")),
         ("MSR-VTT/IV2/noASR", lambda: comps_msrvtt("internvideo2", "noASR")),
         ("MSR-VTT/IV2/ASR", lambda: comps_msrvtt("internvideo2", "ASR"))]

# Canonical QPP predictors from a query's gallery score vector s (cosine sims). k = top-k depth.
# Each returns "higher = more confident / better predicted performance"; the single-feature nested
# ridge learns the sign to align with escalation gain, so orientation is not assumed here.
QPP_NAMES = ["QPP_max", "QPP_sd", "QPP_nqc", "QPP_wig", "QPP_clarity"]
QPP_CITE = {"QPP_max": "max score (magnitude)", "QPP_sd": "score SD",
            "QPP_nqc": "NQC, Shtok 2012", "QPP_wig": "WIG, Zhou & Croft 2007",
            "QPP_clarity": "Clarity/peakedness, Cronen-Townsend 2002"}


def qpp_predictors(s, k=10):
    s = s.double()
    ssort, _ = torch.sort(s, descending=True)
    topk = ssort[:k]
    mu, sd = s.mean(), s.std()
    p = torch.softmax(s, dim=0)
    ent = float(-(p * (p + 1e-12).log()).sum())
    return {
        "QPP_max": float(ssort[0]),
        "QPP_sd": float(sd),
        "QPP_nqc": float(topk.std() / (mu.abs() + 1e-9)),          # collection-mean normalized (Shtok)
        "QPP_wig": float((topk.mean() - mu) / (sd + 1e-9)),        # top-k gain over collection (Zhou & Croft)
        "QPP_clarity": float(-ent),                                # peaked retrieval distribution = confident
    }


def build(queries, target, comps):
    """One cell -> DataFrame with A_COLS confidence features + the QPP predictors, and true gain g."""
    fA = fuse(comps, LADDER["A_visual"]); fB = fuse(comps, LADDER["B_noevents"])
    ndA = per_query(fA, target)[0].numpy(); ndB = per_query(fB, target)[0].numpy()
    rows = []
    for i, q in enumerate(queries):
        r = {"charlen": len(q), "wordlen": len(q.split())}
        for k, v in conf_feats(fA[i]).items():
            r[f"A_{k}"] = v
        r.update(qpp_predictors(fA[i]))
        rows.append(r)
    return pd.DataFrame(rows), (ndB - ndA)


def oof_regress(est, X, g):
    return cross_val_predict(est, X, g, cv=KFold(5, shuffle=True, random_state=0))


def oof_logistic(X, g):
    """Classification framing: predict sign(gain>0), route by predicted probability. Falls back to
    NaN if a cell's gain is single-signed (no two classes to fit)."""
    y = (g > 0).astype(int)
    if y.min() == y.max():
        return None
    pipe = Pipeline([("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=2000, C=1.0))])
    return cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                             method="predict_proba")[:, 1]


def scores_for(pred, g):
    return {"tau": float(kendalltau(pred, g).statistic),
            "rho": float(np.corrcoef(pred, g)[0, 1]),
            "gap": float(gap_at(FIXED_F, pred, g) * 100)}


def models():
    """Estimator zoo for the model-class ablation (regressors); logistic handled separately."""
    return {
        "ridge": mk(),
        "gbm": Pipeline([("sc", StandardScaler()),
                         ("m", GradientBoostingRegressor(random_state=0))]),
        "rf": Pipeline([("sc", StandardScaler()),
                        ("m", RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1))]),
        "svr": Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf", C=1.0))]),
    }


def main():
    out = {"fixed_f": FIXED_F, "qpp_cite": QPP_CITE, "cells": {}}
    for name, fn in CELLS:
        df, g = build(*fn())
        cell = {"n": int(len(g)), "mean_gain": float(g.mean() * 100), "qpp": {}, "modelclass": {}}

        # (1) QPP single-predictor baselines: each predictor through the nested ridge as 1 feature
        for qp in QPP_NAMES:
            pred = oof_regress(mk(), df[[qp]].values, g)
            cell["qpp"][qp] = scores_for(pred, g)

        # (2) model-class ablation on the full A_COLS feature set
        for mname, est in models().items():
            pred = oof_regress(est, df[A_COLS].values, g)
            cell["modelclass"][mname] = scores_for(pred, g)
        logit = oof_logistic(df[A_COLS].values, g)
        cell["modelclass"]["logistic"] = scores_for(logit, g) if logit is not None else None

        out["cells"][name] = cell
        best_qpp = max(cell["qpp"].items(), key=lambda kv: kv[1]["tau"])
        print(f"\n### {name}  (n={cell['n']}, mean gain {cell['mean_gain']:+.2f})")
        print(f"  full router (ridge, {len(A_COLS)} feats):  tau={cell['modelclass']['ridge']['tau']:+.3f}  "
              f"gap@0.5={cell['modelclass']['ridge']['gap']:+.2f}")
        print(f"  best single QPP predictor ({best_qpp[0]}): tau={best_qpp[1]['tau']:+.3f}  "
              f"gap@0.5={best_qpp[1]['gap']:+.2f}")
        for qp in QPP_NAMES:
            s = cell["qpp"][qp]
            print(f"      {qp:12s} ({QPP_CITE[qp]:34s}) tau={s['tau']:+.3f} gap={s['gap']:+.2f}")
        print("  model class (full A_COLS):")
        for mname in ["ridge", "logistic", "gbm", "rf", "svr"]:
            s = cell["modelclass"].get(mname)
            if s is None:
                print(f"      {mname:10s}  (skipped: single-signed gain)")
            else:
                print(f"      {mname:10s}  tau={s['tau']:+.3f}  rho={s['rho']:+.3f}  gap@0.5={s['gap']:+.2f}")

    # ---- summaries across cells ----
    def mean_over(getter):
        vals = [getter(out["cells"][c]) for c in out["cells"]]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(vals))

    summary = {"mean_tau": {}, "mean_gap": {}}
    for mname in ["ridge", "logistic", "gbm", "rf", "svr"]:
        summary["mean_tau"][f"model:{mname}"] = mean_over(
            lambda c, m=mname: (c["modelclass"].get(m) or {}).get("tau"))
        summary["mean_gap"][f"model:{mname}"] = mean_over(
            lambda c, m=mname: (c["modelclass"].get(m) or {}).get("gap"))
    for qp in QPP_NAMES:
        summary["mean_tau"][f"qpp:{qp}"] = mean_over(lambda c, q=qp: c["qpp"][q]["tau"])
        summary["mean_gap"][f"qpp:{qp}"] = mean_over(lambda c, q=qp: c["qpp"][q]["gap"])
    out["summary"] = summary

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    print("\n[mean over 6 cells]  method                         tau      gap@0.5")
    ranked = sorted(summary["mean_tau"].items(), key=lambda kv: -kv[1])
    for k, v in ranked:
        print(f"  {k:34s} {v:+.3f}   {summary['mean_gap'][k]:+.2f}")


if __name__ == "__main__":
    main()
