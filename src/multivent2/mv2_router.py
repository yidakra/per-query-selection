"""Expected-gain router on the real MultiVENT 2.0 A->B gains (from mv2_ab.py).

Same protocol as the original cells: predict per-query gain g = nDCG_B - nDCG_A from tier-A confidence
features (out-of-fold ridge), escalate the top-f. Reports Kendall tau + permutation p + bootstrap CI,
the nested (selection-bias-free) frontier gap, APGR, and CPT. Self-contained (no heavy imports); reads
results/ablations/mv2_ab.json. CPU-only.
"""
import os
import sys
import json
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import kendalltau

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AB = os.path.join(_ROOT, "results", "ablations", "mv2_ab.json")
OUT = os.path.join(_ROOT, "results", "ablations", "mv2_router.json")
RNG = np.random.default_rng(0)
FRACS = np.arange(0.02, 1.0, 0.02)


def mk():
    return Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])


def gap_at(f, ghat, g):
    n = len(g); k = int(round(f * n))
    if k == 0:
        return 0.0
    S = np.argsort(-ghat)[:k]
    return (k / n) * (g[S].mean() - g.mean())


def apgr(ghat, g):
    r = [gap_at(f, ghat, g) / gap_at(f, g, g) for f in FRACS if gap_at(f, g, g) > 1e-9]
    return float(np.mean(r)) if r else float("nan")


def cpt(ghat, g, x):
    tot = g.sum()
    if tot <= 0:
        return float("nan")
    cum = np.cumsum(g[np.argsort(-ghat)]) / tot
    k = int(np.searchsorted(cum >= x, True))
    return float((k + 1) / len(g)) if k < len(g) else float("nan")


def nested_gap(X, g):
    outer = KFold(5, shuffle=True, random_state=1); gaps = []
    for tr, te in outer.split(np.arange(len(g))):
        m = mk().fit(X[tr], g[tr])
        gh = cross_val_predict(mk(), X[tr], g[tr], cv=KFold(5, shuffle=True, random_state=2))
        fstar = max(FRACS, key=lambda f: gap_at(f, gh, g[tr]))
        gaps.append(gap_at(fstar, m.predict(X[te]), g[te]) * 100)
    return float(np.mean(gaps)), float(np.std(gaps, ddof=1) / np.sqrt(len(gaps)))


def main():
    d = json.load(open(AB))
    order = d["feature_order"]
    qids = [q for q in d["features"] if q in d["per_query"]]
    X = np.array([d["features"][q] for q in qids])
    g = np.array([d["per_query"][q]["ndB"] - d["per_query"][q]["ndA"] for q in qids])
    n = len(g)
    print(f"n={n}  sd(gain)={100*g.std():.2f}  mean(gain)={100*g.mean():+.2f}")

    ghat = cross_val_predict(mk(), X, g, cv=KFold(5, shuffle=True, random_state=0))
    tau = float(kendalltau(ghat, g).statistic)
    perm = np.array([kendalltau(RNG.permutation(ghat), g).statistic for _ in range(2000)])
    p_tau = float((1 + (perm >= tau).sum()) / 2001)
    bt = np.array([kendalltau(ghat[i := RNG.integers(0, n, n)], g[i]).statistic for _ in range(2000)])
    tau_ci = [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]
    ng, ngs = nested_gap(X, g)

    out = {"n": n, "sd_gain": float(100 * g.std()), "mean_gain": float(100 * g.mean()),
           "ndcgA": d["ndcgA"], "ndcgB": d["ndcgB"],
           "tau": tau, "p_tau": p_tau, "tau_ci95": tau_ci,
           "nested_gap": ng, "nested_sem": ngs,
           "apgr": apgr(ghat, g), "cpt50": cpt(ghat, g, 0.5), "cpt80": cpt(ghat, g, 0.8)}
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"tau(pred,true gain) = {tau:+.3f}  perm p={p_tau:.4f}  CI95=[{tau_ci[0]:+.3f},{tau_ci[1]:+.3f}]")
    print(f"NESTED frontier gap = {ng:+.2f} +/- {ngs:.2f} nDCG")
    print(f"APGR = {out['apgr']:.3f}   CPT50 = {out['cpt50']:.2f}   CPT80 = {out['cpt80']:.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
