#!/usr/bin/env python
"""Honest significance test for the expected-gain cascade.

At escalation fraction f, selecting set S by predicted gain gives
    gap(f) = f * ( mean_{i in S} g_i  -  mean_i g_i ),   g_i = ndcg_B(i) - ndcg_A(i)
i.e. the ONLY thing that beats the fixed-policy chord is whether the ranker's top-f has
above-average true gain.  Randomly escalating a fraction f reproduces the chord exactly in
expectation, so the chord IS the random-escalation baseline.

We therefore report, at every f:
  - gap(f) with a bootstrap CI over queries,
  - a permutation null (shuffle predicted gains) giving an exact one-sided p-value,
and a NESTED operating-point selection (f chosen on train folds, scored on held-out) so the
headline number carries no selection bias.  CPU-only.
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_cascade_exp import build, CN  # noqa: E402
from router_gain_cascade import A_COLS, B_COLS  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402

RNG = np.random.default_rng(0)
NPERM, NBOOT = 10000, 5000
FRACS = np.arange(0.05, 1.0, 0.05)
OUT = "/home/ubuntu/q2e_repro/results/ablations/router_gain_curve.json"


def mk():
    return Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])


def gap_at(f, ghat, g):
    """gap(f) = f * (mean gain of top-f by ghat  -  overall mean gain)."""
    n = len(g); k = int(round(f * n))
    if k == 0:
        return 0.0
    S = np.argsort(-ghat)[:k]
    return (k / n) * (g[S].mean() - g.mean())


def run(setting):
    df, _ = build(setting)
    ndA, ndB, ndF = df["ndcgA"].values, df["ndcgB"].values, df["ndcgFull"].values
    g = ndB - ndA
    n = len(g)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    ghat = cross_val_predict(mk(), df[A_COLS].values, g, cv=cv)

    rho = np.corrcoef(ghat, g)[0, 1]
    # permutation null on rho itself (equivalently on the whole gap curve)
    perm_rho = np.array([np.corrcoef(RNG.permutation(ghat), g)[0, 1] for _ in range(2000)])
    p_rho = (1 + (perm_rho >= rho).sum()) / (1 + len(perm_rho))

    print(f"\n### MultiVENT {setting} full_pool (n={n})   [A->B escalation]")
    print(f"  OOF rho(predicted gain, true gain) = {rho:+.3f}   perm p = {p_rho:.4f}")
    print(f"  mean gain A->B = {g.mean()*100:+.2f} NDCG   (Fixed-A {ndA.mean()*100:.2f} -> Fixed-B {ndB.mean()*100:.2f})")
    print(f"  {'f':>5} {'cost':>6} {'NDCG':>7} {'chord':>7} {'gap':>7} {'95% CI':>16} {'perm p':>8}")

    curve = []
    for f in FRACS:
        k = int(round(f * n))
        S = np.argsort(-ghat)[:k]
        sel = np.zeros(n, bool); sel[S] = True
        nd = np.where(sel, ndB, ndA).mean() * 100
        cost = np.where(sel, CN["-Events"], CN["A_visual"]).mean()
        chord = (ndA.mean() + (k / n) * g.mean()) * 100
        gap = gap_at(f, ghat, g) * 100

        # bootstrap over queries: resample, recompute gap with the SAME fixed ranking
        bs = np.empty(NBOOT)
        for b in range(NBOOT):
            idx = RNG.integers(0, n, n)
            bs[b] = gap_at(f, ghat[idx], g[idx]) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])

        # permutation null: shuffle predicted gains, gap should center on 0
        pm = np.array([gap_at(f, RNG.permutation(ghat), g) * 100 for _ in range(NPERM // 10)])
        p = (1 + (pm >= gap).sum()) / (1 + len(pm))

        star = "*" if p < 0.05 and lo > 0 else " "
        print(f"  {f:5.2f} {cost:6.2f} {nd:7.2f} {chord:7.2f} {gap:+7.2f} [{lo:+6.2f},{hi:+6.2f}] {p:8.4f}{star}")
        curve.append(dict(f=float(f), cost=float(cost), ndcg=float(nd), chord=float(chord),
                          gap=float(gap), ci=[float(lo), float(hi)], p=float(p)))

    # ---- nested selection: pick f on 4 folds, score on the 5th. No selection bias. ----
    outer = KFold(n_splits=5, shuffle=True, random_state=1)
    nested_gaps, picked = [], []
    for tr, te in outer.split(np.arange(n)):
        # refit the gain model inside the training half only (no leakage into te)
        m = mk().fit(df[A_COLS].values[tr], g[tr])
        gh_tr = cross_val_predict(mk(), df[A_COLS].values[tr], g[tr], cv=KFold(5, shuffle=True, random_state=2))
        f_star = max(FRACS, key=lambda f: gap_at(f, gh_tr, g[tr]))
        gh_te = m.predict(df[A_COLS].values[te])
        nested_gaps.append(gap_at(f_star, gh_te, g[te]) * 100)
        picked.append(float(f_star))
    ng = np.array(nested_gaps)
    print(f"  NESTED (f chosen on train folds, scored held-out): gap {ng.mean():+.2f} "
          f"+/- {ng.std(ddof=1)/np.sqrt(len(ng)):.2f} (sem)  f* = {picked}")

    # ---- oracle ceiling on the same policy class ----
    orc = max(gap_at(f, g, g) * 100 for f in FRACS)
    print(f"  ORACLE (rank by true gain): best gap {orc:+.2f}  <- ceiling")

    # ---- is there ANY signal for B->Full? (the tier the router never buys) ----
    g2 = ndF - ndB
    gh2 = cross_val_predict(mk(), df[B_COLS].values, g2, cv=cv)
    rho2 = np.corrcoef(gh2, g2)[0, 1]
    perm2 = np.array([np.corrcoef(RNG.permutation(gh2), g2)[0, 1] for _ in range(2000)])
    p2 = (1 + (perm2 >= rho2).sum()) / (1 + len(perm2))
    orc2 = max(gap_at(f, g2, g2) * 100 for f in FRACS)
    print(f"  [B->Full] rho={rho2:+.3f} perm p={p2:.4f}; mean gain {g2.mean()*100:+.2f}; oracle gap {orc2:+.2f}")

    return dict(setting=setting, n=n, rho=float(rho), p_rho=float(p_rho), curve=curve,
                nested_gap=float(ng.mean()), nested_sem=float(ng.std(ddof=1) / np.sqrt(len(ng))),
                nested_f=picked, oracle_gap=float(orc),
                bf=dict(rho=float(rho2), p=float(p2), mean_gain=float(g2.mean() * 100), oracle_gap=float(orc2)))


res = [run(s) for s in ["noASR", "ASR"]]
json.dump(res, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
