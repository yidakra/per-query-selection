#!/usr/bin/env python
"""Expected-gain cascade: instead of classifying each query into a tier, regress the
NDCG GAIN from escalating and spend the compute budget on the queries predicted to gain
most.  Sweeping the escalation fraction traces an achievable accuracy-compute curve that
is compared against the convex hull of the fixed-tier policies (the honest bar).

Cost accounting: tier A's similarity component is a subset of B's, which is a subset of
Full's, so a cascade that scores A, inspects its confidence, then escalates pays only for
the components it ends up scoring (1 / 2 / 5).  Escalation itself is free.

Cascade discipline: the A->B decision may use only A-side features (B has not been paid
for yet); the B->Full decision may use A- and B-side features.  CPU-only.
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_cascade_exp import build, CN, LADDER_TIERS  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402

A_COLS = ["A_margin12", "A_margin13", "A_z1", "A_entropy", "A_maxp", "A_top5mass", "A_std", "A_top1",
          "charlen", "wordlen"]
B_COLS = A_COLS + ["B_margin12", "B_margin13", "B_z1", "B_entropy", "B_maxp", "B_top5mass",
                   "B_std", "B_top1", "AB_disagree"]
CV = KFold(n_splits=5, shuffle=True, random_state=0)


def models():
    yield "ridge", lambda: Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])
    yield "gbm", lambda: GradientBoostingRegressor(random_state=0, n_estimators=200, max_depth=2,
                                                   learning_rate=0.05, subsample=0.9)


def hull(fixed):
    """Upper concave envelope of the fixed-tier (cost, ndcg) points; returns ndcg(cost)."""
    pts = sorted((c, n) for n, c in fixed.values())
    keep = [pts[0]]
    for c, n in pts[1:]:
        while len(keep) >= 2:
            (c0, n0), (c1, n1) = keep[-2], keep[-1]
            if (n1 - n0) * (c - c0) <= (n - n0) * (c1 - c0):  # keep[-1] below chord -> drop
                keep.pop()
            else:
                break
        keep.append((c, n))
    xs = np.array([c for c, _ in keep]); ys = np.array([n for _, n in keep])
    return lambda c: float(np.interp(c, xs, ys))


def run(setting):
    df, _ = build(setting)
    ndA, ndB, ndF = (df["ndcgA"].values, df["ndcgB"].values, df["ndcgFull"].values)
    fixed = {t: (v.mean() * 100, CN[t]) for t, v in zip(LADDER_TIERS, [ndA, ndB, ndF])}
    H = hull(fixed)
    n = len(df)
    print(f"\n### MultiVENT {setting} full_pool (n={n})")
    for t in LADDER_TIERS:
        print(f"  Fixed-{t:9} NDCG {fixed[t][0]:.2f} @ cost {fixed[t][1]:.2f}")

    # oracle: per-query true gains, same escalation machinery -> upper bound on this policy class
    for mname, mk in models():
        gAB_hat = cross_val_predict(mk(), df[A_COLS].values, ndB - ndA, cv=CV)
        gBF_hat = cross_val_predict(mk(), df[B_COLS].values, ndF - ndB, cv=CV)
        best = None
        for fB in np.linspace(0, 1, 21):          # fraction escalated A->B
            kB = int(round(fB * n))
            toB = np.zeros(n, bool)
            if kB: toB[np.argsort(-gAB_hat)[:kB]] = True
            for fF in np.linspace(0, 1, 21):      # fraction of the B-set escalated B->Full
                idxB = np.flatnonzero(toB)
                kF = int(round(fF * len(idxB)))
                toF = np.zeros(n, bool)
                if kF:
                    toF[idxB[np.argsort(-gBF_hat[idxB])[:kF]]] = True
                nd = np.where(toF, ndF, np.where(toB, ndB, ndA)).mean() * 100
                cost = np.where(toF, CN["Full"], np.where(toB, CN["-Events"], CN["A_visual"])).mean()
                gap = nd - H(cost)
                if best is None or gap > best[0]:
                    best = (gap, nd, cost, fB, fF)
        gap, nd, cost, fB, fF = best
        rho_AB = np.corrcoef(gAB_hat, ndB - ndA)[0, 1]
        rho_BF = np.corrcoef(gBF_hat, ndF - ndB)[0, 1]
        verdict = "ABOVE HULL" if gap > 0 else "below hull"
        print(f"  [{mname:5}] best-vs-hull {gap:+.2f}  ({nd:.2f} @ {cost:.2f}, hull {H(cost):.2f}) "
              f"escalate {fB:.0%}->B {fF:.0%}->Full | {verdict}")
        print(f"          OOF gain corr: rho(A->B)={rho_AB:+.3f}  rho(B->Full)={rho_BF:+.3f}")

    # oracle over the SAME policy class (true gains, same top-k escalation)
    best = None
    for fB in np.linspace(0, 1, 21):
        kB = int(round(fB * n)); toB = np.zeros(n, bool)
        if kB: toB[np.argsort(-(ndB - ndA))[:kB]] = True
        for fF in np.linspace(0, 1, 21):
            idxB = np.flatnonzero(toB); kF = int(round(fF * len(idxB)))
            toF = np.zeros(n, bool)
            if kF: toF[idxB[np.argsort(-(ndF - ndB)[idxB])[:kF]]] = True
            nd = np.where(toF, ndF, np.where(toB, ndB, ndA)).mean() * 100
            cost = np.where(toF, CN["Full"], np.where(toB, CN["-Events"], CN["A_visual"])).mean()
            gap = nd - H(cost)
            if best is None or gap > best[0]: best = (gap, nd, cost)
    print(f"  [oracle] best-vs-hull {best[0]:+.2f}  ({best[1]:.2f} @ {best[2]:.2f})  <- ceiling of this policy class")


for s in ["noASR", "ASR"]:
    run(s)
