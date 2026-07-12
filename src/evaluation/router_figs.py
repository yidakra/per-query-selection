#!/usr/bin/env python
"""Final figures for the adaptive-routing extension.

Fig 1 (frontier): per-cell accuracy-compute curve for the expected-gain cascade, against the
  convex hull of the FIXED-tier policies.  The hull between Fixed-A and Fixed-B is exactly the
  cost-matched random-escalation baseline, so it is the honest bar; we draw it, not just Fixed-B.
  Bootstrap band over queries.  Oracle curve shown as the ceiling.
Fig 2 (heterogeneity): sd(per-query A->B gain) vs routing headroom, across all six cells.
  The load-bearing claim: routing value scales with query-complexity heterogeneity.

CPU-only.  Writes reports/figures/router_frontier.{pdf,png} and router_heterogeneity.{pdf,png}.
"""
import os, sys, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_hetero import (comps_multivent, comps_msrvtt, mk, gap_at, A_COLS, FRACS, RNG)  # noqa: E402
from oracle_router_headroom import fuse, per_query, LADDER  # noqa: E402
from router_cascade_exp import conf_feats, CN  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
FIG = f"{_ROOT}/reports/figures"
OUT = f"{_ROOT}/results/ablations/router_curves.json"
NBOOT = 2000
CELLS = [("MultiVENT noASR", lambda: comps_multivent("noASR")),
         ("MultiVENT ASR", lambda: comps_multivent("ASR")),
         ("MSR-VTT (mCLIP) noASR", lambda: comps_msrvtt("multiclip", "noASR")),
         ("MSR-VTT (mCLIP) ASR", lambda: comps_msrvtt("multiclip", "ASR")),
         ("MSR-VTT (IV2) noASR", lambda: comps_msrvtt("internvideo2", "noASR")),
         ("MSR-VTT (IV2) ASR", lambda: comps_msrvtt("internvideo2", "ASR"))]


def curve(queries, target, comps):
    fA = fuse(comps, LADDER["A_visual"]); fB = fuse(comps, LADDER["B_noevents"])
    ndA = per_query(fA, target)[0].numpy(); ndB = per_query(fB, target)[0].numpy()
    ndF = per_query(fuse(comps, LADDER["C_full"]), target)[0].numpy()
    rows = []
    for i, q in enumerate(queries):
        r = {"charlen": len(q), "wordlen": len(q.split())}
        for k, v in conf_feats(fA[i]).items():
            r[f"A_{k}"] = v
        rows.append(r)
    X = pd.DataFrame(rows)[A_COLS].values
    g = ndB - ndA
    n = len(g)
    ghat = cross_val_predict(mk(), X, g, cv=KFold(5, shuffle=True, random_state=0))

    pts = []
    for f in FRACS:
        k = int(round(f * n)); S = np.argsort(-ghat)[:k]
        sel = np.zeros(n, bool); sel[S] = True
        nd = np.where(sel, ndB, ndA).mean() * 100
        cost = np.where(sel, CN["-Events"], CN["A_visual"]).mean()
        chord = (ndA.mean() + (k / n) * g.mean()) * 100
        orc = (ndA.mean() + gap_at(f, g, g) + (k / n) * g.mean()) * 100
        bs = np.array([gap_at(f, ghat[i2], g[i2]) * 100
                       for i2 in (RNG.integers(0, n, n) for _ in range(NBOOT))])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pts.append(dict(f=float(f), cost=float(cost), ndcg=float(nd), chord=float(chord),
                        oracle=float(orc), lo=float(chord + lo), hi=float(chord + hi)))
    fixed = dict(A=(CN["A_visual"], float(ndA.mean() * 100)), B=(CN["-Events"], float(ndB.mean() * 100)),
                 Full=(CN["Full"], float(ndF.mean() * 100)))
    return pts, fixed, float(g.std() * 100)


data = {}
for name, fn in CELLS:
    pts, fixed, het = curve(*fn())
    data[name] = dict(pts=pts, fixed=fixed, het_sd=het)
    print(f"[curve] {name}: sd(gain)={het:.2f}")
json.dump(data, open(OUT, "w"), indent=2)

# ---------------- Fig 1: frontier ----------------
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True)
for ax, (name, _) in zip(axes.ravel(), CELLS):
    d = data[name]; pts = d["pts"]
    c = [p["cost"] for p in pts]
    ax.fill_between(c, [p["lo"] for p in pts], [p["hi"] for p in pts], alpha=.18, color="C0", lw=0)
    ax.plot(c, [p["oracle"] for p in pts], color="0.55", ls=":", lw=1.6, label="Oracle (true gain)")
    ax.plot(c, [p["chord"] for p in pts], color="0.25", ls="--", lw=1.5,
            label="Fixed hull = cost-matched\nrandom escalation")
    ax.plot(c, [p["ndcg"] for p in pts], color="C0", lw=2.1, label="Expected-gain cascade")
    fx = d["fixed"]
    for lab, (cc, nn) in fx.items():
        ax.plot([cc], [nn], "o", ms=5.5, color="C3", zorder=5)
        ax.annotate(f"Fixed-{lab}", (cc, nn), textcoords="offset points", xytext=(5, -10), fontsize=7.5)
    ax.axhline(fx["B"][1], color="C3", lw=.7, alpha=.45)
    ax.set_title(f"{name}   sd(gain)={d['het_sd']:.1f}", fontsize=9.5)
    ax.grid(alpha=.25, lw=.5)
    # the router lives strictly between tier A and tier B; Fixed-Full sits at 1.0, off to the right.
    ax.set_xlim(CN["A_visual"] * 0.55, CN["-Events"] * 1.30)
    ax.axvline(CN["-Events"], color="C3", lw=.6, alpha=.35, ls=":")
for ax in axes[-1]:
    ax.set_xlabel("cost (measured J/query, normalized to Full = 1.0)")
for ax in axes[:, 0]:
    ax.set_ylabel("nDCG")
axes[0, 0].legend(fontsize=7, loc="lower right", framealpha=.9)
fig.suptitle("Per-query expected-gain routing beats the cost-matched fixed baseline in the low-budget regime\n"
             "(the cascade never buys the Full tier: event-decomposition gain is not predictable per query)",
             fontsize=10.5)
# the whole frontier sits in the cheapest ~1.6% of the Full budget -- the measured-joules payoff
fig.text(0.5, 0.005,
         f"x-axis is MEASURED energy: tier B = {CN['-Events']*100:.2f}% of Full, tier A = {CN['A_visual']*100:.2f}% "
         f"(Full = {int(round(1418.36))} J/query, off-axis at 1.0). "
         "The component-count proxy overstated this range as 20-40% of Full.",
         ha="center", fontsize=7.6, color="0.35")
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
for e in ("pdf", "png"):
    fig.savefig(f"{FIG}/router_frontier.{e}", dpi=160, bbox_inches="tight")
print(f"wrote {FIG}/router_frontier.{{pdf,png}}")

# ---------------- Fig 2: heterogeneity ----------------
het = json.load(open(f"{_ROOT}/results/ablations/router_hetero.json"))
sd = np.array([r["het_sd"] for r in het])
orc = np.array([r["oracle_gap"] for r in het])
gap = np.array([r["nested_gap"] for r in het])
sem = np.array([r["nested_sem"] for r in het])
lbl = [r["cell"] for r in het]
mv = np.array(["MultiVENT" in l for l in lbl])


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
for ax, y, ye, ttl in [(axes[0], orc, None, "Oracle headroom"),
                       (axes[1], gap, sem, "Achieved (nested) frontier gap")]:
    for m, col, nm in [(mv, "C3", "MultiVENT"), (~mv, "C0", "MSR-VTT")]:
        if ye is None:
            ax.scatter(sd[m], y[m], c=col, s=55, label=nm, zorder=3)
        else:
            ax.errorbar(sd[m], y[m], yerr=ye[m], fmt="o", c=col, ms=7, capsize=3, label=nm, zorder=3)
    k, b = np.polyfit(sd, y, 1)
    xs = np.linspace(sd.min() - .6, sd.max() + .6, 20)
    ax.plot(xs, k * xs + b, color="0.4", ls="--", lw=1.2, zorder=1)
    ax.set_title(f"{ttl}\nSpearman rho = {spearman(sd, y):+.2f}  (n=6 cells)", fontsize=10)
    ax.set_xlabel("query-complexity heterogeneity:  sd(per-query A$\\to$B nDCG gain)")
    ax.set_ylabel("nDCG over cost-matched baseline")
    ax.grid(alpha=.25, lw=.5)
    ax.legend(fontsize=8)
fig.suptitle("Routing value scales with query-complexity heterogeneity — not with dataset identity",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
for e in ("pdf", "png"):
    fig.savefig(f"{FIG}/router_heterogeneity.{e}", dpi=160, bbox_inches="tight")
print(f"wrote {FIG}/router_heterogeneity.{{pdf,png}}")
print(f"\nSpearman sd(gain) vs oracle headroom : {spearman(sd, orc):+.3f}")
print(f"Spearman sd(gain) vs achieved gap    : {spearman(sd, gap):+.3f}")
