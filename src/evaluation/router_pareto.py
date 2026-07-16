#!/usr/bin/env python
"""The effectiveness hook: the router MOVES the efficiency-effectiveness Pareto frontier.

`router_findings.md` reports the vertical (iso-cost) reading: at a fixed budget the router beats the
cost-matched random mixture by +gap nDCG. Reviewers of a full IR track also want the EFFECTIVENESS
reading, which is the same fact read horizontally: to reach a given nDCG, the router costs LESS than
the random-escalation baseline. This script computes both, in measured joules, on the existing A->B
caches -- no new retrieval.

Setup on the accuracy-compute plane, per cell (cost = marginal J/query, tier_cost.py):
  cost(f)        = J_A + f*(J_B - J_A)
  chord_ndcg(f)  = ndA + f*(ndB - ndA)              # escalate a RANDOM fraction f (the honest baseline)
  router_ndcg(f) = ndA + (1/n) sum_{top-f by predicted gain} g_i   =  chord_ndcg(f) + gap(f)

So the router frontier is the chord lifted by gap(f) >= 0 across the plateau -> it WEAKLY DOMINATES
the random frontier on the plane. Two readings:
  - ISO-COST  (vertical): +gap(f) nDCG at equal cost. (the headline we already report)
  - ISO-ACCURACY (horizontal): to match the nDCG that random escalation reaches at f=0.5, the router
    needs only f_router < 0.5 of the escalation budget -> a cost SAVING, in joules and %.

The negatives are the frontier's shape, not a failure: B->Full (177x tier B, prize is label noise)
and Tier C are DOMINATED escalations; a Pareto-optimal router prunes them. The router is efficient
*because* it declines them. CPU-only; predicted gain is OOF (same protocol as router_hetero.py).
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_hetero import comps_multivent, comps_msrvtt, mk, A_COLS  # noqa: E402
from oracle_router_headroom import fuse, per_query, LADDER  # noqa: E402
from router_cascade_exp import conf_feats  # noqa: E402
from tier_cost import JOULES  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402

J_A, J_B, J_F = JOULES["A_visual"], JOULES["-Events"], JOULES["Full"]
DJ = J_B - J_A                                   # A->B escalation increment, 15.52 J
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{_ROOT}/results/ablations/router_pareto.json"
FIG = f"{_ROOT}/reports/figures"

CELLS = [("MultiVENT/mCLIP/noASR", lambda: comps_multivent("noASR")),
         ("MultiVENT/mCLIP/ASR", lambda: comps_multivent("ASR")),
         ("MSR-VTT/mCLIP/noASR", lambda: comps_msrvtt("multiclip", "noASR")),
         ("MSR-VTT/mCLIP/ASR", lambda: comps_msrvtt("multiclip", "ASR")),
         ("MSR-VTT/IV2/noASR", lambda: comps_msrvtt("internvideo2", "noASR")),
         ("MSR-VTT/IV2/ASR", lambda: comps_msrvtt("internvideo2", "ASR"))]


def cell(name, queries, target, comps):
    fA = fuse(comps, LADDER["A_visual"]); fB = fuse(comps, LADDER["B_noevents"])
    ndA = per_query(fA, target)[0].numpy(); ndB = per_query(fB, target)[0].numpy()
    ndFull = per_query(fuse(comps, LADDER["C_full"]), target)[0].numpy()
    g = ndB - ndA
    n = len(g)
    ndA_m, ndB_m, ndF_m = ndA.mean(), ndB.mean(), ndFull.mean()

    df = pd.DataFrame([{**{"charlen": len(q), "wordlen": len(q.split())},
                        **{f"A_{k}": v for k, v in conf_feats(fA[i]).items()}}
                       for i, q in enumerate(queries)])
    ghat = cross_val_predict(mk(), df[A_COLS].values, g, cv=KFold(5, shuffle=True, random_state=0))
    order = np.argsort(-ghat)
    cumg = np.concatenate([[0.0], np.cumsum(g[order])])          # cumg[k] = sum of top-k true gains

    def router_nd(f):
        k = int(round(f * n)); return ndA_m + cumg[k] / n
    def chord_nd(f):
        return ndA_m + f * (ndB_m - ndA_m)

    fs = np.arange(0.0, 1.0001, 0.01)
    router = np.array([router_nd(f) for f in fs])
    chord = np.array([chord_nd(f) for f in fs])
    cost = J_A + fs * DJ

    # ISO-COST (vertical): max nDCG lift over the chord, in points
    iso_cost_gap = float((router - chord).max() * 100)
    f_at_max = float(fs[int(np.argmax(router - chord))])

    # ISO-ACCURACY (horizontal): to reach the nDCG random reaches at f=0.5, router needs f_router
    y_mid = chord_nd(0.5)
    reach = np.where(router >= y_mid - 1e-12)[0]
    f_router = float(fs[reach[0]]) if len(reach) else 1.0
    cost_router = J_A + f_router * DJ
    cost_chord = J_A + 0.5 * DJ
    saving_J = float(cost_chord - cost_router)                  # J/query saved to hit the same nDCG
    saving_pct_escal = float((0.5 - f_router) / 0.5 * 100)      # % of the escalation budget saved

    # whole-frontier: max horizontal budget saving over achievable targets
    best = 0.0
    for f_c in np.arange(0.05, 1.0, 0.01):
        y = chord_nd(f_c)
        rr = np.where(router >= y - 1e-12)[0]
        if len(rr):
            best = max(best, f_c - fs[rr[0]])
    max_budget_saving_pct = float(best * 100)

    dominates = bool(np.all(router >= chord - 1e-9))
    print(f"### {name}")
    print(f"  Fixed-A {ndA_m*100:.2f}@{J_A:.1f}J  Fixed-B {ndB_m*100:.2f}@{J_B:.1f}J  "
          f"Fixed-Full {ndF_m*100:.2f}@{J_F:.0f}J")
    print(f"  ISO-COST   : +{iso_cost_gap:.2f} nDCG over the chord at equal cost (f*={f_at_max:.2f})")
    print(f"  ISO-ACCURACY: match random@50% nDCG={y_mid*100:.2f} with f={f_router:.2f} of budget "
          f"-> save {saving_J:.2f} J/q = {saving_pct_escal:.0f}% of the escalation cost")
    print(f"  frontier weakly dominates the chord: {dominates};  max budget saving {max_budget_saving_pct:.0f}%")
    return dict(cell=name, n=n, ndA=float(ndA_m*100), ndB=float(ndB_m*100), ndFull=float(ndF_m*100),
                J_A=J_A, J_B=J_B, J_Full=J_F,
                iso_cost_gap=iso_cost_gap, f_at_max_gap=f_at_max,
                iso_acc_target_ndcg=float(y_mid*100), iso_acc_f_router=f_router,
                iso_acc_saving_J=saving_J, iso_acc_saving_pct_escalation=saving_pct_escal,
                max_budget_saving_pct=max_budget_saving_pct, dominates_chord=dominates,
                frontier={"cost": cost.tolist(), "router": (router*100).tolist(),
                          "chord": (chord*100).tolist()})


def figure(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, r in zip(axes.ravel(), res):
            c = np.array(r["frontier"]["cost"]); rt = np.array(r["frontier"]["router"])
            ch = np.array(r["frontier"]["chord"])
            ax.fill_between(c, ch, rt, where=rt >= ch, color="#4C78A8", alpha=0.18)
            ax.plot(c, ch, "--", color="#888", lw=1.3, label="random-f (chord)")
            ax.plot(c, rt, "-", color="#4C78A8", lw=2.0, label="router")
            ax.scatter([r["J_A"], r["J_B"]], [r["ndA"], r["ndB"]], color="#111", zorder=5, s=22)
            ax.annotate("A", (r["J_A"], r["ndA"]), textcoords="offset points", xytext=(4, -9), fontsize=8)
            ax.annotate("B", (r["J_B"], r["ndB"]), textcoords="offset points", xytext=(2, 4), fontsize=8)
            ax.set_title(r["cell"], fontsize=9)
            ax.set_xlabel("marginal J/query"); ax.set_ylabel("nDCG@10")
            ax.tick_params(labelsize=8)
        axes.ravel()[0].legend(fontsize=8, loc="lower right")
        fig.suptitle("Router moves the efficiency–effectiveness frontier in the A→B budget region "
                     "(Full is 60–177× further right, off-scale)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        for e in ("pdf", "png"):
            fig.savefig(f"{FIG}/router_pareto.{e}", dpi=160, bbox_inches="tight")
        print(f"wrote {FIG}/router_pareto.{{pdf,png}}")
    except Exception as e:
        print(f"[fig skipped] {e}")


def main():
    res = [cell(name, *fn()) for name, fn in CELLS]
    json.dump({"cells": res, "tier_cost_J": JOULES}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    figure(res)


if __name__ == "__main__":
    main()
