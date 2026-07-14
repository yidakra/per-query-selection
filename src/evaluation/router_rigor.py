#!/usr/bin/env python
"""Rigor pack: put the A->B router on the vocabulary and error bars a routing/IR reviewer expects.

Adds, on the existing cached A->B results (no new retrieval), four things the literature now demands:

  1. FRONTIER ENVELOPES -- at each escalation fraction f, the router's realized gap vs the two bounds
     it lives between: RANDOM-f (escalate a random fraction; expected gap = 0, the cost-matched chord)
     and ORACLE-f (escalate the top-f by TRUE gain). This is the oracle/random envelope RouterBench /
     RouteLLM / AcuRank plot.
  2. APGR -- Average Performance Gap Recovered (RouteLLM): mean over f of router_gap(f)/oracle_gap(f).
     The fraction of the recoverable (oracle-over-random) headroom the router captures, one scalar.
     NB the oracle denominator is IN-SAMPLE (router_findings.md sec 4 shows it is optimistic), so this
     APGR is a CONSERVATIVE reading -- it divides by an inflated ceiling.
  3. CPT -- Call-Performance Threshold (RouteLLM): the escalation fraction needed to capture x% of the
     full tier-B improvement. Lower is better. Reported at 50% and 80%.
  4. BOOTSTRAP CIs on Kendall tau and the gap, and BH-FDR correction across the six-cell tau family --
     so "significant" is family-wise honest, not per-test.

Self-contained: recomputes the per-cell OOF predictions with the same protocol as router_hetero.py
(the tau/gap it prints reproduce that script). CPU-only; reads cached component tensors.
"""
import os, sys, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_hetero import comps_multivent, comps_msrvtt, mk, gap_at, A_COLS  # noqa: E402
from oracle_router_headroom import fuse, per_query, LADDER  # noqa: E402
from router_cascade_exp import conf_feats  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

RNG = np.random.default_rng(0)
FRACS = np.arange(0.02, 1.0, 0.02)          # fine grid for smooth APGR / envelopes
F_REF = 0.5                                  # operating point for the bootstrap CI on the gap
N_BOOT = 2000
N_PERM = 2000
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{_ROOT}/results/ablations/router_rigor.json"

CELLS = [("MultiVENT/mCLIP/noASR", lambda: comps_multivent("noASR")),
         ("MultiVENT/mCLIP/ASR", lambda: comps_multivent("ASR")),
         ("MSR-VTT/mCLIP/noASR", lambda: comps_msrvtt("multiclip", "noASR")),
         ("MSR-VTT/mCLIP/ASR", lambda: comps_msrvtt("multiclip", "ASR")),
         ("MSR-VTT/IV2/noASR", lambda: comps_msrvtt("internvideo2", "noASR")),
         ("MSR-VTT/IV2/ASR", lambda: comps_msrvtt("internvideo2", "ASR"))]


def build(queries, target, comps):
    fA = fuse(comps, LADDER["A_visual"]); fB = fuse(comps, LADDER["B_noevents"])
    ndA = per_query(fA, target)[0].numpy(); ndB = per_query(fB, target)[0].numpy()
    rows = []
    for i, q in enumerate(queries):
        r = {"charlen": len(q), "wordlen": len(q.split())}
        for k, v in conf_feats(fA[i]).items():
            r[f"A_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows), (ndB - ndA)


def apgr(ghat, g):
    """Mean over f of router_gap(f)/oracle_gap(f), where oracle_gap(f) > 0."""
    num, ratios = 0, []
    for f in FRACS:
        og = gap_at(f, g, g)
        if og > 1e-9:
            ratios.append(gap_at(f, ghat, g) / og)
    return float(np.mean(ratios)) if ratios else float("nan")


def cpt(ghat, g, x):
    """Min escalation fraction to capture x of the full A->B gain (RouteLLM CPT). nan if gain<=0."""
    total = g.sum()
    if total <= 0:
        return float("nan")
    order = np.argsort(-ghat)
    cum = np.cumsum(g[order]) / total
    k = np.searchsorted(cum >= x, True)          # first index where fraction achieved >= x
    return float((k + 1) / len(g)) if k < len(g) else float("nan")


def envelope(ghat, g):
    return {"f": [float(f) for f in FRACS],
            "router": [float(gap_at(f, ghat, g) * 100) for f in FRACS],
            "oracle": [float(gap_at(f, g, g) * 100) for f in FRACS],
            "random": [0.0 for _ in FRACS]}


def cell(name, queries, target, comps):
    df, g = build(queries, target, comps)
    X = df[A_COLS].values
    n = len(g)
    ghat = cross_val_predict(mk(), X, g, cv=KFold(5, shuffle=True, random_state=0))

    tau = float(kendalltau(ghat, g).statistic)
    perm = np.array([kendalltau(RNG.permutation(ghat), g).statistic for _ in range(N_PERM)])
    p_tau = float((1 + (perm >= tau).sum()) / (1 + N_PERM))

    # bootstrap over queries (predictor fixed): CI on tau and on gap@F_REF
    bt, bg = np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        bt[b] = kendalltau(ghat[idx], g[idx]).statistic
        bg[b] = gap_at(F_REF, ghat[idx], g[idx]) * 100
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    res = {"cell": name, "n": n, "mean_gain": float(g.mean() * 100),
           "tau": tau, "p_tau": p_tau, "tau_ci95": ci(bt),
           "gap_at_0.5": float(gap_at(F_REF, ghat, g) * 100), "gap_ci95": ci(bg),
           "apgr": apgr(ghat, g), "cpt50": cpt(ghat, g, 0.5), "cpt80": cpt(ghat, g, 0.8),
           "envelope": envelope(ghat, g)}
    print(f"### {name}  (n={n})")
    print(f"  tau={tau:+.3f} (perm p={p_tau:.4f}) CI95=[{res['tau_ci95'][0]:+.3f},{res['tau_ci95'][1]:+.3f}]")
    print(f"  gap@0.5={res['gap_at_0.5']:+.2f} CI95=[{res['gap_ci95'][0]:+.2f},{res['gap_ci95'][1]:+.2f}]")
    print(f"  APGR={res['apgr']:.3f}  CPT50={res['cpt50']:.2f}  CPT80={res['cpt80']:.2f}")
    return res


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev
    return adj.tolist()


def main():
    res = [cell(name, *fn()) for name, fn in CELLS]
    p_raw = [r["p_tau"] for r in res]
    p_adj = bh_fdr(p_raw)
    for r, pa in zip(res, p_adj):
        r["p_tau_bh"] = float(pa)
    out = {"cells": res, "bh_family": [r["cell"] for r in res],
           "p_raw": p_raw, "p_bh": p_adj, "fdr_level": 0.05,
           "survive_fdr": [res[i]["cell"] for i in range(len(res)) if p_adj[i] < 0.05]}
    json.dump(out, open(OUT, "w"), indent=2)

    print("\n[BH-FDR across the six-cell tau family]")
    for r, pa in zip(res, p_adj):
        star = "*" if pa < 0.05 else " "
        print(f"  {r['cell']:24s} p_raw={r['p_tau']:.4f}  p_BH={pa:.4f}{star}")
    print(f"  survive FDR<0.05: {out['survive_fdr']}")
    print(f"\n[summary]  {'cell':24s} {'APGR':>6} {'CPT50':>6} {'CPT80':>6}  gap@0.5 (CI95)")
    for r in res:
        print(f"  {r['cell']:24s} {r['apgr']:6.3f} {r['cpt50']:6.2f} {r['cpt80']:6.2f}  "
              f"{r['gap_at_0.5']:+.2f} [{r['gap_ci95'][0]:+.2f},{r['gap_ci95'][1]:+.2f}]")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
