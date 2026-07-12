#!/usr/bin/env python
"""THE heterogeneity test: adaptive routing's value should scale with how heterogeneous a
dataset's per-query tier preference is.  MultiVENT (multi-gold, multilingual gallery) should
show a real A->B escalation gap over the cost-matched chord; MSR-VTT (1.01 gold/query, 0.86
majority tier prior) should show ~none.

Reports, per cell: the escalation-gain correlation with a permutation p-value, the nested
(selection-bias-free) frontier gap, the oracle ceiling, and a heterogeneity statistic
(std of the per-query A->B gain).  CPU-only; reads cached component tensors.
"""
import os, sys, json, numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from oracle_router_headroom import canonical_order, fuse, per_query, TEXT4, LADDER  # noqa: E402
from router_cascade_exp import conf_feats, CN  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402

RNG = np.random.default_rng(0)
FRACS = np.arange(0.05, 1.0, 0.05)
A_COLS = ["A_margin12", "A_margin13", "A_z1", "A_entropy", "A_maxp", "A_top5mass", "A_std", "A_top1",
          "charlen", "wordlen"]
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
RUNS = f"{_ROOT}/runs"
OUT = f"{_ROOT}/results/ablations/router_hetero.json"

MV = {"noASR": ("multivent_textonly_noASR", "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"),
      "ASR": ("multivent_textonly_ASR", "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR")}
# Absolute, not cwd-relative: importing oracle_router_headroom chdirs into the upstream tree, so a
# relative "data/..." here silently resolves against external/q2e_official/ instead of the repo root.
MSRVTT_DS = f"{_ROOT}/data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_{{s}}"


def mk():
    return Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])


def gap_at(f, ghat, g):
    n = len(g); k = int(round(f * n))
    if k == 0:
        return 0.0
    S = np.argsort(-ghat)[:k]
    return (k / n) * (g[S].mean() - g.mean())


def comps_multivent(setting):
    text_tag, ds_name = MV[setting]
    ds = load_from_disk(f"{_ROOT}/data/MultiVENT/{ds_name}")
    queries, video_ids, target = canonical_order(ds)
    VID = os.path.join(RUNS, "multivent_video_multiclip")
    qv = torch.load(os.path.join(VID, "cache", "query_vs_video.pt"))
    order = json.load(open(os.path.join(VID, "video_order.json")))
    qi = {q: i for i, q in enumerate(order["queries"])}
    vi = {v: i for i, v in enumerate(order["video_ids"])}
    rmap = torch.tensor([qi[q] for q in queries]); cmap = torch.tensor([vi[v] for v in video_ids])
    comps = {"query_vs_video": qv[rmap][:, cmap]}
    cdir = os.path.join(RUNS, text_tag, "cache")
    for p in TEXT4:
        comps[p] = torch.load(os.path.join(cdir, f"{p}.pt"))
    return queries, target, comps


def comps_msrvtt(enc, setting):
    ds = load_from_disk(MSRVTT_DS.format(s=setting))
    queries, _, target = canonical_order(ds)
    cdir = os.path.join(RUNS, f"msrvtt_{enc}_{setting}", "cache")
    comps = {p: torch.load(os.path.join(cdir, f"{p}.pt")) for p in ["query_vs_video"] + TEXT4}
    return queries, target, comps


def cell(name, queries, target, comps):
    fA = fuse(comps, LADDER["A_visual"]); fB = fuse(comps, LADDER["B_noevents"])
    fC = fuse(comps, LADDER["C_full"])
    ndA = per_query(fA, target)[0].numpy(); ndB = per_query(fB, target)[0].numpy()
    ndF = per_query(fC, target)[0].numpy()
    rows = []
    for i, q in enumerate(queries):
        fa = conf_feats(fA[i])
        r = {"charlen": len(q), "wordlen": len(q.split())}
        for k, v in fa.items():
            r[f"A_{k}"] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    g = ndB - ndA
    n = len(g)
    gold_per_q = float(target.sum(1).double().mean())

    ghat = cross_val_predict(mk(), df[A_COLS].values, g, cv=KFold(5, shuffle=True, random_state=0))
    rho = np.corrcoef(ghat, g)[0, 1]
    perm = np.array([np.corrcoef(RNG.permutation(ghat), g)[0, 1] for _ in range(2000)])
    p_rho = (1 + (perm >= rho).sum()) / (1 + len(perm))

    # nested, selection-bias-free frontier gap
    outer = KFold(n_splits=5, shuffle=True, random_state=1)
    gaps = []
    for tr, te in outer.split(np.arange(n)):
        m = mk().fit(df[A_COLS].values[tr], g[tr])
        gh_tr = cross_val_predict(mk(), df[A_COLS].values[tr], g[tr],
                                  cv=KFold(5, shuffle=True, random_state=2))
        f_star = max(FRACS, key=lambda f: gap_at(f, gh_tr, g[tr]))
        gaps.append(gap_at(f_star, m.predict(df[A_COLS].values[te]), g[te]) * 100)
    gaps = np.array(gaps)
    orc = max(gap_at(f, g, g) * 100 for f in FRACS)

    het = float(g.std() * 100)          # per-query heterogeneity of the A->B gain
    frac_neg = float((g < -1e-9).mean())  # queries HURT by adding captions
    print(f"\n### {name}  (n={n}, gold/q={gold_per_q:.2f})")
    print(f"  Fixed-A {ndA.mean()*100:.2f}@{CN['A_visual']:.4f}  Fixed-B {ndB.mean()*100:.2f}@{CN['-Events']:.4f}  "
          f"Fixed-Full {ndF.mean()*100:.2f}@{CN['Full']:.2f}  (cost = J/query, Full=1.0)")
    print(f"  heterogeneity: sd(gain A->B) = {het:.2f} NDCG; {frac_neg:.0%} of queries HURT by captions")
    print(f"  rho(pred,true gain) = {rho:+.3f}  perm p = {p_rho:.4f}")
    print(f"  NESTED frontier gap = {gaps.mean():+.2f} +/- {gaps.std(ddof=1)/np.sqrt(len(gaps)):.2f} (sem)")
    print(f"  ORACLE ceiling      = {orc:+.2f}   -> captured {100*gaps.mean()/orc if orc>0 else 0:.0f}%")
    return dict(cell=name, n=n, gold_per_q=gold_per_q, het_sd=het, frac_hurt=frac_neg,
                rho=float(rho), p_rho=float(p_rho), nested_gap=float(gaps.mean()),
                nested_sem=float(gaps.std(ddof=1) / np.sqrt(len(gaps))), oracle_gap=float(orc),
                fixedA=float(ndA.mean() * 100), fixedB=float(ndB.mean() * 100), fixedFull=float(ndF.mean() * 100))


def main():
    res = []
    for s in ["noASR", "ASR"]:
        res.append(cell(f"MultiVENT/multiclip/{s}", *comps_multivent(s)))
    for enc in ["multiclip", "internvideo2"]:
        for s in ["noASR", "ASR"]:
            res.append(cell(f"MSR-VTT/{enc}/{s}", *comps_msrvtt(enc, s)))

    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    print("\n[heterogeneity vs routing value]")
    print(f"  {'cell':32} {'gold/q':>7} {'sd(gain)':>9} {'nested gap':>12} {'oracle':>8}")
    for r in res:
        print(f"  {r['cell']:32} {r['gold_per_q']:7.2f} {r['het_sd']:9.2f} "
              f"{r['nested_gap']:+7.2f}+/-{r['nested_sem']:.2f} {r['oracle_gap']:+8.2f}")


if __name__ == "__main__":
    main()
