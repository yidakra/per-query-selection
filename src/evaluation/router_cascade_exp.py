#!/usr/bin/env python
"""Decisive experiment: does the routing signal live in the CHEAP TIER'S RETRIEVAL
CONFIDENCE (not the query text)? Builds per-query confidence features from the fused
tier-A / tier-B score matrices (top-1 margin, z-score, entropy over the gallery, and
A-vs-B top-rank disagreement) and compares three feature sets against the fixed-tier
convex hull under honest out-of-fold CV.  CPU-only; reads cached component tensors.
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from oracle_router_headroom import canonical_order, fuse, per_query, TEXT4, LADDER  # noqa: E402
import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import pandas as pd

RUNS = "/home/ubuntu/q2e_repro/runs"
LAD = {"A_visual": 1, "-Events": 2, "Full": 5}; CN = {t: LAD[t] / 5 for t in LAD}
LADDER_TIERS = ["A_visual", "-Events", "Full"]
SETTINGS = {"noASR": ("multivent_textonly_noASR", "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"),
            "ASR":   ("multivent_textonly_ASR",   "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR")}


def conf_feats(fused_row):
    """Retrieval-confidence features from one query's fused scores over the gallery."""
    s = fused_row.double()
    p = torch.softmax(s, dim=0)
    ssort, idx = torch.sort(s, descending=True)
    psort = torch.softmax(ssort, dim=0)
    ent = float(-(p * (p + 1e-12).log()).sum())
    return {
        "top1": float(ssort[0]), "margin12": float(ssort[0] - ssort[1]),
        "margin13": float(ssort[0] - ssort[2]), "z1": float((ssort[0] - s.mean()) / (s.std() + 1e-9)),
        "entropy": ent, "maxp": float(psort[0]), "top5mass": float(psort[:5].sum()),
        "std": float(s.std()), "argtop": int(idx[0]),
    }


def build(setting):
    text_tag, ds_name = SETTINGS[setting]
    ds = load_from_disk(f"data/MultiVENT/{ds_name}")
    queries, video_ids, target = canonical_order(ds)
    VID = os.path.join(RUNS, "multivent_video_multiclip")
    qv = torch.load(os.path.join(VID, "cache", "query_vs_video.pt"))
    order = json.load(open(os.path.join(VID, "video_order.json")))
    qv_v = {v: i for i, v in enumerate(order["video_ids"])}; qv_q = {q: i for i, q in enumerate(order["queries"])}
    rmap = torch.tensor([qv_q[q] for q in queries]); cmap = torch.tensor([qv_v[v] for v in video_ids])
    comps = {"query_vs_video": qv[rmap][:, cmap]}
    cdir = os.path.join(RUNS, text_tag, "cache")
    for pth in TEXT4:
        comps[pth] = torch.load(os.path.join(cdir, f"{pth}.pt"))

    fusedA = fuse(comps, LADDER["A_visual"]); fusedB = fuse(comps, LADDER["B_noevents"])
    ndA = per_query(fusedA, target)[0]; ndB = per_query(fusedB, target)[0]
    ndFull = per_query(fuse(comps, LADDER["C_full"]), target)[0]
    nd = {"A_visual": ndA, "-Events": ndB, "Full": ndFull}

    recs = []
    for i, q in enumerate(queries):
        fa = conf_feats(fusedA[i]); fb = conf_feats(fusedB[i])
        row = {"query": q, "charlen": len(q), "wordlen": len(q.split())}
        for k, v in fa.items(): row[f"A_{k}"] = v
        for k, v in fb.items(): row[f"B_{k}"] = v
        row["AB_disagree"] = int(fa["argtop"] != fb["argtop"])   # does adding captions change the top hit?
        row["ndcgA"] = float(ndA[i]); row["ndcgB"] = float(ndB[i]); row["ndcgFull"] = float(ndFull[i])
        recs.append(row)
    return pd.DataFrame(recs), nd


def label(df, eps):
    v = np.stack([df["ndcgA"].values, df["ndcgB"].values, df["ndcgFull"].values], 1)  # N x 3
    best = v.max(1, keepdims=True)
    within = v >= (best - eps)
    # cheapest tier within eps
    lab = np.where(within[:, 0], "A_visual", np.where(within[:, 1], "-Events", "Full"))
    return lab


def stats(df, tiers):
    nd = {"A_visual": df["ndcgA"].values, "-Events": df["ndcgB"].values, "Full": df["ndcgFull"].values}
    sel = np.array([nd[t][i] for i, t in enumerate(tiers)])
    cost = np.array([CN[t] for t in tiers])
    return sel.mean() * 100, cost.mean()


CONF_COLS = ["A_margin12", "A_margin13", "A_z1", "A_entropy", "A_maxp", "A_top5mass",
             "A_std", "A_top1", "B_margin12", "B_margin13", "B_z1", "B_entropy",
             "B_maxp", "B_top5mass", "B_std", "B_top1", "AB_disagree", "charlen", "wordlen"]


def oof(df, y, feats, n_splits=5):
    y = np.asarray(y); _, c = np.unique(y, return_counts=True)
    if len(c) < 2: return y.copy()
    k = max(2, int(min(n_splits, c.min())))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    if feats == "text":
        est = Pipeline([("f", FeatureUnion([
            ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True))])),
            ("clf", LogisticRegression(max_iter=2000, C=4.0))])
        X = df["query"].values
    elif feats == "conf":
        est = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=1.0))])
        X = df[CONF_COLS].values
    else:  # text+conf
        est = ColumnTransformer([
            ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True), "query"),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True), "query"),
            ("num", StandardScaler(), CONF_COLS)])
        est = Pipeline([("ct", est), ("clf", LogisticRegression(max_iter=2000, C=2.0))])
        X = df
    return cross_val_predict(est, X, y, cv=cv)


def run(setting):
    df, nd = build(setting)
    fixed = {t: stats(df, [t] * len(df)) for t in LADDER_TIERS}
    print(f"\n### MultiVENT {setting} full_pool (n={len(df)})")
    for t in LADDER_TIERS:
        print(f"  Fixed-{t:9} NDCG {fixed[t][0]:.2f} @ cost {fixed[t][1]:.2f}")
    bN, bC = fixed["-Events"]; fN = fixed["Full"][0]
    for feats in ["text", "conf", "text+conf"]:
        pts = []
        for eps in [0.0, 0.01, 0.03, 0.05, 0.08, 0.12]:
            y = label(df, eps)
            pred = list(oof(df, y, feats))
            n, c = stats(df, pred)
            acc = float(np.mean(np.asarray(pred) == np.asarray(y)))
            pts.append((n, c, eps, acc))
        # report the point with best NDCG, and the cheapest point matching Full-0.5
        bestN = max(pts, key=lambda p: p[0])
        # does any point beat Fixed-B convex position (>= B ndcg at <= B cost)?
        domB = any(n >= bN - 1e-9 and c <= bC + 1e-9 for n, c, _, _ in pts)
        # cheapest point that matches Full within 0.3 NDCG
        match = [p for p in pts if p[0] >= fN - 0.3]
        mm = min(match, key=lambda p: p[1]) if match else None
        tag = "DOMINATES Fixed-B" if domB else "no dominance"
        line = f"  [{feats:9}] bestNDCG {bestN[0]:.2f}@{bestN[1]:.2f}(eps{bestN[2]},acc{bestN[3]:.2f}) | {tag}"
        if mm:
            line += f" | matches Full({fN:.2f}) at cost {mm[1]:.2f} = {100*(1-mm[1]):.0f}% cheaper"
        print(line)


if __name__ == "__main__":
    for s in ["noASR", "ASR"]:
        run(s)
