#!/usr/bin/env python
"""Diagnostic: can query-text features route AT ALL? Tests configs against the
fixed-baseline convex hull (the bar the trained router must beat to matter)."""
import json, os, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.dummy import DummyClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows_all = [json.loads(l) for l in open(os.path.join(ROOT, "results/ablations/router_trainset.jsonl"))]
LADDER = ["A_visual", "-Events", "Full"]; COST = {"A_visual": 1, "-Events": 2, "Full": 5}
CN = {t: COST[t] / 5 for t in LADDER}

def sel(ds, enc, s, pool): return [r for r in rows_all if r["dataset"]==ds and r["encoder"]==enc and r["setting"]==s and r.get("pool") in (pool,None)]
def cwe(r, eps):
    v={t:r["_ndcg"][t] for t in LADDER}; b=max(v.values())
    for t in LADDER:
        if v[t]>=b-eps: return t
    return LADDER[-1]
def stats(rows, ch):
    return (np.mean([r["_ndcg"][t] for r,t in zip(rows,ch)])*100, np.mean([CN[t] for t in ch]))

def feats(texts):
    return FeatureUnion([
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, sublinear_tf=True)),
    ])

def oof(texts, y, cw):
    y=np.asarray(y); _,c=np.unique(y,return_counts=True)
    if len(c)<2: return y.copy()
    k=int(min(5,c.min()));  k=max(k,2)
    cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    pipe=Pipeline([("f",feats(texts)),("clf",LogisticRegression(max_iter=2000,C=4.0,class_weight=cw))])
    return cross_val_predict(pipe, texts, y, cv=cv)

def run(name, rows):
    texts=[r["query"] for r in rows]
    fixed={t:stats(rows,[t]*len(rows)) for t in LADDER}
    print(f"\n### {name} (n={len(rows)})  fixed: "+"  ".join(f"{t}={fixed[t][0]:.2f}@{fixed[t][1]:.2f}" for t in LADDER))
    # bar to beat: best fixed NDCG at any cost = Fixed-Full; and Fixed-B as cheap strong baseline
    fullN=fixed["Full"][0]; bN,bC=fixed["-Events"]
    for cw in [None, "balanced"]:
        best=None
        for eps in [0.0,0.01,0.03,0.05,0.08,0.12,0.2]:
            y=[cwe(r,eps) for r in rows]
            pred=list(oof(texts,y,cw)); n,c=stats(rows,pred)
            acc=np.mean(np.asarray(pred)==np.asarray(y))
            # does this point beat Fixed-B? (>= its NDCG at <= its cost, or better tradeoff)
            if best is None or n>best[0]: best=(n,c,eps,acc)
        n,c,eps,acc=best
        dom_B = "DOMINATES-B" if (n>=bN-1e-9 and c<=bC+1e-9) else ("beats-B-ndcg" if n>bN else "< B")
        print(f"  LR cw={str(cw):8} best router pt: NDCG {n:.2f} @ cost {c:.2f} (eps={eps}, tier_acc={acc:.2f})  vs FixedB {bN:.2f}@{bC:.2f} -> {dom_B}")
    # majority-class dummy tier_acc reference at eps=0
    y0=[cwe(r,0.0) for r in rows]
    dm=cross_val_predict(DummyClassifier(strategy="most_frequent"), np.zeros((len(rows),1)), y0, cv=5)
    print(f"  [ref] majority-tier acc={np.mean(np.asarray(dm)==np.asarray(y0)):.2f}; oracle eps=0 NDCG={stats(rows,y0)[0]:.2f}@{stats(rows,y0)[1]:.2f}")

run("MultiVENT noASR fullpool", sel("MultiVENT","multiclip","noASR","full_pool"))
run("MultiVENT ASR fullpool", sel("MultiVENT","multiclip","ASR","full_pool"))
run("MSRVTT noASR", sel("MSR-VTT","multiclip","noASR","full"))
