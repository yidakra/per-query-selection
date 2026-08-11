"""Embedding-based pre-retrieval predictors as a boundary probe.

The reworded boundary says: predictors that aggregate corpus term statistics cannot select an
evidence source, and predictors that read option outcomes can. Embedding-based pre-retrieval
predictors (Arabzadeh et al.'s specificity line) are the family the wording makes a prediction
about: they aggregate no corpus statistics, and they also observe no option outcomes, so the
boundary predicts they fail at source selection too. This runs four label-free query-embedding
features through the standard symmetric protocol so the prediction is tested, not asserted.

Features per query, all computable before any retrieval and without any index or label:
  norm       L2 norm of the sentence embedding (specificity proxy)
  density10  mean cosine distance to the 10 nearest other queries (query-space density; no labels,
             unlike QSD, so it is leakage-free by construction)
  centroid   cosine distance to the mean query embedding
  length     token count

  python src/multivent2/mv2_embed_probe.py
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_queries, load_qrels  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from mv2_qpp_table import CELLS, route_nested  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
MARGIN = 5e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_embed_probe.json"))
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.model_selection import GroupKFold

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    model = SentenceTransformer(a.model, device="cpu")

    out = {"model": a.model, "cells": {}}
    for label, fn in CELLS:
        cell = json.load(open(os.path.join(ABL, fn)))
        qids = [q for q in cell["per_query"] if q in queries]
        ndA = np.array([cell["per_query"][q]["ndA"] for q in qids])
        ndB = np.array([cell["per_query"][q]["ndB"] for q in qids])
        g = ndB - ndA
        E = model.encode([queries[q] for q in qids], batch_size=64, convert_to_numpy=True,
                         show_progress_bar=False)
        En = E / np.linalg.norm(E, axis=1, keepdims=True)
        sim = En @ En.T
        np.fill_diagonal(sim, -1.0)
        top10 = np.sort(sim, axis=1)[:, -10:]
        feats = {
            "norm": np.linalg.norm(E, axis=1),
            "density10": 1.0 - top10.mean(axis=1),
            "centroid": 1.0 - En @ (En.mean(axis=0) / np.linalg.norm(En.mean(axis=0))),
            "length": np.array([len(queries[q].split()) for q in qids], dtype=float),
        }
        grp = event_groups(qids, qrels)
        splits = list(GroupKFold(5).split(np.arange(len(qids)), groups=grp))
        orig = max(ndA.mean(), ndB.mean())

        rows = {}
        for name, x in feats.items():
            (nd, tau, _, frac, _), _cal = route_nested(x, g, ndA, ndB, splits, grp, seed=a.seed)
            rows[name] = {"routed": nd, "tau": tau, "frac": frac,
                          "underlined": bool(nd > orig + MARGIN)}
        X = np.stack(list(feats.values()), axis=1)
        (nd, tau, _, frac, _), _cal = route_nested(X, g, ndA, ndB, splits, grp, seed=a.seed)
        rows["all4_ridge"] = {"routed": nd, "tau": tau, "frac": frac,
                             "underlined": bool(nd > orig + MARGIN)}
        out["cells"][label] = {"original": float(orig), "rows": rows}
        print(f"{label}: orig={orig:.4f}  " + "  ".join(
            f"{n}={r['routed']:.4f}/τ{r['tau']:+.3f}" for n, r in rows.items()), flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
