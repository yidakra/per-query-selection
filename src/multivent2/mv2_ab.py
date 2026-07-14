"""Tier A and tier B on the real MultiVENT 2.0 test set (CPU, reuses shipped assets).

Tier A = the provided CLIP visual run (validated: nDCG@10 0.30364). Tier B = rerank tier-A's top-K
candidates by fusing the CLIP visual rank with a caption-relevance rank (TF-IDF cosine of the query
against the shipped Qwen3-Omni captions), via reciprocal-rank fusion. Emits per-query nDCG@10 for A
and B (the A->B gain the router will predict) plus tier-A confidence features.

This is the cheap, model-free version of tier B (lexical captions). A dense/ColBERT caption scorer is
the obvious upgrade; the pipeline is identical.
"""
import os
import sys
import json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run, load_queries  # noqa: E402
from mv2_eval import ndcg10  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402
import jsonlines  # noqa: E402
import ir_measures  # noqa: E402
from ir_measures import nDCG  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
OUT = os.path.join(_ROOT, "results", "ablations", "mv2_ab.json")
RRF_K = 60


def load_captions(path):
    caps = {}
    with jsonlines.open(path) as r:
        for line in r:
            caps[str(line["doc_id"])] = line["text"]
    return caps


def per_query_ndcg(qrels, run):
    out = {}
    for m in ir_measures.iter_calc([nDCG @ 10], qrels, run):
        out[m.query_id] = m.value
    return out


def rrf(rank_a, rank_b):
    return 1.0 / (RRF_K + rank_a) + 1.0 / (RRF_K + rank_b)


def main():
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    runA = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    caps = load_captions(os.path.join(DATA, "qwen_captions_test.jsonl"))
    print(f"queries {len(queries)} | tierA queries {len(runA)} | captions {len(caps)}")

    # TF-IDF over all captions once; query vectors on the fly
    cap_ids = list(caps)
    vec = TfidfVectorizer(min_df=2, max_features=200_000, sublinear_tf=True)
    cap_mat = vec.fit_transform(caps[c] for c in cap_ids)
    cap_row = {c: i for i, c in enumerate(cap_ids)}
    print(f"tfidf: {cap_mat.shape[0]} caps x {cap_mat.shape[1]} terms")

    runB = {}
    feats, gains_rows = [], []
    for qid, cand_scores in runA.items():
        cands = list(cand_scores)                                   # tier-A top-K candidate ids
        qv = vec.transform([queries[qid]])
        rows = [cap_row[c] for c in cands if c in cap_row]
        present = [c for c in cands if c in cap_row]
        cap_sim = (cap_mat[rows] @ qv.T).toarray().ravel() if rows else np.zeros(0)
        cap_of = dict(zip(present, cap_sim))

        # ranks within the candidate set (0 = best)
        clip_order = sorted(cands, key=lambda v: -cand_scores[v])
        clip_rank = {v: i for i, v in enumerate(clip_order)}
        cap_order = sorted(cands, key=lambda v: -cap_of.get(v, -1.0))
        cap_rank = {v: i for i, v in enumerate(cap_order)}
        runB[qid] = {v: rrf(clip_rank[v], cap_rank[v]) for v in cands}

        f = conf_features(np.array([cand_scores[v] for v in clip_order]))
        feats.append((qid, [f[k] for k in FEATURE_ORDER]))

    ndA_all = ndcg10(qrels, runA)
    ndB_all = ndcg10(qrels, runB)
    pqA = per_query_ndcg(qrels, runA)
    pqB = per_query_ndcg(qrels, runB)
    common = [q for q in pqA if q in pqB]
    g = np.array([pqB[q] - pqA[q] for q in common])

    print(f"\nTier A (CLIP)  nDCG@10 = {ndA_all:.5f}   (provided baseline 0.30364)")
    print(f"Tier B (+caps) nDCG@10 = {ndB_all:.5f}   delta {100*(ndB_all-ndA_all):+.2f}")
    print(f"per-query A->B gain: mean {100*g.mean():+.2f}  sd {100*g.std():.2f}  "
          f"help {100*(g>1e-9).mean():.0f}%  hurt {100*(g<-1e-9).mean():.0f}%")

    json.dump({"ndcgA": ndA_all, "ndcgB": ndB_all,
               "per_query": {q: {"ndA": pqA[q], "ndB": pqB.get(q, pqA[q])} for q in pqA},
               "features": {qid: fv for qid, fv in feats}, "feature_order": FEATURE_ORDER},
              open(OUT, "w"))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
