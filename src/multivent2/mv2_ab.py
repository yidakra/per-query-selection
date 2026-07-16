"""Tier A and tier B on the real MultiVENT 2.0 test set (CPU, reuses shipped assets).

Tier A = the provided CLIP visual run (validated: nDCG@10 0.30364). Tier B = rerank tier-A's top-K
candidates by fusing the CLIP visual rank with a caption-relevance rank over the shipped Qwen3-Omni
captions, via reciprocal-rank fusion. Two caption scorers:
  --scorer tfidf   TF-IDF cosine (model-free, fast)
  --scorer dense   sentence-embedding cosine (all-MiniLM-L6-v2 by default; caption embeddings cached)

Emits per-query nDCG@10 for A and B plus tier-A confidence features -> the router's input.
CPU-only. Run e.g.:
  CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_ab.py --scorer dense
"""
import os
import sys
import json
import argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run, load_queries  # noqa: E402
from mv2_eval import ndcg10  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402
import jsonlines  # noqa: E402
import ir_measures  # noqa: E402
from ir_measures import nDCG  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
RRF_K = 60


def load_captions(path):
    caps = {}
    with jsonlines.open(path) as r:
        for line in r:
            caps[str(line["doc_id"])] = line["text"]
    return caps


def per_query_ndcg(qrels, run):
    return {m.query_id: m.value for m in ir_measures.iter_calc([nDCG @ 10], qrels, run)}


def tfidf_caption_scores(queries, runA, caps):
    from sklearn.feature_extraction.text import TfidfVectorizer
    cap_ids = list(caps)
    vec = TfidfVectorizer(min_df=2, max_features=200_000, sublinear_tf=True)
    cap_mat = vec.fit_transform(caps[c] for c in cap_ids)
    row = {c: i for i, c in enumerate(cap_ids)}
    print(f"tfidf: {cap_mat.shape[0]} caps x {cap_mat.shape[1]} terms")
    out = {}
    for qid, cand_scores in runA.items():
        cands = list(cand_scores)
        qv = vec.transform([queries[qid]])
        rows = [row[c] for c in cands if c in row]
        present = [c for c in cands if c in row]
        sim = (cap_mat[rows] @ qv.T).toarray().ravel() if rows else np.zeros(0)
        out[qid] = dict(zip(present, sim.tolist()))
    return out


def dense_caption_scores(queries, runA, caps, model_name):
    from sentence_transformers import SentenceTransformer
    cap_ids = list(caps)
    cache = os.path.join(DATA, f"capemb_{model_name.split('/')[-1]}.npz")
    model = SentenceTransformer(model_name, device="cpu")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        cap_emb = z["emb"]; cached_ids = list(z["ids"])
        assert cached_ids == cap_ids, "cached caption ids differ; delete the cache"
        print(f"dense: loaded cached caption embeddings {cap_emb.shape}")
    else:
        print(f"dense: encoding {len(cap_ids)} captions with {model_name} (cached after) ...", flush=True)
        cap_emb = model.encode([caps[c] for c in cap_ids], batch_size=256,
                               normalize_embeddings=True, show_progress_bar=True)
        np.savez(cache, emb=cap_emb.astype(np.float32), ids=np.array(cap_ids, dtype=object))
        print(f"dense: encoded + cached {cap_emb.shape}")
    row = {c: i for i, c in enumerate(cap_ids)}
    qvecs = model.encode([queries[q] for q in runA], batch_size=256, normalize_embeddings=True)
    out = {}
    for qi, (qid, cand_scores) in enumerate(runA.items()):
        cands = list(cand_scores)
        rows = [row[c] for c in cands if c in row]
        present = [c for c in cands if c in row]
        sim = (cap_emb[rows] @ qvecs[qi]) if rows else np.zeros(0)
        out[qid] = dict(zip(present, sim.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["tfidf", "dense"], default="tfidf")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(_ROOT, "results", "ablations",
                                     f"mv2_ab_{a.scorer}.json" if a.scorer == "dense" else "mv2_ab.json")

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    runA = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    caps = load_captions(os.path.join(DATA, "qwen_captions_test.jsonl"))
    print(f"queries {len(queries)} | tierA queries {len(runA)} | captions {len(caps)} | scorer {a.scorer}")

    cap_scores = (tfidf_caption_scores if a.scorer == "tfidf" else
                  (lambda q, r, c: dense_caption_scores(q, r, c, a.model)))(queries, runA, caps)

    runB, feats = {}, {}
    for qid, cand_scores in runA.items():
        cands = list(cand_scores)
        capq = cap_scores[qid]
        clip_order = sorted(cands, key=lambda v: -cand_scores[v])
        clip_rank = {v: i for i, v in enumerate(clip_order)}
        cap_order = sorted(cands, key=lambda v: -capq.get(v, -1.0))
        cap_rank = {v: i for i, v in enumerate(cap_order)}
        runB[qid] = {v: 1.0 / (RRF_K + clip_rank[v]) + 1.0 / (RRF_K + cap_rank[v]) for v in cands}
        f = conf_features(np.array([cand_scores[v] for v in clip_order]))
        feats[qid] = [f[k] for k in FEATURE_ORDER]

    ndA, ndB = ndcg10(qrels, runA), ndcg10(qrels, runB)
    pqA, pqB = per_query_ndcg(qrels, runA), per_query_ndcg(qrels, runB)
    g = np.array([pqB.get(q, pqA[q]) - pqA[q] for q in pqA])
    print(f"\nTier A (CLIP)  nDCG@10 = {ndA:.5f}   (provided baseline 0.30364)")
    print(f"Tier B (+caps) nDCG@10 = {ndB:.5f}   delta {100*(ndB-ndA):+.2f}   [{a.scorer}]")
    print(f"per-query A->B gain: mean {100*g.mean():+.2f}  sd {100*g.std():.2f}  "
          f"help {100*(g>1e-9).mean():.0f}%  hurt {100*(g<-1e-9).mean():.0f}%")

    json.dump({"scorer": a.scorer, "ndcgA": ndA, "ndcgB": ndB,
               "per_query": {q: {"ndA": pqA[q], "ndB": pqB.get(q, pqA[q])} for q in pqA},
               "features": feats, "feature_order": FEATURE_ORDER}, open(out_path, "w"))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
