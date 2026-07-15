"""Full tier on MultiVENT 2.0: fuse the LLM event decomposition into tier B, then route B->Full.

Tier B = CLIP visual rank + caption rank (dense MiniLM over the Qwen captions), RRF-fused. The Full
tier adds three more rank lists -- prequel / during / sequel -- each the dense similarity of that event
description (from mv2_events.py) against the candidate captions. Full = RRF over all five rank lists,
the same 5-component C_full ladder the original Q2E used.

Then the B->Full routing question: can features available *at tier B* (before any LLM call) predict
which queries the event tier will help? We emit a router-ready JSON (gain = ndFull - ndB, features =
tier-B confidence) and run mv2_router.py on it. On the original small cells B->Full was unpredictable
ex ante; this tests that at scale. Also emits the LLM token cost for the energy frontier.

  CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_full.py
  CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_router.py --in results/ablations/mv2_full.json
"""
import os
import sys
import json
import argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run, load_queries  # noqa: E402
from mv2_eval import ndcg10  # noqa: E402
from mv2_ab import load_captions, per_query_ndcg  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
RRF_K = 60
EVENTS = ["prequel", "during", "sequel"]


def rrf_ranks(cands, scores, default=-1.0):
    """RRF contribution 1/(K+rank) for one score dict over cands (higher score = better rank)."""
    order = sorted(cands, key=lambda v: -scores.get(v, default))
    rank = {v: i for i, v in enumerate(order)}
    return {v: 1.0 / (RRF_K + rank[v]) for v in cands}


def load_events(path):
    ev = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ev[r["qid"]] = r
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=os.path.join(DATA, "events_qwen7b.jsonl"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "ablations", "mv2_full.json"))
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    runA = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    caps = load_captions(os.path.join(DATA, "qwen_captions_test.jsonl"))
    events = load_events(a.events)
    print(f"queries {len(queries)} | tierA {len(runA)} | captions {len(caps)} | events {len(events)}")

    # dense caption embeddings (reuse the cache mv2_ab.py --scorer dense wrote)
    cap_ids = list(caps)
    cache = os.path.join(DATA, f"capemb_{a.model.split('/')[-1]}.npz")
    model = SentenceTransformer(a.model, device="cpu")
    z = np.load(cache, allow_pickle=True)
    cap_emb = z["emb"]; assert list(z["ids"]) == cap_ids, "caption-embedding cache is stale"
    row = {c: i for i, c in enumerate(cap_ids)}
    print(f"loaded cached caption embeddings {cap_emb.shape}")

    # encode queries + the 3 event descriptions per query in the same MiniLM space
    qids = [q for q in runA if q in events]
    qvec = model.encode([queries[q] for q in qids], batch_size=256, normalize_embeddings=True)
    evec = {e: model.encode([events[q][e] for q in qids], batch_size=256, normalize_embeddings=True)
            for e in EVENTS}

    # Event contribution as ONE rank list per query: max-pool the 3 event similarities per candidate
    # (Q2E scores an event component by max-pooling over the LLM's descriptions), then fuse with a
    # tunable RRF weight w so the speculative events don't automatically get 3/5 of the vote.
    WSWEEP = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    runB = {}
    ev_rrf_q, base_q, feats = {}, {}, {}
    for qi, qid in enumerate(qids):
        cand_scores = runA[qid]
        cands = list(cand_scores)
        capm = cap_emb[np.array([row[c] for c in cands])]           # every cand has a caption here
        clip_rrf = rrf_ranks(cands, cand_scores)
        cap_rrf = rrf_ranks(cands, dict(zip(cands, (capm @ qvec[qi]).tolist())))
        ev_sim = np.max([capm @ evec[e][qi] for e in EVENTS], axis=0)   # max-pool over the 3 events
        ev_rrf = rrf_ranks(cands, dict(zip(cands, ev_sim.tolist())))
        base_q[qid] = {v: clip_rrf[v] + cap_rrf[v] for v in cands}
        ev_rrf_q[qid] = ev_rrf
        runB[qid] = base_q[qid]
        f = conf_features(np.array([base_q[qid][v] for v in cands]))    # tier-B confidence = legal pre-Full
        feats[qid] = [f[k] for k in FEATURE_ORDER]

    def full_run(w):
        return {qid: {v: base_q[qid][v] + w * ev_rrf_q[qid][v] for v in base_q[qid]} for qid in qids}

    ndB = ndcg10(qrels, runB)
    pqB = per_query_ndcg(qrels, runB)
    print(f"\nTier B (CLIP+caps) nDCG@10 = {ndB:.5f}")
    print("event-weight sweep (Full = B + w * maxpool-event RRF):")
    sweep = {}
    for w in WSWEEP:
        nd = ndcg10(qrels, full_run(w))
        sweep[w] = nd
        print(f"  w={w:<4} Full nDCG@10 = {nd:.5f}   delta vs B {100*(nd-ndB):+.2f}")
    w_best = max(WSWEEP[1:], key=lambda w: sweep[w])                 # best nonzero weight

    runF = full_run(w_best)
    ndF = sweep[w_best]
    pqF = per_query_ndcg(qrels, runF)
    common = [q for q in qids if q in pqB and q in pqF]
    g = np.array([pqF[q] - pqB[q] for q in common])
    print(f"\nrouter uses best nonzero weight w={w_best}: Full nDCG@10 = {ndF:.5f}  delta {100*(ndF-ndB):+.2f}")
    print(f"per-query B->Full gain: mean {100*g.mean():+.2f}  sd {100*g.std():.2f}  "
          f"help {100*(g>1e-9).mean():.0f}%  hurt {100*(g<-1e-9).mean():.0f}%")

    ptok = sum(events[q].get("prompt_tok", 0) for q in qids)
    gtok = sum(events[q].get("gen_tok", 0) for q in qids)
    print(f"LLM cost: {ptok} prompt + {gtok} gen tokens over {len(qids)} queries "
          f"({(ptok+gtok)/max(1,len(qids)):.0f} tok/query)")

    # router-ready: gain = ndFull - ndB, "ndA"->tier B base, "ndB"->Full, features legal at tier B
    json.dump({"scorer": "full_event_qwen7b", "ndcgA": ndB, "ndcgB": ndF, "w_best": w_best,
               "weight_sweep": {str(w): sweep[w] for w in WSWEEP},
               "llm_prompt_tok": ptok, "llm_gen_tok": gtok, "n_queries": len(qids),
               "per_query": {q: {"ndA": pqB[q], "ndB": pqF[q]} for q in common},
               "features": {q: feats[q] for q in common}, "feature_order": FEATURE_ORDER},
              open(a.out, "w"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
