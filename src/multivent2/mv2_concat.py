"""Extension vs decomposition on MultiVENT 2.0 (CPU, reuses shipped assets).

Review question (22 Jul): query EXTENSION (concatenate the expansions into one enriched query and
score once) vs query DECOMPOSITION (score each expansion separately, then fuse). The SCALE-project
finding is that concatenation removes the "some queries great, some terrible" variance that fusion
produces -- so it should shrink the per-query gain heterogeneity the router feeds on, and may raise
mean effectiveness too.

Same tier-B base as mv2_ab / mv2_full (CLIP visual + caption[query], RRF). Three ways to add the LLM
event descriptions (prequel/during/sequel from mv2_events.py):
  DECOMP : 3 events encoded SEPARATELY, max-pooled per candidate, added as a weighted RRF component
           (this is exactly mv2_full.py's Full tier -- the decomposition+fusion baseline).
  CONCAT : the 3 events joined into ONE string, encoded once, added as a weighted RRF component.
  EXTQ   : the QUERY itself extended -- query+prequel+during+sequel joined into one string, encoded
           once, used as the caption-side query (replacing plain-query caption rank), RRF'd with CLIP.

For each: nDCG@10 at the best nonzero event weight, and per-query gain vs tier B (mean/sd/help%/hurt%).
sd is the heterogeneity statistic; lower sd at equal-or-higher mean supports "concatenation wins".
Also prints a matched-weight sd table so the DECOMP vs CONCAT variance comparison is not confounded by
each picking a different weight.

  CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_concat.py
  CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_concat.py --events data/multivent2/events_qwen14b.jsonl
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
from mv2_full import rrf_ranks, load_events, EVENTS  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
WSWEEP = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def gain_stats(pq_base, pq_var, qids):
    common = [q for q in qids if q in pq_base and q in pq_var]
    g = np.array([pq_var[q] - pq_base[q] for q in common])
    return {"mean": 100 * g.mean(), "sd": 100 * g.std(),
            "help": 100 * (g > 1e-9).mean(), "hurt": 100 * (g < -1e-9).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=os.path.join(DATA, "events_qwen7b.jsonl"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ev_tag = os.path.basename(a.events).replace("events_", "").replace(".jsonl", "")
    out_path = a.out or os.path.join(_ROOT, "results", "ablations", f"mv2_concat_{ev_tag}.json")

    from sentence_transformers import SentenceTransformer
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    runA = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    caps = load_captions(os.path.join(DATA, "qwen_captions_test.jsonl"))
    events = load_events(a.events)
    print(f"queries {len(queries)} | tierA {len(runA)} | captions {len(caps)} | events {len(events)} | {ev_tag}")

    cap_ids = list(caps)
    model = SentenceTransformer(a.model, device="cpu")
    cache = os.path.join(DATA, f"capemb_{a.model.split('/')[-1]}.npz")
    z = np.load(cache, allow_pickle=True)
    cap_emb = z["emb"]; assert list(z["ids"]) == cap_ids, "caption-embedding cache is stale"
    row = {c: i for i, c in enumerate(cap_ids)}
    print(f"loaded cached caption embeddings {cap_emb.shape}")

    qids = [q for q in runA if q in events]

    def enc(texts):
        return model.encode(texts, batch_size=256, normalize_embeddings=True)

    def ev(q, e):
        return events[q].get(e) or ""

    qvec = enc([queries[q] for q in qids])
    evec = {e: enc([ev(q, e) for q in qids]) for e in EVENTS}
    concat_ev = enc([" ".join(ev(q, e) for e in EVENTS) for q in qids])
    concat_qev = enc([" ".join([queries[q]] + [ev(q, e) for e in EVENTS]) for q in qids])

    base_q, ev_rrf_decomp, ev_rrf_concat, extq_run, feats = {}, {}, {}, {}, {}
    for qi, qid in enumerate(qids):
        cands = list(runA[qid])
        capm = cap_emb[np.array([row[c] for c in cands])]
        clip_rrf = rrf_ranks(cands, runA[qid])
        cap_rrf = rrf_ranks(cands, dict(zip(cands, (capm @ qvec[qi]).tolist())))
        base_q[qid] = {v: clip_rrf[v] + cap_rrf[v] for v in cands}
        # tier-B confidence features -> legal at tier B, for the B->Full router (same as mv2_full.py)
        cf = conf_features(np.array([base_q[qid][v] for v in cands]))
        feats[qid] = [cf[k] for k in FEATURE_ORDER]
        # DECOMP: 3 events scored separately, max-pooled over the events (Q2E-style)
        ev_sim = np.max([capm @ evec[e][qi] for e in EVENTS], axis=0)
        ev_rrf_decomp[qid] = rrf_ranks(cands, dict(zip(cands, ev_sim.tolist())))
        # CONCAT: the 3 events joined into one string, encoded once
        ev_rrf_concat[qid] = rrf_ranks(cands, dict(zip(cands, (capm @ concat_ev[qi]).tolist())))
        # EXTQ: enriched query (query+events) replaces the plain caption query, RRF'd with CLIP
        qev_rrf = rrf_ranks(cands, dict(zip(cands, (capm @ concat_qev[qi]).tolist())))
        extq_run[qid] = {v: clip_rrf[v] + qev_rrf[v] for v in cands}

    runB = base_q
    ndB = ndcg10(qrels, runB)
    pqB = per_query_ndcg(qrels, runB)
    print(f"\nTier B (CLIP + caption[query]) nDCG@10 = {ndB:.5f}")

    def full_run(ev_rrf, w):
        return {qid: {v: base_q[qid][v] + w * ev_rrf[qid][v] for v in base_q[qid]} for qid in qids}

    results = {"ev_tag": ev_tag, "model": a.model, "ndcgB": ndB, "n_queries": len(qids), "variants": {}}

    abl = os.path.join(_ROOT, "results", "ablations")
    for name, ev_rrf in [("DECOMP", ev_rrf_decomp), ("CONCAT", ev_rrf_concat)]:
        sweep = {w: ndcg10(qrels, full_run(ev_rrf, w)) for w in WSWEEP}
        w_best = max(WSWEEP[1:], key=lambda w: sweep[w])
        pq_full = per_query_ndcg(qrels, full_run(ev_rrf, w_best))
        stats = gain_stats(pqB, pq_full, qids)
        results["variants"][name] = {"w_best": w_best, "ndcg": sweep[w_best],
                                     "delta_vs_B": 100 * (sweep[w_best] - ndB),
                                     "sweep": {str(w): sweep[w] for w in WSWEEP}, "gain": stats}
        print(f"\n{name}: best w={w_best}  nDCG@10 = {sweep[w_best]:.5f}  delta {100*(sweep[w_best]-ndB):+.2f}")
        print(f"   per-query gain vs B: mean {stats['mean']:+.2f}  sd {stats['sd']:.2f}  "
              f"help {stats['help']:.0f}%  hurt {stats['hurt']:.0f}%")
        # router-ready JSON (mv2_full.json schema) so mv2_router.py / mv2_frontier.py consume the B->Full step
        common = [q for q in qids if q in pqB and q in pq_full and q in feats]
        rj = {"scorer": f"{name.lower()}_full_{ev_tag}", "ndcgA": ndB, "ndcgB": sweep[w_best], "w_best": w_best,
              "per_query": {q: {"ndA": pqB[q], "ndB": pq_full[q]} for q in common},
              "features": {q: feats[q] for q in common}, "feature_order": FEATURE_ORDER}
        json.dump(rj, open(os.path.join(abl, f"mv2_full_{name.lower()}_{ev_tag}.json"), "w"))

    ndE = ndcg10(qrels, extq_run)
    statsE = gain_stats(pqB, per_query_ndcg(qrels, extq_run), qids)
    results["variants"]["EXTQ"] = {"ndcg": ndE, "delta_vs_B": 100 * (ndE - ndB), "gain": statsE}
    print(f"\nEXTQ (query extended w/ events, one caption rank): nDCG@10 = {ndE:.5f}  delta {100*(ndE-ndB):+.2f}")
    print(f"   per-query gain vs B: mean {statsE['mean']:+.2f}  sd {statsE['sd']:.2f}  "
          f"help {statsE['help']:.0f}%  hurt {statsE['hurt']:.0f}%")

    print("\nmatched-weight per-query gain sd (heterogeneity), DECOMP vs CONCAT:")
    matched = {}
    for w in WSWEEP[1:]:
        sd_d = gain_stats(pqB, per_query_ndcg(qrels, full_run(ev_rrf_decomp, w)), qids)["sd"]
        sd_c = gain_stats(pqB, per_query_ndcg(qrels, full_run(ev_rrf_concat, w)), qids)["sd"]
        matched[str(w)] = {"decomp_sd": sd_d, "concat_sd": sd_c}
        tag = "concat lower" if sd_c < sd_d else "decomp lower"
        print(f"  w={w:<4} decomp sd {sd_d:5.2f}   concat sd {sd_c:5.2f}   {tag}")
    results["matched_weight_sd"] = matched

    json.dump(results, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
