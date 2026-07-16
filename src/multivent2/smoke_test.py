"""Synthetic end-to-end smoke test for the two-stage cascade.

Builds tiny fake data with a planted signal (each query's gold video sits near it in embedding space),
runs the whole stage1 -> router -> stage2 -> eval flow, and asserts the wiring works: every query gets
a ranking, the right fraction escalates, and nDCG@10 computes and reflects the planted signal. No real
data, models, or GPU. Run:
  CUDA_VISIBLE_DEVICES="" python src/multivent2/smoke_test.py
"""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from retrieve import TwoStageRetriever, FEATURE_ORDER  # noqa: E402
from mv2_eval import evaluate  # noqa: E402


def build(rng, n_videos=60, n_queries=8, d=32):
    vids = [f"v{i}" for i in range(n_videos)]
    emb = rng.standard_normal((n_videos, d))
    queries, qvecs, qrels, gold = {}, {}, {}, {}
    for q in range(n_queries):
        qid = f"q{q}"
        qv = rng.standard_normal(d)
        g = q % n_videos
        emb[g] = qv + 0.15 * rng.standard_normal(d)        # plant: gold video near the query
        queries[qid] = f"query {q}"
        qvecs[qid] = qv / (np.linalg.norm(qv) + 1e-9)
        qrels[qid] = {vids[g]: 3}
        gold[qid] = vids[g]
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return vids, emb, queries, qvecs, qrels, gold


def main():
    rng = np.random.default_rng(0)
    vids, emb, queries, qvecs, qrels, gold = build(rng)

    # tier B and Full: small positive boosts to the true gold (captions/events help some queries)
    def caption_scorer(text, cand):
        q = text.split()[-1]; g = gold[f"q{q}"]
        return {g: 0.05} if g in cand else {}

    def event_scorer(text, cand):
        q = text.split()[-1]; g = gold[f"q{q}"]
        return {g: 0.03} if g in cand else {}

    # router stub: predict gain from the z1 confidence feature (deterministic)
    zi = FEATURE_ORDER.index("z1")
    def gain_predictor(F):
        return F[:, zi]

    r = TwoStageRetriever(vids, emb, caption_scorer, event_scorer, gain_predictor, k=1000)
    run, info = r.run_all(queries, qvecs, f_escalate=0.5)

    assert len(run) == len(queries), "every query must get a ranking"
    assert all(len(run[q]) == len(vids) for q in run), "candidates == all videos when k>=N"
    n_esc = sum(v["escalated"] for v in info.values())
    assert n_esc == round(0.5 * len(queries)), f"expected 4 escalated, got {n_esc}"

    m = evaluate(qrels, run)
    print("smoke nDCG@10", round(m["nDCG@10"], 4), "| escalated", n_esc, "/", len(queries))
    assert 0.0 <= m["nDCG@10"] <= 1.0
    assert m["nDCG@10"] > 0.5, "planted signal should give high nDCG"
    print("OK: two-stage control flow works end-to-end.")


if __name__ == "__main__":
    main()
