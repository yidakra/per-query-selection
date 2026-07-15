# First real result on MultiVENT 2.0: the A→B router at scale

The router works on the community benchmark our collaborators built, at ~10× the query count of the
original cells, on graded multi-gold judgments. CPU-only, reuses the shipped features (no raw video,
no re-encoding). `src/multivent2/mv2_ab.py` (tiers) + `mv2_router.py` (router).

## Setup
- Tier A = the provided CLIP visual run. Our harness scores it at **nDCG@10 = 0.30364**, matching the
  official evaluator to the last digit — tier A is validated, not reimplemented.
- Tier B = rerank tier-A's top-K by fusing the visual rank with a caption rank (TF-IDF cosine of the
  query against the shipped Qwen3-Omni captions, reciprocal-rank fusion). This is the cheap, model-free
  version of the caption tier; a dense/ColBERT scorer is the obvious upgrade and drops into the same
  pipeline.
- 2,546 test queries; eval = official nDCG@10.

## Captions help, and the gain is very heterogeneous
| | nDCG@10 |
|---|---|
| Tier A (CLIP visual) | 0.30364 |
| Tier B (+ captions) | **0.36052** (+5.69) |

Per-query A→B gain: mean **+5.69**, sd **24.58**, **43% of queries helped, 23% hurt**. That sd is far
above any original cell (which ran 7.95–17.02), and 23% actively hurt by captions is the strongest
"one-size fusion is wrong" signal we have seen. This is the regime where routing has the most to do.

## The router captures it
Predict the gain from tier-A confidence features (out-of-fold ridge), escalate the top-f:

| metric | value |
|---|---|
| Kendall τ (pred, true gain) | **+0.127**  (perm p = .0005, bootstrap CI [+0.101, +0.151]) |
| nested frontier gap vs the cost-matched chord | **+2.23 ± 0.18 nDCG** |
| APGR | 0.217 |
| CPT₅₀ / CPT₈₀ | 0.24 / 0.47 |

The nested gap **+2.23** is larger than every original cell (which ran +0.73 to +1.68), and it lands
where the heterogeneity thesis predicts: bigger sd(gain) → bigger routing value. CPT₅₀ = 0.24 means the
router captures half the caption-tier improvement by escalating only 24% of queries. The result is
significant under permutation and its bootstrap CI excludes zero, on out-of-fold predictions.

## Dense captions don't beat lexical here
Swapping the TF-IDF caption score for a dense sentence embedder (all-MiniLM-L6-v2, cosine over the
shipped Qwen captions, same RRF fusion) does *not* help:

| Tier B scorer | nDCG@10 | Δ vs A | sd(gain) | help / hurt | τ | nested gap | CPT₅₀ |
|---|---|---|---|---|---|---|---|
| TF-IDF (lexical) | **0.36052** | +5.69 | 24.58 | 43% / 23% | +0.127 | +2.23 | 0.24 |
| Dense (MiniLM) | 0.35442 | +5.08 | 22.89 | 43% / 21% | +0.117 | +1.96 | 0.25 |

A small off-the-shelf embedder loses to TF-IDF on these captions — the captions are keyword-dense and
lexical overlap with the query is a strong signal. The routing result is the same either way (τ ≈ 0.12,
nested gap ≈ +2), which is the point: the router's job is ordering queries by A→B gain, and it does that
regardless of which caption scorer produces the gain. A stronger dense tier (a retrieval-tuned or
ColBERT-style scorer) is still the obvious upgrade, but "any dense model beats lexical" is false here.

## The Full event tier: barely helps, and can't be routed
We built the Full tier on 2.0 — the LLM step of the cascade. For each query a local instruction model
(qwen2.5:7b-instruct on GPU1; a 70B like the original Q2E used does not fit on a 15 GB A2, so this is a
feasible stand-in and a lower bound) writes prequel / during / sequel visual descriptions of the event.
Each is scored against the candidate captions and fused into tier B. `mv2_events.py` (generation, 213
tokens/query over 2,544 queries) + `mv2_full.py` (fusion + sweep).

**It barely helps, and only at a carefully tuned weight.** Fusing the events at full weight *hurts* (the
speculative descriptions bury the CLIP+caption signal). Giving them a fair, down-weighted vote and
sweeping the weight on the test set:

| event weight w | Full nDCG@10 | Δ vs tier B |
|---|---|---|
| 0 (tier B) | 0.35403 | — |
| 0.1 | 0.35587 | +0.18 |
| **0.25 (best)** | **0.35960** | **+0.56** |
| 0.5 | 0.35794 | +0.39 |
| 1.0 | 0.34047 | −1.36 |
| 2.0 | 0.28995 | −6.41 |

Even at the weight picked *on the test set* (an optimistic ceiling), the event tier adds **+0.56 nDCG**,
with low per-query heterogeneity (sd 7.54, 25% help / 20% hurt) — a quarter of the A→B tier's sd. This
replicates the project's earlier "Tier C is dead" conclusion, now on the community benchmark our
collaborators built.

**And the tiny gain that exists cannot be routed.** Predicting per-query B→Full gain from tier-B
features (out-of-fold ridge, same protocol):

| metric | B→Full (Full tier) | A→B (caption tier), for contrast |
|---|---|---|
| Kendall τ (pred, true gain) | **+0.002** (perm p = .44, CI [−0.024, +0.031]) | +0.127 (p = .0005) |
| nested frontier gap | **+0.15 ± 0.04** (ns) | +2.23 ± 0.18 |
| APGR | 0.065 | 0.217 |

τ is statistically zero and the nested gap is negligible: from tier-B confidence there is no signal for
which queries the event tier will help — the same negative we found on the small cells, now at 10× scale.

**This is the paper's point, shown in both directions on real data.** The A→B step gives a large,
heterogeneous, *predictable* gain — route it, and the router captures +2.23 nDCG. The B→Full step gives
a tiny, fragile, *unpredictable* gain for ~213 LLM tokens/query plus the event-scoring cost — so a good
router declines it, and you should not buy the Full tier at all. The value of routing is not uniform
across a cascade; it concentrates where the per-query gain is both large and predictable.

## Scope and next step
- Both cascade steps are now measured on real MultiVENT 2.0: A→B (route it) and B→Full (don't). This is
  the routing-**quality** result — whether the router orders queries by true gain — and it lands the way
  the heterogeneity thesis predicts in both directions.
- The **cost** axis is in tokens, not yet joules. The A→B tiers are both cheap (CLIP + caption cosine),
  so that step's energy gap is small; the interesting cost is the Full tier's 213 LLM tokens/query, which
  buys +0.56 unroutable nDCG. A measured-joules frontier (wrap the event generation + scoring in the
  CodeCarbon harness, explicit gpu_ids) would put the "never buy Full" call in energy terms — the
  remaining build.
- Caveat carried forward: qwen2.5:7b is a lower bound on the Full tier vs the original 70B. A stronger
  decomposer could lift the +0.56, but the routing null (τ≈0) is about *predictability from tier-B
  features*, which a better generator does not obviously fix.

## Reproduce
```bash
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_ab.py                     # tiers A, B      -> mv2_ab.json
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_router.py                 # A->B router     -> mv2_router.json
# Full tier (needs Ollama on GPU1 w/ qwen2.5:7b-instruct):
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_events.py                 # event decomp    -> events_qwen7b.jsonl
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_full.py                   # Full tier+sweep -> mv2_full.json
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_router.py --in results/ablations/mv2_full.json  # B->Full router
```
Needs the gitignored `data/multivent2/` files (provided CLIP run, qrels, queries, Qwen captions).
