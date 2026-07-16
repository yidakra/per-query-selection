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

## Scope and next step
- This is the routing-**quality** result (does the router order queries by true A→B gain) on real
  MultiVENT 2.0. It is the strongest such result we have.
- The **cost** axis here is not yet in joules: with lexical captions both tiers are cheap, so the A→B
  cost gap is small. The measured-energy frontier and the "never buy Full" story need the Full tier
  (LLaMA event decomposition), which is the next build (needs a GPU + the 4 GB features).
- Upgrade path, in order: dense/ColBERT caption tier (stronger B), then the Full event tier, then the
  measured-joules frontier on 2.0. The pipeline and harness are in place for all three.

## Reproduce
```bash
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_ab.py       # tiers A, B -> mv2_ab.json
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_router.py   # router      -> mv2_router.json
```
Needs the gitignored `data/multivent2/` files (provided CLIP run, qrels, queries, Qwen captions).
