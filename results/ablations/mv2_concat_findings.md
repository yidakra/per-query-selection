# Extension beats decomposition: concatenating the LLM events wins on both effectiveness and variance

Supervision (22 Jul) raised a finding from the JHU SCALE project: **query extension** (concatenate the
expansions into one enriched query, score once) tends to beat **query decomposition** (score each
expansion separately, then fuse), because fusion produces a "some queries great, some terrible" spread
that concatenation flattens. We tested it directly on MultiVENT 2.0, CPU-only, reusing shipped assets
(`src/multivent2/mv2_concat.py`, no new generation).

Same tier-B base as `mv2_ab`/`mv2_full` (CLIP visual + caption[query], RRF). Three ways to add the LLM
event descriptions (prequel/during/sequel from `mv2_events.py`) on top of it:

- **DECOMP**: 3 events encoded separately, max-pooled per candidate, added as a weighted RRF component. This is exactly `mv2_full.py`'s Full tier (decomposition + fusion).
- **CONCAT**: the 3 events joined into one string, encoded once, added as a weighted RRF component.
- **EXTQ**: the query itself extended (query + prequel + during + sequel → one string), used as the caption-side query in place of the plain query, RRF'd with CLIP.

## Result: concatenation wins on effectiveness, at both decomposer sizes

| decomposer | variant | best w | nDCG@10 | Δ vs tier B | per-query gain sd | help / hurt |
|---|---|---|---|---|---|---|
| **qwen7B** | Tier B | -- | 0.35403 | -- | -- | -- |
| | DECOMP (current) | 0.25 | 0.35960 | +0.56 | 7.54 | 25% / 20% |
| | **CONCAT** | 0.5 | **0.36244** | **+0.84** | 10.45 | 31% / 24% |
| | EXTQ | -- | 0.35573 | +0.17 | 12.81 | 28% / 27% |
| **qwen14B** | Tier B | -- | 0.35435 | -- | -- | -- |
| | DECOMP (current) | 0.5 | 0.36244 | +0.81 | 11.00 | 29% / 25% |
| | **CONCAT** | 0.5 | **0.37001** | **+1.57** | 10.68 | 32% / 21% |
| | EXTQ | -- | 0.36342 | +0.91 | 12.52 | 30% / 25% |

- **CONCAT > DECOMP on effectiveness at both sizes**, and the gap *widens* with the decomposer: +0.84 vs +0.56 (7B), **+1.57 vs +0.81 (14B)**. On 14B both pick the same weight (w=0.5), so it is a like-for-like comparison: concatenation nearly doubles the gain at identical contribution strength.
- **EXTQ (replacing the query wholesale) is the wrong way to extend**: worst effectiveness and highest variance at both sizes. The win comes from concatenating the *events* as a weighted component, not dissolving the original query into them.

## The variance claim: concatenation is lower-variance at matched contribution

The headline `sd` column above is confounded: each variant picks its own best weight, and a bigger
weight mechanically inflates both mean and variance. Holding the event weight fixed removes that:

| event weight | 7B DECOMP sd | 7B CONCAT sd | 14B DECOMP sd | 14B CONCAT sd |
|---|---|---|---|---|
| 0.1 | 4.64 | **4.37** | 4.67 | **4.32** |
| 0.25 | 7.54 | **6.86** | 7.67 | **7.50** |
| 0.5 | 11.18 | **10.45** | 11.00 | **10.68** |
| 1.0 | 16.61 | **15.84** | 16.21 | **15.50** |
| 2.0 | 23.75 | **22.66** | 23.18 | **21.67** |

**At every weight and both sizes, concatenation has lower per-query gain variance than decomposition.**
This is the SCALE claim confirmed: scoring the events as one string spreads the benefit more evenly
across queries than scoring them separately and fusing. The effect is modest (~5–10% relative sd
reduction) but perfectly consistent.

## What it means for the routing thesis

Two things point in opposite directions and roughly cancel, leaving the project's conclusion intact:

1. **Lower variance is (slightly) less for the router to exploit.** The router's value tracks
   `sd(per-query gain)` (ρ=+0.943, `router_findings.md` §2). Concatenation shrinks that sd, so on a
   concatenated Full tier the router has marginally less headroom. The shrink is small.
2. **But the Full tier becomes less dominated.** Concatenation roughly halves the Full tier's
   energy-per-nDCG-point: at 14B, +1.57 nDCG for the same ~259.7 J/query is **≈165 J/point** vs
   DECOMP's ≈321 (`mv2_findings.md`); at 7B, ≈171 vs ≈257. The "don't buy Full" verdict softens
   from "dominated" toward "still expensive but closer to the frontier."

Net: the A→B routing story is unchanged (that gain lives in the caption tier, not the events). The
honest update is that **how you fold in the LLM expansion is a real design axis we had under-explored**:
concatenation is a free (same LLM cost) ~1.5–2× effectiveness improvement on the Full tier over the
decompose-and-fuse default, with a small variance reduction as a side effect.

## Follow-up: does concatenation change the Full tier's routability and energy frontier?

Ran both CPU checks on the concatenated 14B Full tier (`mv2_router.py`, `mv2_frontier.py`):

| B→Full step (14B) | Kendall τ (perm p) | nested gap | APGR | CPT₅₀ | energy / nDCG@10 pt | oracle headroom captured |
|---|---|---|---|---|---|---|
| DECOMP (decompose + fuse) | +0.035 (.0045) | +0.39 ± 0.07 | 0.082 | 0.18 | 321 J | 8% |
| **CONCAT** (extend + concatenate) | **+0.042 (.0015)** | **+0.41 ± 0.11** | 0.090 | 0.26 | **166 J** | **13%** |

- **Routability is marginally better, not worse.** The variance shrink did not cost predictability: CONCAT's B→Full gain is if anything slightly more orderable (τ +0.042 vs +0.035, though both are ~6× weaker than the A→B τ of +0.127). The worry that flattening heterogeneity would starve the router did not materialise at this scale.
- **Energy frontier: less dominated, still expensive.** Concatenation halves the Full tier's cost-per-point (166 vs 321 J/nDCG@10), and the routed curve now sits above the random-escalation chord at every f, capturing **13%** of a +2.99 oracle headroom, vs 8% for decompose-and-fuse. But even fully escalated the tier buys +1.57 nDCG for 259.7 J/query, while A→B buys +5.69 nearly free, so Full stays off the efficient frontier.

**Verdict update.** "Route A→B, don't buy Full" stands, but softens: concatenation is the right way to
build the Full tier *if you build it*: ~2× the effectiveness and ~2× cheaper per point than the
decompose-and-fuse default, with a small but real routable gain. The negative on Full is now "expensive
and weakly routable," not "dominated and unroutable."

## Reproduce follow-up
```bash
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_router.py --in results/ablations/mv2_full_concat_qwen14b.json
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_frontier.py --full results/ablations/mv2_full_concat_qwen14b.json \
    --energy results/ablations/mv2_energy_qwen14b.json --out results/ablations/mv2_frontier_concat_qwen14b.json
```

## Reproduce
```bash
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_concat.py                                   # qwen7b
CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_concat.py --events data/multivent2/events_qwen14b.jsonl
# -> results/ablations/mv2_concat_{qwen7b,qwen14b}.json
```
CPU-only; reuses `10pyscene_clip.json`, `qwen_captions_test.jsonl`, `capemb_all-MiniLM-L6-v2.npz`,
`events_qwen{7,14}b.jsonl`, and the test qrels.
