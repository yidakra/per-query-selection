# Two-stage cost cascade on MultiVENT 2.0: design

Our original pipeline scores every query against every video and caches the full score tensor. That
does not scale here: MultiVENT 2.0 has ~110K test videos, so the query×video product is ~280M pairs
and the per-(query,video,event) tensors would be multiple TB. The fix is a two-stage retrieve-then-
rerank, with the router deciding per query whether to pay for the expensive second stage.

## Pipeline

```
query
  |
  v
[stage 1: cheap, all 110K videos]
  tier A  = query vs video     (dense cosine over CLIP/SigLIP video embeddings)
  tier B  = A + query vs captions (ColBERT MaxSim over shipped VLM captions)
  -> ranking over all videos, and the top-K candidate set (K ~ 100-1000)
  -> cheap-tier confidence features (margin, entropy, top-5 mass, score sd) over the ranking
  |
  v
[router]  predict per-query gain of escalating; escalate the top-f fraction
  |                                         |
  no escalate                              escalate
  |                                         v
return tier-B ranking          [stage 2: expensive, top-K only]
                                 LLM decomposes query -> prequel/during/sequel events
                                 score ~30 event components vs the top-K captions
                                 rerank the K candidates -> final ranking
```

Everything the router needs (stage-1 scores, confidence features) is available before paying for the
generation, so the escalation decision is cascade-legal: same discipline as the original work.

## What we reuse vs build

Reuse (shipped in the dataset `features/`, ~16 GB, no raw video needed):
- CLIP (10 keyframes) or SigLIP (16 frames) video embeddings -> tier A.
- Qwen3-Omni / Florence captions -> tier B (ColBERT).
- Whisper ASR + OCR -> optional extra tier-B signal (this is where MMMORRF gets most of its lift).
- qrels (graded 0-3, multi-gold) + the official nDCG@10 metric -> our validated harness
  (`validate_harness.py` reproduces the CLIP baseline 0.30364 to the last digit).

Build:
- **The two-stage top-K reranker** (`retrieve.py`), the one substantive new component.
- Query-side encoding for tier A: the shipped embeddings are video-side only, so we need the matching
  text tower (SigLIP text encoder) to embed the ~2,545 test queries. Small (short texts), one-off.
- Caption -> ColBERT tokenization for tier B.
- Wire the existing expected-gain router (the ridge from `router_hetero.py`) onto stage-1 features.

## Cost accounting

The tier structure and the routing argument are unchanged; only absolute joules move (bigger gallery,
different encoders). Stage 1 is paid on every query; stage 2 (the ~30 LLaMA-70B generations, the
dominant cost) only on escalated queries. The top-K rerank means stage 2 scores K captions, not 110K,
so the per-escalation cost is bounded by K, not the corpus. Re-measure the per-tier joules on this
setup with the same NVML method (`tracking.py`); do not reuse the original-MultiVENT absolute numbers.

## Two decisions to make (good to co-scope with the JHU collaborators, whose benchmark this is)

1. **Reuse the shipped features vs regenerate with our own encoder/captioner.** Reuse is fast, avoids
   200-4,400 GPU-hrs, and decouples us from video link-rot. The cost: our numbers use CLIP/SigLIP +
   Qwen captions, not our own models, so they are not apples-to-apples with the original-MultiVENT
   runs. Recommendation: **reuse**, and frame MultiVENT 2.0 as a new, larger cell in the grid: the
   router is encoder-agnostic (transfer result: it moves across encoders), so the routing claim holds
   regardless of which encoder fills each tier. Regenerate only if a reviewer demands own-encoder parity.
2. **K (rerank depth).** Needs recall@K high enough that stage 2 can find the golds. R@100 of the CLIP
   baseline is 0.60, R@1000 is 0.84, so K=1000 keeps ~84% of golds reachable, K=100 keeps ~60%.
   Start K=1000 for headroom, then sweep down and report the recall/cost trade (K is itself a second
   cost knob the router could exploit).

## Baseline to beat

MMMORRF (SIGIR'25, Yates/Yang et al.): 0.586 nDCG@10 on the same test set, using the same shipped
features + ColBERT-X, fused by modality-aware RRF. Our target is not to beat it on raw nDCG but to
sit on a better efficiency-effectiveness frontier: match MMMORRF-class accuracy while escalating the
expensive tier on only a fraction of queries.

## Status

- Eval harness: done, matches the official evaluator exactly (`validate_harness.py`).
- Two-stage skeleton + synthetic smoke test: `retrieve.py`, `smoke_test.py` (control flow validated on
  fake data; real scorers plug into the documented interfaces).
- Next (needs the 4 GB SigLIP features download + the text tower + GPU): implement the real tier-A/B/Full
  scorers and run on the 1,504 human-judged test queries.
