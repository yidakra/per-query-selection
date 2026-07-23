# Research question and motivation

All numbers are measured on MultiVENT 2.0 (2,546 test queries, graded judgments, nDCG@10).

## The question

**When a retrieval system can draw on several sources of evidence that differ wildly in cost and
reliability, can it decide _per query_ which ones to pay for — using only what the cheapest source has
already produced — and does deciding beat both using the cheap source alone and using everything?**

## Why it is worth asking

Multimodal video retrieval systems score a query against several channels: video frames, speech
transcripts, on-screen text, generated captions, LLM query expansions. Nearly all of them fuse the
channels the same way for every query, at a fixed cost per query. That design assumes every channel is
worth having for every query.

It is not. Whether the speech channel helps depends on whether *that* video has speech and whether
*that* query is about something said. This is a property of the query-corpus interaction, and it varies
enormously query to query.

We can now show that concretely, and the result is stronger than an efficiency argument:

| policy on MultiVENT 2.0 | nDCG@10 |
|---|---|
| visual channel alone | 0.3036 |
| visual + ASR channel, fused for every query (best fixed weight) | **0.2796** |
| visual + ASR, fused only for the queries a router selects | **0.3221** |
| oracle (fuse only where it truly helps) | 0.3653 |

Fusing the speech channel into every query **loses 2.4 nDCG points** — it helps 27% of queries and hurts
35%, with a per-query spread (sd 24.1) roughly ten times its own mean effect. Deciding per query turns
the same channel from a liability into a **+1.84** gain over visual-only, and **+4.25** over fusing it
everywhere. Against cost-matched random escalation the nested-CV gap is **+3.07 ± 0.38**, Kendall
τ = +0.217 (permutation p = .0005).

So the motivation is not "routing saves compute." It is that **for heterogeneous evidence sources,
deciding what to spend on is inseparable from getting a good answer**. Uniform fusion is simultaneously
the expensive option and the worse one.

## What we have established so far

1. **The signal is not in the query text.** A TF-IDF + logistic-regression router over the query string
   performs at or below the majority-class prior. Adaptive-RAG's premise — that you can read difficulty
   off the question — does not transfer to this setting. The signal is in the cheap channel's own
   retrieval confidence (score margins, entropy, top-k mass), which costs nothing extra because the
   cheap channel already ran.
2. **Per-query gain is predictable from those features.** Out-of-fold ridge regression orders queries by
   true gain: τ = +0.217 (channel routing), +0.127 (caption tier). Beats classical QPP predictors
   (Clarity, WIG, NQC) used as routing baselines.
3. **How much routing is worth is itself predictable.** Across six dataset × encoder × ASR cells, the
   spread of per-query gain predicts the achieved routing gap at Spearman ρ = +0.943. That tells you
   whether routing will pay *before* building it.
4. **Some expensive stages should simply be declined.** LLM query expansion costs a measured 143–260 J
   per query on our hardware and is close to unroutable (τ = +0.002 at 7B, +0.035 at 14B); scaling the
   expander 3B → 7B → 14B raises cost faster than benefit. Knowing when *not* to buy a stage is part of
   the same question.

## A scope decision I want your opinion on

The work so far routes between **LLM query-expansion tiers**. The result above suggests routing between
**modality channels** instead, for three reasons:

- The effect is much larger (+3.07 nested gap vs +2.23 for the caption tier and ~0 for the expansion tier).
- The channels ship with the benchmark as ranked lists, so it reproduces on CPU with no extra retrieval cost.
- It is what strong systems on this benchmark actually do — MMMORRF (0.586) and CLaMR (0.585) get their
  effectiveness from weighted fusion over speech and on-screen-text channels. Our contribution then sits
  on the path those systems are already on: *how* to weight the channels, per query, instead of globally.

Honest caveat: our absolute numbers are low (0.30–0.37) because we use the benchmark's cheap provided
channels. The routing claim is about the *policy*, not about beating the leaderboard. The benchmark also
ships the raw ASR and OCR text, so building a stronger channel and re-testing routing on top of it is a
concrete next step rather than a hope.

## What the paper would claim

1. Channel usefulness in multimodal retrieval is strongly query-dependent, to the point that uniform
   fusion of a genuinely informative channel is worse than ignoring it.
2. Which queries benefit is predictable from the cheap channel's own score distribution, at negligible
   cost (measured router overhead: 345 µs/query).
3. The size of that predictability, and therefore the value of routing, is governed by the spread of
   per-query gain — which can be estimated in advance.

## What I would like from you

- Is the pivot from expansion-tier routing to channel routing the right call?
- For efficiency reporting, we now have measured latency distributions, throughput, energy per query
  and risk–coverage/AURC. Which of those should lead, and which are padding for this venue?
