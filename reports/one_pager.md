# Adaptive Q2E: one-page summary

**Claim: QPP-based selection has a boundary, and we found where it is. Cheap pre-retrieval predictors
pick well among query variants and cannot pick among evidence sources at all, because the options in the
second case differ on the document side, where query-collection statistics cannot see.** All numbers are
MultiVENT 2.0 test (2,546 queries, graded multi-gold judgments, nDCG@10), folds grouped by event.

Story B, agreed 31 Jul 2026. Per-query channel routing is the positive control, not the headline.
The story-A version of this file led with the routing gain.

## The argument in three steps

1. **Pre-retrieval QPP does not transfer to source selection.** Arabzadeh et al. (arXiv:2604.22661)
   show cheap pre-retrieval predictors picking well among 30 LLM query variants, ahead of NQC. We ran
   the same families, implemented against the same reference definitions, on a choice among evidence
   channels. All ten pre-retrieval predictors land at τ ≈ 0 and route to within 0.0005 nDCG of doing
   nothing. Score-only post-retrieval reaches |τ| ≈ 0.21. The families swap places.

2. **The choice is predictable, from the other family.** A null alone could mean the decision is not
   predictable, in which case nothing follows about predictors. It is predictable: a selector over the
   channels' own score distributions beats the best fixed policy chosen on the training fold by
   **+7.59 ± 1.01 nDCG** (+5.64 ± 0.93 on the benchmark's shipped channels, permutation p = .0005 in
   both). So the null is about a family of predictors, not about the task.

3. **The reason is structural.** Variant selection compares competing texts, and how a text sits
   against a collection is exactly what pre-retrieval statistics were built to measure. Source
   selection compares channels whose applicability is a property of the document. A silent clip has no
   speech to transcribe. In variant selection every option applies to every document and only quality
   varies.

## Evidence

**The null, and the repair that fails to rescue it.** The obvious objection is that we built one index,
over speech transcripts, so every predictor returned one number per query regardless of channel. So we
gave it every index the benchmark allows: transcripts, on-screen text, and the shipped captions as a
text surrogate for the visual channel, which searches frame embeddings and has no term index of its own.
The features do vary across the three, mean relative range 0.09 to 0.33, so the repair is real. It buys
+0.43 ± 0.38 from speech alone, +0.86 ± 0.52 with on-screen text, +0.81 ± 0.57 with the visual surrogate
added. Stacked on the score features: +7.56 ± 0.89, which is the baseline back again. The surrogate is a
concession we make out loud — pre-retrieval QPP was handed a proxy it does not normally get, and it
still did not help.

This is a bounded null, not a shrug. 2,546 queries against their 56 topics, with intervals that exclude
anything of practical size.

**The positive control.**

| policy over the same three channels | nDCG@10 |
|---|---|
| visual channel alone | 0.3036 |
| best uniform fusion (best weights we found) | 0.3408 |
| pairwise routing: fuse dense ASR or don't | 0.3531 |
| per-query channel selection over 7 policies | **0.4131** |

Multi-target ridge over 30 features: each channel's score-confidence shape, plus how much the channels'
top candidates overlap. That last part is complementarity between evidence sources measured before
either is trusted, which is one of the signals Arabzadeh et al.'s closing section asks for. It is
available to us because our options can be scored side by side. For 72% of queries the selector picks a
single channel.

**The limit case.** Clarity needs a language model over the retrieved documents. Over frames it is
undefined rather than weak, and we report it unavailable rather than substituting a number that looks
like one.

**Downstream, and a prediction of ours that failed.** We expected ranking and grounding to come apart
on our channels, since the visual channel retrieves best and emits embeddings no generator can read
while OCR retrieves worst (0.1223) and emits usable text. Under the QPP-4-RAG nuggetizer protocol on
395 queries with a local judge, the routed system reaches 0.5007 vital-nugget coverage against 0.4648
for the best fixed policy (p = .037), 0.3902 against 0.3432 on strict vital (p = .014). **The ordering
under nugget coverage is the ordering under nDCG.** No utility gap at the policy level. The physical
asymmetry between the channels is real and it did not produce a divergence.

**Robustness.** The null is not an artefact of weak channels. Translating all 109,488 ASR transcripts
with NLLB and re-encoding raises the speech channel from 0.3134 to 0.3332, and the routed gap goes
**+7.59 → +8.01 ± 1.01** while pre-retrieval stays at zero. Improving a channel raises both sides and
the decision layer keeps its margin, which is what should happen if the gain comes from variation in
which channel suits which query rather than from any one channel being bad.

**A methodological note their design invites.** MultiVENT 2.0 carries several phrasings of one event, so
a query-split fold lets a predictor that learns from other queries read its answer off a near-duplicate.
Event grouping (536 groups over 2,546 queries) costs our selector 6% of its gap and leaves the analytic
predictors within 0.002 nDCG, because they read only the current query's scores. QSD loses 52% of its
correlation and BERT-QPP loses 10-29%. The taxonomy is clean: predictors that consume other queries'
performance leak, predictors that consume the current query's scores do not. Thirty variants of one
information need share their relevant documents by construction, so this applies to their design too.

## How we got here

We started by importing Adaptive-RAG's premise into Q2E's LLM-expansion cascade, and it half-failed
usefully. Query text predicts nothing about which queries need the expensive tier (accuracy at or below
the majority prior); the cheap tier's own score distribution predicts a lot. MultiVENT's queries are
259/259 Latin script, so the multilinguality lives in the videos and there is no query-side language
feature to route on. That pointed at the decision among modality channels, which every strong system on
this benchmark makes once, globally, with a single fusion weighting.

## What we are not claiming

Our channels are deliberately cheap, so absolute numbers sit below MMMORRF (0.586) and OmniEmbed
(0.753), which buy their lift with translate-distill dense retrieval per channel. Under story B that is
setting rather than weakness: a boundary condition on someone else's result does not need our retrieval
to be competitive. It would need it if the routing gain were the headline, which is the main reason it
is not.

We also do not claim to beat classical QPP. NQC ties or edges our router on the binary escalate-or-not
decision (0.3193 vs 0.3205 shipped; 0.3531 vs 0.3541 dense). That costs us nothing here — NQC is in the
family that transfers, so it corroborates the claim. A scalar predictor cannot express a k-way policy,
which is why the selector is a selector, but that point is no longer load-bearing.

Ceilings get audited. Picking each query's best policy on half its golds and grading on the other half
wipes out 15 of the oracle's 16 points, so we report no "% of oracle captured" anywhere.

## Negative results we stand behind

The LLM expansion tier is correctly declined by its own router: the oracle prize is +2.55 and does not
survive a held-out gold split, while the price is 63.5× tier B. A 14B decomposer stays Pareto-dominated,
so a 70B would not rescue it. Paraphrase selection ("tier C") does not exist once gold-split audited.
On-screen text is weak evidence however it is scored — dense retrieval that lifted ASR by 4.7 points
moves OCR by 0.1.

## Material that needs a home

Story B has no room for the efficiency work: measured joules per tier, the router at 1.04 ms against
the 9.4 s call it gates, escalating 10% of queries multiplying p99 by 435×, and the risk-coverage /
AURC framing. Same for the heterogeneity result, where sd(per-query gain) predicts the achieved routing
gap at ρ = 0.943 across six cells. Both are real and both are now support at best. Companion
submission, appendix, or a second paper where story A is the point. Undecided.
