# Adaptive Q2E: one-page summary

**Claim: in multimodal video retrieval, which evidence to trust is a per-query decision, and a
1-ms model reading scores the system already computed makes that decision well.** All numbers are
MultiVENT 2.0 test (2,546 queries, graded multi-gold judgments, nDCG@10).

## Research questions

**RQ1.** Can a retrieval system tell, per query and before spending anything, which of its evidence
sources will help? Answer: yes, from the cheap sources' own score distributions; and no from the
query text alone, which refutes Adaptive-RAG's premise in this setting.

**RQ2.** Does acting on that prediction beat every fixed policy, including the best fusion of
everything? Answer: by +7.59 ± 1.01 nDCG on the strong-channel cell, +5.64 ± 0.93 on the shipped
channels, measured leak-free and split by event so near-duplicate queries cannot cross folds.

**RQ3.** When does per-query selection pay, and can that be known in advance? Answer: the payoff
tracks the spread of per-query gain (ρ = 0.943 across six cells), and it vanishes for the LLM
expansion tier, whose gain no feature we tried can rank.

**RQ4.** What makes prediction-based routing different when the documents are video? Answer: three
things, each measured, and this is the axis separating us from a QPP-for-selection literature that is
entirely text.

1. *Predictor and retriever can read different modalities.* Pre-retrieval QPP scores the query against a
   corpus index; in text that is the corpus the retriever searches. Ours searches video, so the only
   index available covers ASR transcripts and describes a different channel than the router is deciding
   about. All ten pre-retrieval predictors land at **τ ≈ 0**, against **|τ| ≈ 0.21** for score-only
   post-retrieval. The same predictors are competitive in text RAG, so this is the setting, not them.
2. *Some predictors do not exist here.* Clarity needs a language model over the retrieved documents.
   Frames have no terms, so it is undefined rather than weak, and we report it unavailable.
3. *Channel applicability is a property of the document.* A silent clip has no speech to transcribe. In
   query-variant selection every variant applies to every document and only quality varies. That is what
   produces the gain spread RQ3 converts into accuracy (sd 23.1 against a mean of 3.72).

A fourth claim is pending the RAG arm now being measured: ranking and grounding may come apart, since
our best cheap retriever (visual) emits embeddings no generator can read while our worst (OCR) emits
usable text. Full argument in `related_work_qpp.md`.

## How we got here

We started by importing Adaptive-RAG's premise into Q2E's LLM-expansion cascade. It half-failed in a
useful way. Query text predicts nothing about which queries need the expensive tier; the cheap tier's
own score distribution predicts a lot. And the expensive step it was meant to gate turned out not
worth gating: LLM decomposition buys at most +0.8 nDCG for 260 J and 9.4 s per query, with per-query
gain that no feature we tried can rank (τ = +0.002, p = .44). The decision that actually matters sits
among the modality channels (frames, speech transcripts, on-screen text, captions), which every strong
system on this benchmark fuses with one global weighting. Ours says the weighting should be per query.

| policy over the same three channels | nDCG@10 |
|---|---|
| visual channel alone | 0.3036 |
| best uniform fusion (best weights we found) | 0.3408 |
| pairwise routing: fuse dense ASR or don't | 0.3545 |
| per-query channel selection over 7 policies | **0.4131** |

The selector is a multi-target ridge over 30 cheap features (each channel's score-confidence shape,
plus how much the channels' top candidates overlap). Out-of-fold, it beats the best fixed policy
chosen on the training fold by **+7.59 ± 1.01** nDCG (+5.64 ± 0.93 with the benchmark's shipped
channels; permutation p = .0005 in every cell). For 72% of queries it picks a single channel. Fusing
everything on every query with unit weights, the field's default, loses to it by 13.6 points, and even
the best weighted fusion we could find loses by 7.2. Where routing pays is also
predictable before building anything: the spread of per-query gain predicts the achieved routing gap
at ρ = 0.943 across our six original cells, and the channel cells land on the same line.

Ceilings get audited here. Picking each query's best policy on half its golds and grading on the
other half wipes out 15 of the oracle's 16 points, so we report no "% of oracle captured". The
selector's gap is immune to that leak by construction: its features never see a label and every
prediction is out-of-fold. Folds are also split by event, not by query, because the benchmark carries
several phrasings of the same event and a plain split puts near-duplicates on both sides of the
boundary. That correction costs the selector 6% of its gap and costs the strongest historical-query
baseline half of its correlation; the taxonomy is in `qpp_baselines.md`.

### RQ2 in full: routing vs. QPP baselines

Twenty QPP predictors implemented to the `QPP-4-RAG` definitions and run as routers, each oriented by an
out-of-fold ridge then used to escalate the queries it flags. Decision metric is routed nDCG@10, ordering
metric is Kendall τ against true gain. Folds are event-grouped. Summary rows below; all twenty, plus
coverage notes, in `reports/qpp_baselines.md`.

| Category | Method | ASR-shipped | τ | ASR-dense | τ |
|---|---|---|---|---|---|
| Original | visual only (cheap) | 0.3036 | — | 0.3036 | — |
| | uniform fusion (best w) | 0.2795 | — | 0.3408 | — |
| Pre-retrieval (10) | best of family | 0.3039 | +0.067 | 0.3408 | +0.044 |
| Post-retrieval (10) | best of family (NQC) | **0.3205** | −0.215 | **0.3541** | −0.154 |
| | clarity | n/a | — | n/a | — |
| Ours | cheap-feature gain ridge | 0.3193 | +0.211 | 0.3531 | +0.160 |
| Oracle | route by true gain | 0.3653 | +1.000 | 0.3910 | +1.000 |

Three readings, one of them against us. **Pre-retrieval QPP fails here**: all ten sit at τ ≈ 0 and route
to within 0.0005 of doing nothing, which is the sharpest evidence for RQ1 and the measured core of RQ4.
**Score-only post-retrieval works**, at |τ| ≈ 0.21. **Our router ties with NQC on this binary decision**,
and if anything trails it (0.3193 vs 0.3205, and 0.3531 vs 0.3541; |τ| 0.211 vs 0.215), so eight features
buy nothing over one good predictor when the only question is whether to escalate. Learning earns its keep on the k-way choice,
which a scalar cannot express: that is why the selector above is the contribution and this cell is the
baseline it clears. An earlier revision claimed we beat every predictor; that was an artifact of our own
implementations, now corrected against the reference.


Negative results we stand behind, briefly. The LLM expansion tier is correctly declined by its own
router (and a 14B decomposer stays Pareto-dominated, so a 70B would not rescue it). Paraphrase
selection ("tier C") does not exist once gold-split audited. On-screen text is weak evidence however
it is scored; dense retrieval that lifted ASR by 4.7 points moves OCR by 0.1.

Scope, honestly: our channels are deliberately cheap, so absolute numbers sit below MMMORRF (0.586)
and OmniEmbed (0.753), which buy their lift with translate-distill dense retrieval per channel. The
contribution is the decision layer those systems lack, and their channels drop into it unchanged.
The gap-closing experiment is already running: NLLB-translating all 71K non-English transcripts to
English, then re-scoring with the same dense encoder, the translate half of MMMORRF's recipe on our
hardware. Results in ~2 days.
