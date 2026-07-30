# Adaptive Q2E: one-page summary

**Claim: in multimodal video retrieval, which evidence to trust is a per-query decision, and a
1-ms model reading scores the system already computed makes that decision well.** All numbers are
MultiVENT 2.0 test (2,546 queries, graded multi-gold judgments, nDCG@10).

## Research questions

**RQ1.** Can a retrieval system tell, per query and before spending anything, which of its evidence
sources will help? Answer: yes, from the cheap sources' own score distributions; and no from the
query text alone, which refutes Adaptive-RAG's premise in this setting.

**RQ2.** Does acting on that prediction beat every fixed policy, including the best fusion of
everything? Answer: by +8.09 ± 0.65 nDCG on the strong-channel cell, +6.20 ± 0.29 on the shipped
channels, measured leak-free.

**RQ3.** When does per-query selection pay, and can that be known in advance? Answer: the payoff
tracks the spread of per-query gain (ρ = 0.943 across six cells), and it vanishes for the LLM
expansion tier, whose gain no feature we tried can rank.

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
| per-query channel selection over 7 policies | **0.4171** |

The selector is a multi-target ridge over 30 cheap features (each channel's score-confidence shape,
plus how much the channels' top candidates overlap). Out-of-fold, it beats the best fixed policy
chosen on the training fold by **+8.09 ± 0.65** nDCG (+6.20 ± 0.29 with the benchmark's shipped
channels; permutation p = .0005 in every cell). For 71% of queries it picks a single channel. Fusing
everything on every query, the field's default, loses to it by ten points. Where routing pays is also
predictable before building anything: the spread of per-query gain predicts the achieved routing gap
at ρ = 0.943 across our six original cells, and the channel cells land on the same line.

Ceilings get audited here. Picking each query's best policy on half its golds and grading on the
other half wipes out 15 of the oracle's 16 points, so we report no "% of oracle captured". The
selector's gap is immune to that leak by construction: its features never see a label and every
prediction is out-of-fold.

### RQ2 in full: routing vs. QPP baselines

The natural baseline for a per-query router is Query Performance Prediction. We implement twenty
predictors to the definitions in `QPP-4-RAG` (Arabzadeh et al. 2026), each computed for a query, oriented
by a single-feature out-of-fold ridge, then used to escalate the queries it flags. Decision metric is the
routed nDCG@10; ordering metric is Kendall τ of the raw predictor against true gain.

| Category | Method | ASR-shipped | τ | ASR-dense | τ | OCR | τ |
|---|---|---|---|---|---|---|---|
| Original | visual only (cheap) | 0.3036 | — | 0.3036 | — | 0.3036 | — |
| | uniform fusion (best w) | 0.2795 | — | 0.3408 | — | 0.2445 | — |
| Pre-retrieval | IDF_avg | 0.3037 | −0.033 | 0.3408 | −0.008 | 0.3036 | +0.022 |
| (ASR text index) | IDF_max | 0.3036 | −0.008 | 0.3408 | +0.024 | 0.3036 | +0.006 |
| | IDF_sum | 0.3034 | +0.057 | 0.3403 | +0.044 | 0.3036 | +0.009 |
| | IDF_std | 0.3036 | +0.016 | 0.3408 | +0.021 | 0.3036 | −0.020 |
| | SCQ_avg | 0.3036 | −0.000 | 0.3408 | −0.013 | 0.3036 | +0.024 |
| | SCQ_max | 0.3032 | +0.064 | 0.3406 | +0.037 | 0.3036 | +0.034 |
| | SCQ_sum | 0.3038 | +0.067 | 0.3408 | +0.040 | 0.3036 | +0.004 |
| | avgICTF | 0.3037 | −0.032 | 0.3408 | −0.007 | 0.3036 | +0.027 |
| | SCS_1 | 0.3033 | −0.040 | 0.3408 | −0.014 | 0.3036 | +0.029 |
| | SCS_2 | 0.3034 | −0.043 | 0.3408 | −0.016 | 0.3036 | +0.027 |
| Post-retrieval | WIG_norm | 0.3151 | −0.168 | 0.3421 | −0.106 | 0.3036 | −0.100 |
| (score-only) | WIG | 0.3036 | +0.006 | 0.3408 | +0.012 | 0.3033 | +0.051 |
| | NQC_norm | 0.3176 | −0.204 | 0.3534 | −0.154 | 0.3036 | −0.162 |
| | **NQC** | **0.3211** | −0.215 | 0.3532 | −0.163 | 0.3039 | −0.174 |
| | SMV_norm | 0.3173 | −0.198 | 0.3518 | −0.145 | 0.3034 | −0.159 |
| | SMV | 0.3198 | −0.208 | 0.3514 | −0.151 | 0.3037 | −0.169 |
| | RSD | 0.3135 | −0.127 | 0.3404 | −0.069 | 0.3034 | −0.099 |
| | σ_max | 0.3179 | −0.196 | 0.3502 | −0.144 | 0.3036 | −0.149 |
| | σ_x0.5 | 0.3160 | −0.182 | 0.3452 | −0.122 | 0.3036 | −0.100 |
| | max | 0.3075 | −0.124 | 0.3436 | −0.112 | 0.3036 | −0.087 |
| Post-retrieval | clarity | n/a | — | n/a | — | n/a | — |
| (needs doc text) | | | | | | | |
| Ours | cheap-feature gain ridge | 0.3205 | +0.217 | **0.3536** | +0.170 | 0.3033 | +0.162 |
| Oracle | route by true gain | 0.3653 | +1.000 | 0.3910 | +1.000 | 0.3305 | +1.000 |

Three things this table says, one of them against us.

**Pre-retrieval QPP does not work here.** Every predictor lands at τ ≈ 0 and routes to within 0.0005 of
either doing nothing or fusing everything. This is the sharpest evidence for RQ1: query-side statistics
carry no usable signal about which channel will help. It also diverges from Arabzadeh et al., who find
cheap pre-retrieval predictors competitive in text RAG, and the reason is structural. A pre-retrieval
predictor measures a query against a corpus index, but the channel being routed here is *visual*. The
only index we can build is over the ASR transcripts, so the statistics describe a different modality
than the decision. That mismatch has no analogue in text retrieval, where predictor and retriever read
the same corpus.

**Post-retrieval score-only predictors do work**, reaching |τ| ≈ 0.21. Clarity is the exception and is
marked n/a rather than zero: it needs an RM1 language model over the retrieved documents, and the
visual channel's documents are frames. The unavailability is structural, so reporting a number would
misrepresent it.

**Our learned router ties with the best single predictor for this binary decision.** NQC reaches 0.3211
on ASR-shipped against our 0.3205, and 0.3532 on ASR-dense against our 0.3536; sweeping the escalation
fraction instead of thresholding at zero keeps them within 0.003 either way (NQC 0.3220 / 0.3532, ours
0.3217 / 0.3555), and |τ| is 0.215 against 0.217. Eight features buy nothing over one well-implemented
predictor when the only question is whether to escalate. The gain from learning appears once the
decision is *which* of several policies to use: a scalar predictor can rank queries by confidence but
cannot name a channel, which is why the k-way selector above is the contribution and this cell is the
baseline it has to clear. An earlier revision of this table reported our router beating every predictor;
that was an artifact of our own predictor implementations, corrected here against `QPP-4-RAG`.

QSD_post and BERT-QPP are still to run and need document embeddings and a trained model respectively.

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
