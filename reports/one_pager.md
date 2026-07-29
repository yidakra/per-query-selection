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
