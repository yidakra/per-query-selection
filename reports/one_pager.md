# Adaptive Q2E one-pager

A video can answer a search query through three kinds of evidence: what was said in it (speech
transcripts), what is written on screen (OCR), and what the frames show. Most systems search all
three and merge the results with fixed weights. We asked a simple question: can a system decide,
for each individual query, which evidence to trust, and can the standard tools of query performance
prediction (QPP) make that decision?

The answer splits cleanly in two. The decision is worth making: choosing evidence per query beats
the best fixed setup by 7.6 nDCG points on MultiVENT 2.0, a large gap in retrieval terms. But the
cheap predictors the QPP literature recommends for decisions like this cannot see it. Every
predictor that works reads the outcome of retrieval itself. Every predictor that only reads the
query against an index of term statistics fails, and we can bound how badly: fed to the same
learner under the same conditions, term-statistic features buy at most +0.86 nDCG where retrieval
outcomes buy +7.59.

This matters because a recent study (Arabzadeh et al., SIGIR 2026) showed the opposite in a
neighbouring setting: cheap term-statistic predictors picked well among LLM rewrites of a query.
Both settings pick one option from several before spending money. The difference is where the
options differ. Rewrites differ on the query side, which term statistics can see. Evidence channels
differ on the document side, which they cannot.

**Claim.** QPP-based selection has a boundary. Predictors built on corpus term statistics carry
roughly ten times less usable signal for choosing an evidence source than predictors that read
retrieval outcomes. The pre-retrieval versus post-retrieval labels the field organises by do not
mark this line. We do not claim term statistics can never help: when the fixed setup is deliberately
mistuned, a few of them recover a little of the slack. The ten-to-one contrast is what survives
everything we tested.

## How to read the numbers

The benchmark is MultiVENT 2.0: 2,546 queries over 109,724 videos, with graded relevance judgments.
A **cell** is one binary decision, per query: stay with the cheap visual search, or escalate to a
fusion with one more channel. There are three cells (shipped speech, dense speech, on-screen text).
The **fixed policy** is the better of the two options applied to every query. A predictor earns
credit only for beating it. Every predictor, simple or learned, goes through the identical
procedure: its threshold is chosen on held-out training data, never on the test queries, and folds
keep queries about the same event together so near-duplicates cannot leak answers. Significance is
tested at the event level with a Holm correction, and a null is only called a null when its
confidence interval fits inside ±0.005 nDCG of the fixed policy.

---

## RQ1. Does QPP-based selection transfer from query rewrites to evidence sources?

Not usefully, for the term-statistic family. We implemented all eleven of the study's term-statistic
predictors (IDF, ICTF, SCQ, SCS and query length, in their aggregations) from the authors' own
reference code. Across the three cells, that is 33 chances to beat the fixed policy. They succeed 0
times. After the significance correction, none comes close, and 21 of the 33 are formally equivalent
to doing nothing.

The obvious objection: term statistics need an index, and we only had one, over the transcripts. So
we built one per channel, using the shipped captions as a stand-in index for the visual channel.
The predictors' values then genuinely differ across channels, and it still does not help: the
repaired features buy +0.43, +0.86 and +0.81 nDCG (each ± about 0.5), against +7.59 from retrieval
outcomes under the identical learner. Stacking both feature sets gives +7.56, the retrieval-outcome
baseline back again.

The signal these predictors miss is real and learnable. A ridge regression over the channels' own
score distributions, thirty features in total, beats the fixed policy by **+7.59 ± 1.01 nDCG**
(p < 0.0005 at the event level). So the task is not the problem. The predictor family is.

The result also survives being attacked from its weakest side, the worry that our channels are too
weak to be representative. We strengthened the speech channel four separate ways, ending with the
retriever family the benchmark's best system uses, and strengthened two channels at once in a fifth
test. The fixed baseline climbed from 0.337 to 0.368. The per-query selection gap stayed between
+7.4 and +8.7 the whole way. On a second collection (MSR-VTT), the term-statistic family again
scored 0 for 88. Retrieval-outcome predictors transferred there only partially, so the claim we
carry forward is the failure of the term-statistic family, not a guarantee for everything else.

---

## RQ2. How do the standard QPP predictors compare when used as selectors?

The table shows the dense-speech cell, laid out like the source study's own results table so the
two can be read side by side. Underline means the predictor beat the fixed policy by their margin;
bold is the best in its block. The other two cells and all metrics are in
`results/ablations/mv2_table1_nested_grouped.md`.

| Category | Method | nDCG@10 | τ | R@100 | N_all | N_strict |
|---|---|---|---|---|---|---|
| Original | best fixed policy (no selection) | 0.3408 | -- | 0.7268 | 0.3672 | 0.2621 |
| | visual only | 0.3036 | -- | 0.6027 | 0.3235 | 0.2256 |
| | uniform fusion (best w) | 0.3408 | -- | 0.7268 | 0.3672 | 0.2621 |
| Pre-retrieval | IDF_avg | 0.3397 | −0.008 | 0.7230 | 0.3655 | 0.2601 |
| | IDF_max | 0.3369 | +0.024 | 0.7075 | 0.3634 | 0.2590 |
| | IDF_sum | 0.3381 | +0.044 | 0.7204 | 0.3654 | 0.2600 |
| | IDF_std | 0.3388 | +0.021 | 0.7166 | 0.3648 | 0.2596 |
| | ICTF_avg | 0.3393 | −0.007 | 0.7235 | 0.3653 | 0.2596 |
| | SCQ_avg | 0.3396 | −0.013 | 0.7240 | 0.3639 | 0.2598 |
| | SCQ_max | 0.3398 | +0.037 | 0.7175 | 0.3652 | 0.2612 |
| | SCQ_sum | 0.3400 | +0.040 | 0.7232 | 0.3672 | 0.2619 |
| | SCS_apx | 0.3404 | −0.014 | 0.7263 | 0.3660 | 0.2609 |
| | SCS_full | 0.3406 | −0.016 | 0.7262 | 0.3674 | 0.2621 |
| | QL | 0.3399 | +0.052 | 0.7184 | 0.3672 | 0.2619 |
| | **QSD_pre** | **<u>0.3466</u>** | +0.134 | 0.7149 | 0.3616 | 0.2562 |
| | DM | *n.i.* | -- | -- | -- | -- |
| Post-retrieval | RSD | 0.3405 | −0.069 | 0.7175 | 0.3755 | 0.2689 |
| | clarity (caption surrogate) | 0.3406 | +0.028 | 0.7232 | -- | -- |
| | NQC | <u>0.3514</u> | −0.163 | 0.7228 | 0.3671 | 0.2623 |
| | NQC_norm | <u>0.3519</u> | −0.154 | 0.7214 | 0.3686 | 0.2631 |
| | σ_max | <u>0.3532</u> | −0.144 | 0.7212 | 0.3716 | 0.2660 |
| | σ_50% | <u>0.3469</u> | −0.122 | 0.7148 | 0.3621 | 0.2564 |
| | SMV | <u>0.3491</u> | −0.151 | 0.7213 | 0.3667 | 0.2615 |
| | SMV_norm | <u>0.3507</u> | −0.145 | 0.7201 | 0.3677 | 0.2618 |
| | WIG | 0.3408 | +0.012 | 0.7231 | 0.3678 | 0.2624 |
| | WIG_norm | <u>0.3449</u> | −0.106 | 0.7161 | 0.3626 | 0.2555 |
| | max | <u>0.3433</u> | −0.112 | 0.7178 | 0.3585 | 0.2573 |
| | QSD_post | <u>0.3486</u> | +0.142 | 0.7087 | 0.3705 | 0.2670 |
| | **BERT-QPP (cross)** | **<u>0.3560</u>** | **+0.210** | 0.7042 | 0.3748 | 0.2725 |
| | BERT-QPP (bi) | 0.3403 | −0.029 | 0.7264 | 0.3676 | 0.2624 |
| Ours | cheap-feature gain ridge | <u>0.3522</u> | +0.160 | 0.7214 | 0.3712 | 0.2634 |
| Oracle | route by true gain | <u>0.3910</u> | +1.000 | 0.6286 | 0.3861 | 0.2841 |

*n.i.* means the study names the predictor but its repository does not contain it, so only the authors
can say what it was. Clarity is computed over the same caption stand-in every other predictor got,
and carries no signal in any cell.

Three things stand out.

**The two blocks split.** Across all three cells the term-statistic block earns 0 underlines out of
33; the retrieval-outcome block earns 17 out of 30, of which 8 survive the significance correction.
Forced to pick a threshold like everyone else, the term-statistic predictors do make choices now.
The choices just make things worse: most of their rows land below the fixed policy, because a
threshold placed on a noise ordering escalates the wrong queries.

**One pre-retrieval predictor stands above the line, and it is the exception that explains the
rule.** QSD_pre reads no index at all. It finds the most similar previously-seen queries and copies
their known outcomes. Its margins over the fixed policy (+0.003 and +0.006 in the speech cells) are
consistent in direction but do not individually survive the significance correction, so we report it
as suggestive. What it shows is that the pre-retrieval label was never the point: this "pre-retrieval"
predictor works exactly as far as it smuggles in outcome information from neighbouring queries.

**A good ordering is not a decision.** BERT-QPP's cross-encoder ranks queries better than anything
else in the table (τ +0.21) yet its raw scores are all positive, so thresholding them at zero fuses
every query and selects nothing. Choosing its threshold on held-out training data instead turns the
same model into the strongest learned row in the table. Its cheaper sibling, the bi-encoder, cannot
rank at all and no threshold saves it. The general version of this finding: across all 78
predictor-cell pairs, a predictor's rank correlation τ correlates at only −0.149 (p = 0.054) with
the value it actually delivers as a selector. An earlier version of this analysis reported a strong
anti-correlation (−0.213, p = 0.008). That turned out to be an artifact of thresholding some
predictors and calibrating others, and we retracted it.

QSD_post, which adds retrieved document text to QSD_pre, changes nothing worth reporting: a few
thousandths of nDCG in either direction depending on the cell.

---

## RQ3. What makes the multimodal case different?

Term-statistic QPP assumes every document has terms. Video breaks that assumption. Of the 109,724
test videos, 10,919 (10.0%) contain no on-screen text at all, and 236 contain no speech. For those
videos one of the channels does not exist, and no statistic computed from the query alone can know
that.

We measured how much of the story this explains, and the honest answer is: part of the selection
signal, none of the family's failure. A single feature counting how many top-ranked candidates lack
a channel routes to +1.93 nDCG on its own, about a quarter of the full +7.59, and the selector's
gains concentrate on exactly the queries whose relevant videos are missing a channel (+10.06 there
against +6.63 elsewhere). But the term-statistic predictors fail just as completely on the 1,843
queries where every relevant video has all its text channels, and just as sharply in the speech
cells where absence is 0.2%. Missing channels are a real signal the working predictors pick up.
They are not the reason the failing predictors fail.

---

## What we are not claiming

Our channels are deliberately cheap, and our absolute numbers sit below the benchmark's best systems
(MMMORRF 0.586, OmniEmbed 0.753). That is the setting, not the finding: a boundary on what a
predictor family can see does not require competitive retrieval, and the channel-strengthening
ladder above is the check that the boundary is not an artifact of weak channels.

We do not claim to beat classical QPP at its own game. On the binary escalate-or-not decision, our
ridge, NQC, σ_max and calibrated BERT-QPP all sit within a point of one another, and every one of
them reads retrieval outcomes.

We do not claim the selector saves compute. Every working predictor needs all channels retrieved
first, so this is selective fusion, an accuracy result. The compute-saving version, skipping a
channel before retrieving from it, is exactly what the cheap family would have enabled, and it is
the family that fails.

We report no "percent of oracle" anywhere: picking each query's best policy on half its relevance
judgments and grading on the other half destroys 15 of the oracle's 16 points, so the oracle rows
are ceilings on these labels, not targets.

Two scope notes. The translated speech channel's average gain redistributes across languages
(Arabic +0.105, English −0.061), so the improvement is not uniform across users. And the source
study's numbers are quoted from its arXiv version and need a re-check against the published
SIGIR 2026 version.

---

Full evidence, caveats and negative results: `reports/evidence.md`. The complete RQ2 table for all
three cells: `results/ablations/mv2_table1_nested_grouped.md`.
