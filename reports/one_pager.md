# Adaptive Q2E one-pager

Standard practice in IR is to let a query performance predictor decide how much machinery a query gets.
Arabzadeh et al. (arXiv:2604.22661) do exactly this to pick among LLM query variants, and it works. We
ask whether it survives when the choice is *which channel to search* in a video corpus: speech,
on-screen text, or the frames themselves. The choice is worth making. A learned selector over the
channels' own score distributions beats the best fixed policy by +7.59 nDCG, and the same learner fed
the corpus-statistic predictors' own per-channel features buys +0.81 at best: the two feature families,
matched on learner, folds and protocol, differ by an order of magnitude. Every predictor in the study
gets the identical leakage-free operating-point calibration, and under that shared protocol the eleven
corpus-statistic predictors clear the source study's margin in 0 of 33 cells, with 21 of 33 formally
equivalent to doing nothing within 0.005 nDCG. What separates the families is dependence on
corpus-aggregate term statistics, collection frequencies read against the query, and in a corpus where
10% of videos carry no text the statistics a channel does not have cannot be aggregated in the first
place. Rank correlation, meanwhile, does not specify a decision threshold: nested calibration changes
BERT-QPP's cross-encoder from an always-fuse failure to the strongest learned QPP baseline and removes
the apparent overall anti-correlation between τ and utility.

**Claim.** QPP-based selection has a boundary: predictors built on corpus-aggregate term statistics
carry roughly an order of magnitude less usable signal for *evidence-source* selection than predictors
that read option outcomes from retrieval, on this benchmark a gap of +0.81 against +7.59 nDCG under a
matched learner. The pre/post-retrieval split the literature organises by is the wrong proxy for that
line: one pre-retrieval predictor that aggregates no corpus statistics (QSD_pre) crosses it, and every
corpus-statistic predictor stays behind it. The boundary is measured, family-level, and bounded.
Nothing here proves a query--collection statistic can never help, and at deliberately suboptimal
fusion weights a few of them do help a little, because routing away from a bad global weight is easy.
The family contrast survives every weight we tested.

MultiVENT 2.0 test, 2,546 queries, graded multi-gold judgments, folds grouped by event (536 groups).
Predictor definitions follow `github.com/Narabzad/QPP-4-RAG`. The source study is published at SIGIR
2026; our quoted numbers are from arXiv v1 and must be re-checked against the camera-ready before
submission.

---

## RQ1. Does QPP-based selection transfer from query variants to evidence sources?

**Not usefully, for the corpus-statistic family.** Arabzadeh et al. (arXiv:2604.22661) show cheap
pre-retrieval predictors picking well among 30 LLM query variants, ahead of NQC. Run the same families
against the same reference definitions on a choice among evidence channels and all eleven land at
τ ≈ 0. Under the identical nested operating-point protocol every learned row gets, they clear the
source study's 0.0005 margin in 0 of 33 cells, none beats the fixed policy after Holm-corrected
group-level tests, and 21 of 33 are equivalent to doing nothing within 0.005 nDCG. The zero is
protocol-robust and weight-sensitive in an instructive way: rebuilt at deliberately suboptimal fusion
weights the family gains up to 5 of 11 small underlines in the dense cell, because a bad global weight
leaves headroom any weak signal can claw back, while the score-only family holds 8 to 10 of 10 at
every weight. The contrast, roughly ten to one under the matched learner, is the finding; the literal
zero is a property of the well-tuned baseline.

The obvious objection is that one index over transcripts gives every predictor one number per query
regardless of channel. So we built an index per channel: transcripts, on-screen text, and the shipped
captions as a text surrogate for the visual channel, which has none of its own. Features do vary across
the three (mean relative range 0.09–0.33). The repair buys +0.43 ± 0.38, +0.86 ± 0.52 and +0.81 ± 0.57.
Stacked on the score features it gives +7.56 ± 0.89, which is the score-feature baseline back again. We
handed pre-retrieval QPP a proxy it does not normally get and it still did not help.

**The choice itself is predictable.** A selector over the channels' own score distributions beats the
best fixed policy chosen on the training fold by **+7.59 ± 1.01 nDCG** (group-level sign-flip
p < 5 × 10⁻⁴ for both the pooled selection gain and the nested gap itself, with events as the
exchangeable unit). The failure above therefore sits with the predictor family. The task gives a
learner plenty to find. We treat this selector as the positive control, and the paper's claim is about
which predictors can read the signal it proves exists.

The boundary survives two harder checks, with useful qualifications. A translated speech channel raises
the fixed baseline from 0.3408 to 0.3452: 3 of 11 corpus-statistic rows now clear the 0.0005 margin, but
only to 0.3468--0.3472, against 8 of 10 score-only rows and 0.3920 for the control. On MSR-VTT-1kA,
across direct visual-versus-caption choice and visual-to-fusion escalation under two encoders and two
evidence conditions, the corpus-statistic family is 0/88. The direct choice has 5.57--10.37 nDCG of
oracle headroom and the control beats fixed in all four conditions. Score-only transfer on that second
collection is mixed, so the claim that survives both checks is the corpus-statistic boundary. The
score-only family earns no universal guarantee from us.

---

## RQ2. How do the standard QPP predictors compare when used as routers?

Laid out like Arabzadeh et al.'s Table 1 so the two read side by side: <u>underline</u> beats the
Original row, **bold** is best in section. Their Original is the unmodified query, so ours is the best
fixed policy, the default when you do no selection. **ASR-dense cell** below, the one where the
expensive channel pays. All three cells and the full artifact are in
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

*n.i.* = no equivalent predictor implemented. Every selector row, analytic and learned alike, uses the
same protocol: out-of-fold prediction with an escalation fraction chosen on a group-disjoint subset of
each outer training fold and frozen before outer-test prediction. QSD_pre additionally chooses its
neighbourhood size on the same inner split, so no hyperparameter anywhere reads a test label. Clarity
is computed over the shipped captions, the same text surrogate the per-channel index repair and
BERT-QPP's document side were granted; it carries essentially no signal (τ +0.028 here, −0.010 and
+0.007 in the other cells). Underline follows their rule, a margin above 5 × 10⁻⁴ over the Original
row. The Ours row is a binary escalate-or-not decision, the only decision this cell offers. The k-way
selector over all 7 channel subsets is the separate experiment behind the +7.59 in RQ1 and is not a
row here.

**Across all three cells under the shared protocol, the corpus-statistic block has 0 underlined cells
out of 33; the score-only block has 17 out of 30.** With inference behind the counts: 0 of 33
corpus-statistic rows beat the Original row after Holm-corrected group-level sign-flip tests, 21 of 33
sit inside a ±0.005 nDCG equivalence bound, and 8 of 30 score-only rows are significant after the same
correction. The nested protocol also settles what the old degeneracy statistic meant: forced to choose
an operating point, the corpus-statistic predictors do choose (2 of 33 corner cells, down from 20 at
the raw zero crossing) and mostly land *below* the fixed policy, because an operating point placed on
a noise ordering escalates the wrong queries. Calibration can locate an operating point in a useful
ordering; it cannot create one, and that sentence now covers both families.

**The stronger, matched QSD comparison stays mixed.** QSD_post reads the query, neighbouring queries
and their gains, and retrieved document text. With both variants fully nested, QSD_post is slightly
ahead in shipped speech (+0.0039 nDCG), slightly ahead in dense speech (+0.0020), and behind in OCR
(−0.0014). The earlier one-epoch claim that adding documents makes QSD worse did not survive the
matched comparison. The supported conclusion is narrower: document evidence does not produce a
consistent benefit, so it does not rescue the family comparison, but this experiment no longer
positively locates the boundary by itself.

**Correlation is not a decision rule, and calibration changes the conclusion.** The three-epoch BERT-QPP
cross-encoder orders well (τ +0.223 / +0.210 / +0.180), but its predictions remain all positive and its
raw zero crossing fuses every query. Choosing its escalation fraction on group-disjoint inner folds
routes 52.4% / 65.2% / 12.0% instead and reaches 0.3204 / 0.3560 / 0.3042, the best post-retrieval
row in all three cells now that every row shares the nested protocol. On the judged subset its
N_all also beats both fixed endpoints in the two speech cells.

The bi-encoder, the cheaper and deployable one whose document side encodes offline, cannot order at all
even after the same treatment: τ −0.017 / −0.029 / −0.015 and routed nDCG 0.3006 / 0.3403 / 0.3028,
below fixed everywhere. Calibration can locate an operating point in a useful ordering; it cannot create
one.

With every row, analytic and learned, on the shared nested protocol, Kendall τ against utility over
the best fixed policy is **−0.149** over 78 rows (p = 0.054); the old −0.213 (p = 0.008) headline was
a calibration artefact. Against cheap-only it is −0.057 (p = 0.458). Restricted populations stay
modestly negative without reaching a headline. The supported lesson is that τ scores an ordering
while a deployable selector also needs leakage-free operating-point calibration.

---

## RQ3. What makes the multimodal case different?

Classical pre-retrieval QPP is built on tf-idf-style corpus statistics. Our documents are video: there
is no lexical index over frames, and the text channels are noisy, multilingual, and **often absent**.
Of the 109,724 test videos, **10,919 (10.0%) yield no on-screen text at all** and 236 (0.2%) yield no
speech.

For those 10,919 videos there is no on-screen text channel to score. Modality *applicability* is a
property of the document, and no query-side statistic can see it. Text retrieval never poses the
question: every document in a text collection has terms, so a modality is always available.

Measured, the mechanism is real and partial, and we scope it to what it explains. A bare
availability feature (the fraction of top-ranked visual candidates lacking each text channel) routes
to a +1.93 ± 0.69 nested gap on its own, roughly a quarter of the selector's +7.59, and the
selector's per-query gains concentrate where applicability bites (+10.06 nDCG on the 711 queries
with a channel-absent relevant video against +6.63 on the rest). Appending the flags to the score
features buys nothing (+7.28 ± 1.09), so the score distributions already subsume availability. And
the corpus-statistic null does not depend on the absence tail: restricted to the 1,843 queries whose
every relevant document carries on-screen text, the family is still 0 of 11 above the fixed policy.
Absence explains part of the routing signal; it does not explain the family's failure, which shows
up just as sharply in the speech cells where absence is 0.2%.

*(The efficiency material, measured joules, p99 latency and risk-coverage, is scoped to a second
paper, see `paper2_scope.md`.)*

---

## What we are not claiming

Our channels are deliberately cheap, so absolute numbers sit below MMMORRF (0.586) and OmniEmbed (0.753).
For this claim the cheap channels are the setting: a boundary condition on someone else's result does
not need our retrieval to be competitive. We do not claim to beat classical QPP: under the shared
nested protocol our ridge edges NQC in both speech cells (0.3161 vs 0.3144 shipped, 0.3522 vs 0.3514
dense) and trails σ_max and calibrated BERT-QPP in dense speech, so the honest statement is that
several score-family predictors and our router sit within a point of one another on the binary
decision, and all of them are in the family that transfers. Ceilings get audited: picking each query's best policy on half its golds and grading on the other
half wipes out 15 of the oracle's 16 points, so we report no "% of oracle captured" anywhere.

Nor do we claim the selector saves compute. Every predictor that works here reads the channels' own
retrieval output, so all channels are retrieved before the decision fires: this is selective fusion,
an accuracy result, and the selector's marginal cost is a feature computation and a ridge pass. The
compute-saving version of the question, skip a channel before retrieving from it, is exactly what the
cheap pre-retrieval family would have enabled, and it is the family that fails. That is why the
boundary matters to practice. Two further scope notes: the translated channel's aggregate gain
redistributes across languages (Arabic +0.105, English −0.061), so the fused improvement is not
uniform across users, and MSR-VTT's caption channel uses human-written captions, an upper bound on
what a production captioner would provide.

**Open:** the QPP-4-RAG suite is complete, with every predictor named, BERT-QPP in both flavours, both halves
of QSD. The one row we cannot fill is DM, which appears in their Table 1, nowhere in their repository,
and whose row equals their Original row in all eight columns. Only the authors can say what it was.

---

Full evidence, caveats and negative results: `reports/evidence.md`. Complete RQ2 table with all three
cells: `reports/qpp_baselines.md` and `results/ablations/mv2_table1_nested_grouped.md`.
