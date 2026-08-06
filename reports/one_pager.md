# Adaptive Q2E one-pager

Standard practice in IR is to let a query performance predictor decide how much machinery a query gets.
Arabzadeh et al. (arXiv:2604.22661) do exactly this to pick among LLM query variants, and it works. We
ask whether it survives when the choice is *which channel to search* in a video corpus: speech,
on-screen text, or the frames themselves. The choice is worth making (a selector over the channels' own
score distributions beats the best fixed policy by +7.59 nDCG), but the standard corpus-statistic
family does not see it reliably. Eleven such predictors sit flat in the three reference cells, Clarity
cannot even be computed over frames, and the one pre-retrieval predictor that escapes needs no index at
all. What separates the predictors that work from the ones that fail is whether they depend on
document-side language statistics, and in a corpus where 10% of videos carry no text, that dependence
decides a tenth of the collection outright. Rank correlation, meanwhile,
does not specify a decision threshold: nested calibration changes BERT-QPP's cross-encoder from an
always-fuse failure to the strongest learned QPP baseline and removes the apparent overall
anti-correlation between τ and utility.

**Claim.** QPP-based selection has a boundary: the reference corpus-statistic family does not provide
a reliable *evidence-source* selector, while signals that read option outcomes from retrieval transfer
much more reliably. The pre/post-retrieval split the literature organises by is a proxy for that, and
it is the wrong proxy here. The boundary is measured at the family level. Nothing here proves that a
query--collection statistic can never help.

MultiVENT 2.0 test, 2,546 queries, graded multi-gold judgments, folds grouped by event (536 groups).
Predictor definitions follow `github.com/Narabzad/QPP-4-RAG`.

---

## RQ1. Does QPP-based selection transfer from query variants to evidence sources?

**No, for the corpus-statistic family.** Arabzadeh et al. (arXiv:2604.22661) show cheap pre-retrieval
predictors picking well among 30 LLM query variants, ahead of NQC. Run the same families against the same
reference definitions on a choice among evidence channels and all eleven land at τ ≈ 0, routing to within
0.0005 nDCG of doing nothing in each of the three reference cells.

The obvious objection is that one index over transcripts gives every predictor one number per query
regardless of channel. So we built an index per channel: transcripts, on-screen text, and the shipped
captions as a text surrogate for the visual channel, which has none of its own. Features do vary across
the three (mean relative range 0.09–0.33). The repair buys +0.43 ± 0.38, +0.86 ± 0.52 and +0.81 ± 0.57.
Stacked on the score features it gives +7.56 ± 0.89, which is the score-feature baseline back again. We
handed pre-retrieval QPP a proxy it does not normally get and it still did not help.

**The choice itself is predictable.** A selector over the channels' own score distributions beats the
best fixed policy chosen on the training fold by **+7.59 ± 1.01 nDCG** (permutation p = .0005). The
failure above therefore sits with the predictor family. The task gives a learner plenty to find. We
treat this selector as the positive control, and the paper's claim is about which predictors can read
the signal it proves exists.

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
| Pre-retrieval | IDF_avg | 0.3408 | −0.008 | 0.7268 | 0.3672 | 0.2621 |
| | IDF_max | 0.3408 | +0.024 | 0.7268 | 0.3672 | 0.2621 |
| | IDF_sum | 0.3400 | +0.044 | 0.7258 | 0.3682 | 0.2622 |
| | IDF_std | 0.3408 | +0.021 | 0.7268 | 0.3672 | 0.2621 |
| | ICTF_avg | 0.3408 | −0.007 | 0.7268 | 0.3672 | 0.2621 |
| | SCQ_avg | 0.3408 | −0.013 | 0.7268 | 0.3672 | 0.2621 |
| | SCQ_max | 0.3408 | +0.037 | 0.7268 | 0.3672 | 0.2621 |
| | SCQ_sum | 0.3408 | +0.040 | 0.7268 | 0.3672 | 0.2621 |
| | SCS_apx | 0.3408 | −0.014 | 0.7268 | 0.3672 | 0.2621 |
| | SCS_full | 0.3408 | −0.016 | 0.7268 | 0.3672 | 0.2621 |
| | QL | 0.3408 | +0.052 | 0.7268 | 0.3678 | 0.2625 |
| | **QSD_pre** | **<u>0.3469</u>** | +0.152 | 0.7112 | 0.3630 | 0.2559 |
| | DM | *n.i.* | -- | -- | -- | -- |
| Post-retrieval | RSD | 0.3388 | −0.069 | 0.7202 | 0.3660 | 0.2611 |
| | clarity | n/a | -- | -- | -- | -- |
| | NQC | <u>0.3527</u> | −0.163 | 0.7198 | 0.3669 | 0.2630 |
| | NQC_norm | <u>0.3541</u> | −0.154 | 0.7169 | 0.3716 | 0.2671 |
| | σ_max | <u>0.3507</u> | −0.144 | 0.7223 | 0.3676 | 0.2640 |
| | σ_50% | <u>0.3446</u> | −0.122 | 0.7190 | 0.3629 | 0.2572 |
| | SMV | <u>0.3509</u> | −0.151 | 0.7201 | 0.3682 | 0.2638 |
| | SMV_norm | <u>0.3527</u> | −0.145 | 0.7174 | 0.3677 | 0.2631 |
| | WIG | 0.3408 | +0.012 | 0.7268 | 0.3672 | 0.2621 |
| | WIG_norm | 0.3411 | −0.106 | 0.7186 | 0.3649 | 0.2597 |
| | max | <u>0.3430</u> | −0.112 | 0.7232 | 0.3638 | 0.2608 |
| | QSD_post | <u>0.3486</u> | +0.142 | 0.7087 | 0.3705 | 0.2670 |
| | **BERT-QPP (cross)** | **<u>0.3560</u>** | **+0.210** | 0.7042 | 0.3748 | 0.2725 |
| | BERT-QPP (bi) | 0.3403 | −0.029 | 0.7264 | 0.3676 | 0.2624 |
| Ours | cheap-feature gain ridge | <u>0.3531</u> | +0.160 | 0.7207 | 0.3684 | 0.2642 |
| Oracle | route by true gain | <u>0.3910</u> | +1.000 | 0.6286 | 0.3861 | 0.2841 |

*n.i.* = no equivalent predictor implemented; `n/a` = undefined over frames. Analytic predictors use
their out-of-fold gain crossing. QSD and BERT-QPP use an escalation fraction chosen on a group-disjoint
subset of each outer training fold and fixed before outer-test prediction. Underline follows their
rule, a margin above 5 × 10⁻⁴ over the Original row. The Ours row is a binary escalate-or-not decision,
the only decision this cell offers. The k-way selector over all 7 channel subsets is the separate
experiment behind the +7.59 in RQ1 and is not a row here.

**Across all three cells, the corpus-statistic block has 0 underlined cells out of 33; the score-only
block has 16 out of 30.** The underline threshold is not doing the work: the largest margin anywhere in
those 33 cells is **+0.0003**. The escalation fractions show what is actually happening: in the OCR
cell all eleven escalate exactly 0% of queries, and in ASR-dense six of eleven escalate exactly 100%. A
prediction that never varies is a fixed policy wearing a predictor's name. Counting only exact 0 or
exact 1, **20 of 33 corpus-statistic cells are degenerate against 2 of 30 score-only cells**.

**The stronger, matched QSD comparison is mixed.** QSD_post reads the query, neighbouring queries and
their gains, and retrieved document text. After five epochs and group-disjoint nested calibration, it is
effectively tied with equally calibrated QSD_pre in shipped speech (+0.0001 nDCG), modestly better in
dense speech (+0.0017), and worse in OCR (−0.0023). τ changes −0.042 / −0.010 / +0.005. The earlier
one-epoch claim that adding documents makes QSD worse did not survive this. The supported conclusion is
narrower: document evidence does not produce a consistent benefit, so it does not rescue the family
comparison, but this experiment no longer positively locates the boundary by itself.

**Correlation is not a decision rule, and calibration changes the conclusion.** The three-epoch BERT-QPP
cross-encoder orders well (τ +0.223 / +0.210 / +0.180), but its predictions remain all positive and its
raw zero crossing fuses every query. Choosing its escalation fraction on group-disjoint inner folds
routes 52.4% / 65.2% / 12.0% instead and reaches 0.3204 / 0.3560 / 0.3042. It nearly ties the best
post-retrieval row in shipped speech and is best in dense speech and OCR. On the judged subset its
N_all also beats both fixed endpoints in the two speech cells.

The bi-encoder, the cheaper and deployable one whose document side encodes offline, cannot order at all
even after the same treatment: τ −0.017 / −0.029 / −0.015 and routed nDCG 0.3006 / 0.3403 / 0.3028,
below fixed everywhere. Calibration can locate an operating point in a useful ordering; it cannot create
one.

Replacing every learned row with its nested-calibration decision weakens Kendall τ against utility over
the best fixed policy from −0.213 to **−0.099** over 78 rows (p = 0.213). Against cheap-only it is −0.046
(p = 0.558). Restricted populations remain modestly negative, but the overall anti-correlation headline
was a calibration artefact. The supported lesson is that τ scores an ordering while a deployable
selector also needs leakage-free operating-point calibration.

---

## RQ4. What makes the multimodal case different?

Classical pre-retrieval QPP is built on tf-idf-style corpus statistics. Our documents are video: there
is no lexical index over frames, and the text channels are noisy, multilingual, and **often absent**.
Of the 109,724 test videos, **10,919 (10.0%) yield no on-screen text at all** and 236 (0.2%) yield no
speech.

For those 10,919 videos there is no on-screen text channel to score. Modality *applicability* is a
property of the document, and no query-side statistic can see it. Text retrieval never poses the
question: every document in a text collection has terms, so a modality is always available. Clarity is
the limit case. It needs a language model over the retrieved documents, so over frames it has no
definition at all, and we report it unavailable instead of substituting a number that looks like one.

*(RQ3 is the efficiency material: measured joules, p99 latency, risk-coverage. Scoped to a second paper,
see `paper2_scope.md`.)*

---

## What we are not claiming

Our channels are deliberately cheap, so absolute numbers sit below MMMORRF (0.586) and OmniEmbed (0.753).
For this claim the cheap channels are the setting: a boundary condition on someone else's result does
not need our retrieval to be competitive. We do not claim to beat classical QPP: NQC ties or edges our
router on the binary decision, and since NQC is in the family that transfers, that corroborates the
claim. Ceilings get audited: picking each query's best policy on half its golds and grading on the other
half wipes out 15 of the oracle's 16 points, so we report no "% of oracle captured" anywhere.

**Open:** the QPP-4-RAG suite is complete, with every predictor named, BERT-QPP in both flavours, both halves
of QSD. The one row we cannot fill is DM, which appears in their Table 1, nowhere in their repository,
and whose row equals their Original row in all eight columns. Only the authors can say what it was.

---

Full evidence, caveats and negative results: `reports/evidence.md`. Complete RQ2 table with all three
cells: `reports/qpp_baselines.md` and `results/ablations/mv2_table1_nested_grouped.md`.
