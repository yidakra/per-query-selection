# Adaptive Q2E — one-pager

**Claim.** QPP-based selection has a boundary, and it is not the pre/post-retrieval split the literature
organises by. Predictors that read the query against a corpus index cannot choose an *evidence source*.
Predictors that read evidence back from retrieval can. The dividing line is whether a predictor needs
document-side language statistics — which in a video collection may not exist at all.

MultiVENT 2.0 test, 2,546 queries, graded multi-gold judgments, folds grouped by event (536 groups).
Predictor definitions follow `github.com/Narabzad/QPP-4-RAG`.

---

## RQ1 — Does QPP-based selection transfer from query variants to evidence sources?

**No, for the corpus-statistic family.** Arabzadeh et al. (arXiv:2604.22661) show cheap pre-retrieval
predictors picking well among 30 LLM query variants, ahead of NQC. Run the same families against the same
reference definitions on a choice among evidence channels and all eleven land at τ ≈ 0, routing to within
0.0005 nDCG of doing nothing.

The obvious objection is that one index over transcripts gives every predictor one number per query
regardless of channel. So we built an index per channel — transcripts, on-screen text, and the shipped
captions as a text surrogate for the visual channel, which has none of its own. Features do vary across
the three (mean relative range 0.09–0.33). The repair buys +0.43 ± 0.38, +0.86 ± 0.52 and +0.81 ± 0.57;
stacked on the score features it gives +7.56 ± 0.89, which is the score-feature baseline back again. We
handed pre-retrieval QPP a proxy it does not normally get and it still did not help.

**The null is about the family, not the task.** A selector over the channels' own score distributions
beats the best fixed policy chosen on the training fold by **+7.59 ± 1.01 nDCG** (permutation p = .0005).
So the choice is predictable; these predictors just cannot see it. This is the positive control, not the
headline.

---

## RQ2 — How do the standard QPP predictors compare when used as routers?

Laid out like Arabzadeh et al.'s Table 1 so the two read side by side: <u>underline</u> beats the
Original row, **bold** is best in section. Their Original is the unmodified query, so ours is the best
fixed policy — the default when you do no selection. **ASR-dense cell** below, the one where the
expensive channel pays; all three cells and the full artifact in
`results/ablations/mv2_table1_grouped.md`.

| Category | Method | nDCG@10 | τ | R@100 | N_all | N_strict |
|---|---|---|---|---|---|---|
| Original | best fixed policy (no selection) | 0.3408 | — | 0.7268 | 0.3672 | 0.2621 |
| | visual only | 0.3036 | — | 0.6027 | 0.3235 | 0.2256 |
| | uniform fusion (best w) | 0.3408 | — | 0.7268 | 0.3672 | 0.2621 |
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
| | **QSD_pre** | **<u>0.3466</u>** | +0.152 | 0.7074 | 0.3545 | 0.2500 |
| | DM | *n.i.* | — | — | — | — |
| Post-retrieval | RSD | 0.3388 | −0.069 | 0.7202 | 0.3660 | 0.2611 |
| | clarity | n/a | — | — | — | — |
| | NQC | <u>0.3527</u> | −0.163 | 0.7198 | 0.3669 | 0.2630 |
| | **NQC_norm** | **<u>0.3541</u>** | −0.154 | 0.7169 | 0.3716 | 0.2671 |
| | σ_max | <u>0.3507</u> | −0.144 | 0.7223 | 0.3676 | 0.2640 |
| | σ_50% | <u>0.3446</u> | −0.122 | 0.7190 | 0.3629 | 0.2572 |
| | SMV | <u>0.3509</u> | −0.151 | 0.7201 | 0.3682 | 0.2638 |
| | SMV_norm | <u>0.3527</u> | −0.145 | 0.7174 | 0.3677 | 0.2631 |
| | WIG | 0.3408 | +0.012 | 0.7268 | 0.3672 | 0.2621 |
| | WIG_norm | 0.3411 | −0.106 | 0.7186 | 0.3649 | 0.2597 |
| | max | <u>0.3430</u> | −0.112 | 0.7232 | 0.3638 | 0.2608 |
| | QSD_post | <u>0.3423</u> | +0.102 | 0.6983 | 0.3557 | 0.2496 |
| | BERT-QPP (cross) | 0.3408 | **+0.229** | 0.7268 | 0.3672 | 0.2621 |
| | BERT-QPP (bi) | 0.3125 | −0.025 | 0.6414 | 0.3443 | 0.2424 |
| Ours | k-way channel selector | <u>0.3531</u> | +0.160 | 0.7207 | 0.3684 | 0.2642 |
| Oracle | route by true gain | <u>0.3910</u> | +1.000 | 0.6286 | 0.3861 | 0.2841 |

*n.i.* = no equivalent predictor implemented; `n/a` = undefined over frames. Every nDCG cell is the
predictor's own decision at its zero crossing, including BERT-QPP's — swept operating points are
reported separately in `qpp_baselines.md` and are not comparable to this column. Underline follows their
rule, a margin above 5 × 10⁻⁴ over the Original row.

**Across all three cells, the corpus-statistic block has 0 underlined cells out of 33; the score-only
block has 16 out of 30.** The underline threshold is not doing the work: the largest margin anywhere in
those 33 cells is **+0.0003**. The escalation fractions put it more bluntly — a corpus-statistic
predictor here does not choose badly, it does not choose. In the OCR cell all eleven escalate exactly 0%
of queries; in ASR-dense six of eleven escalate exactly 100%. Counting only exact 0 or exact 1, **20 of
33 corpus-statistic cells are degenerate against 2 of 30 score-only cells**.

**Both halves of QSD are now in, and the post-retrieval one is worse.** QSD_post reads everything the
suite has — the query, its neighbours in Query Space with their known gains, and the retrieved document
text — and it lands below QSD_pre on *both* metrics in *all three* cells (−0.008 / −0.004 / −0.003 nDCG,
τ −0.027 / −0.050 / −0.003). QSD_pre uses no document evidence whatsoever. Adding it, plus a trained
transformer, did not help. That is the boundary argued above, tested inside a single predictor family
rather than across families. Caveat carried in the table notes: one epoch, bert-base, CPU, so this
bounds the variant at that budget rather than at any budget.

**Correlation and decision come apart, and they do it systematically.** BERT-QPP (cross) has the best τ
in the table (+0.237 / +0.229 / +0.167) and the worst decision in it: its predictions are all positive,
minimum +0.38, so it escalates every query in all three cells and lands exactly on uniform fusion — worse
than doing nothing in two of them. Its nugget row is identical to the uniform-fusion row, which is what a
selector that never selects looks like under a generation metric.

The two BERT-QPP variants fail in different ways, which is why both are worth carrying. The
cross-encoder orders well and cannot decide. The bi-encoder — the cheaper, deployable one, whose document
side encodes offline — cannot order at all: τ −0.016 / −0.025 / −0.030. It is *not* degenerate, escalating
31–35% of queries, but the decisions are noise, and it loses to the best fixed policy in all three cells.
Neither flavour of the suite's one supervised predictor yields a usable decision.

This is not one bad row. Over all 78 predictor-cell rows, Kendall τ against utility over the best fixed
policy is **−0.213** (p = 0.008), and it *strengthens* to −0.288 when degenerate rows are dropped and
−0.386 among rows escalating between 5% and 95%. τ scores the whole ordering; a selection reads one point
of it.

---

## RQ4 — What makes the multimodal case different?

This is the differentiator, and it is measurable rather than rhetorical. Classical pre-retrieval QPP is
built on tf-idf-style corpus statistics. Our documents are video: there is no lexical index over frames,
and the text channels are noisy, multilingual, and **often absent**. Of the 109,724 test videos,
**10,919 (10.0%) yield no on-screen text at all** and 236 (0.2%) yield no speech.

For those videos a channel does not underperform — it does not exist. Modality *applicability* is a
property of the document, and no query-side statistic can see it. Text retrieval has no analogue: every
document in a text collection has terms, so the question of whether a modality is available never
arises. Clarity is the limit case: it needs a language model over the retrieved documents, so over frames
it is undefined rather than weak, and we report it unavailable rather than substituting a number that
looks like one.

*(RQ3 is the efficiency material — measured joules, p99 latency, risk-coverage. Scoped to a second paper;
see `paper2_scope.md`.)*

---

## What we are not claiming

Our channels are deliberately cheap, so absolute numbers sit below MMMORRF (0.586) and OmniEmbed (0.753).
Under this story that is setting rather than weakness: a boundary condition on someone else's result does
not need our retrieval to be competitive. We do not claim to beat classical QPP — NQC ties or edges our
router on the binary decision, and since NQC is in the family that transfers, that corroborates the
claim. Ceilings get audited: picking each query's best policy on half its golds and grading on the other
half wipes out 15 of the oracle's 16 points, so we report no "% of oracle captured" anywhere.

**Open:** the QPP-4-RAG suite is complete — every predictor named, BERT-QPP in both flavours, both halves
of QSD. The one row we cannot fill is DM, which appears in their Table 1, nowhere in their repository,
and whose row equals their Original row in all eight columns. That is a question for the authors rather
than a gap in our implementation.

---

Full evidence, caveats and negative results: `reports/evidence.md`. Complete RQ2 table with all three
cells: `reports/qpp_baselines.md` and `results/ablations/mv2_table1_grouped.md`.
