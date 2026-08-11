# ECIR 2027 full paper: outline and evidence map

**Story: B with A as the engine** (agreed 31 Jul 2026). The claim is a boundary condition on
QPP-for-selection, and per-query channel routing is the positive control that makes the boundary
interpretable. An earlier version of this file assumed story A and has been replaced.

Working document, not for commit.

## Configuration

ECIR 2027 full paper track. **Confirmed against the call on 2 Aug 2026** (previously assumed):

| | |
|---|---|
| Length | 12 pages, unlimited additional pages for references |
| **Appendices** | **count toward the 12** and must sit before the references |
| Template | Springer LNCS, LaTeX or Word; ORCIDs encouraged |
| Review | **double-blind**, first-stage review then meta-reviewer discussion |
| Abstract due | **21 Sep 2026** |
| Paper due | **5 Oct 2026** |
| Notification | 7 Dec 2026 |
| Conference | 21–25 Mar 2027, Southampton FC, Southampton UK, in person |
| Data release | no explicit requirement in the full-paper call |

Two of these change plans rather than confirm them.

**Double-blind was never in this outline.** The submission cannot self-identify, which reaches further
than the author block: `github.com/yidakra/adaptive-q2e` cannot be linked as-is, any artifact URL has to
be anonymised, and the MultiVENT collaboration cannot be named in a way that identifies us. Decide the
artifact story before writing rather than after.

**Appendices counting toward the 12 pages** removes the obvious parking space. The full RQ2 table is 33
rows × 4 metrics × 3 cells and cannot be a free appendix. It has to be either cut down for the body or
held for a companion artifact. The one-pager's approach (full rows and all four metrics for one cell,
the other two summarised) is the version that fits.

Authorship with the JHU collaborators unsettled and now more consequential. See open items. Under a
21 Sep abstract deadline this is the item with the least slack, since an author list is required before
there is a paper.

---

## The logical spine

This is the part to agree before any prose gets written, because everything else follows from it.

> Arabzadeh et al. show that when the options are **query variants**, a cheap pre-retrieval predictor
> picks well. We ask whether that transfers when the options are **evidence sources**. It does not:
> eleven corpus-statistic predictors sit at τ ≈ 0 and route to within 0.0005 nDCG of doing nothing.
> The one pre-retrieval predictor that escapes (QSD_pre) reads no corpus index, only other queries'
> labels, which locates the real boundary: document-side language statistics, not pre/post-retrieval.
>
> That null is only interesting if the choice is predictable *at all*. It is: a selector over the
> channels' own score distributions beats the best fixed policy by +7.59 ± 1.01 nDCG on the same
> event-grouped folds. **So the null is a statement about the predictor family, not about the task.**
>
> The reason is structural. Variant selection compares competing texts, and how a text sits against a
> collection is exactly what corpus statistics measure. Source selection compares channels whose
> applicability is a property of the *document* (a silent clip has no speech to transcribe), and no
> query-side statistic can see that.

Three sentences, three sections. If the reader takes only the middle one they have story A, which is
why A cannot be cut and must not lead.

**What this buys us:** the headline no longer depends on our channels being competitive with MMMORRF.
A boundary condition on someone else's result survives "your retrieval is weak" in a way a +7.59
system claim does not. That was the reason for choosing B.

---

## Title candidates

1. *Query Performance Prediction Selects Queries, Not Sources*
2. *The Options Have to Differ Where the Predictor Can See: A Boundary on QPP-Based Selection*
3. *When Pre-Retrieval QPP Stops Working: Selecting Evidence Sources in Multimodal Video Retrieval*

(1) is the sharpest and the most falsifiable-sounding, which is a virtue. (3) is the safe one.

---

## Section plan (~6,300 words body)

### 1. Introduction (900 w)

1. QPP has been recast from *how hard is this query* to *which candidate should I run*, and it works:
   cheap pre-retrieval predictors are competitive at picking among LLM query variants.
2. The natural next question is what else that selects. Retrieval systems choose among more than
   rewritings. In multimodal video retrieval they choose among evidence channels on every query.
3. We run the same predictor families, implemented to the same reference definitions, on that choice.
   The result inverts: corpus-statistic prediction collapses, score-only post-retrieval carries it.
4. State the spine (above) in three sentences. This is the introduction's real work.
5. Contributions:
   - A measured boundary on QPP-based selection, with the per-channel-index repair that rules out the
     obvious artefact.
   - The positive control: k-way channel selection, +7.59 ± 1.01, permutation p = .0005, event-grouped
     and nested.
   - The structural account of *why* the family flips, located precisely by its two endpoints: QSD_pre
     escapes the null by needing no document statistics, and Clarity is undefined over frames by
     needing the most.
   - A calibration result with wider QPP implications: nested operating-point choice rescues the
     BERT-QPP cross-encoder and removes the apparent overall anti-correlation between τ and utility.
   - A leakage taxonomy for evaluating predictors-as-selectors, which their setting also needs.

Do not open with fused weighting. That was story A's opening.

### 2. Related work (800 w)

- **QPP as selection.** Arabzadeh et al. in full and fairly: 30 variants per need, 56 TREC-RAG topics,
  8 pre-retrieval + 12 post-retrieval predictors, IDF_max lifting nugget quality 0.273 → 0.398 ahead of
  NQC. Concede completely. The paper's whole value depends on the reader believing we take their result
  seriously, so the concession must be the most generous paragraph in the paper.
- **QPP families**, with the note that we implement against their reference repository so the
  comparison is same-vocabulary. iQPP and VQPP for the both-ρ-and-τ convention and for "no predictor is
  consistently best."
- **QPP-GenRE** as the accuracy-first alternative and its cost.
- **Multimodal video retrieval**, briefly. MultiVENT 2.0, MMMORRF, OmniEmbed. Under story B this is
  setting, not competition: say what the channels are and move on. Two or three sentences, not a
  paragraph defending our numbers.

### 3. Two instantiations of one selection problem (600 w)

The section that makes the comparison legitimate. Define selection abstractly: a set of options, a
predictor scoring each, a decision rule. Instantiate twice (options as query variants over one corpus,
options as evidence channels over one query) and name precisely what differs: **the option set varies
on the query side in one and on the document side in the other.** Everything in §5.3 is a consequence
of that sentence, so it is worth its own section rather than a paragraph in the method.

### 4. Setup and method (900 w)

- Channels and the 7-policy space. Weighted RRF, k = 60. Table 1: per-channel nDCG@10.
- Predictors evaluated: 11 corpus-statistic pre-retrieval, 10 score-only post-retrieval, plus supervised
  BERT-QPP and both halves of QSD, all to the QPP-4-RAG definitions. Analytic rows use an out-of-fold
  one-feature ridge; learned rows use group-disjoint nested operating-point calibration.
- The positive-control selector: 30 features (per-channel confidence shape + pairwise top-10/top-100
  overlap), multi-target RidgeCV, argmax. Say here that every feature reads scores the cascade has
  already computed, and give the 1.04 ms in one clause.
- Protocol: nested outer CV with the best fixed policy re-chosen per training fold. Event-grouped folds
  from union-find over shared relevant documents, 536 groups over 2,546 queries. Both cost us margin.
- MultiVENT 2.0 test, 2,546 queries, **109,724 videos** (109,488 carry speech; 98,805 carry on-screen
  text), nDCG@10, **2,000-sample** permutation tests. Both figures were wrong in an earlier version of
  this outline: 109,488 is the speech channel's document count and not the collection, and the
  permutation test is 2,000 samples, so the reported p = .0005 is its floor of 1/2001 rather than a
  measured value.

### 5. Results

#### 5.1 Corpus statistics do not select sources (700 w)

The null, then the repair, then the null again, then the exception. Eleven predictors at τ ≈ 0, routing
within 0.0005 of doing nothing. The obvious objection is that we built one index over transcripts, so every predictor returned
one number per query regardless of channel. So: an index per channel, meaning transcripts, on-screen text, and
the shipped captions as a **text surrogate** for the visual channel, which has no term index of its own.
Features do vary across the three (mean relative range 0.09 to 0.33), so the repair is a real repair.
It buys +0.43 ± 0.38 / +0.86 ± 0.52 / +0.81 ± 0.57, none distinguishable from routing nothing, and
+7.56 ± 0.89 when stacked on the score features, which is the baseline back again.

One sentence conceding the surrogate, plainly. The honest framing is that we handed pre-retrieval QPP a
proxy it does not normally get and it still did not help.

Power has to be addressed here, not in limitations: 2,546 queries against their 56 topics, and a
confidence interval that excludes anything of practical size. The null is bounded.

Then QSD_pre, which is the section's best paragraph. Fully nested (k and fraction both chosen on the
inner split) it beats fixed in the two speech cells and sits just under it in OCR
(0.3065 / 0.3466 / 0.3028), because it reads no corpus index at all: only the historical queries
nearest this one, and their known effectiveness. So the boundary is not the category label the
literature organises by. It is corpus-aggregate term statistics, worded so BERT-QPP's
document-reading success does not falsify it. QSD still trails the score cluster and the control, and
52% of what it has is duplicate detection (§5.5), so it sharpens the claim without denting it. The
matched QSD_post comparison stays mixed at a few thousandths of nDCG, ruling out consistent benefit
or harm from adding document text.

#### 5.2 The choice is predictable, from the other family (700 w)

The positive control. Table 2: visual .3036 / best uniform fusion .3408 / pairwise routing .3531 /
k-way selection **.4131**. Nested gap +7.59 ± 1.01, p = .0005; +5.64 ± 0.93 on shipped channels. 72%
single-channel picks. Score-only post-retrieval reaches |τ| ≈ 0.21 where pre-retrieval sits at zero.

Include the reading against us, and under story B it costs less than it did: **NQC ties or beats our
router on the binary escalate-or-not decision** (.3193 vs .3205). That is fine here. The paper's claim
is about which *family* transfers, and NQC is in the family that does. The k-way point stays (a scalar
cannot express a k-way policy) but it is no longer load-bearing.

Then the calibration correction. Three-epoch BERT-QPP cross has τ +.223 / +.210 / +.180 but still
fuses everything at its raw zero crossing. Fractions chosen on group-disjoint inner folds route to
.3204 / .3560 / .3042, the best learned QPP row. The bi-encoder remains below fixed. Recomputing all 78
rows weakens τ-versus-utility from −.213 to −.099 (p=.213), so the headline is calibration, not
anti-correlation.

#### 5.3 Why the family flips (600 w)

The structural argument from §3, now with evidence:
1. Channel applicability is a document property. A silent clip has no speech. In variant selection
   every option applies to every document and only quality varies.
2. That is what produces the gain spread the selector exploits: sd 23.1 against a mean of 3.72.
3. The limit case: Clarity needs a language model over retrieved documents. Over frames it is
   **undefined, not weak**, and we report it unavailable rather than substituting a number.

#### 5.4 Does the boundary matter downstream? (500 w)

Their utility gap says ranking and answer quality come apart, and NQC, the predictor that edges us on
the binary cell, correlates −0.038 with answer quality against 0.329 with nDCG in their setting.

We measured it three ways and it appears in two of them.

Nuggets, QPP-4-RAG nuggetizer, 395 queries, run under both grounding protocols and two assignment
checkpoints. Ground every policy on all text for the documents it retrieved and routing lifts vital
coverage 0.4648 → 0.5007 with 14B (p=.037) and 0.4381 → 0.4781 with 7B (p=.038). Ground each policy on
only the channels it selected and the same ranked lists give non-significant gains under both: +.0076
(p=.67) and +.0255 (p=.22), on an unchanged +7.4 nDCG lead. Exact assignment agreement is 71--72%,
κ=.53. Both checkpoints are Qwen2.5, so this is capacity/checkpoint rather than cross-family robustness.
**Selection that also narrows the generator's evidence hands the retrieval gain back**, because the
router picks a single channel for 73% of queries where the fixed policy always has two.

Recall: scoring the same decisions under R@100, the ASR-dense cell goes 0.7268 fixed / 0.7207 ours /
0.6286 oracle while nDCG goes 0.3408 / 0.3531 / 0.3910. **The better the nDCG selection, the worse the
recall**, with no generator involved at all.

So their gap reproduces twice over, and the section's job is to say what governs it in each case. The
recall half also obliges the paper to note, wherever it reports nDCG@10 alone, that this is the metric
the selection was fitted to.

We predicted the split for the wrong reason, and the paper should say so. `related_work_qpp.md` expected
it because the visual channel emits embeddings no generator can read while OCR emits usable text. The
asymmetry is real, but what produced the divergence is narrower evidence per query, not the visual
channel's illegibility, and we only found that by running both arms. Whether the NQC inversion holds
across modalities is then an open question we can pose but not settle.

#### 5.5 Robustness of the boundary (400 w)

- The null is not an artefact of weak channels: improving speech by translation raises the $k$-way
  routed gap, +7.59 → +8.01 ± 1.01. In the matched binary cell 3/11 corpus-statistic rows make small
  gains, versus 8/10 score-only rows and a much larger control gain.
- Second collection: MSR-VTT-1kA, two encoders × ASR/no-ASR × direct choice/escalation. Corpus-statistic
  rows 0/88. The direct task has 5.57--10.37 nDCG oracle headroom and the control clears fixed in 4/4.
  Score-only transfer is mixed (5/80), which bounds the generalisation claim.
- Leakage taxonomy: predictors consuming other queries' performance leak (QSD −52%, BERT-QPP −10 to
  −29% under event grouping); predictors reading only the current query's scores lose ≤ 2%. Their
  30-variants-per-need design shares relevant documents by construction, so this applies to them too.

### 6. Discussion and limitations (500 w)

The caption surrogate. Judge sensitivity for §5.4. Absolute nDCG below MMMORRF and OmniEmbed, which
under story B is a statement about setting, not a weakness in the claim, and should be phrased that
way. MSR-VTT supplies a within-video second collection, but the remaining generalisation question is
whether the boundary holds for non-video multi-source retrieval or is about modality specifically.

### 7. Conclusion (250 w)

The transferable finding: the predictor family that works depends on what the choice is *over*.
Options that differ on the query side are visible to query-collection statistics; options that differ on
the document side are not, and need evidence from retrieval itself.

---

## What story B demotes

Real work that loses most of its page space. Flagging it because it is a cost, not an oversight.

| material | under story A | under story B |
|---|---|---|
| Efficiency: latency, energy, p99 tail, AURC | own section + figure | one clause in §4 (router is 1.04 ms) |
| sd(gain) → achieved gap, ρ = 0.943 over 6 cells | RQ3, own figure | one sentence in §5.3, or cut |
| LLM-expansion tier declines itself; oracle audit | RQ3 companion | footnote or appendix |
| Per-language translation table | §5.4 support | §5.5, one sentence; the table goes to appendix |
| Multi-gold oracle ceilings are label noise | methodological contribution | appendix, or the companion in open item 4 |

Decided 31 Jul: the efficiency material becomes **a second paper**, not an appendix. Scope, the framing
problem (routing among equal-cost channels saves nothing, so paper 2 cannot be "routing saves compute"),
and the overlap to manage are in `paper2_scope.md`. This paper goes first and paper 2 cites it; the
+7.59 is a contribution here and prior work there.

The rows above still leave this paper thin in one place. §5.3 loses the heterogeneity correlation, which
was its only quantitative support for "the gain comes from variation." The sd 23.1 against mean 3.72
figure carries it alone now. Watch that section.

---

## Evidence map

| § | Claim | Number | Source |
|---|---|---|---|
| 2 | Their pre-retrieval result | IDF_max .273 → .398; NQC .381 | `related_work_qpp.md` (v1 PDF, recheck v2) |
| 4 | Per-channel effectiveness | visual .3036, ASR shipped .2666, OCR .1223, ASR dense .3134, +MT .3332 | `mv2_translate_findings.md` |
| 4 | Event grouping | 536 groups / 2,546 queries | `qpp_baselines.md`, `router_event_groups.py` |
| 5.1 | Pre-retrieval null, symmetric protocol | 0/33 clear margin; 0/33 survive Holm; 21/33 equivalent within .005 | `mv2_qpp_table_sym_grouped.json`, `mv2_row_inference.json` |
| 5.1 | Per-channel repair | +0.43 ± .38 / +0.86 ± .52 / +0.81 ± .57; stacked +7.56 ± .89 | `mv2_qpp_prechannel.py`, `.json` |
| 5.1 | Features do vary | mean relative range 0.09 (SCQ_max) – 0.33 (IDF_std) | `mv2_qpp_prechannel.json` |
| 5.1 | Weight sensitivity | pre 0-5/11 only at suboptimal w, post 8-10/10 at every w | `mv2_w_sensitivity_asr_{dense,shipped}.json` |
| 5.2 | Positive control | +7.59 ± 1.01, group sign-flip p < 5e-4 (pooled and nested); +5.64 ± 0.93 shipped | `mv2_channel_select_dense_m3_grouped.json` |
| 5.2 | Matched-learner family ratio | QPP features +0.81 best vs score features +7.59, same ridge and folds | `mv2_qpp_prechannel.json` |
| 5.2 | Policy table | .3036 / .3408 / .3522 / .4131 | one-pager |
| 5.2 | Post-retrieval works | 17/30 underlined, 8/30 survive Holm (NQC τ −0.215) | `mv2_table1_nested_grouped.md`, `mv2_row_inference.json` |
| 5.2 | Score cluster on binary | ours .3161 vs NQC .3144; ours .3522 vs σ_max .3532, BERT .3560 | `mv2_qpp_table_sym_grouped.json` |
| 5.2 | Matched QSD | post-minus-pre nDCG +.0039 / +.0020 / −.0014 | `mv2_qsd_pre_nestedk_grouped.json`, `mv2_qsd_post_5ep_nested_grouped.json` |
| 5.2 | Nested BERT-QPP cross | .3204 / .3560 / .3042, best post-retrieval row in all cells; bi below fixed | `mv2_bertqpp_cross_3ep_nested_grouped.json`, `mv2_bertqpp_bi_3ep_nested_grouped.json` |
| 5.2 | Clarity concession row | caption surrogate: .3008 / .3406 / .3036, τ ≈ 0 | `mv2_clarity_surrogate_grouped.json` |
| 5.2 | τ does not specify utility | all rows −.149, p=.054 descriptive; cheap-only −.057 | `mv2_qpp_utility_nested_sym_grouped.json` |
| 5.3 | Applicability, measured | flags alone +1.93 ± .69; gains +10.06 absent-gold vs +6.63; null persists on all-present strata 0/11 | `mv2_applicability.json` |
| 5.5 | Channel-strength ladder | fixed .3372/.3428/.3527/.3678 → gap +7.59/+8.01/+8.68/+7.41, band +7.4..+8.7; picks 72→86%; fusion harmful at top rung | `mv2_channel_select_rr_grouped.json`, `mv2_channel_select_plaidx_grouped.json` |
| 5.1 | Embedding probe | 4 query-embedding features + ridge: 0 underlines, best τ +0.098 | `mv2_embed_probe.json` |
| 5.2 | Single-channel picks | 72% (76% translated) | `mv2_translate_findings.md` |
| 5.3 | Gain spread | ASR-shipped -2.41 +/- 24.07, ASR-dense +3.72 +/- 23.12, OCR -5.92 +/- 19.42 | computed from the three cell JSONs, 2 Aug 2026 |
| 5.3 | Clarity undefined | n/a | `qpp_baselines.md` |
| 5.4 | Nugget coverage, retrieval isolated | vital .5007 vs .4648, p = .037; strict vital +.047, p = .014 | `metrics_n400_all.json` |
| 5.4 | Same, own evidence only | vital .4822 vs .4746, p = .67; strict vital +.020, p = .29 | `metrics_n400_own.json` |
| 5.4 | Second assignment checkpoint | all +.0399, p=.038; own +.0255, p=.22; agreement .712/.720, κ=.525/.535 | `mv2_rag_judge_comparison.json` |
| 5.4 | Single-channel picks on the RAG subset | 73% (asr 165, visual 123, ocr 4 of 395) | `evidence.md`; routed-policy assignments in the RAG artifacts |
| 5.4 | Their NQC inversion | −0.038 answer vs 0.329 nDCG | `related_work_qpp.md` |
| 5.5 | Better channel, wider gap | +7.59 → +8.01 ± 1.01, p = .0005 | `mv2_translate_findings.md` |
| 5.5 | Translated binary qualification | corpus 3/11 vs score-only 8/10; control .3920 vs fixed .3452 | `mv2_qpp_table_mt_grouped.json` |
| 5.5 | Second-collection replication | corpus 0/88; score-only 5/80; direct-task control 4/4 | `mv2_msrvtt_source_replication.json` |
| 5.5 | Leakage taxonomy | QSD −52%, BERT-QPP −10..−29%, analytic ≤ 2% | `qpp_baselines.md` |

Must not appear as headline claims: any "% of oracle captured" figure, and the 22.35 J tier-B number
from the original Q2E cost model.

## Figures

| # | Content | Status |
|---|---|---|
| 1 | The two instantiations of §3: variant selection vs source selection, side by side | to draw; this is the paper's picture |
| 2 | Predictor family × routed nDCG: pre-retrieval clustered at zero, post-retrieval spread, control at +7.59 | data exists |
| 3 | *(optional)* per-channel pre-retrieval feature variation, to show the repair is real | data exists |

Figure 1 is now the important one. Under story A the picture was the cascade; under story B it is the
structural difference, and if that diagram is good the paper is much easier to review.

## Open items

No locally runnable experiment in the submission checklist remains open. Remaining items require an
external decision, a new domain, or new source material:

1. **Authorship and affiliations.** Story B is a boundary condition argued on the JHU benchmark and
   engages Arabzadeh et al. directly. Settle before the abstract deadline.
2. **Anonymous artifact decision and hosting.** Both repositories are private and cannot be linked from
   the review submission as they stand.
3. **Beyond-video generalisation or a cross-family judge.** MSR-VTT supplies a second video collection.
   The second assignment checkpoint is from the same Qwen2.5 family. Neither closes those wider claims.
4. **Recheck Arabzadeh et al. against any v2** if one appears. Current values use the 24 Apr 2026 v1.
