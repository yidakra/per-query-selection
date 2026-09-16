# ECIR 2027 short paper: outline, page budget, and what is cut

Track decided 16 Sep 2026 after supervision approved the story ("the overall results look good to
me, go ahead") and advised the short track on the grounds that we propose no new method. The full
paper outline in `paper_outline.md` is kept for a later extended version and is no longer the plan.

## Configuration, checked against the call on 16 Sep 2026

| | |
|---|---|
| Length | 6 pages, plus additional pages for references |
| Anonymity | fully anonymised, double-anonymous review |
| Abstract due | 5 Oct 2026 |
| Paper due | 12 Oct 2026 |
| Notification | 7 Dec 2026 |
| Submission | EasyChair |

Six pages is the binding constraint. The study has six experimental blocks and room for about two,
so most of this document is the cut list rather than the contents.

## The spine

Title: *Selection Needs Outcomes: Three Per-Query Choices in Multilingual Video Retrieval*.

One claim, stated once and defended: **a retrieval system can make three different per-query choices
that are each worth making, and the cheap pre-retrieval predictors the QPP literature recommends for
exactly this job convert none of them, while predictors that read a retrieval outcome convert all
three.** The line that separates the two families is not the pre-retrieval versus post-retrieval
split the field organises by, and not whether the options differ as queries or as documents. It is
whether the predictor sees an outcome.

A short paper has to earn its claim on one pass. The argument is therefore: the decisions exist
(Section 3), the cheap family fails on all three (Section 4), the failure is not an artefact of the
obvious confounds (Section 4, one paragraph), and the boundary is about outcomes rather than the
usual taxonomy (Section 5, with the language model router as the sharpest case).

## Page budget

| Section | Pages | Carries |
|---|---|---|
| 1 Introduction | 1.0 | The three choices, the claim, the contribution list |
| 2 Background | 0.5 | QPP predictor families, the source study, video retrieval setup |
| 3 Setup | 0.9 | MultiVENT 2.0, the channels, the three option sets, the protocol |
| 4 The three decisions | 1.9 | Headline table, the confound paragraph, the positive control |
| 5 Where query-only information works | 0.7 | The router split, and the mechanism sentence |
| 6 What an oracle cannot tell you | 0.6 | The label-splitting audit, and why no oracle is reported |
| 7 Conclusion | 0.3 | The boundary restated, one sentence on what would move it |
| Total | 5.9 | |

## What goes in

**The headline table.** Three rows, one per decision, two columns: how many individual
corpus-statistic predictors beat the default (0 of 33, 0 of 11, 0 of 11) and what an
outcome-reading selector returns (+7.59, +2.39, +2.16 nDCG@10). Every gain in the last column
significant under event-grouped tests with Holm correction. No oracle column, for the reason in
Section 6.

**The protocol, in four lines.** Thresholds chosen on held-out training data only, folds grouped by
event so near-duplicate phrasings cannot leak, group sign-flip tests with Holm within family,
cluster-bootstrap intervals, and a stated equivalence bound of 0.005 so that "no effect" is a
measurement rather than a failure to reject.

**The confound paragraph.** One paragraph, seven controls, one clause each: translated index,
English-only subset, two captioner sizes, the union index, seven query formulations, a second
collection, and a pipeline audit. The full accounting goes in the repository and is cited as such.

**The positive control.** The same machinery converts the channel decision, so a predictor that
misses is distinguishable from a decision nothing can predict. This is the single most important
defence of a null and it is not negotiable for space.

**The router split.** A language model reading only the query text converts the language decision
(+3.0, no labels, no extra retrieval) and fails the channel decision (-9.9, and -11.1 when forced to
choose one channel, with its picks right only at the base rate). Query-only information converts a
decision exactly when the query carries the attribute the decision is about.

**The oracle audit.** Picking on half of each query's relevance labels and grading on the other half
removes 15.2, 12.5 and 10.8 points from the three oracles, which in each case is the whole thing.
Hence no oracle and no percentage-of-oracle anywhere in the paper. This doubles as the answer to a
reviewer asking why the headroom is not reported.

## What is cut, and why

Each of these is finished, committed, and reachable from `reports/evidence.md`. None of them is
needed to establish the claim, and a short paper that gestures at all of them establishes nothing.

| Cut | Why it survives being cut |
|---|---|
| Depth ablation | Refines how much of a ranked list a score predictor needs; does not bear on the boundary |
| Error-tolerance curves | Orthogonal to the claim. The break-even agreement rate is a deployment question |
| Fixed and query-conditioned fusion | Shows selection beats fusion, and the claim does not depend on beating fusion |
| Axis composition | Channel and language gains add, which is a second claim we cannot afford |
| Generation-side nuggets | Directional only, and it is the source study's setting rather than ours |
| Supervision cost curves | Deployment context, which the claim explicitly disclaims |
| Video-embedding index | One sentence in the confound paragraph, no more |
| MVEB | Its positive control fails, so it cannot corroborate the boundary claim. One sentence in Section 6, as a second case of an oracle nothing converts |

## Open items before drafting

1. EasyChair account and submission record. Author's task, not started.
2. Anonymisation pass: the repository link has to become an anonymised artefact link.
3. MVEB query text and the five missing id lists, requested from supervision on 16 Sep. Only affects
   the Section 6 sentence, so drafting is not blocked on the reply.
4. Decide whether Section 2 cites the source study as replication or as motivation. Replication is
   more honest and costs a sentence.
