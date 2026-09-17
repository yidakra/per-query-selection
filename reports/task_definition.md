# Task definition and evaluation benchmark

What this project evaluates, on which data, with which judgments and under which protocol. It is
written to be read on its own, without the evidence file. Every number here is reproducible from the
committed artifacts; `reports/evidence.md` is the source of record for results.

## What the benchmark measures

Retrieval systems normally settle a configuration question once and apply the answer to every query.
This benchmark measures what happens when the same question is settled per query instead. A task
here is therefore not "rank these documents" but: **given a set of options and a per-query predictor,
choose one option for each query, run it, and report the retrieval quality the user receives.**

That framing has two consequences that shape everything else. The system must commit to one option
per query rather than blend them, so a predictor that orders options badly loses the spread between
them rather than merely failing to gain. And the quantity reported is always delivered nDCG@10 after
selection, never the predictor's correlation with anything, because a predictor can order queries
well and still select badly.

## Decision tasks

| Task | Options per query | Default it must beat | Collection |
|---|---|---|---|
| Evidence channel | Speech transcript, on-screen text, and their fusion | Best fixed channel policy, chosen offline | MultiVENT 2.0 |
| Query variant | The user's query plus 30 rewritings, 31 in total | The user's original query | MultiVENT 2.0 |
| Query language | English, Chinese, Korean, Russian, Arabic | Asking in English | MultiVENT 2.0 |
| Retrieval system | Two finished first-stage systems | The better system overall | MVEB, 7 pools |

The default matters as much as the option set. Each default is the strongest choice a deployer could
make without per-query information, so a selector is credited only for what per-query knowledge adds,
not for beating a weak baseline.

## Collections and judgments

**MultiVENT 2.0** is the primary collection: multilingual news video, 2,546 test queries over 56
topics. Each video carries a speech transcript and its on-screen text. Judgments are graded 0 to 3,
covering 4,222 documents of which 3,473 are relevant, so a query has about five relevant videos.
Judgments also carry metadata used only in the error analysis: the relevant video's language, the
event type, the production style, and which modality the relevance was drawn from.

**MSR-VTT-1kA** is the second collection for the predictor null, where the caption is the entire
document side.

**MVEB** supplies seven pools for the system-selection task: AudioCaps, ActivityNet Captions,
DiDeMo, MSR-VTT, VATEX, and VGGSound twice, once with video-caption queries and once with
audio-caption queries. Pools range from 665 to 4,884 queries. Judgments are single-gold, one relevant
video per query, which is why the label-splitting audit cannot be run there.

## Query sets

The MultiVENT queries are the benchmark's own. The variant pool is generated: six reformulation
methods at five samples each, produced with a self-hosted 7B instruction model at temperature 0.6
through a public reformulation toolkit, giving 31 candidates per query including the original. The
language variants are machine translations of each query into four further languages. Generation
settings are pre-registered in `results/ablations/mv2_variant_prereg.md` with the outcome readings
fixed before the experiment ran.

## Evaluation protocol

The same protocol applies to every predictor and every task.

- **Metric.** nDCG@10 after selection.
- **Calibration.** Each predictor is turned into a selector by a single-feature ridge whose sign and
  scale are fitted out of fold, so no predictor is penalised for pointing the wrong way.
- **Folds.** Grouped by event, so near-duplicate phrasings of one event cannot sit on both sides of a
  fold boundary. MSR-VTT and MVEB have one phrasing per need and use shuffled five-fold instead.
- **Significance.** Group-level sign-flip test, Holm corrected within each predictor family.
- **Intervals.** Cluster bootstrap over event groups.
- **Equivalence.** A bound of 0.005 nDCG stated in advance, so a null result is a measurement rather
  than a failure to reject.
- **Positive control.** Every null is reported beside a selector that converts the same decision
  under the same folds. Without it, a predictor that misses cannot be distinguished from a decision
  that nothing predicts.

## Predictor families under test

Eleven pre-retrieval predictors from corpus term statistics, ten score-only post-retrieval
predictors, a clarity variant over document text, a supervised cross-encoder and bi-encoder, and a
ridge over cheap retrieval-outcome features. A zero-shot language model reading only the query text
is evaluated as a further selector on the channel and language tasks.

## Artifacts and formats

Ranked lists are JSON objects mapping a query id to a mapping of document id to score. Per-task
results, including every predictor's routed score, its correlation and its test outcome, are the
JSON files under `results/ablations/`, each with a markdown twin for reading. Large inputs are not
version controlled; `README.md` records how to rebuild them.

## Reproduction

Each result names its own entry point in `reports/evidence.md`. The main table is
`mv2_qpp_table.py`, selection per axis is `mv2_channel_select.py`, `mv2_variant_selection.py` and
`mv2_language_selection.py`, the oracle audit is `mv2_axis_goldsplit.py` and
`mv2_variant_goldsplit.py`, and the error analysis is `mv2_error_analysis.py`.

## What is deliberately not measured

Efficiency. Selection as defined here retrieves every option before choosing, so it buys retrieval
quality and saves no compute. Cost figures appear in the evidence file as deployment context and are
not part of any claim.
