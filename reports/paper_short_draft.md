# Selection Needs Outcomes: Three Per-Query Choices in Multilingual Video Retrieval

Draft 1, 16 Sep 2026. Target: ECIR 2027 short track, 6 pages plus references. Word budget follows
`paper_outline_short.md`. Numbers here are the committed ones in `reports/evidence.md`; every claim
should be checked against that file before submission. Anonymisation is not applied yet.

## Abstract (target 150 to 200 words)

Retrieval systems fix most of their choices once and apply them to every query. On a multilingual
video collection at least three of those choices could be made per query instead: which evidence
channel to search, which rewriting of the query to run, and which language to ask in. We measure
what each decision is worth on MultiVENT 2.0 and ask whether query performance prediction can make
it. All three decisions pay. A selector that reads retrieval outcomes gains 7.6, 2.4 and 2.2 nDCG@10
over the default each decision replaces, all significant under event-grouped tests with correction.
The pre-retrieval predictors built from corpus term statistics, the family the literature recommends
for exactly this job, convert none of the three across 55 tests, and on the two query-side decisions
they lose up to 7 points against simply keeping the default. The null survives translated indexes,
regenerated captions, an English-only subset, seven query formulations and a second collection. A
language model reading only the query text recovers the language decision and fails the channel
decision. What separates the predictors that work from those that fail is not the pre-retrieval and
post-retrieval split the field organises by. It is whether the predictor sees an outcome.

## 1 Introduction (target 480 words)

A retrieval system settles the same questions for every query it answers, and usually freezes one
global answer into a configuration. On a multilingual video collection there are at least three such
questions. Which evidence should be searched: what was said in the video, or what is written on
screen? Which phrasing of the query should be run, the user's own or one of its rewritings? And when
the videos are not in the language of the query, which language should the system ask in?

Each could be decided per query instead, and query performance prediction is the obvious machinery
for deciding. The pre-retrieval family in particular is what the literature recommends for this job,
because it costs nothing at query time: term statistics over the corpus, computed before any
retrieval runs. A recent study reports that these cheap predictors select well among rewritings of a
query when the rewritings are judged by generated-answer quality.

We ask whether that transfers to ranking, on three decisions of different kinds, under one protocol.
It does not. The corpus-statistic family converts none of the three decisions in 55 tests, and on
the two query-side decisions it actively hurts. Predictors that read what retrieval returned convert
all three. The line between the two families is not the pre-retrieval versus post-retrieval split
the field organises by, and it is not whether the options differ as queries or as documents, since
the cheap family fails on the query-side decisions too. It is whether the predictor gets to see an
outcome.

Contributions. (i) We show that three per-query decisions on a multilingual video collection are
each worth making, and measure what an outcome-reading selector returns on each. (ii) We show that
the cheap corpus-statistic family converts none of them, with the null bounded by stated equivalence
intervals rather than by a failure to reject, and surviving seven controls including a second
collection. (iii) We show where query-only information does work: a language model reading only the
query text converts the language decision and fails the channel decision, which locates the boundary
at whether the query carries the attribute the decision is about. (iv) We show that the per-query
oracles routinely reported as headroom for this kind of work are, on this collection, almost
entirely label luck, and we report none.

## 2 Background (target 240 words)

TODO. Three paragraphs.

Paragraph 1: QPP predictor families. Pre-retrieval from corpus term statistics (IDF, ICTF, SCQ,
SCS), post-retrieval from the score distribution (WIG, NQC, SMV, sigma), and supervised predictors
(BERT-QPP). Frame the usual taxonomy explicitly, because the result cuts across it.

Paragraph 2: the source study. Cheap predictors selecting among LLM rewritings, judged by
generated-answer quality. State plainly that we reran their task with their toolkit and pool size,
and that their effect appears in our data in their direction but is not significant at our sample.
Replication framing, not motivation framing.

Paragraph 3: multimodal video retrieval and MultiVENT 2.0. Enough to make the channels concrete.

## 3 Setup (target 430 words)

TODO. Collection: MultiVENT 2.0, 2,546 test queries over 56 topics, multilingual video with
speech transcripts and on-screen text.

The three option sets:
- Channel: shipped speech, dense speech, on-screen text, and their fusions.
- Query variant: the original query plus 30 rewritings, six methods at five samples, generated with
  a self-hosted 7B instruction model at temperature 0.6.
- Language: the query asked in English, Chinese, Korean, Russian or Arabic, translated with NLLB.

Protocol, four sentences: thresholds are chosen on held-out training data only. Folds are grouped by
event, so near-duplicate phrasings of one event cannot sit on both sides of a boundary.
Significance is a group sign-flip test with Holm correction within family. The equivalence bound of
0.005 is stated in advance, so that no effect is a measurement rather than a failure to reject.

## 4 The three decisions (target 600 words plus table)

TODO. Lead with the table, then three short paragraphs: what the table says, why a bad ordering
costs more than no decision, and the confound paragraph.

| Decided per query | Corpus term statistics | Model reading retrieval outcomes |
|---|---|---|
| Which evidence channel to search | 0 of 33 | +7.59 |
| Which rewriting of the query to run | 0 of 11 | +2.39 |
| Which language to ask in | 0 of 11 | +2.16 |

Gains are nDCG@10 against the default the decision replaces: the best fixed channel policy, the
user's original query, and asking in English. The middle column counts how many individual
term-statistic predictors beat that same default.

Paragraph on the cost of a bad ordering: the system commits to one option rather than adjusting a
fraction of queries, so a predictor that orders the options badly loses much of the spread between
them, and the worst costs 7 nDCG points against keeping the default.

Positive control paragraph: the same machinery converts the channel decision, so a predictor that
misses is distinguishable from a decision that nothing predicts. This is the load-bearing sentence
of the whole paper and it must not be cut for space.

Confound paragraph, one clause each: translated index, English-only subset, two captioner sizes, the
union index holding caption and transcript and on-screen text together, seven query formulations, a
second collection, and a six-point pipeline audit. Cite the artefact repository for the full
accounting.

## 5 Where query-only information works (target 340 words)

TODO. The router experiment and the sentence it buys.

A language model reading only the query text, no labels and no extra retrieval, converts the
language decision at +3.0 nDCG and fails the channel decision at -9.9. Given a three-way prompt it
answers "search everything" on 2,535 of 2,546 queries, which is the fixed-fusion policy. Forced to
choose one channel with no third option it loses 11.1, and its picks are right at the base rate of
each channel being better, so they carry no information about which channel holds the answer.

The sentence: query-only information converts a decision exactly when the query carries the
attribute the decision is about. The language of the relevant coverage is in the query. Which
evidence channel holds the answer is not, and neither is which rewriting will rank best.

## 6 What an oracle cannot tell you (target 290 words)

TODO. The label-splitting audit and why no oracle appears anywhere above.

Per-query oracles are the usual way to report headroom for this kind of work. Picking each query's
best option with the same relevance labels that then grade the pick rewards label luck as well as
real advantage. We measured how much: choosing on half of each query's labels and grading on the
other half removes 15.2 points from the channel oracle, 12.5 from the variant oracle and 10.8 from
the language oracle, which in each case is the whole thing. What survives runs from half a point
below the default to two thirds of a point above it.

With about five relevant videos per query, a half holds two or three, and an argmax over up to 31
options scored on two or three labels is reading label placement rather than option quality. We
therefore report no oracle and no percentage of oracle captured anywhere in this paper. The
selectors are unaffected, because they are graded on the full labels and never read one when they
decide.

One sentence on the second collection: on a benchmark whose judgments have a single relevant video
per query, a system-selection decision with seven points of apparent headroom is converted by
nothing we tried, including the outcome-reading learner that converts the channel decision here,
which is the same symptom seen from the other side.

## 7 Conclusion (target 145 words)

TODO. Restate the boundary in two sentences. One sentence on what would move it: a cheap predictor
that reads something outcome-like without paying for retrieval, or a collection whose judgments are
dense enough to estimate a per-query oracle.

## Checks before submission

- Anonymise: replace the repository link with an anonymised artefact link.
- Every number re-checked against `reports/evidence.md`.
- Confirm the equivalence bound and the test descriptions match the protocol section exactly.
- LNCS template, 6 pages, references unlimited.
