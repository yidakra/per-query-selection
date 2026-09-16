# Selection Needs Outcomes: Three Per-Query Choices in Multilingual Video Retrieval

Draft 2, 16 Sep 2026. All sections written. Target: ECIR 2027 short track, 6 pages plus references. Word budget follows
`paper_outline_short.md`. Numbers here are the committed ones in `reports/evidence.md`; every claim
should be checked against that file before submission. Anonymisation is not applied yet.

## Abstract

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

## 1 Introduction

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

## 2 Background

Query performance prediction estimates how well a system will do on a query. The literature divides
its predictors by what they are allowed to read. Pre-retrieval predictors read the query and corpus
term statistics only, which is why they are the family recommended whenever a decision has to be
made before retrieval runs: inverse document frequency and its variants, collection query
similarity, simplified clarity. Post-retrieval predictors read the score distribution a retrieval
already produced, such as weighted information gain, normalised query commitment and score
magnitude variance. Supervised predictors learn the mapping directly, with BERT-QPP the usual
reference. We keep this taxonomy explicit because our result cuts across it rather than along it.

The immediate motivation is Arabzadeh et al. (arXiv:2604.22661), who report that cheap
pre-retrieval predictors pick well among thirty large language model rewritings of a query. We treat
that as a replication target rather than as background. We reran their task with their toolkit and
their pool size and scored it their way, on generated-answer quality: their effect appears in our
data in their direction, and is not significant at our sample of judged queries, so we call it
consistent rather than confirmed. Our own question is whether the same predictors transfer to
ranking quality, which is the setting in which a deployed system would use them.

MultiVENT 2.0 supplies the collection: multilingual news video, with a speech transcript and
on-screen text per video, and queries in English against video whose spoken language is often not
English.

## 3 Setup

We evaluate on the MultiVENT 2.0 test set: 2,546 queries over 56 events, against multilingual news
video. Each video carries a speech transcript and its on-screen text, and we use dense retrieval
over each, plus the shipped sparse speech run, giving three text channels and their fusions.

Three option sets define the three decisions.

*Evidence channel.* Retrieve over the speech transcript, over the on-screen text, or over their
fusion. The default is the best fixed channel policy, chosen offline on the same data, which is the
strongest baseline a deployer could pick without per-query information.

*Query variant.* The user's own query plus thirty rewritings, six reformulation methods at five
samples each, generated with a self-hosted 7B instruction model at temperature 0.6 through a public
reformulation toolkit. The default is the user's original query.

*Query language.* The query asked in English, Chinese, Korean, Russian or Arabic, translated with a
public multilingual translation model, each against the speech channel in that language. The default
is asking in English.

Every predictor goes through one protocol. Each predictor is calibrated by a single-feature ridge
whose sign and scale are fitted out of fold, so a predictor is never penalised for pointing the
wrong way, and is then used to select: the option with the higher predicted gain is the one the
system runs. Thresholds are chosen on held-out training data only. Folds are grouped by event, so
near-duplicate phrasings of one event cannot sit on both sides of a boundary. Significance is a
group-level sign-flip test with Holm correction within each predictor family, and intervals are
cluster bootstrap. We state an equivalence bound of 0.005 nDCG in advance, so that a null is a
measurement rather than a failure to reject.

The decision metric is nDCG@10 after selection, the quantity a deployed system delivers. We also
report Kendall correlation between the raw predictor and the true per-query gain, because a
predictor can order queries well and still select badly, and the difference between those two is
part of what we find.

## 4 The three decisions

All three decisions are worth making, and one family of predictors makes none of them.

| Decided per query | Corpus term statistics | Model reading retrieval outcomes |
|---|---|---|
| Which evidence channel to search | 0 of 33 | +7.59 |
| Which rewriting of the query to run | 0 of 11 | +2.39 |
| Which language to ask in | 0 of 11 | +2.16 |

Gains are nDCG@10 against the default the decision replaces: the best fixed channel policy, the
user's original query, and asking in English. Every gain in the last column is significant under the
grouped tests with correction. The middle column counts how many of the individual term-statistic
predictors beat that same default, across 55 tests in total, and the answer is none of them
anywhere. Each of the eleven predictors is tested in three channel settings, which is where 33 of
the tests come from; the query-side decisions admit one setting each.

On the two query-side decisions the cheap family does worse than nothing. This is a property of
selection rather than of prediction. A system that selects has to commit to one option per query
rather than adjust a fraction of them, so a predictor that orders the options badly does not simply
fail to help, it spends the spread between the options in the wrong direction. The worst of these
predictors costs 7 nDCG points against keeping the default, which is larger than anything the good
predictors gain.

The null is a measurement, not an absence of evidence. Of the 33 channel outcomes, 21 fall inside
the stated equivalence bound of 0.005, meaning the predictor is demonstrably doing nothing rather
than doing something we cannot detect.

**The positive control.** The same machinery, on the same queries, under the same folds, converts
the channel decision: a ridge over cheap retrieval-outcome features returns 0.3522 against the best
fixed policy at 0.3408, and a calibrated cross-encoder reaches 0.3560. A predictor that misses is
therefore distinguishable from a decision that nothing predicts, which is what makes the null
readable at all.

**The null survives the obvious objections.** With the lexical index rebuilt over English-translated
transcripts the family is 0 of 33 again, and the same holds on the 448 queries whose relevant videos
are all English, which answers the language-mismatch objection. Newly generated captions from two
captioner sizes change nothing as the document-side index on either collection, and neither does a
union index holding each video's caption, transcript and on-screen text together, which answers the
caption-quality objection. The channel null holds separately inside each of seven query formulation
pools, 77 tests without a pass, which answers the objection that we tested one phrasing. A six-point
audit of the pipeline found no bug behind the zero. Strengthening the speech channel four ways
raises the fixed baseline from 0.337 to 0.368 while the selection gap stays between 7.4 and 8.7
points throughout, so the result is not an artefact of a weak baseline.

## 5 Where query-only information works

The obvious modern objection is that none of this matters because a language model can read the
query and choose. We ran that baseline: a self-hosted instruction model, temperature 0, one
constrained prompt per decision, scored on the existing runs exactly like every other selector, with
no labels and no extra retrieval.

It converts the language decision and fails the channel decision. On language it gains 2.99 nDCG,
above the outcome-reading ridge's 2.16, at zero supervision cost. On channel it loses 9.90. Given a
three-way prompt it answers "search everything" on 2,535 of 2,546 queries, which is simply the fixed
fusion policy under another name. Forced to choose one channel with no third option it loses 11.05,
and its picks are right at the base rate of each channel being the better one, so they carry no
information about which channel holds the answer. Its agreement with the two-channel oracle is
42.7%, against 87.1% for the rule "always pick speech".

The pattern locates the boundary more precisely than our own selectors do. Query-only information
converts a decision exactly when the query carries the attribute the decision is about. A query
about Taipei politics says that the relevant coverage will be in Chinese, and the model reads that.
Nothing in the words of a query says whether the answer was spoken aloud or written on screen, and
nothing in them says which rewriting will rank best. This also rewrites the cost of the language
decision: the supervision and extra retrieval that the outcome-reading selector needs are not
intrinsic to the decision, they are the price of learning it from labels, and a model that reads the
query pays neither and gains more.

## 6 What an oracle cannot tell you

Work of this kind normally reports a per-query oracle as headroom. We report none, and this section
is why.

An oracle picks each query's best option using the same relevance labels that then grade the pick,
so it rewards label luck as well as real advantage. We measured how much. Splitting each query's
relevant documents in half, choosing on one half and grading on the other, removes 15.2 points from
the channel oracle, 12.5 from the variant oracle and 10.8 from the language oracle. In each case
that is the entire oracle: what survives runs from half a point below the default to two thirds of a
point above it. With about five relevant videos per query a half holds two or three, and an argmax
over up to thirty-one options scored on two or three labels reads label placement rather than option
quality.

This removes a headroom estimate, not a decision. The selectors are graded on the full labels and
never read a label when they decide, so their gains are unaffected and still clear the significance
tests. What disappears is the denominator that a percentage-of-oracle figure would need.

A second collection shows the same thing from the other side. On a seven-pool video retrieval
benchmark with two finished first-stage systems, choosing between them per query shows 1.5 to 7.2
points of apparent headroom, and nothing converts it: corpus statistics 0 of 77, score-based
predictors 0 of 70, and the outcome-reading learner that converts the channel decision here fails on
all seven pools. Those judgments are single-gold, so the audit above cannot even be run, and between
41% and 88% of queries are ties. A large apparent oracle that no predictor reaches is what label
luck looks like when it cannot be measured directly.

## 7 Conclusion

Three per-query decisions on a multilingual video collection are each worth making, and the cheap
pre-retrieval predictors that the literature recommends for exactly this job convert none of them.
Predictors that read a retrieval outcome convert all three. The separating line is not the
pre-retrieval and post-retrieval split that the field organises by, and not whether the options
differ as queries or as documents, since the cheap family fails on the query-side decisions too. It
is whether the predictor sees an outcome. The one exception locates the same boundary from outside:
a language model reading the query alone converts the one decision whose answer the query carries.

Two things would move this result. A cheap predictor that reads something outcome-like without
paying for retrieval would break the boundary as stated. A collection whose judgments are dense
enough to estimate a per-query oracle would tell us how much of each decision is really there.

## Checks before submission

- Anonymise: replace the repository link with an anonymised artefact link.
- Every number re-checked against `reports/evidence.md`.
- Confirm the equivalence bound and the test descriptions match the protocol section exactly.
- LNCS template, 6 pages, references unlimited.
