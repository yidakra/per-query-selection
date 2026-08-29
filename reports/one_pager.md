# Three choices worth making per query, and what cheap prediction cannot see

A retrieval system settles the same questions for every query it answers, usually by freezing one
global answer into a configuration file. On a multilingual video collection there are at least three
such questions. Which evidence should be searched: what was said in the video, what is written on
screen, or what the frames show? Which phrasing of the query should be run, the user's own or one of
its rewritings? And when the videos are not in the language of the query, which language should the
system ask in? Each of these can instead be decided per query. We measured what the three decisions
are worth, and whether the standard tools of query performance prediction (QPP) can make them.

All three are worth making. One family of predictors makes none of them.

| Decided per query | Best possible choice is worth | Corpus term statistics | Model reading retrieval outcomes |
|---|---|---|---|
| Which evidence channel to search | see the caveat below | 0 of 33 | **+7.59** |
| Which rewriting of the query to run | +12.4 | 0 of 11 | **+2.39** |
| Which language to ask in | +10.4 | 0 of 11 | **+2.16** |

Gains are nDCG@10 against the default the decision replaces: the best fixed channel policy, the
user's original query, and asking in English. Every gain in the last column is significant under
group-level tests with corrections. The middle column counts how many of the individual
term-statistic predictors beat that same default, across 55 tests in total, and the answer is none
of them anywhere. On the two query-side decisions they do worse than that. Because the system has to
commit to one option rather than adjust a fraction of queries, a predictor that orders the options
badly loses much of the spread between them, and the worst of these predictors costs 7 nDCG points
against simply keeping the default.

This is the result the project now rests on. Choosing per query is worth having, on three different
kinds of choice. Predictors built from corpus term statistics, the cheap pre-retrieval family the
QPP literature recommends for exactly this job, convert none of the three. Predictors that read what
retrieval actually returned convert all three. The line that separates the two families is not the
pre-retrieval versus post-retrieval split the field organises by, and it is not whether the options
differ as queries or as documents, since the query-side decisions fail for the cheap family too. It
is whether the predictor gets to see an outcome.

There is one exception, and it belongs to the study that motivated this work. Arabzadeh et al.
(SIGIR 2026) report that cheap term-statistic predictors pick well among LLM rewritings of a query
when the rewritings are judged by the quality of the answer generated from them, rather than by
ranking. We reran their task here, with their toolkit and their pool size, and scored it their way.
Their effect appears in our data too, in their direction: the rewritings our term statistics pick
produce better nugget-scored answers than the original query, while ranking worse. That gain is not
significant at our 395 judged queries, so we call it consistent rather than confirmed, and note that
their own 56 information needs could not have separated these outcomes either.

**Claim.** Cheap corpus-statistic QPP does not convert per-query selection decisions. We tested
three decisions of different kinds on one collection under one protocol, and the family fails on all
three when the measure is ranking quality. The signal it does carry, in the setting the source study
identified, is about generated-answer quality on query rewritings, and is directional here. What
works instead is reading retrieval outcomes, which converts every one of the three decisions. This
is an effectiveness claim: selection buys retrieval quality on the same channels. It saves no
compute, and the costs reported later are deployment context, not the contribution.

**The evidence, in brief.** Every predictor goes through the identical protocol: thresholds chosen
on held-out training data only, folds grouped by event so near-duplicate queries cannot leak
answers, significance tested with corrections. Each of the eleven term-statistic predictors is tested in three settings (one per text channel:
shipped speech, dense speech, on-screen text), giving 33 chances to beat the best fixed policy. They
succeed zero times; 21 of the 33 outcomes are formally equivalent
to doing nothing. Giving them a dedicated index per channel does not rescue them: we built one
index per channel, using the shipped video captions as the document-side index for the visual
channel, and the repaired features buy at most +0.86 nDCG against +7.59 from retrieval outcomes
under the same learner (stacking both gives +7.56, nothing added). The
retrieval-outcome predictors clear the bar in 17 of 30 cases. The table shows the setting where escalation pays most (dense speech retrieval),
laid out like the source study's own results table. Underline means the predictor beat the fixed
policy by the source study's margin, bold is best in its block, τ is the predictor's rank correlation with the true
per-query gain, R@100 is recall, and N_all / N_strict are answer-quality (nugget coverage) scores
from a separate answer-generation evaluation. The other two cells are in
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


Three details carry the story. The one "pre-retrieval" predictor above the baseline, QSD_pre, reads
no index at all: it copies the outcomes of similar previously-seen queries, so it works exactly as
far as it smuggles in outcome information (its margins are consistent but do not survive the
significance correction). BERT-QPP's cross-encoder shows that a good ordering is not a decision:
its raw scores select nothing, and threshold calibration on training data is what turns the best τ
in the table into the best selector. And clarity, handed the same caption stand-in every other
predictor got, still carries no signal.

**The result holds up under attack.** Strengthening the speech channel four separate ways (ending
with the retriever family the benchmark's best system uses), and two channels at once in a fifth
test, raises the fixed baseline from 0.337 to 0.368 while the selection gap stays between +7.4 and
+8.7 throughout. On a second collection the term-statistic family scores 0 for 88. Part of the
mechanism is measurable: 10% of videos have no on-screen text at all, a fact no query-side
statistic can know, and a bare channel-availability feature recovers about a quarter of the
selection gap. But absence does not explain the family's failure, which is just as sharp where
every channel exists.

**The objections we were given, tested.** Language mismatch, raised because the queries are English
and most videos are not: with the lexical index rebuilt over English-translated transcripts the
family is 0 of 33 again, and the same holds on the 448 queries whose relevant videos are all
English. Caption quality, raised because the shipped captions are old and weak: newly generated
Qwen3.5-9B captions as the document-side index change nothing on either collection (0 of 33 here, 0
of 88 on MSR-VTT, where the caption is the entire document side), and neither does a union index
holding each video's caption, transcript and on-screen text together. Query formulation, raised
because we had tested only one phrasing: the channel-selection null holds separately inside each of
seven formulation pools, 77 tests without a pass. Fixed alternatives to selecting lose as well.
Concatenating every expansion into one query costs 3.1 points against the original, and fusing all
31 candidates' result lists costs 0.6. A related check answers the question of whether routing
should depend on language: it should not. Across queries asked in five languages the best channel,
the headroom, and the oracle's picks barely move, so language carries no channel-routing signal even
though choosing the language itself is worth 10.4 points.

**What we are not claiming.** The oracle columns are upper bounds, not targets. Picking each query's
best option with the same relevance labels that then grade the pick rewards label luck as well as
real advantage, and we measured how much: for the channel decision, choosing on half of each query's
labels and grading on the other half removes 15 of the oracle's 16 points, which is why that row
carries no number and why we never report a percentage of oracle captured. The two query-side
oracles are optimistic for the same reason and are quoted only to show that a decision exists to be
made. Our channels are deliberately cheap; the boundary claim does not need
competitive retrieval, and the channel-strengthening ladder is the check. We do not beat classical
QPP at its own binary game: our ridge, NQC and calibrated BERT-QPP sit within a point of one
another, and all of them read retrieval outcomes. The selector saves no compute, since all channels
are retrieved before it decides; the compute-saving version of the question is exactly the one the
cheap family fails. And an earlier headline from this project, that τ anti-correlates with
delivered value, did not survive a symmetric protocol and was retracted.

Full evidence and caveats: `reports/evidence.md`. The complete predictor table, all cells and
metrics: `results/ablations/mv2_table1_nested_grouped.md`.
