# Choosing the right evidence per query: what QPP can and cannot see

A video can answer a search query through three kinds of evidence: what was said in it (speech
transcripts), what is written on screen, and what the frames show. Most systems search all three
and merge the results with fixed weights. We asked: can a system decide, for each individual
query, which evidence to trust, and can the standard tools of query performance prediction (QPP)
make that decision?

The answer has two parts. The decision is worth making: choosing evidence per query beats the best
fixed setup by 7.6 nDCG points on MultiVENT 2.0 (2,546 queries, 110K videos), a large gap in
retrieval terms. But the cheap predictors the QPP literature recommends for decisions like this
cannot see that opportunity. Every predictor that works reads the outcome of retrieval itself. Every predictor
that only reads the query against an index of term statistics fails, and we can bound how badly:
fed to the same learner under the same conditions, term-statistic features buy at most +0.86 nDCG
where retrieval outcomes buy +7.59.

This matters because a recent study (Arabzadeh et al., SIGIR 2026) showed the opposite in a
neighbouring setting: cheap term-statistic predictors picked well among LLM rewrites of a query.
The difference is where the options differ. Rewrites are different query texts, so a statistic
computed from the query can tell them apart. Evidence channels share one query and differ only in
the documents behind them, so to a query-side statistic every channel looks the same. We no longer
have to argue this from theory. We reran their task on our own collection, with their toolkit and
their pool size, and both halves came out: term statistics picked rewrites that trend better on
generated-answer quality, the direction they report, though not significantly at our sample size,
and picked nothing when the options were evidence sources. Corpus statistics show query-selection
signal and no source-selection signal, on one dataset under one protocol.

**Claim.** QPP-based selection has a boundary. Predictors built on corpus term statistics carry
roughly ten times less usable signal for choosing an evidence source than predictors that read
retrieval outcomes, and the pre-retrieval versus post-retrieval labels the field organises by do
not mark this line. This is an effectiveness claim: per-query selection buys retrieval quality on
the same channels. It saves no compute, and the costs we report later are deployment context, not
the contribution.

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

**Three named objections, tested.** Language mismatch: with the index rebuilt over
English-translated transcripts the family is 0 of 33 again, and the same holds on the 448 queries
whose relevant videos are all English. Caption quality: newly generated Qwen3.5-9B captions as the
document-side index change nothing on either collection (0 of 33 here, 0 of 88 on MSR-VTT, where
the caption is the entire document side). Query formulation: we generated the source study's full
pool, six reformulation methods, five samples each, 31 candidates per query, and asked the
predictors to pick among them. On the retrieval metric, term statistics select above the original
query 0 of 11 times while score-reading predictors do it 7 of 10. The channel-selection null also
holds separately inside every one of the seven formulation pools (77 tests, none pass). Fixed
alternatives lose too: concatenating all expansions into one query costs 3.1 points and fusing all
31 result lists costs 0.6. On the generation metric, where the source study's pre-retrieval result
actually lives, their pattern appears here: the variants that term statistics pick worsen ranking
yet sit above the original on every nugget metric and above both use-everything policies. The
paired tests keep it honest: that gain is directional, not significant, at 395 judged queries,
while the per-query oracle's generation gain is significant on all four metrics. Selecting
evidence channels still beats every one of these variant policies on both metrics. A related
check: channel choice is not language-driven. Across
queries asked in five languages, the best channel, the selection headroom, and the oracle's picks
barely move. What does move is the query itself: asking in the video's own language beats English
by 9 to 12 nDCG points on the speech channel, a per-query decision our framework could target next.

**What we are not claiming.** Our channels are deliberately cheap; the boundary claim does not need
competitive retrieval, and the channel-strengthening ladder is the check. We do not beat classical
QPP at its own binary game: our ridge, NQC and calibrated BERT-QPP sit within a point of one
another, and all of them read retrieval outcomes. The selector saves no compute, since all channels
are retrieved before it decides; the compute-saving version of the question is exactly the one the
cheap family fails. And an earlier headline from this project, that τ anti-correlates with
delivered value, did not survive a symmetric protocol and was retracted.

Full evidence and caveats: `reports/evidence.md`. The complete predictor table, all cells and
metrics: `results/ablations/mv2_table1_nested_grouped.md`.
