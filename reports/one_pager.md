# Choosing the right evidence per query: what QPP can and cannot see

A video can answer a search query through three kinds of evidence: what was said in it (speech
transcripts), what is written on screen, and what the frames show. Most systems search all three
and merge the results with fixed weights. We asked: can a system decide, for each individual
query, which evidence to trust, and can the standard tools of query performance prediction (QPP)
make that decision?

The answer has two parts. The decision is worth making: choosing evidence per query beats the best
fixed setup by 7.6 nDCG points on MultiVENT 2.0 (2,546 queries, 110K videos), a large gap in
retrieval terms. But the cheap predictors the QPP literature recommends for decisions like this
cannot see it. Every predictor that works reads the outcome of retrieval itself. Every predictor
that only reads the query against an index of term statistics fails, and we can bound how badly:
fed to the same learner under the same conditions, term-statistic features buy at most +0.86 nDCG
where retrieval outcomes buy +7.59.

This matters because a recent study (Arabzadeh et al., SIGIR 2026) showed the opposite in a
neighbouring setting: cheap term-statistic predictors picked well among LLM rewrites of a query.
The difference is where the options differ. Rewrites differ on the query side, which term
statistics can see. Evidence channels differ on the document side, which they cannot.

**Claim.** QPP-based selection has a boundary. Predictors built on corpus term statistics carry
roughly ten times less usable signal for choosing an evidence source than predictors that read
retrieval outcomes, and the pre-retrieval versus post-retrieval labels the field organises by do
not mark this line.

**The evidence, in brief.** Every predictor goes through the identical protocol: thresholds chosen
on held-out training data only, folds grouped by event so near-duplicate queries cannot leak
answers, significance tested with corrections. The eleven term-statistic predictors get 33 chances
to beat the best fixed policy and succeed zero times; 21 of the 33 outcomes are formally equivalent
to doing nothing. Giving them a dedicated index per channel does not rescue them. The
retrieval-outcome predictors clear the bar in 17 of 30 cases. The table shows the densest cell
(nDCG@10 against a fixed-policy baseline of 0.3408; τ is each predictor's rank correlation with the
true per-query gain):

| Method | nDCG@10 | τ |
|---|---|---|
| best fixed policy (no selection) | 0.3408 | -- |
| best of 11 term-statistic predictors | 0.3406 | +0.052 |
| QSD_pre (copies outcomes of similar past queries) | 0.3466 | +0.134 |
| clarity (given a caption index) | 0.3406 | +0.028 |
| NQC (retrieval score spread) | 0.3514 | −0.163 |
| BERT-QPP cross-encoder, calibrated | **0.3560** | +0.210 |
| BERT-QPP bi-encoder (the deployable one) | 0.3403 | −0.029 |
| our 30-feature score-distribution ridge | 0.3522 | +0.160 |
| oracle (true best choice per query) | 0.3910 | +1.000 |

(The best-of-11 row takes each column's own maximum: no single term-statistic predictor achieves
both.)

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

**What we are not claiming.** Our channels are deliberately cheap; the boundary claim does not need
competitive retrieval, and the channel-strengthening ladder is the check. We do not beat classical
QPP at its own binary game: our ridge, NQC and calibrated BERT-QPP sit within a point of one
another, and all of them read retrieval outcomes. The selector saves no compute, since all channels
are retrieved before it decides; the compute-saving version of the question is exactly the one the
cheap family fails. And an earlier headline from this project, that τ anti-correlates with
delivered value, did not survive a symmetric protocol and was retracted.

Full evidence and caveats: `reports/evidence.md`. The complete predictor table, all cells and
metrics: `results/ablations/mv2_table1_nested_grouped.md`.
