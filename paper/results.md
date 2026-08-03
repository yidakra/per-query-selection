# 5. Results

Draft for the ECIR 2027 submission. Section numbering follows `reports/paper_outline.md`. Every number
here is traceable to `reports/evidence.md` and the generated artifacts under `results/ablations/`.

All results are on the MultiVENT 2.0 test collection: 2,546 queries, 109,724 videos, graded multi-gold
judgments, nDCG@10 unless stated. Folds are grouped by event (536 groups, union-find over shared
relevant documents), and the best fixed policy is re-chosen on each training fold, so every comparison
is nested. Significance is by 10,000-sample permutation.

## 5.1 Corpus statistics do not select sources

Eleven pre-retrieval predictors from the reference suite — IDF in four aggregations, ICTF, SCQ in three,
SCS in two, and QL — produce no usable routing decision on any of our three channel pairs. Across the
33 predictor-cell combinations, not one beats the fixed policy by the 5 × 10⁻⁴ margin the source table
uses for its underline notation. The largest margin anywhere in the block is +0.0003.

Correlations tell the same story from the other side: every one of the 33 sits at |τ| < 0.08, against
|τ| ≈ 0.21 for the score-only post-retrieval family on identical folds.

The escalation fractions are blunter than either. A corpus-statistic predictor here does not choose
badly; it does not choose. In the OCR cell all eleven escalate exactly 0% of queries. In ASR-dense six of
eleven escalate exactly 100%, landing precisely on uniform fusion. Counting only exact 0 and exact 1,
20 of 33 corpus-statistic cells are degenerate, against 2 of 30 score-only cells. The predictor emits a
number, the number never crosses zero, and the system does the same thing 2,546 times.

**The obvious objection, and the repair that fails to rescue it.** A reader will say we built one index,
over speech transcripts, so every predictor returned one number per query no matter which channel was
being considered. Under that reading the null is an artefact of our setup rather than a property of the
predictors.

So we built an index per channel: speech transcripts, on-screen text, and the benchmark's shipped
captions as a text surrogate for the visual channel, which searches frame embeddings and has no term
index of its own. The features do move. Mean relative range across the three indices runs from 0.09
(SCQ_max) to 0.33 (IDF_std), so the repair is a real repair and not a relabelling.

It buys +0.43 ± 0.38 nDCG from speech alone, +0.86 ± 0.52 with on-screen text added, and +0.81 ± 0.57
with the visual surrogate on top. None of those is distinguishable from routing nothing. Stack the
per-channel features on the score features and the result is +7.56 ± 0.89, which is the score-feature
baseline back again, unimproved.

We should be plain about the surrogate. Handing pre-retrieval QPP a caption index for a visual channel
gives it a proxy it would not have in deployment, and it is a concession in the predictors' favour. It
still did not help.

QL makes the structural point without any of this machinery. The reference implementation defines it as
the query's token count, a number identical across every channel by construction, and the suite asks it
to choose between speech, on-screen text, and frames.

This is a bounded null rather than an absence of evidence. We have 2,546 queries where the original
study had 56 topics, and the confidence intervals exclude anything of practical size.

**One pre-retrieval predictor escapes, and it locates the boundary.** QSD-QPP's pre-retrieval variant
beats the fixed policy in all three cells: 0.3137, 0.3466 and 0.3039, at τ +0.164, +0.152 and +0.095.
It is the only pre-retrieval row in the table that does.

What separates it from the other eleven is what it reads. QSD_pre embeds the query, finds the historical
queries nearest to it in that space, and interpolates their known effectiveness. It touches no corpus
index at all. The eleven that collapse all score query terms against collection statistics.

That moves the boundary off the label the literature organises by. Pre-retrieval versus post-retrieval
turns out to be a proxy for something else, and the thing it proxies is whether a predictor needs
document-side language statistics. On a video collection those statistics may not exist.

Two things keep QSD from denting the null it escapes. It still trails both NQC and our control on the
honest split. And most of what it has is duplicate detection: under a plain query split it reaches
τ +0.343 and beats everything including our selector, and event grouping removes 52% of that, because a
query's nearest neighbour shares 60% of its relevant documents on average against 0.0018 for a random
query. What survives grouping is real and small.

**The same test inside one predictor family.** QSD's post-retrieval variant reads everything the suite
has: the query, its neighbours in query space with their known gains, and the retrieved document text.
It is QSD_pre plus exactly the evidence QSD_pre does without. If document-side statistics were the
missing ingredient, it should win.

It loses, on both metrics, in all three cells:

| cell | QSD_pre | QSD_post | Δ nDCG | Δ τ |
|---|---|---|---|---|
| ASR-shipped | 0.3137 / +0.164 | 0.3054 / +0.137 | −0.0083 | −0.027 |
| ASR-dense | 0.3466 / +0.152 | 0.3423 / +0.102 | −0.0043 | −0.050 |
| OCR | 0.3039 / +0.095 | 0.3013 / +0.092 | −0.0027 | −0.003 |

QSD_post is not broken. It escalates 52%, 66% and 15% of queries and still beats the fixed policy in two
of three cells, so it is a working predictor that gains nothing from the documents. A trained
transformer over three kinds of evidence lands below a training-free interpolation over one. We trained
it for one epoch on bert-base with k = 5 neighbours, so this bounds the variant at that budget rather
than at any budget, and a better-resourced version is the first thing a reviewer will ask for. What the
comparison does establish is that the ingredient QSD_pre lacks is not the one holding it back.

## 5.2 The choice is predictable, from the other family

A null on its own is ambiguous. If nothing predicts the right channel, the failure might belong to the
task rather than to the predictors, and nothing follows about QPP.

The choice is predictable. A selector reading only the channels' own score distributions beats the best
fixed policy by **+7.59 ± 1.01 nDCG** (p = .0005), and by +5.64 ± 0.93 on the benchmark's shipped
channels. Both numbers are nested: the fixed policy is re-chosen on each training fold, so the selector
is not being compared against a baseline picked with knowledge of the test set.

| policy over the same three channels | who decides | nDCG@10 | against its own best fixed policy |
|---|---|---|---|
| visual channel alone | nobody, fixed | 0.3036 | the cheap run both experiments start from |
| best weighted fusion, one setting for every query | chosen offline | 0.3408 | this is the fixed baseline for row 3 |
| binary routing, ASR-dense cell | ridge over cheap features | 0.3531 | +0.0123 |
| k-way selection over 7 channel subsets | ridge over cheap features | **0.4131** | **+7.59 ± 1.01** over `asr+visual` 0.3372 |

Rows 2 and 3 come from the binary cell, where fusion is weighted and the only available decision is
escalate or don't. Row 4 is the k-way selector over the seven non-empty subsets of the three channels,
with unweighted fusion, so its own best fixed policy is `asr+visual` at 0.3372. The last column exists
because subtracting across those two conventions gives a number that means nothing.

The selector uses 30 features: each channel's score-confidence shape, plus how far the channels' top
candidates overlap. That second group measures complementarity between evidence sources before either
source is trusted, which is one of the signals the original study's closing section asks for and which
is available to us because our options can be scored side by side. For 72% of queries the selector
resolves to a single channel.

Score-only post-retrieval prediction works here too, without any training. NQC reaches |τ| ≈ 0.21 and
routes to 0.3205, 0.3527 and 0.3040, escalating 42.5%, 79.3% and 6.7% of queries. The families split
cleanly along the line drawn in §5.1.

**The reading against us.** NQC ties or edges our binary router: 0.3205 against our 0.3193 on
ASR-shipped, and NQC_norm 0.3541 against our 0.3531 on ASR-dense. We report this because it costs the
argument nothing. The claim is about which family transfers, and NQC belongs to the family that does, so
a strong NQC corroborates the boundary rather than threatening it. A scalar predictor cannot express a
k-way policy, which is why the control is a selector, but that observation is no longer load-bearing.

**Correlation and decision come apart, and they do it systematically.** BERT-QPP's cross-encoder has the
best correlations in the table, τ +0.237, +0.229 and +0.167, and the worst decisions in it. Its
predictions are all positive, minimum +0.38, so the zero crossing carries no information even though the
ordering does. It escalates every query in all three cells, including the OCR cell where fusing costs
0.059 nDCG. It routes to 0.2795 and 0.2445 in the two cells where fusion hurts, both below doing
nothing. Read by correlation it is the best post-retrieval predictor available. Read by decision it is
the worst, and only the second reading tells you what happens if you deploy it.

The bi-encoder fails the other way, which is why we carry both. It is the deployable variant, since the
document side encodes offline and query time costs one short forward pass rather than one per candidate.
Trained on the same target, folds and budget, it reaches τ −0.016, −0.025 and −0.030. It is not
degenerate: it escalates 31–35% of queries and produces 2,485 distinct predictions out of 2,546, so the
model trained and did not collapse. The decisions are simply noise, and it lands below the fixed policy
everywhere. The cross-encoder orders and cannot decide; the bi-encoder cannot order.

And this is not one bad row. Over all 78 predictor-cell rows, Kendall τ against utility over the best
fixed policy is **−0.213** (p = 0.008, Pearson −0.397). The obvious objection is that an inert predictor
scores exactly zero utility and is thereby counted harmless, dragging the correlation down by itself.
That does not explain it: dropping the degenerate rows strengthens the relationship to −0.288 (n = 53),
and restricting to rows that escalate between 5% and 95% strengthens it again to −0.386 (n = 33). Per
cell it reads −0.408, −0.200 and −0.109. Utility is measured against the best fixed policy rather than
against the cheap channel, because a selector that beats cheap-only while losing to always-fusing has
bought nothing; the more flattering denominator gives −0.118.

τ scores an entire ordering. A selection reads one point of it. In this setting the two point in
opposite directions, which is a problem for the metric the QPP literature selects its own methods on.

## 5.3 Why the family flips

Variant selection compares competing texts against one collection, and how a text sits against a
collection is exactly what corpus statistics were built to measure. Source selection compares channels
whose applicability is a property of the document.

That difference is measurable here rather than rhetorical. Of the 109,724 test videos, **10,919 (10.0%)
yield no on-screen text at all**, 236 (0.2%) yield no speech, and 101 yield neither. For those videos the
channel does not underperform. It does not exist. No query-side statistic can see that in advance,
because availability is a fact about the document and the query has not met the document yet.

Text retrieval has no analogue. Every document in a text collection has terms, so the question of
whether a modality is available never arises, and the QPP literature has had no reason to develop a
predictor that could answer it.

The spread this creates is what the control exploits. Per-query escalation gain, in nDCG points, runs
−2.41 ± 24.07 on ASR-shipped, +3.72 ± 23.12 on ASR-dense and −5.92 ± 19.42 on OCR. In two of the three
cells the average query is hurt by the expensive channel while the standard deviation is four times the
size of the mean, so a fixed policy is choosing between harming most queries and abandoning the ones
that need rescuing. That is the whole opportunity, and it is invisible to any statistic computed before
the document side is consulted.

Clarity is the limit case at the other end of the axis QSD sits on. It needs a language model over the
retrieved documents, so over frames it is undefined rather than weak. We report it unavailable instead
of substituting a number that would look like one. Maximum dependence on document-side statistics,
therefore no definition at all.

**What the expensive channel actually does.** A channel can only raise nDCG@10 by pulling a relevant
document into the top ten that the cheap channel did not already have there. Those documents sit just
below the cut. In all three cells, 91–96% of them are at visual rank 11–50 and the rest at 51–100, with
**none past rank 100 and none absent from the visual list** (medians 18, 20, 21). The expensive channel
re-orders documents the cheap channel had already found and nearly ranked. It does not discover
documents the cheap channel missed.

They are also few, 214 to 573 distinct documents per cell, and in two of three cells fusion pushes more
relevant documents out of the top ten than it pulls in: 632 against 491 on ASR-shipped, 557 against 214
on OCR. That is where the negative mean gains come from, and it bounds what channel fusion can be asked
to do at all.

## 5.4 Does the boundary matter downstream?

The original study reports a utility gap: variants that maximise ranking metrics often fail to produce
the best generated answers, and NQC — the predictor that edges our router on the binary decision —
correlates −0.038 with answer quality against 0.329 with nDCG in their setting.

We measured it two ways and it appears in both.

**Generation.** Under the reference nuggetizer protocol on 395 queries with a local 14B judge, run under
two grounding rules:

| grounding | routed vital | best fixed vital | Δ | p |
|---|---|---|---|---|
| all text for the documents each policy retrieved | 0.5007 | 0.4648 | +0.0359 | .037 |
| only the channels each policy selected | 0.4822 | 0.4746 | +0.0076 | .67 |

Same ranked lists in both rows, and the same +7.4 nDCG lead for the routed system. Isolate retrieval and
the ranking gain converts into answer quality. Hold each policy to the evidence it chose and it does
not. The fixed policy is unmoved between the rows because `asr+visual` is two text sources under either
rule; the routed system drops because it picks a single channel for 73% of these queries, so a query
routed to the visual channel gets written from captions rather than from what was said in the video.

The gap is therefore a design choice rather than a property of routing. Select the channel for
retrieval, then ground the generator on everything reachable for the documents you found. Selection that
also narrows the evidence hands the retrieval gain straight back.

**Recall.** This one needs no generator at all. Scoring the same decisions under Recall@100, the
ASR-dense cell reads 0.7268 for the fixed policy, 0.7207 for our selector and 0.6286 for an oracle
routing on true nDCG gain, while nDCG@10 goes 0.3408, 0.3531, 0.3910. The better the nDCG selection, the
worse the recall, monotonically, and a perfect nDCG selector gives up nearly ten points of it.

So the gap reproduces twice, once against recall and once against nugget coverage under realistic
grounding, and it closes only in the arm that isolates retrieval. Any table of ours reporting nDCG@10
alone is reporting the metric the selection was fitted to, and we say so wherever we do it.

We predicted the split for the wrong reason and it is worth recording why. We expected ranking and
grounding to diverge because the visual channel retrieves best while emitting embeddings no generator
can read, and OCR retrieves worst (0.1223) while emitting usable text. That asymmetry is real, but what
produced the divergence is narrower evidence per query, not the visual channel's illegibility. We found
that only by running both arms.

## 5.5 Robustness of the boundary

**The null is not an artefact of weak channels.** Translating all 109,488 speech transcripts with NLLB
and re-encoding raises the speech channel from 0.3134 to 0.3332. The routed gap goes up with it,
+7.59 → +8.01 ± 1.01, while the corpus-statistic block stays at zero. Improving a channel raises both
sides and the decision layer keeps its margin, which is what should happen if the gain comes from
variation in which channel suits which query rather than from any single channel being bad.

**Leakage, and a taxonomy that the source setting needs too.** MultiVENT 2.0 carries several phrasings
of one event, so a query-split fold lets any predictor that learns from other queries read its answer
off a near-duplicate. Event grouping costs our selector 6% of its gap and leaves the analytic predictors
within 0.002 nDCG, because they read only the current query's scores. QSD_pre loses 52%. BERT-QPP loses
between 10% and 29%.

The split is clean, and it falls in the same place as the one in §5.1 seen from the other direction.
Predictors that consume other queries' performance leak. Predictors that consume the current query's
scores do not. The one pre-retrieval predictor that escapes the null is also the one that leaks, because
both properties follow from it reading labels rather than documents.

Thirty variants of one information need share their relevant documents by construction, so a
variant-selection study needs this control as much as we do.
