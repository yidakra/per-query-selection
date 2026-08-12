# Related work and framing: selection is not one problem

Draft of what becomes §2 and §3 of the paper under story B. Rewritten 31 Jul 2026 from the story-A
version, which argued from our routing gain outward. The argument now runs the other way: their result,
our test of it, what the test shows about both. Citation keys are placeholders.

---

**Query performance prediction as a selection problem.** Arabzadeh et al. (2026, arXiv:2604.22661)
recast query performance prediction from estimating how hard a query is to choosing which of several
candidates to run. Their candidates are LLM reformulations: 30 variants per information need, produced
by six methods over GPT-4o, evaluated on the 56 TREC-RAG 2024 topics against MS MARCO v2.1. A selector
picks one variant before any retrieval or generation cost is paid. They evaluate eight pre-retrieval and
twelve post-retrieval predictors under both correlation and decision metrics.

Two of their findings matter here, and the first is the surprising one. **Cheap pre-retrieval predictors
do well.** IDF_max on BM25 lifts nugget-level answer quality from 0.273 to 0.398, and from 0.227 to
0.377 on the strict variant, ahead of NQC at 0.381 and 0.355. A predictor that reads term statistics off
an index, before a single document is scored, picks better among thirty rewritings than a predictor that
reads the retrieved scores. The second finding is that ranking quality and answer quality come apart.
Selecting the variant with the best nDCG@5 reaches 0.644 nDCG and only 0.344 strict-nugget; selecting
for strict-nugget reaches 0.536 there and falls to 0.348 nDCG. They call this the utility gap.
QPP-GenRE (Meng et al., 2024, arXiv:2404.01012) sits alongside as the accuracy-first alternative,
reaching state-of-the-art prediction by having an LLM judge each retrieved document, at the cost of an
LLM pass over the candidate list.

We take the first finding seriously enough to test it somewhere else, which is the whole of this paper.

**Two instantiations of one problem.** Strip both settings to the same shape. There is a set of options,
a predictor that scores each option before the expensive work is done, and a decision rule that picks
one. In their instantiation the options are rewritings of a question, searched over one corpus. In ours
the options are the evidence channels a video carries: what was said in it, what is written on screen,
and what the frames look like. Same shape, and one difference that turns out to decide everything.
**Their options vary on the query side. Ours vary on the document side.**

That sounds like bookkeeping. It sets what a predictor is able to see.

**The boundary has ancestors, and the paper must own them.** Choosing which of several searchable
sources to query, per query, is the resource selection problem of federated and distributed IR, and
that literature drew a version of our dividing line thirty years ago: lexicon-based methods that score
a source by its collection term statistics (CORI, Callan et al., SIGIR 1995; gGlOSS, Gravano and
Garcia-Molina) were overtaken by sample-based methods that run the query against document samples and
read retrieval scores back (ReDDE, Si and Callan, SIGIR 2003; CRCS; SUSHI; survey in Shokouhi and Si,
Federated Search, FnTIR 2011). Taily (Aly et al., SIGIR 2013) is the standing counterexample, a
shard-selection method built on corpus term statistics that works well, and the paper must engage it
rather than cite around it: Taily selects among shards of one homogeneous text collection, where every
shard has term statistics of the same kind, which is precisely the condition the visual channel breaks.
Vertical selection in aggregated search (Arguello et al., SIGIR 2009; Diaz, WSDM 2009; Arguello, FnTIR
2017) is the same decision one level up, choosing among media verticals per query, and it too found
collection-side term evidence weak for verticals without text representations. Our claim is therefore
not that the dividing line is new. It is that the line re-emerges, measurably and per query, inside
the QPP-for-selection paradigm on the reference definitions of a published QPP study, in a setting
with a mechanism text federations lack: a document can lack the channel entirely, so applicability is
per document, not per collection. The positive control has ancestry too: query-adaptive multimodal
fusion with score-distribution features was an active TRECVID-era line (Yan, Yang and Hauptmann, ACM
MM 2004 [verify exact venue]; Wilkins et al., ACM MIR 2006 [verify]), and the selector should be
presented as the control it is, not as a novelty. Selective query expansion via difficulty prediction
(Amati et al., ECIR 2004; Yom-Tov et al., SIGIR 2005) and QPP for fusion decisions (Markovits et al.,
CIKM 2012, the closest precedent to our binary cells) anchor the opening claim that QPP decides how
much machinery a query gets.

**The corpus-statistic family does not transfer.** We implemented eleven pre-retrieval predictors
against the same reference repository their definitions come from, so this is a same-vocabulary
comparison rather than a same-spirit one. All eleven land at τ ≈ 0 and route to within 0.0005 nDCG of
doing nothing. The score-only post-retrieval family, run identically, reaches |τ| ≈ 0.21.

The protocol is symmetric, which closes the strongest objection to the contrast. Every family gets
the identical nested operating-point calibration: escalation fraction chosen on a group-disjoint
subset of each outer training fold, never on a test label. Under that shared protocol the
corpus-statistic family stops being degenerate (2 of 33 cells at a corner, down from 20 under the
raw zero crossing) and still clears the source study's margin nowhere, 0 of 33, mostly landing below
the fixed policy: forcing an operating point onto a noise ordering escalates the wrong queries. The
score-only family holds 17 of 30 under the same treatment. With per-row inference at the event-group
level, 0 of 33 corpus-statistic rows beat the fixed policy after Holm correction and 21 of 33 are
formally equivalent to it within 0.005 nDCG, against 8 of 30 significant score-only rows. The
zero-crossing degeneracy contrast (20/33 vs 5/33) survives as a diagnostic of raw calibration, not as
the family evidence.

One pre-retrieval predictor escapes, and it is the exception that fixes the rule. QSD_pre embeds the
query, finds the historical queries nearest to it, and interpolates their known effectiveness, so it
reads no corpus index at all. Under the fully nested protocol (neighbourhood size and operating point
both chosen inside the training fold) it clears the fixed policy in the two speech cells and no
term-statistic predictor does anywhere. So the boundary is not the pre/post-retrieval split the
literature organises by. It is whether a predictor depends on corpus-aggregate term statistics,
collection frequencies read against the query. That wording matters: BERT-QPP's cross-encoder reads
retrieved document text and sits on the successful side, so "document-side language" would be
falsified by our own table, while "corpus-aggregate statistics" separates the families cleanly. Do
not cite the WIG-vs-NQC contrast as a corpus-dependence gradient: our adaptation replaces the
reference formulas' collection score with the list mean on similarity channels, so our WIG carries no
collection-frequency information (fact-check, 11 Aug).

Two things stop the exception from weakening the result. QSD_pre still trails both NQC and our
selector. And most of what it has is duplicate detection: under a plain query split it reaches
τ +0.343 and beats everything, and event grouping removes 52% of that (see the methodological note
below). Zendel et al. (SIGIR 2019) is the precedent: other queries of the same information need are
potent predictors, which is exactly the near-duplicate mechanism the grouping quantifies. QSD_pre is
less a pre-retrieval anomaly than a third predictor category the pre/post taxonomy has no name for,
supervision transfer from labelled neighbours, and the paper should present it that way.

The first thing to rule out is our own setup. A pre-retrieval predictor reads query terms against a
corpus index, and we had built one index, over the speech transcripts, so every predictor returned a
single number per query no matter which channel was under consideration. That alone would produce the
null and would say nothing about the method.

So we gave it every index the benchmark allows: the transcripts, the on-screen text, and the shipped
captions as a text surrogate for the visual channel, which searches frame embeddings and has no term
index of its own. The features do vary across the three, with a mean relative range from 0.09 on
SCQ_max to 0.33 on IDF_std, so the repair works as a repair. It buys almost nothing. Per-channel
pre-retrieval features reach +0.43 ± 0.38 nDCG from the speech index alone, +0.86 ± 0.52 with on-screen
text added, and +0.81 ± 0.57 once the visual surrogate joins them, on event-grouped folds. Stacked on
top of the score features they give +7.56 ± 0.89, which is the baseline back again
(`mv2_qpp_prechannel.py`).

The surrogate deserves its own sentence, plainly. The visual channel retrieves over CLIP embeddings; the
captions describe the same videos in text. Anyone who knows MultiVENT 2.0 will notice, and the honest
framing is that we handed pre-retrieval QPP a proxy it does not normally get and it still did not help.

**The null is bounded.** 2,546 queries against their 56 topics, and intervals that exclude
anything of practical size. This is evidence of absence, at that power.

**And the choice is predictable, from the other family.** A null on its own could mean the decision is
simply not predictable, in which case nothing about predictor families follows. It is predictable. A
selector over the channels' own score distributions, thirty features, same event-grouped folds, beats
the best fixed policy chosen on the training fold by +7.59 ± 1.01 nDCG, and by +5.64 ± 0.93 on the
benchmark's shipped channels, permutation p = .0005 in both. Among those thirty features are the
cross-channel ones: how much two channels' top-10 and top-100 candidates overlap, which is
complementarity between evidence sources measured before either is trusted. That is one of the signals
their closing section calls for, and it is available here because the options can be scored side by
side.

So the pre-retrieval null is a statement about a family of predictors, not about the task.

**Why the family flips.** Corpus term statistics describe how hard a query looks against a collection.
Which evidence source will answer it is not a property of the query's vocabulary, and no amount of
per-channel indexing makes it one. Their setting is different in exactly the way that matters: the
options there are competing texts, and how a text sits against the collection is precisely what those
predictors were built to measure. Ours are channels whose applicability is a property of the document.
A silent clip has no speech to transcribe. A talking-head clip has frames that show nothing the query
could match. In variant selection every option applies to every document and only quality varies.

Clarity is the limit case and it does not even get a surrogate, since it needs a language model over the
retrieved documents. For a channel whose documents are images it is undefined rather than weak, and we
report it unavailable rather than substituting something that looks like a number.

The finding that cheap pre-retrieval prediction suffices is not wrong, then. It is bounded, and the
boundary is that the options have to differ in ways query-collection statistics can see.

**The utility gap appears here too, and what governs it is where the generator's evidence comes from.**
Their gap is a divergence between two objectives measured on the same text. Ours looked like it should
have a physical cause underneath it: the frame channel retrieves best of our cheap sources and emits
embeddings, which no generator can read, while OCR retrieves worst at 0.1223 and emits text a generator
can use directly. Ranking ability and grounding ability are carried by different things here, so we
predicted a divergence sharper than theirs.

We measured it under their nuggetizer protocol on 395 stratified queries and ran two grounding arms.
The 14B checkpoint produces nuggets, reports and primary assignments; a 7B checkpoint from the same
model family repeats assignment while holding nuggets and reports fixed.
In the first arm every policy is grounded on all available text for the documents it returned, which
isolates retrieval quality and is the closest analogue of their setting. In the second each policy may
read only the channels it selected, which is what a deployed cascade would have.

The two arms disagree. Isolating retrieval, the routed system reaches 0.5007 vital-nugget coverage
against 0.4648 for the best fixed policy (p = .037) and 0.3902 against 0.3432 on strict vital (p = .014),
and the policy ordering under nugget coverage is the ordering under nDCG. Holding each policy to its own
evidence, the same routed system reaches 0.4822 against 0.4746 (p = .67) and 0.3674 against 0.3473
(p = .29): indistinguishable, while still leading that policy by +7.4 nDCG.

The 7B assignment repeats the distinction: all-text vital coverage is 0.4781 against 0.4381
(delta +0.0399, p=.038), while selected-channel coverage is 0.4583 against 0.4328
(delta +0.0255, p=.22). Exact label agreement is 71--72% and Cohen's κ is 0.53. Thus absolute judgments
move, but the arm-level conclusion survives this capacity/checkpoint change. Because both are Qwen2.5,
this is not evidence of cross-family judge robustness.

The asymmetry is real, then, and it sits in the grounding rather than in the retrieval. The best fixed
policy is unaffected by the restriction because `asr+visual` gives it two text sources either way; the
routed system loses 2 to 4 points because it selects a single channel for 73% of these queries, and a
query routed to the visual channel is written from captions instead of from speech. The reports are the
same length, over the same five documents. What changes is what is in them. The design consequence is
worth stating in the paper: select the channel for retrieval, then ground on everything reachable for
the documents you found.

The gap turns up a second time without any generator at all. Scoring the same decisions under
Recall@100, the ASR-dense cell goes 0.7268 for the best fixed policy, 0.7207 for our selector, 0.6286
for an oracle routing on true nDCG gain, while nDCG goes 0.3408 / 0.3531 / 0.3910. The better the nDCG
selection, the worse the recall, monotonically, exactly as their Oracle-ndcg@5 and Oracle-recall@100
rows diverge. Any table of ours reporting nDCG@10 alone is reporting the metric the selection was fitted
to, and the paper should say so where it reports one.

One stake stays open. NQC, which sits within a point of our router on the binary escalation
decision, is the same predictor they find correlating at −0.038 with answer quality while correlating at
0.329 with nDCG. We cannot say from our data whether that inversion is a property of their setting or
of the predictor, and it is worth posing as a question rather than answering it with a guess.

**A methodological note their design invites.** MultiVENT 2.0 carries several phrasings of the same
event, so a fold split by query lets any predictor that learns from other queries' labels read its
answer off a near-duplicate. Grouping folds by event, 536 groups over 2,546 queries, costs our selector
6% of its gap and leaves the analytic predictors within 0.002 nDCG, because they read only the current
query's score distribution. The historical-query family is another matter: QSD loses 52% of its
correlation and supervised BERT-QPP loses between 10% and 29%. The taxonomy is clean. Predictors that
consume other queries' performance leak; predictors that consume the current query's scores do not.

Thirty variants of one information need share their relevant documents by construction, which is a
stronger version of the same structure, and QSD appears in both their pre- and post-retrieval lists.
Their correlation analysis is within-topic and so unaffected. Any predictor of theirs fitted on a pool
of other variants is exposed, and we can quantify the exposure because we have already paid for it.

---

## Notes for revision (not for the paper)

- Numbers for their side are from the v1 PDF, 24 Apr 2026. **Checked 2 Aug 2026: arXiv still shows only
  v1**, so the numbers stand. But the paper is now published (SIGIR 2026, doi:10.1145/3805712.3808571,
  *Proceedings of the 49th International ACM SIGIR Conference*), and the camera-ready is the version of
  record. Cite the ACM version, not the preprint. The ACM page is paywalled to us, so the arXiv v1 and
  the published version have **not** been diffed. If any of their numbers moved in camera-ready, ours
  would be quoting a superseded table. Get the published PDF before submission and re-check the figures
  we quote.
- The QSD citation was wrong and is fixed: the title is *Query Performance Prediction Using Neural Query
  Space Proximity* (not "Estimating..."), ACM TIST, doi:10.1145/3762197. Worth knowing that **Negar
  Arabzadeh is an author of both** that paper and the variant-selection paper we bound, so QSD_pre and
  QSD_post are not a neutral third-party baseline. They are the same group's method, which makes the
  QSD_post result a stronger rather than weaker thing to report.
- Do NOT write that we beat classical QPP. Under the shared nested protocol the cluster is within a
  point either way per cell: ours 0.3161 vs NQC 0.3144 on shipped, ours 0.3522 vs σ_max 0.3532 and
  BERT-QPP 0.3560 on dense. Under story B this costs nothing: everything in that cluster is in the
  family that transfers, so it corroborates the claim. Do not let a later draft turn it into either a
  wound or a win.
- The degeneracy counts come from each predictor's recorded escalation fraction, exactly 0 or exactly 1,
  not from its routed nDCG sitting on a fixed policy's. Inferring it from the nDCG at a 5e-4 tolerance
  gave 46 cells against the measured 25, because a predictor that escalates six queries out of 2,546
  scores the same as one that escalates none and is not the same predictor. Do not let a later edit
  reintroduce the cheaper version.
- Do not resurrect the constant-feature argument. An earlier draft argued the null was structural,
  that a query-side feature is constant across channels because the query never changes. That is wrong
  once you build an index per channel, and the measured version is stronger.
- The caption surrogate concession is load-bearing under story B, since §5.1 is now the paper's main
  result. It cannot be compressed away for space.
- The utility-gap paragraph has been rewritten twice. First it predicted a split; then the `all` arm
  came back without one and it said so; then the `own` arm came back with one. Both arms are now
  reported and neither may be quoted alone, because "routing improves answer quality" is true of one and
  false of the other. Do not let a later draft pick the flattering arm.
- We predicted the `own` arm would widen the gap, reasoning that single-channel policies would fall
  hardest. They did, and so did the routed system, which is single-channel on 73% of queries. Keep that
  on the record rather than presenting the mechanism as though we had it in advance.
- Their 56 topics against our 2,546 queries is a fair power contrast, but the RAG arm runs on 395, so
  do not claim scale on both axes in the same breath.
- Both RAG arms and the second assignment checkpoint are in. Numbers live in
  `results/ablations/rag/metrics_n400_{all,own}{,_qwen7b}.json`; the current write-up is in
  `evidence.md` and `experiment_gap_closure.md`.
- Table 1 is complete as of 1 Aug 2026, nugget columns included, for every row rather than the
  section-best rows. They are mixed per query from two judged runs per cell, which is exact and not an
  approximation: a report's coverage is a property of the ranked list it was written from. The mix
  aborts unless both endpoints reproduce the judged runs. Do not let a later draft describe these
  columns as estimated or partial.
- The old τ-versus-utility headline does not survive symmetric nested calibration. With every row on
  the shared protocol it is −0.149 over 78 rows (p=.054), −0.168 after exact degeneracies are removed,
  −0.143 for 5--95% escalation, and −0.057 against cheap-only, with the row p-values descriptive (78
  dependent rows). The portable result is that correlation does not supply a deployable operating
  point, demonstrated by the cross-encoder changing from always-fuse to the strongest learned baseline
  under inner-fold calibration. Do not reinstate the anti-correlation claim.
- The corpus-axis experiments (1 Aug 2026) are a negative result and belong in the paper as one. In
  sample they look outstanding (100.0% of the ASR-dense gain at 38% of the corpus, an oracle document
  set beating full extraction by +0.069), and every bit of it is selection-on-test. Train and held-out
  carrier sets overlap by zero documents in 15 of 15 folds. This is the tier-C failure mode again, and
  the reason to publish it is that the in-sample number is exactly what a less careful paper would have
  reported. What survives: the depth rule at 38% of corpus for 58% of the gain, +0.026 over random.
  Scripts are in the scratchpad, not the repo, pending a decision on whether this section exists.
- QPP-GenRE as a live baseline would quantify the cost gap against our 1 ms router, since it needs an
  LLM pass over the candidate list. Still to do, and under story B it is optional rather than expected.
- Under story A this file was related work. Under story B most of it is §2 and §3 of the paper.
