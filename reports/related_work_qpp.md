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

**The corpus-statistic family does not transfer.** We implemented eleven pre-retrieval predictors
against the same reference repository their definitions come from, so this is a same-vocabulary
comparison rather than a same-spirit one. All eleven land at τ ≈ 0 and route to within 0.0005 nDCG of
doing nothing. The score-only post-retrieval family, run identically, reaches |τ| ≈ 0.21.

One pre-retrieval predictor escapes, and it is the exception that fixes the rule. QSD_pre embeds the
query, finds the historical queries nearest to it, and interpolates their known effectiveness, so it
reads no corpus index at all. It beats the fixed policy in every cell (0.3137 / 0.3466 / 0.3039), which
no term-statistic predictor does anywhere. So the boundary is not the pre/post-retrieval split the
literature organises by. It is whether a predictor needs document-side language statistics.

Two things stop that from weakening the result. QSD_pre still trails both NQC and our selector. And most
of what it has is duplicate detection: under a plain query split it reaches τ +0.343 and beats
everything, and event grouping removes 52% of that (see the methodological note below). The predictor
that escapes the null is the same one that leaks, and both follow from it reading other queries' labels
rather than documents.

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

**The null is bounded, not a shrug.** 2,546 queries against their 56 topics, and intervals that exclude
anything of practical size. We are not reporting an absence of evidence.

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

**The utility gap did not appear here, and we expected it to.** Their gap is a divergence between two
objectives measured on the same text. Ours looked like it should have a physical cause underneath it:
the frame channel retrieves best of our cheap sources and emits embeddings, which no generator can read,
while OCR retrieves worst at 0.1223 and emits text a generator can use directly. Ranking ability and
grounding ability are carried by different things here, so we predicted a divergence sharper than
theirs.

We measured it under their nuggetizer protocol, 395 stratified queries, local judge, with every policy
grounded on all available text for the documents it returned, which isolates retrieval quality and is
the closest analogue of their setting. The routed system reaches 0.5007 vital-nugget coverage against
0.4648 for the best fixed policy (p = .037), and 0.3902 against 0.3432 on strict vital (p = .014). **The
policy ordering under nugget coverage is the ordering under nDCG.** The one apparent swap separates two
policies by five parts in a hundred thousand and is a tie.

The physical asymmetry is real and it did not produce a divergence there. On this cascade, at the policy
level, retrieval nDCG is a faithful stand-in for downstream answer quality.

It is not a faithful stand-in for recall, and that is where their gap turns up. Scoring the same
decisions under Recall@100, the ASR-dense cell goes 0.7268 for the best fixed policy, 0.7207 for our
selector, 0.6286 for an oracle routing on true nDCG gain, while nDCG goes 0.3408 / 0.3531 / 0.3910. The
better the nDCG selection, the worse the recall, monotonically, exactly as their Oracle-ndcg@5 and
Oracle-recall@100 rows diverge. So the utility gap survives the change of setting; it just does not run
between the two objectives we assumed. Any table of ours reporting nDCG@10 alone is reporting the metric
the selection was fitted to, and the paper should say so where it reports one.

One stake stays open. NQC, the predictor that edges our router on the binary escalation
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

- Numbers for their side are from the v1 PDF, 24 Apr 2026. Re-check against any v2 before submission.
- Do NOT write that we beat classical QPP. With formulas pinned to their repo and folds grouped by
  event, NQC edges our router on the binary decision (0.3205 vs 0.3193 shipped, 0.3541 vs 0.3531 dense,
  |τ| 0.215 vs 0.211). Under story B this costs nothing: NQC is in the family that transfers, so it
  corroborates the claim. Do not let a later draft turn it back into a wound.
- Do not resurrect the constant-feature argument. An earlier draft argued the null was structural,
  that a query-side feature is constant across channels because the query never changes. That is wrong
  once you build an index per channel, and the measured version is stronger.
- The caption surrogate concession is load-bearing under story B, since §5.1 is now the paper's main
  result. It cannot be compressed away for space.
- The utility-gap paragraph used to be written as work in progress, predicting a split. The split did
  not appear. The rewritten paragraph reports that, and the temptation to re-hedge it into "no gap was
  detected" should be resisted; we predicted one, we looked, it is not there.
- Their 56 topics against our 2,546 queries is a fair power contrast, but the RAG arm runs on 395, so
  do not claim scale on both axes in the same breath.
- The `--evidence own` arm is still running. If it changes the ordering, the utility-gap paragraph moves
  again and the change is worth reporting rather than smoothing.
- QPP-GenRE as a live baseline would quantify the cost gap against our 1 ms router, since it needs an
  LLM pass over the candidate list. Still to do, and under story B it is optional rather than expected.
- Under story A this file was related work. Under story B most of it is §2 and §3 of the paper.
