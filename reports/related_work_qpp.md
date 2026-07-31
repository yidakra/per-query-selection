# Related work: QPP as a selection problem (draft)

Draft positioning against the two QPP papers the supervision flagged. Citation keys are placeholders to
swap for the paper's bib. Revised 31 Jul 2026 after reading arXiv:2604.22661 in full.

---

**Query performance prediction as a selection problem.** The closest work to ours is Arabzadeh et al.
(2026, arXiv:2604.22661), who recast query performance prediction from estimating how hard a query is to
choosing which of several candidates to run. Their candidates are LLM reformulations: 30 variants per
information need, produced by six methods over GPT-4o, evaluated on the 56 TREC-RAG 2024 topics against
MS MARCO v2.1. A selector picks one variant before any retrieval or generation cost is paid. They
evaluate eight pre-retrieval and twelve post-retrieval predictors under both correlation and decision
metrics, and report two findings we have to answer. Cheap pre-retrieval predictors do well: IDF_max on
BM25 lifts nugget-level answer quality from 0.273 to 0.398 and from 0.227 to 0.377 on the strict
variant, ahead of NQC at 0.381 and 0.355. And ranking quality and answer quality come apart. Selecting
the variant with the best nDCG@5 reaches 0.644 nDCG but only 0.344 strict-nugget, while selecting for
strict-nugget reaches 0.536 there and falls to 0.348 nDCG. They call this the utility gap. QPP-GenRE
(Meng et al., 2024, arXiv:2404.01012) sits alongside as the accuracy-first alternative, reaching
state-of-the-art prediction by having an LLM judge each retrieved document, at the cost of an LLM pass
over the candidate list.

We share their premise. A per-query choice can be predicted cheaply, and the prediction is worth more
than the ranking metric it is usually validated against. Our predictors are implemented against the same
repository their definitions come from, so the comparison is same-vocabulary rather than same-spirit.
What differs is what gets chosen. They select among rewritings of one question; we select among the
evidence channels a video carries, its speech, its on-screen text, and the frames themselves. That
sounds like a change of domain and it is really a change of structure, because it decides what a
predictor is even able to see.

**Why the pre-retrieval result does not carry over.** Their pre-retrieval family is competitive. Ours
collapses: all ten predictors land at τ ≈ 0 and route to within 0.0005 nDCG of doing nothing, while the
score-only post-retrieval family reaches |τ| ≈ 0.21. The first thing to rule out is our own setup. A
pre-retrieval predictor reads query terms against a corpus index, and we built one index, over the
speech transcripts, so every predictor returned a single number per query regardless of which channel
was under consideration. That would explain the null on its own and would say nothing about the method.

So we gave it every index the benchmark allows: the transcripts, the on-screen text, and the shipped
captions as a text surrogate for the visual channel, which retrieves over frame embeddings and has no
term index of its own. The features do vary across the three, with a mean relative range from 0.09
(SCQ_max) to 0.33 (IDF_std), so the repair works as a repair. It buys almost nothing. Against the
+7.59 ± 1.01 nDCG that the channels' own score distributions deliver on the same event-grouped folds,
the per-channel pre-retrieval features reach +0.43 ± 0.38 from the speech index alone, +0.86 ± 0.52 with
on-screen text added, and +0.81 ± 0.57 once the visual surrogate joins them. None of those is
distinguishable from routing nothing. Stacked on top of the score features they give +7.56 ± 0.89,
which is the baseline back again (`mv2_qpp_prechannel.py`).

That is a stronger result than the indexing story we first reached for, and it points elsewhere. Corpus
term statistics describe how hard a query looks against a collection. Which *evidence source* will
answer it is not a property of the query's vocabulary, and no amount of per-channel indexing makes it
one. Variant selection is different in exactly the way that matters: the options there are competing
texts, and how a text sits against the collection is precisely what those predictors were built to
measure. Clarity is the limit case and it does not even get a surrogate, since it needs a language model
over the retrieved documents; for a channel whose documents are images it is undefined rather than weak,
and we report it unavailable rather than substituting something that looks like a number. The finding
that cheap pre-retrieval prediction suffices is not wrong, then. It is bounded, and the boundary is that
the options have to differ in ways query-collection statistics can see.

**The utility gap, when the modalities are not interchangeable.** Their gap is a divergence between two
objectives measured on the same text. Ours has a physical cause underneath it. The frame channel
retrieves best of our cheap sources and emits embeddings, which no generator can read. OCR retrieves
worst at 0.1223 and emits text a generator can use directly. Ranking ability and grounding ability are
carried by different things here, so the gap is not a subtle divergence to be traded off but a
constraint on which channels can answer at all. We are measuring it now on ~400 stratified queries under
their nuggetizer protocol, with a local judge, in two evidence conditions: each policy grounded on the
text of the channels it actually retrieved, and every policy grounded on all available text for the
documents it returned. The second isolates retrieval quality and is the closest analogue of their
setting. One detail sharpens the stake. NQC, the predictor that edges our router on the binary
escalation decision, is the same predictor they find correlates at −0.038 with answer quality while
correlating at 0.329 with nDCG. If that inversion holds across modalities, the ranking cell everyone
reports is the wrong place to judge either method.

**A methodological note their design invites.** MultiVENT 2.0 carries several phrasings of the same
event, so a fold split by query lets any predictor that learns from other queries' labels read its
answer off a near-duplicate. Grouping folds by event (536 groups over 2,546 queries) costs our selector
6% of its gap and leaves the analytic predictors within 0.002 nDCG, because they read only the current
query's score distribution. What it does to the historical-query family is another matter: QSD loses
52% of its correlation, and supervised BERT-QPP loses between 10% and 29%. The taxonomy is clean.
Predictors that consume other queries' performance leak; predictors that consume the current query's
scores do not. Thirty variants of one information need share their relevant documents by construction,
which is a stronger version of the same structure, and QSD appears in both their pre- and post-retrieval
lists. Their correlation analysis is within-topic and so unaffected. Any predictor of theirs that fits
on a pool of other variants is exposed, and we can quantify the exposure because we have already paid
for it.

**Where we end up.** Their closing call is for generation-aware predictors that treat retrieval and
generation as one system, and for signals of complementarity, coverage and redundancy rather than
ranking quality alone. Our 30 features already include the cross-channel part of that: how much two
channels' top-10 and top-100 candidates overlap, which is complementarity between evidence sources
measured before either is trusted. On the k-way choice this buys +7.59 ± 1.01 nDCG over the best fixed
policy chosen on the training fold (+5.64 ± 0.93 on the shipped channels, permutation p = .0005), while
a scalar predictor cannot express a k-way policy at all. On the binary escalate-or-not decision a single
good predictor matches us, and we say so.

---

## Notes for revision (not for the paper)

- Numbers above are from the v1 PDF, 24 Apr 2026. Re-check against any v2 before submission.
- Do NOT write that we beat classical QPP. With formulas pinned to their repo and folds grouped by
  event, NQC edges our router on the binary decision (0.3205 vs 0.3193 shipped, NQC_norm 0.3541 vs
  0.3531 dense, |τ| 0.215 vs 0.211). The defensible claim is that a scalar cannot express a k-way policy
  choice, so the selector is the contribution and the pairwise cell is the baseline it clears.
- Their 56 topics against our 2,546 queries is a fair power contrast, but the RAG arm runs on ~400, so
  do not claim scale on both axes in the same breath.
- The "802 of 2,546" figure is from the dense-m3 grouped selector picks. Recompute if that cell changes.
- The per-channel pre-retrieval experiment has now run (`mv2_qpp_prechannel.py`, results in
  `results/ablations/mv2_qpp_prechannel.json`). An earlier draft of this section argued the null was
  structural, that a query-side feature is constant across channels because the query never changes.
  That is wrong once you build an index per channel, and the measured version is better: the features do
  vary and still buy +0.86 ± 0.52 at best. Do not resurrect the constant-feature argument.
- The caption index is a surrogate and the paper must say so in one sentence. The visual channel searches
  CLIP embeddings over frames; the captions describe the same videos in text. Reviewers who know
  MultiVENT 2.0 will notice, and the honest framing is that we handed pre-retrieval QPP a proxy it does
  not normally get and it still did not help.
- QPP-GenRE as a live baseline would quantify the cost gap against our 1 ms router, since it needs an LLM
  pass over the candidate list. Still to do.
- The utility-gap paragraph is written as work in progress on purpose. When the RAG arm reports, if
  ranking and grounding do come apart on our channels, this section should open with that instead of the
  pre-retrieval argument, and the NQC inversion becomes the headline.
