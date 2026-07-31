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

**Why the pre-retrieval result does not carry over, and what that tells us.** Their pre-retrieval family
is competitive. Ours collapses: all ten predictors land at τ ≈ 0 and route to within 0.0005 nDCG of
doing nothing at all, while the score-only post-retrieval family reaches |τ| ≈ 0.21. The gap is not
tuning. A pre-retrieval predictor reads query terms against a corpus index, and in variant selection the
options *are* different query texts, so the feature moves as the choice moves. That is the whole signal.
When the options are channels the query text never changes, so an IDF or SCS value is one number per
query no matter which channel is under consideration. It can say this query looks hard. It cannot say
prefer speech to frames here, because it takes the same value for both.

The obvious repair is a term index per channel, which would restore the variation. It works for speech
and for on-screen text. It cannot work for the frames, which carry no terms at all, and the frame
channel is the one our selector picks alone for 802 of 2,546 queries and the strongest single channel we
have at 0.3036. Clarity fails the same way and more visibly: it needs a language model over the
retrieved documents, so for a channel whose documents are images it is undefined rather than weak, and
we report it unavailable instead of substituting something that looks like a number. So the finding that
cheap pre-retrieval prediction suffices is not wrong. It is bounded, and the boundary is worth naming:
pre-retrieval QPP discriminates between options only when the options differ in the query text, and in
multimodal selection they do not.

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
- **Experiment this section is currently promising and has not run:** per-channel pre-retrieval
  predictors. We build the lexical index over ASR only (`mv2_qpp_table.py`), so our pre-retrieval values
  are one scalar per query, which is exactly the structural point. Building a second index over the OCR
  text would give the argument its own measurement: even with per-channel variation restored for the two
  text channels, the selector still cannot see the frame channel, which is the one most often picked.
  Cheap, CPU-only, and it converts the paragraph from reasoning to evidence. Worth doing before this
  section is defended.
- QPP-GenRE as a live baseline would quantify the cost gap against our 1 ms router, since it needs an LLM
  pass over the candidate list. Still to do.
- The utility-gap paragraph is written as work in progress on purpose. When the RAG arm reports, if
  ranking and grounding do come apart on our channels, this section should open with that instead of the
  pre-retrieval argument, and the NQC inversion becomes the headline.
