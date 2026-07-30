# Related work: QPP as a selection problem (draft)

Draft paragraph positioning the router against the two QPP papers the supervision flagged. Citation keys
are placeholders to swap for the paper's bib.

---

**Query performance prediction as a selection problem.** The closest work to ours is Arabzadeh et al.
(2026, arXiv:2604.22661), who recast query performance prediction (QPP) from estimating a query's
difficulty to selecting among candidates. They use QPP to choose, among many LLM-generated
reformulations of a query, the single variant to run through a retrieval-augmented generation (RAG)
pipeline, deciding before the pipeline's retrieval and generation costs are paid. On TREC-RAG 2024 they
report that even cheap pre-retrieval predictors such as IDF recover much of the gain a perfect selector
would achieve, with nugget-level answer quality for BM25 rising 46% to 66% over the original query. They
also surface a "utility gap": the variant that maximizes ranking quality is frequently not the one that
yields the best generated answer, so that an nDCG-optimal oracle can underperform a QPP method tuned for
answer quality. A related line, QPP-GenRE (Meng et al., 2024, arXiv:2404.01012), reaches
state-of-the-art QPP by having an LLM generate per-document relevance judgments, at the cost of an LLM
pass over the candidate list.

Our work shares the premise that a per-query choice can be predicted cheaply, but the setting is
multimodal, and that changes the problem in a way the text formulation does not confront. We implement
their predictor suite over our channels and find that the pre-retrieval family, which is competitive in
their study, collapses to τ ≈ 0 here, while the score-only post-retrieval family reaches |τ| ≈ 0.21.
The reason is structural rather than a matter of tuning. A pre-retrieval predictor scores the query
against a corpus index, and in text that index covers the same corpus the retriever searches. Our
retriever searches videos. The only index we can build is over their ASR transcripts, so the statistics
describe a different channel than the one the router is deciding about, and the transcripts are
themselves noisy, frequently absent, and multilingual. Clarity is the sharper case: it builds a
language model from the retrieved documents, so for a channel whose documents are frames it is undefined
rather than weak, and we report it as unavailable instead of substituting an approximation. What remains
computable is the cheap channel's own score distribution, which is why our features sit after the cheap
retrieval and before the expensive one. Prior reports that cheap pre-retrieval prediction suffices are
therefore not contradicted so much as bounded: they hold where predictor and retriever read the same
modality.

Beyond the domain, we differ in what is chosen and how. Arabzadeh et al. select among query
reformulations; we route among heterogeneous evidence sources, the modality channels (speech, on-screen
text, captions) and an LLM-expansion tier, whose costs differ by orders of magnitude and whose
applicability is itself a per-video property. Whether the speech channel is worth consulting depends on
whether the video contains speech, a kind of heterogeneity with no direct analogue in text query
variants. Rather than scoring each option with an off-the-shelf QPP predictor, we regress the escalation
gain directly from the cheap tier's score distribution, so the expensive source is never invoked to make
the decision. Our claims are also broader: a law relating the heterogeneity of per-query gain to the
value of routing (Spearman ρ = 0.94), the finding that uniformly fusing a weak channel is worse than
ignoring it, and a measured joules-and-latency cost model. The utility gap they identify concerns
systems with a generation stage; we evaluate on retrieval quality directly and are not exposed to it,
though it would return if a generation tier were added.

---

## Notes for revision (not for the paper)

- The 46–66% figures are BM25 + IDF_max on Nugget-All / Nugget-Strict (0.273→0.398, 0.227→0.377). Swap
  for the dense-retriever numbers if we want a stronger or more conservative anchor.
- Their benchmark is 56 judged queries; if a reviewer leans on our 2,546-query scale as a strength,
  this contrast is worth a half-sentence.
- Do NOT write that we beat classical QPP. Once the formulas were pinned to their repo, NQC matched our
  router on the binary escalate decision (0.3211 vs 0.3205 shipped, 0.3532 vs 0.3536 dense, |τ| 0.215 vs
  0.217). The defensible claim is that a scalar predictor cannot express a k-way policy choice, so the
  selector is the contribution and the pairwise cell is the baseline it clears.
- Their predictor set and our baselines share Clarity/WIG/NQC, so the comparison is same-vocabulary.
  Worth stating, now that it is a tie rather than a win.
- Their benchmark is 56 judged queries against our 2,546. Useful contrast if a reviewer questions power,
  but the RAG arm will run on ~400, so do not overclaim scale across both axes at once.
- QPP-GenRE as a live baseline would let us quantify the cost gap against our 1 ms router, since it needs
  an LLM pass over the candidate list. Still to do, along with QSD_post and BERT-QPP.
- The utility-gap sentence at the end of the second paragraph is a promise until the RAG arm reports. If
  ranking and grounding do come apart on our channels, that paragraph should lead with it rather than
  treat it as a caveat.
