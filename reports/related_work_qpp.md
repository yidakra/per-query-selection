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
multimodal, and that changes the problem in a way the text-RAG formulation does not confront. Classical
pre-retrieval QPP predicts difficulty from corpus term statistics such as IDF, ICTF, and clarity, all of
which presume a lexical index over the documents. Our documents are videos. The visual channel has no
term statistics at all, and the only text is derived from ASR and OCR, which is noisy, frequently absent
(a silent clip has no speech to transcribe), and multilingual, so the collection frequency of a query
term is either undefined or unreliable. A pre-retrieval predictor in the classical sense is therefore
unavailable for the modality that carries most of the signal. The cheapest usable signal is instead the
score distribution of the cheap visual channel once it has run, namely its top score, its margins, and
its entropy. This also resolves an apparent tension with prior reports that pre-retrieval QPP suffices
in text: our features are pre-expensive-retrieval yet post-cheap-retrieval, and in the multimodal
setting there is no free query-only predictor to fall back on.

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
- Clarity/WIG/NQC appear in *both* their predictor set and our routing baselines, so "we beat classical
  QPP as routing baselines" (RQ2) is a direct, same-vocabulary comparison — worth making explicit.
- If we add QPP-GenRE as an actual baseline (an LLM over the candidate list), the para can gain a
  sentence quantifying the cost gap against our ~1 ms cheap-feature router.
