# 3. Two instantiations of one selection problem

Draft for the ECIR 2027 submission. This is the section Figure 1 illustrates, and §5.3 refers back to it.

Selection, stated once and abstractly, has three parts. A set of options *O*. A predictor *f* that scores
each option before any of them is run in full. A decision rule that turns those scores into a choice. The
whole point is to commit before paying, so *f* is restricted to what is cheap to compute, and the
interesting question is always what *f* is allowed to look at.

Two lines of work instantiate this differently, and the difference is not obvious until you write both
down in the same notation.

**Options as query variants.** Given a query *q*, an LLM produces rewritings *q₁ … q₃₀*. Each is run
against the same collection *D*. The predictor scores each variant and the system runs the winner. This
is the setting of Arabzadeh et al., and pre-retrieval predictors do well in it.

**Options as evidence sources.** Given a query *q*, a video collection is indexed several ways at once:
speech transcripts, on-screen text, frame embeddings. Each option is a channel or a fused subset of
channels. The query does not change. What changes is which representation of the documents the query is
matched against.

Written side by side, the two differ in exactly one place. In variant selection the option set varies on
the **query** side and the collection is held fixed. In source selection the option set varies on the
**document** side and the query is held fixed.

Everything in §5 follows from that sentence.

A pre-retrieval predictor is a function of the query text and statistics of the collection. In variant
selection its first argument varies across options, so it has something to discriminate with, and the
literature's results are results about that. In source selection its first argument is constant across
options by construction. The only way such a predictor can tell two options apart is through its second
argument, the collection statistics — which is why we built a separate index per channel and measured
whether that repair works (§5.1). It is also why QL is the clean limit case rather than a straw man: the
reference implementation defines it as the query's token count, a function of the first argument alone,
so in source selection it returns the same number for every option no matter what one does to the
indices.

This framing predicts which predictors should escape, and the prediction is testable before the
experiment. A predictor survives the move to source selection if its input varies with the option on the
document side. Score-only post-retrieval predictors read the retrieved list, and each channel returns a
different list, so they qualify. QSD's pre-retrieval variant reads historical queries together with their
observed effectiveness *under the option being scored*, so it qualifies too, despite sitting in the
pre-retrieval category. Clarity needs a language model estimated over the retrieved documents, which for
a frame index does not exist, so it fails in the strongest possible way: not weakly, but undefined.

The second difference is smaller to state and harder to work around. In variant selection every option
applies to every document, and only quality varies between them. A rewriting that retrieves badly still
*retrieves*. In source selection an option may not apply to a document at all. A silent clip has no
speech to transcribe; a clip with no titles or captions burned into the frame has no on-screen text. This
is a property of the document, decided long before the query arrives, and no amount of query-side
statistics can recover it.

Text retrieval offers no precedent because the situation cannot arise there. Every document in a text
collection has terms. The question of whether a modality is available never comes up, so no predictor has
ever needed to answer it.

We are not claiming these two instantiations exhaust selection. Retrieval systems choose among rerankers,
among retrieval depths, among expansion budgets, and each of those varies on its own side of the query–
document divide. The claim is narrower and, we think, more useful: **the predictor family that works
depends on which side the options differ on**, and the pre/post-retrieval taxonomy the field organises by
tracks that only by accident.
