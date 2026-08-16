# Pre-registration: the query-variant experiments

Written and committed before any variant is generated or scored (2026-08-16). Supervision raised
two confounds against the pre-retrieval null: the queries are one fixed formulation, and the source
study varied formulation across six generation methods. These experiments answer that, and the
readings of every outcome are fixed here, in advance, so no result gets to choose its own story.

## Design

**Task B (run first): replicate the source study's query-variant selection task on MultiVENT 2.0.**
Variants are generated with the source study's own toolkit (QueryGym, Apache-2.0), all six methods
(GenQR, GenQR-Ensemble, MuGI, QA-Expand, Query2Doc, Query2Exp), five samples per method plus the
original query: 31 candidates per query, 2,546 queries. Every method concatenates its expansion onto
the original query per its QueryGym recipe; per-method temperature defaults from the toolkit are
pinned in the committed config, and MuGI's `adaptive_times` is pinned to 5 (the value its
concatenation code actually uses). Variants are executed against the dense speech channel (bge-m3,
the strongest re-runnable channel), with the speech+OCR RRF fusion as the secondary pipeline.
Predictors: the same analytic suite as Table 1, computed per variant. Scoring: within-query
correlation aggregated across queries, plus the selection task (predictor-picked variant executed,
against original-query, best-single-reformulator, and oracle rows), nDCG@10 and R@100.

**Task A: the channel-selection null, re-tested per formulation.** For each of the six method
pools (and the original), the symmetric nested protocol runs on the speech-vs-OCR binary cell and
the {speech, OCR, both} k-way cell. The visual channel is excluded: the benchmark ships it as a
frozen ranked list with no frame embeddings or text tower, so it cannot be recomputed for new query
text, and a proxy channel inside a robustness test would prove nothing. Folds stay event-grouped;
all variants of a query inherit the query's event group, so no variant of a test-fold query is ever
seen in training.

**Language variants (route-by-language).** The 2,546 queries translated into the four largest
non-English corpus languages (zh, ko, ru, ar) with NLLB-200-distilled-600M (queries are single
sentences; the corpus side used the 1.3B model, and a seeded 50-query back-translation sample is
committed for inspection). Each language variant runs the same channel cells; the analysis reports
whether the best channel and the routing gap move with query language.

## Pre-registered readings

1. **Task B, replication succeeds** (pre-retrieval predictors select variants above the original
   query, directionally consistent with the source study): the boundary claim gets its strongest
   form, both halves on one collection: corpus statistics select queries and do not select sources
   under one protocol. This is the expected outcome.
2. **Task B, replication fails** (no predictor family selects variants above original here): the
   story is no longer a boundary between decisions; it becomes "the source study's result does not
   transfer to multimodal corpora." Publishable, different, and the one-pager headline changes.
   We commit to reporting this outcome as prominently as outcome 1.
3. **Task A, null holds across formulations** (corpus-statistic family stays not-significant in
   every method pool): the formulation confound is closed, the null is a property of the decision,
   and the one-pager says so in one sentence.
4. **Task A, any formulation rescues the family** (significant after the same Holm-corrected
   group tests, in any pool): the claim is rewritten as formulation-conditional, the rescuing
   formulation is named in the claim block, and the abstract's "0 of 33" framing is retired. We
   commit to this rewrite in advance.
5. **Route-by-language**: if the picked channel distribution or the routing gap shifts materially
   by query language (differences beyond the cluster-bootstrap CIs), language enters the selector
   as a feature and the one-pager gains a language paragraph; if not, the answer to supervision is
   "channel choice is not language-driven on this benchmark," with the table as evidence.

## Fixed protocol details

Same as Table 1 throughout: nested escalation-fraction calibration on group-disjoint inner splits,
event-grouped 5-fold outer CV, group-level sign-flip tests with Holm correction within family,
cluster-bootstrap 95% CIs, equivalence bounds at ±0.005 nDCG. No protocol knob changes between
formulations or languages. The generation endpoint, model, and every prompt go into the repo with
the outputs; generated variants are data and are committed once, then frozen.

## Known deviations from the source study, stated in advance

- Generator model is not GPT-4o (theirs); QueryGym's own reproducibility runs used open Qwen
  models, so generator identity is not load-bearing. Ours is pinned in the committed config.
- Their collection is text (MS MARCO v2.1, 56 needs); ours is multimodal video (2,546 queries).
  Statistical power moves from pool size to query count.
- Their retrievers are BM25 and Cohere-dense; ours is bge-m3 dense (primary) and RRF fusion
  (secondary).
- Visual channel excluded from Task A (frozen artifact), stated above.
