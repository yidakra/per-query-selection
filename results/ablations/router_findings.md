# Adaptive routing over Q2E fusion tiers — findings

Ladder (cost = number of similarity components scored; Full = 5 → normalized 1.0):

| tier | components | cost |
|---|---|---|
| A (visual) | `query_vs_video` | 0.2 |
| B (−Events) | `+ query_vs_captions` | 0.4 |
| Full | all 5 (adds LLM event decomposition) | 1.0 |

A's component is a **subset** of B's, which is a subset of Full's, so a cascade pays only for the
components it ends up scoring. Escalation is free; there is no re-scoring penalty.

## 1. Negative result: the routing signal is not in the query text

`router_frontier.py`, `router_diag.py`. TF-IDF (word 1–2 gram + char_wb 3–5 gram) → logistic
regression over the query string, labelled with Adaptive-RAG's `cheapest_within_eps` rule,
scored strictly out-of-fold (`cross_val_predict`, vectorizer refit per fold).

Tier accuracy lands **at or below the majority-class prior**: MultiVENT noASR 0.37 vs prior 0.36;
MSR-VTT 0.90 vs prior 0.86. Every operating point is dominated by the constant Fixed-B policy.

Adaptive-RAG's premise — that query complexity is predictable from the question's surface form —
**does not transfer to text-to-video retrieval**. Whether event decomposition helps depends on the
query × gallery *interaction*, not on the query alone. This is consistent with MultiVENT's queries
being 259/259 Latin script: the multilinguality lives in the videos and captions, not the queries,
so there is no query-side language feature to route on.

## 2. What works: expected-gain cascade on cheap-tier retrieval confidence

Route on the **cheap tier's own retrieval signal** — top-1 margin, margin@1-3, z-score of the top
hit, softmax entropy over the gallery, top-5 mass, score sd — and regress the *gain* rather than
classify the tier. Cascade discipline: the A→B decision may use only A-side features.

Predict `g_i = nDCG_B(i) − nDCG_A(i)` (ridge, OOF), escalate the top-`f` by predicted gain. The
frontier gap over the cost-matched baseline then has a closed form:

```
gap(f) = f · ( mean_{i∈S} g_i  −  mean_i g_i )
```

so the entire claim reduces to *does the ranker order queries by true gain?* — exactly
permutation-testable. Randomly escalating a fraction `f` reproduces the chord between Fixed-A and
Fixed-B in expectation, so **the chord is the honest baseline**, not Fixed-B alone.

Operating point `f` is chosen by **nested CV** (picked on training folds, scored held-out), so the
headline number carries no selection bias.

| cell | gold/q | sd(gain) | ρ(pred,true) | perm p | nested gap | oracle | captured |
|---|---|---|---|---|---|---|---|
| MSR-VTT mCLIP noASR | 1.01 | 7.95 | +0.048 | .066 | +0.32 ± 0.10 | +1.46 | 22% |
| MSR-VTT mCLIP ASR | 1.01 | 9.81 | +0.093 | .003 | +0.36 ± 0.10 | +2.15 | 17% |
| MSR-VTT IV2 noASR | 1.01 | 11.24 | +0.044 | .087 | +0.54 ± 0.15 | +1.96 | 28% |
| MSR-VTT IV2 ASR | 1.01 | 13.81 | +0.127 | .0005 | +0.96 ± 0.13 | +3.09 | 31% |
| MultiVENT noASR | 9.24 | 15.22 | +0.164 | .0065 | +0.73 ± 0.22 | +5.04 | 14% |
| MultiVENT ASR | 9.24 | 17.02 | +0.233 | .0005 | +1.68 ± 0.25 | +6.03 | 28% |

Significance is a broad plateau over `f ≈ 0.35–0.90`, not a single lucky point.

### Caveats — do not overclaim

- **The router never beats Fixed-B in absolute nDCG.** At cost 0.39 (≈ B's 0.40) MultiVENT noASR
  reaches 74.15 vs Fixed-B's 74.28. The win lives *strictly between* Fixed-A and Fixed-B, i.e. at
  budgets where B cannot be run on every query and the only fixed alternative is a cost-matched
  random mixture. This is a legitimate accuracy–compute frontier claim. It is **not** "we beat Q2E".
- **The Full tier is never purchased.** ρ(B→Full) = +0.094 (p = .07) on MultiVENT noASR, and even the
  *oracle* gap for B→Full is only +2.55. Event decomposition's per-query benefit is not predictable
  from retrieval confidence. That is itself a finding, and it bounds the whole approach.
- The router captures only 14–31% of oracle headroom.

## 3. The load-bearing claim: routing value scales with heterogeneity

Across all six cells, `sd(per-query A→B gain)` predicts both the oracle headroom and the achieved
gap with **Spearman ρ = +0.943** (n = 6; one-tailed critical value at α = .05 is 0.829).

Crucially the driver is heterogeneity, *not dataset identity*: MSR-VTT/internvideo2/ASR has
sd = 13.81, close to MultiVENT's 15.22, and its gap (+0.96) lands where the trend predicts —
despite 1.01 gold/query. The fraction of headroom the router *captures* stays roughly flat
(14–31%); it is the headroom itself that scales.

Related: **17% of MultiVENT queries are actively hurt** by adding captions (vs 2–4% on MSR-VTT).
That is what a router is for, and it is the quiet critique of Q2E's one-size-fits-all fusion.

## Reproduce

```
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_diag.py        # negative result (§1)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_gain_curve.py  # permutation-tested curve (§2)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_hetero.py      # all 6 cells (§3)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_figs.py        # figures
```

CPU-only; reads cached component tensors under `runs/<tag>/cache/`.
