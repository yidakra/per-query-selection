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

| cell | gold/q | sd(gain) | ρ(pred,true) | perm p | nested gap | oracle (in-sample) | "captured" |
|---|---|---|---|---|---|---|---|
| MSR-VTT mCLIP noASR | 1.01 | 7.95 | +0.048 | .066 | +0.32 ± 0.10 | +1.46 | 22% |
| MSR-VTT mCLIP ASR | 1.01 | 9.81 | +0.093 | .003 | +0.36 ± 0.10 | +2.15 | 17% |
| MSR-VTT IV2 noASR | 1.01 | 11.24 | +0.044 | .087 | +0.54 ± 0.15 | +1.96 | 28% |
| MSR-VTT IV2 ASR | 1.01 | 13.81 | +0.127 | .0005 | +0.96 ± 0.13 | +3.09 | 31% |
| MultiVENT noASR | 9.24 | 15.22 | +0.164 | .0065 | +0.73 ± 0.22 | +5.04 | 14% |
| MultiVENT ASR | 9.24 | 17.02 | +0.233 | .0005 | +1.68 ± 0.25 | +6.03 | 28% |

Significance is a broad plateau over `f ≈ 0.35–0.90`, not a single lucky point.

> **The `oracle` column is an in-sample quantity and the `captured` column is not a measure of
> remaining headroom.** Both are kept above because they are what the experiment computed, but see
> §4: the oracle orders queries by a gain measured on the same labels it is scored against, and
> most of that gain is irreducible label noise. The `nested gap` column is out-of-fold and stands.

### Caveats — do not overclaim

- **The router never beats Fixed-B in absolute nDCG.** At cost 0.39 (≈ B's 0.40) MultiVENT noASR
  reaches 74.15 vs Fixed-B's 74.28. The win lives *strictly between* Fixed-A and Fixed-B, i.e. at
  budgets where B cannot be run on every query and the only fixed alternative is a cost-matched
  random mixture. This is a legitimate accuracy–compute frontier claim. It is **not** "we beat Q2E".
- **The Full tier is never purchased.** ρ(B→Full) = +0.094 (p = .07) on MultiVENT noASR, and even the
  *oracle* gap for B→Full is only +2.55. Event decomposition's per-query benefit is not predictable
  from retrieval confidence. That is itself a finding, and it bounds the whole approach.
- The router captures only 14–31% of oracle headroom — **but that fraction is not meaningful**, and
  §4 shows why. Its denominator is inflated by label noise.

## 3. The load-bearing claim: routing value scales with heterogeneity

Across all six cells, `sd(per-query A→B gain)` predicts both the oracle headroom and the achieved
gap with **Spearman ρ = +0.943** (n = 6; one-tailed critical value at α = .05 is 0.829).

Crucially the driver is heterogeneity, *not dataset identity*: MSR-VTT/internvideo2/ASR has
sd = 13.81, close to MultiVENT's 15.22, and its gap (+0.96) lands where the trend predicts —
despite 1.01 gold/query. The fraction of headroom the router *captures* stays roughly flat
(14–31%); it is the headroom itself that scales.

Related: **17% of MultiVENT queries are actively hurt** by adding captions (vs 2–4% on MSR-VTT).
That is what a router is for, and it is the quiet critique of Q2E's one-size-fits-all fusion.

## 4. The oracle ceilings are in-sample, and mostly label noise

Added after the tier-C study (`tierC_findings.md`), which found that an oracle choosing paraphrase
subsets against the test labels reports +2.00 nDCG of headroom that no method can reach. The same
critique applies to the `oracle` column above, so we measured it rather than assumed it:
`router_oracle_goldsplit.py`.

The ceiling is computed as `gap_at(f, ghat=g, g=g)` — queries are ordered by the **true** per-query
gain `g`, measured on the very relevance judgments used to score the result. That is in-sample
selection. MultiVENT's 9.24 golds/query let us split each query's golds into halves and re-run it.

| MultiVENT | A→B ceiling, full golds (9.24/q) | in-sample, half golds (4.62/q) | out-of-sample | optimism |
|---|---|---|---|---|
| noASR | +5.04 | +6.77 ± 0.20 | −1.58 ± 0.45 | 8.35 |
| ASR | +6.03 | +7.54 ± 0.19 | −1.17 ± 0.50 | 8.71 |

Two things follow, and the first needs no transfer argument at all:

**Halving the gold set raises the ceiling** (+5.04 → +6.77, +6.03 → +7.54). Deleting labels strictly
removes information. A ceiling that measured recoverable headroom could not rise; a ceiling that
measures an oracle's capacity to *fit* the labels must. It rises.

**The ordering does not transfer.** Escalate the queries chosen on one gold half, grade on the
other, and you do worse than escalating none of them.

The same holds for B→Full (optimism 3.36 / 3.76), which reinforces §2's caveat that the Full tier is
never purchasable.

### What this does not mean

- **The nested gaps (+0.73 / +1.68) are unaffected.** They are out-of-fold, permutation-tested, and
  remain the honest headline. Only the oracle *denominator* moves.
- **The out-of-sample gap being negative does not mean achievable gain is negative.** That oracle
  estimates each query's gain from ~4 golds and is variance-dominated: it escalates queries whose
  gain is *noisily* large. The nested-CV router pools across training queries, trading variance for
  bias, and wins. A pooled learner beating a per-query oracle fed noisy labels is not a paradox.
- **The ceiling is still a valid bound**, just a very loose one. On a fixed label set `gap(f)` is
  maximised by ordering on the true gain, so no router can exceed it *on those labels*. It is a
  bound, not a target — and "% of oracle captured" is therefore not a measure of what is left.
- **MSR-VTT cannot be tested this way** (1.01 golds/query — no second half to grade on). Its
  ceilings (+1.46 to +3.09) stay labelled in-sample.

### The heterogeneity thesis survives

§3 correlates `sd(gain)` with *both* the oracle headroom and the achieved gap (ρ = +0.943 for each).
The first of those correlations inherits this contamination. The second does not: `nested gap` is
out-of-fold, and on its own it carries the claim. Report the achieved-gap correlation as the
load-bearing one; the oracle-headroom correlation is at best corroborative and is partly a statement
about how much label noise each cell has.

### Methodological upshot

Report oracle headroom with a held-out label split whenever the dataset is multi-gold. It costs one
extra evaluation. Where it is impossible (single-gold data), label the ceiling in-sample and do not
build a "% captured" narrative on it.

## Reproduce

```
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_diag.py        # negative result (§1)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_gain_curve.py  # permutation-tested curve (§2)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_hetero.py      # all 6 cells (§3)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_oracle_goldsplit.py  # ceiling audit (§4)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_figs.py        # figures
```

CPU-only; reads cached component tensors under `runs/<tag>/cache/`.
