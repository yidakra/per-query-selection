# Adaptive routing over Q2E fusion tiers — findings

Ladder (cost = **measured GPU energy per query**, J; normalized so Full = 1.0). Single source of
truth `tier_cost.py`; measurement in `cost_model_findings.md`.

| tier | components | J/query | cost (Full = 1) |
|---|---|---|---|
| A (visual) | `query_vs_video` | 6.83 | 0.0048 |
| B (−Events) | `+ query_vs_captions` | 22.35 | 0.0158 |
| Full | all 5 (adds LLM event decomposition) | 1418.36 | 1.0 |

A's component is a **subset** of B's, which is a subset of Full's, so a cascade pays only for the
components it ends up scoring. Escalation is free; there is no re-scoring penalty.

> **The cost axis is measured energy, not a component count.** Earlier figures used
> `cost = # similarity components scored` ({A,B,Full} = {0.2, 0.4, 1.0}); that proxy rated all five
> components at unit cost, but they differ by up to 68× (`cost_model_findings.md`). Every frontier
> figure and the tables below now use the measured marginal joules from `tier_cost.py`. The router's
> whole operating range is the cheapest **0.5–1.6% of the Full budget**, not the 20–40% the proxy
> implied. The gaps are **identical** either way — cost is affine in the escalated fraction `f`, so
> cost-matched == `f`-matched in any unit; only the x-positions moved. Verified: regenerating the
> frontier JSONs changed only the `cost` field, everything else byte-for-byte. See §5.

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

| cell | gold/q | sd(gain) | ρ (perm p) | τ (perm p) | nested gap | oracle (in-sample) | "captured" |
|---|---|---|---|---|---|---|---|
| MSR-VTT mCLIP noASR | 1.01 | 7.95 | +0.048 (.066) | +0.014 (.280) | +0.32 ± 0.10 | +1.46 | 22% |
| MSR-VTT mCLIP ASR | 1.01 | 9.81 | +0.093 (.003) | +0.057 (.012) | +0.36 ± 0.10 | +2.15 | 17% |
| MSR-VTT IV2 noASR | 1.01 | 11.24 | +0.044 (.087) | +0.067 (.006) | +0.54 ± 0.15 | +1.96 | 28% |
| MSR-VTT IV2 ASR | 1.01 | 13.81 | +0.127 (.0005) | +0.122 (.0005) | +0.96 ± 0.13 | +3.09 | 31% |
| MultiVENT noASR | 9.24 | 15.22 | +0.164 (.0065) | +0.121 (.0035) | +0.73 ± 0.22 | +5.04 | 14% |
| MultiVENT ASR | 9.24 | 17.02 | +0.233 (.0005) | +0.171 (.0005) | +1.68 ± 0.25 | +6.03 | 28% |

Significance is a broad plateau over `f ≈ 0.35–0.90`, not a single lucky point.

> **Both correlations are reported, following the QPP-benchmark convention.** iQPP (Poesina et al.,
> 2023) and VQPP (Lutu et al., 2026) grade a query-performance predictor by its Pearson *r* **and**
> Kendall *τ* against true effectiveness; our per-query gain predictor is a (differential) QPP model,
> so we report the same pair. ρ is linear; τ is rank-only and outlier-robust, and both are computed
> on the same OOF predictions with a shared 2000-shuffle permutation null. τ agrees with ρ on the
> five stronger cells (all p < .05). The exception is the lowest-heterogeneity cell, MSR-VTT/mCLIP/
> noASR, whose rank association is not significant (τ = +0.014, p = .28) though its linear ρ grazes it
> — the honest reading is that its per-query signal is marginal, exactly as its low sd(gain) predicts.
> **What we predict is a *differential* — nDCG_B − nDCG_A — not absolute AP**, a noisier target than
> the single-system effectiveness those benchmarks predict, so these magnitudes are not comparable to
> iQPP's τ ≈ 0.65 ceiling.

> **The `oracle` column is an in-sample quantity and the `captured` column is not a measure of
> remaining headroom.** Both are kept above because they are what the experiment computed, but see
> §4: the oracle orders queries by a gain measured on the same labels it is scored against, and
> most of that gain is irreducible label noise. The `nested gap` column is out-of-fold and stands.

### Caveats — do not overclaim

- **The router never beats Fixed-B in absolute nDCG.** At a budget just under tier B (cost 0.0152 ≈
  B's 0.0158) MultiVENT noASR reaches 74.15 vs Fixed-B's 74.28. The win lives *strictly between*
  Fixed-A and Fixed-B, i.e. at budgets where B cannot be run on every query and the only fixed
  alternative is a cost-matched
  random mixture. This is a legitimate accuracy–compute frontier claim. It is **not** "we beat Q2E".
- **The Full tier is never purchased.** The original phrasing here — *"ρ(B→Full) = +0.094 (p = .07);
  event decomposition's benefit is not predictable from retrieval confidence"* — cited only the cell
  that supported it. Both cells:

  | | ρ(B→Full) | perm p | oracle B→Full gap |
  |---|---|---|---|
  | MultiVENT noASR | +0.094 | .070 | +2.55 |
  | MultiVENT ASR | +0.137 | **.011** | +2.56 |

  The ASR cell **is** significant. So B→Full gain is *weakly* predictable, not unpredictable, and
  "not predictable" was an overclaim by omission.

  The conclusion is unchanged, but the reason is different: what kills the escalation is the size of
  the prize, not the inability to see it. The oracle gap is only ≈ +2.55 either way — and §4 shows
  that figure is itself an in-sample quantity whose advantage does not survive a held-out label split
  (optimism 3.36 / 3.76, out-of-sample −0.47 / −0.65). There is nothing there worth buying, which
  still bounds the whole approach.

  §5 adds an independent reason on the *price* rather than the prize: the Full tier costs a measured
  **63.5× tier B**, not the 2.5× the component-count proxy implies.
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

The heterogeneity thesis also predicts **where the router transfers** — see `transfer_findings.md`
(`router_transfer.py`). Training the A→B ridge on one cell and deploying it unchanged on another
transfers at within-cell strength across encoders (mCLIP ↔ IV2, mean ρ +0.14) and across the
ASR/noASR setting (+0.15), but **flips sign across the MultiVENT ↔ MSR-VTT boundary** (−0.05): the
transferable signal is bounded by the query-complexity *regime*, not the encoder. A single router
pooled over five cells (leave-one-cell-out) deploys with significant positive gain to **5 of 6**
cells, and *beats* local fitting on the four MSR-VTT cells whose own A→B signal is thin. The router
is a portable mechanism, not a per-cell curve-fit.

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

## 5. The cost axis is measured energy — how the proxy was retired

The frontier figures and §2 tables now plot the **measured joules** below (via `tier_cost.py`),
not the old component count. This section is the measurement and the proof that the swap was safe.
Full write-up: `cost_model_findings.md`. Measured marginal joules per query, MultiVENT noASR, GPU1:

| component | measured | proxy |
|---|---|---|
| `query_vs_video` | 6.83 J | 1 unit |
| `query_vs_captions` | 15.51 J | 1 unit |
| one event component (as published, `mx_q=30`) | 465.34 J | 1 unit |

Four of the five components share one code path and differ only in how many query-side strings they
feed it (259 vs 7,770). Cost is affine in that count — `E(N) = 3167 + 0.912·N` J per doc slot,
R² = 0.9994, verified by held-out extrapolation to N=7,770 within 1.3%.

| tier | measured | normalised | proxy |
|---|---|---|---|
| A | 6.83 J | 0.0048 | 0.2 |
| B | 22.35 J | 0.0158 | 0.4 |
| Full | 1,418.36 J | 1.0 | 1.0 |

**The gaps in §2 are invariant — and this was checked, not just argued.** Cost of escalating a
fraction `f` from A to B is `cost(A) + f·(cost(B) − cost(A))` — affine in `f` under any cost
assignment — so a cost-matched baseline is an `f`-matched baseline in either unit. Regenerating
`router_gain_curve.json` and `router_curves.json` under the joules axis changed **only** the `cost`
field of each point; `ndcg`, `chord`, `gap`, the CIs, permutation `p`, `nested_gap` (+0.73 / +1.68)
and `oracle_gap` (+5.04 / +6.03) are byte-for-byte identical. Nothing in §2, §3 or §4 moves.

**Two things do move.** The B→Full escalation is now dead on price as well as prize (63.5× vs the
proxy's 2.5×), and the reported savings against Full are revealed as a lower bound, since the proxy
understates Full's cost — before even counting the ~30 LLaMA-70B generations per query that only
Full pays.

Along the way: tier A's query path runs the ViT-H **vision tower on a batch of black images** and
discards the output (`vision_embedder.py:148`). Bypassing it makes tier A 8.3× cheaper (6.83 → 0.83
J/query). We report the as-shipped figure, since that is what the published nDCG paid.

## Reproduce

```
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_diag.py        # negative result (§1)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_gain_curve.py  # permutation-tested curve (§2)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_hetero.py      # all 6 cells (§3)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_oracle_goldsplit.py  # ceiling audit (§4)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_figs.py        # figures
CUDA_VISIBLE_DEVICES=1  python src/evaluation/component_energy_bench.py --padded  # cost model (§5)
CUDA_VISIBLE_DEVICES=1  python src/evaluation/component_energy_tierA.py           # cost model (§5)
```

CPU-only except the two `component_energy_*` scripts, which need GPU1. All read cached component
tensors under `runs/<tag>/cache/`.
