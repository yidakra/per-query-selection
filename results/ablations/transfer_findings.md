# Does the expected-gain router generalize across cells?

**Verdict: the router is a real, transferable mechanism — not a per-cell artifact — but it transfers
within a retrieval *regime*, not across one.** The cheap-tier confidence signal moves cleanly across
encoders (MultiCLIP ↔ InternVideo2) and across the ASR/noASR setting, but it does **not** cross the
MultiVENT ↔ MSR-VTT boundary, where it flips sign. A single router pooled over five cells transfers
with significant positive gain to five of six held-out cells under Pearson ρ, and to **all six** under
the rank-based Kendall τ.

`router_transfer.py`. Train the A→B gain ridge on one cell, apply it **unchanged** to another, and
measure the predictor–gain correlation on the target — both **Pearson ρ** (linear) and **Kendall τ**
(rank-only, outlier-robust), the pair the QPP benchmarks iQPP (Poesina et al., 2023) and VQPP (Lutu
et al., 2026) report for predictor-vs-effectiveness. Off-diagonal transfer is inherently honest
(disjoint data); the diagonal is within-cell out-of-fold. CPU-only, reads cached component tensors.

The diagonal reproduces `router_hetero.py`'s within-cell OOF correlations **exactly** — ρ
`[0.164, 0.233, 0.048, 0.093, 0.044, 0.127]` and τ `[0.121, 0.171, 0.014, 0.057, 0.067, 0.122]` —
which cross-validates the reimplementation.

## 1. Transfer splits cleanly by regime

Mean off-diagonal transfer correlation, grouped by how far the target is from the source
(confidence-only features; full `A_COLS` in parentheses). τ is smaller in magnitude than ρ by
construction, but tells the identical regime story:

| regime | mean ρ | mean τ | reads as |
|---|---|---|---|
| same encoder, across ASR/noASR | **+0.139** (+0.147) | **+0.106** (+0.111) | transfers as well as within-cell |
| across encoder, same dataset (mCLIP ↔ IV2) | **+0.135** (+0.128) | **+0.086** (+0.082) | transfers freely |
| across dataset (MultiVENT ↔ MSR-VTT) | **−0.037** (−0.057) | **−0.031** (−0.052) | does not transfer; sign flips |

The regime split is metric-invariant: both correlations are positive at within-cell strength for the
two intra-regime shifts and negative across the dataset boundary. The flat +0.04 off-diagonal
*average* is an artifact of mixing these: two regimes transfer at
within-cell strength, one is negative and drags the mean to zero. The heatmap
(`reports/figures/router_transfer.png`) shows it directly — a **4×4 positive block** over all four
MSR-VTT cells (both encoders, both settings, ρ = +0.09…+0.19), with **negative corners** wherever
MultiVENT trains MSR-VTT or vice-versa.

**Why the dataset boundary is the hard one.** MSR-VTT is single-gold (1.01 relevant videos/query);
MultiVENT is multi-gold (mean 9.24). The A→B gain being predicted — *does adding caption evidence
raise nDCG@10* — has a different structure when there is one right answer vs ten, so a ranker fitted
to one regime mis-orders the other. This is the **heterogeneity thesis** (`router_findings.md` §3)
appearing again, now as a transfer boundary: routing value and routing *transferability* both track
the query-complexity regime, not the encoder or the audio channel.

## 2. Train-once, deploy-anywhere: leave-one-cell-out

Pool the other five cells, fit once, deploy on the held-out cell. This is the number a deployment
sees. Both correlations on the held-out cell, each with a permutation p over the same 2000 shuffles,
and the realized frontier gap at a fixed f = 0.5 (full `A_COLS`):

| held-out cell | LOCO ρ (p) | LOCO τ (p) | within-cell ρ | gap@0.5 |
|---|---|---|---|---|
| MSR/IV2/noASR | **+0.156** (.0005) | **+0.094** (.0005) | +0.044 | +0.69 |
| MSR/IV2/ASR | **+0.143** (.0005) | **+0.089** (.001) | +0.127 | +0.64 |
| MSR/mCLIP/ASR | **+0.127** (.0005) | **+0.074** (.003) | +0.093 | +0.38 |
| MSR/mCLIP/noASR | **+0.124** (.0005) | **+0.056** (.013) | +0.048 | +0.33 |
| MultiVENT/mCLIP/ASR | **+0.180** (.017) | **+0.151** (.0005) | +0.233 | +1.06 |
| MultiVENT/mCLIP/noASR | +0.064 (.116) | **+0.149** (.001) | +0.164 | +1.39 |

**Five of six transfer significantly under ρ; all six under τ.** For all four MSR-VTT cells the pooled
router *beats* the router trained on that cell's own data (LOCO ρ > within-cell ρ) — pooling denoises
cells whose own A→B signal is weak (within-cell ρ as low as +0.04). The pooled model is not just
portable, it is **better than local fitting** where the local signal is thin.

The one cell where the two metrics disagree is **MultiVENT/noASR**: ρ = +0.064 (p = .12, not
significant) but τ = +0.149 (p = .001, significant). This is not a metric we cherry-picked — it is
diagnostic. MultiVENT's gain distribution is heavy-tailed (sd 15.22; 17% of queries actively *hurt*
by captions), and a handful of extreme-gain queries deflate the linear ρ while the pooled router's
**rank ordering of the cell stays sound**, which is exactly what τ measures. The honest reading: the
router does order MultiVENT/noASR queries by true gain (τ, gap@0.5 = +1.39), but a linear fit to that
cell is outlier-sensitive. We report both and flag the discrepancy rather than claim only the
favorable one — and it is a concrete argument for τ being the more appropriate metric on multi-gold
cells, the same reason the QPP benchmarks report it.

## 3. Confidence features carry the transfer; text features do not

Dropping the two query-text surface features (`charlen`, `wordlen`) barely moves the overall
transfer — the off-diagonal mean *rises* slightly (ρ +0.044 vs +0.033; τ +0.028 vs +0.016) and the
regime pattern is unchanged. The transferable signal is entirely in the cheap tier's **retrieval
confidence** (top-1 margin, gallery entropy, top-5 mass, score sd), consistent with `router_diag.py`'s
finding that the query surface form carries no routing signal. The lone exception is transfer *into*
MultiVENT, where the length features help the pooled model tell long news queries apart from short
MSR-VTT ones — a dataset cue, not a complexity cue — which is why full `A_COLS` LOCO beats
confidence-only on the two MultiVENT cells but not the four MSR-VTT ones. **Kendall τ makes this
sharper than ρ did:** on MultiVENT/ASR, removing the length features flips LOCO τ from +0.151 to
**−0.034** (a sign change, not just a shrinkage), and on MultiVENT/noASR the confidence-only pool
still holds under τ (+0.131, p = .001) but not under ρ (+0.022, p = .30). The rank metric localizes
the dataset cue to the length features more cleanly than the linear one.

## 4. What this means for the paper

- The A→B router is **not a per-cell curve-fit.** The same confidence→gain relationship holds across
  encoders and audio settings, and a single pooled model deploys with significant positive gain to
  5 of 6 cells under Pearson ρ and all 6 under the rank-based Kendall τ. This is the generalization
  evidence the routing claim needed.
- **Report the QPP pair (ρ and τ), not one.** Following iQPP/VQPP, we grade the gain predictor with
  both; on the multi-gold MultiVENT cells the outlier-robust τ and the linear ρ diverge, and the
  divergence is informative (heavy-tailed gains), not noise. A single-metric table would have hidden
  it.
- Transferability is **bounded by the retrieval regime**, not the encoder. State the deployment
  scope honestly: fit the router on data drawn from the target's gold-multiplicity regime (multi-gold
  news vs single-gold clips); do not expect a MultiVENT-fitted router to route MSR-VTT.
- This is a **generalization** result, reported in the efficiency framing: it does not claim to beat
  Full, only that the cheap-tier routing signal is portable within a regime.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_transfer.py
# -> results/ablations/router_transfer.json, reports/figures/router_transfer.{pdf,png}
```
