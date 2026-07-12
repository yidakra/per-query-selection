# Does the expected-gain router generalize across cells?

**Verdict: the router is a real, transferable mechanism — not a per-cell artifact — but it transfers
within a retrieval *regime*, not across one.** The cheap-tier confidence signal moves cleanly across
encoders (MultiCLIP ↔ InternVideo2) and across the ASR/noASR setting, but it does **not** cross the
MultiVENT ↔ MSR-VTT boundary, where it flips sign. A single router pooled over five cells transfers
with significant positive gain to five of the six.

`router_transfer.py`. Train the A→B gain ridge on one cell, apply it **unchanged** to another, and
measure `ρ(predicted gain, true gain)` on the target. Off-diagonal transfer is inherently honest
(disjoint data); the diagonal is within-cell out-of-fold. CPU-only, reads cached component tensors.

The diagonal reproduces `router_hetero.py`'s within-cell OOF ρ **exactly**
(`[0.164, 0.233, 0.048, 0.093, 0.044, 0.127]`), which cross-validates the reimplementation.

## 1. Transfer splits cleanly by regime

Mean off-diagonal transfer ρ, grouped by how far the target is from the source (confidence-only
features; full `A_COLS` in parentheses):

| regime | mean transfer ρ | reads as |
|---|---|---|
| same encoder, across ASR/noASR | **+0.139** (+0.147) | transfers as well as within-cell |
| across encoder, same dataset (mCLIP ↔ IV2) | **+0.135** (+0.128) | transfers freely |
| across dataset (MultiVENT ↔ MSR-VTT) | **−0.037** (−0.057) | does not transfer; sign flips |

The flat +0.04 off-diagonal *average* is an artifact of mixing these: two regimes transfer at
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
sees. ρ on the held-out cell, permutation p over 2000 shuffles, and the realized frontier gap at a
fixed f = 0.5 (full `A_COLS`):

| held-out cell | LOCO ρ | within-cell ρ | gap@0.5 | p |
|---|---|---|---|---|
| MSR/IV2/noASR | **+0.156** | +0.044 | +0.69 | .0005 |
| MSR/IV2/ASR | **+0.143** | +0.127 | +0.64 | .0005 |
| MSR/mCLIP/ASR | **+0.127** | +0.093 | +0.38 | .0005 |
| MSR/mCLIP/noASR | **+0.124** | +0.048 | +0.33 | .0005 |
| MultiVENT/mCLIP/ASR | **+0.180** | +0.233 | +1.06 | .018 |
| MultiVENT/mCLIP/noASR | +0.064 | +0.164 | +1.39 | .116 |

**Five of six transfer significantly.** For all four MSR-VTT cells the pooled router *beats* the
router trained on that cell's own data (LOCO ρ > within-cell ρ) — pooling denoises cells whose own
A→B signal is weak (within-cell ρ as low as +0.04). The pooled model is not just portable, it is
**better than local fitting** where the local signal is thin.

The one cell that does not transfer is **MultiVENT/noASR** (ρ = +0.064, p = .12). Its pool is
four-fifths MSR-VTT — the wrong regime — so the cross-dataset penalty from §1 lands hardest here.
Note its gap@0.5 point estimate is still +1.39, but ρ is not significant: do not claim it.

## 3. Confidence features carry the transfer; text features do not

Dropping the two query-text surface features (`charlen`, `wordlen`) barely moves anything — the
off-diagonal mean *rises* slightly (+0.044 vs +0.033) and the regime pattern is unchanged. The
transferable signal is entirely in the cheap tier's **retrieval confidence** (top-1 margin, gallery
entropy, top-5 mass, score sd), consistent with `router_diag.py`'s finding that the query surface
form carries no routing signal. The lone exception is transfer *into* MultiVENT, where the
length features help the pooled model tell long news queries apart from short MSR-VTT ones — a
dataset cue, not a complexity cue — which is why full `A_COLS` LOCO beats confidence-only on the two
MultiVENT cells but not the four MSR-VTT ones.

## 4. What this means for the paper

- The A→B router is **not a per-cell curve-fit.** The same confidence→gain relationship holds across
  encoders and audio settings, and a single pooled model deploys to 5 of 6 cells with significant
  positive gain. This is the generalization evidence the routing claim needed.
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
