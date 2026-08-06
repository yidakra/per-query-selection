# Rigor pack: the A→B router on the field's vocabulary and error bars

Puts the positive routing result (`router_findings.md` §2) on the metrics and statistics a
routing/IR reviewer expects, computed on the existing cached A→B results; no new retrieval.
`router_rigor.py`, CPU-only. The per-cell Kendall τ reproduces `router_hetero.py` exactly
(`[0.121, 0.171, 0.014, 0.057, 0.067, 0.122]`), so this is a re-analysis, not a re-run.

**Headline: the result survives hardening.** Under Benjamini–Hochberg correction across the
six-cell family, **5 of 6 cells stay significant**; bootstrap 95% CIs on the realized gap exclude
zero for the **same 5**; and the router recovers **16–33% of the (conservatively estimated) oracle
headroom**, needing to escalate only **21–37% of queries to capture half** the full tier-B
improvement. The one cell that drops is the one that was already weakest.

## 1. Frontier envelopes, APGR, and CPT

At each escalation fraction `f`, the router's realized gap lives between two bounds: **random-f**
(escalate a random fraction; expected gap 0, the cost-matched chord) and **oracle-f** (escalate the
top-f by *true* gain). Full per-`f` envelopes are in `router_rigor.json`. Two scalar summaries from
the LLM-routing literature (RouteLLM):

- **APGR** (Average Performance Gap Recovered): mean over `f` of `router_gap(f) / oracle_gap(f)`, the
  fraction of the recoverable oracle-over-random headroom the router captures.
- **CPT** (Call-Performance Threshold): the escalation fraction needed to capture x% of the full
  tier-B improvement. Lower is better.

| cell | τ (perm p) | gap@0.5 (95% CI) | APGR | CPT₅₀ | CPT₈₀ |
|---|---|---|---|---|---|
| MultiVENT/mCLIP/noASR | +0.121 (.0035) | +1.07 [+0.15, +1.98] | 0.208 | 0.33 | 0.65 |
| MultiVENT/mCLIP/ASR | +0.171 (.0005) | +1.62 [+0.61, +2.72] | 0.261 | 0.37 | 0.65 |
| MSR-VTT/mCLIP/noASR | +0.014 (.280) | +0.22 [−0.02, +0.48] | 0.163 | 0.33 | 0.65 |
| MSR-VTT/mCLIP/ASR | +0.057 (.012) | +0.45 [+0.16, +0.75] | 0.315 | 0.32 | 0.62 |
| MSR-VTT/IV2/noASR | +0.067 (.007) | +0.37 [+0.03, +0.78] | 0.194 | 0.32 | 0.51 |
| MSR-VTT/IV2/ASR | +0.122 (.0005) | +0.97 [+0.50, +1.40] | 0.329 | 0.21 | 0.46 |

**APGR replaces the old "% of oracle captured" (14–31%) with the recognized scalar, and it is a
*conservative* reading.** The denominator is the in-sample oracle, which `router_findings.md` §4
showed is optimistic (it does not survive a gold-split); dividing by an inflated ceiling *understates*
the fraction of *real* headroom recovered. So APGR 0.16–0.33 is a floor, not a ceiling.

**CPT gives the deployment number a practitioner wants:** on the strongest cell (MSR-VTT/IV2/ASR) the
router captures half the tier-B gain by escalating just **21%** of queries; even the multi-gold
MultiVENT cells hit 50% at **33–37%**. Escalating *randomly* would need 50% for 50% by definition, so
CPT₅₀ < 0.5 everywhere is the efficiency win stated in one number.

## 2. Multiple-comparison correction: the story survives

Six cells means six τ significance tests; uncorrected stars invite a multiplicity objection. Applying
**Benjamini–Hochberg (FDR)** across the six-cell family:

| cell | p_raw | p_BH | FDR<0.05 |
|---|---|---|---|
| MultiVENT/mCLIP/ASR | .0005 | .0015 | ✓ |
| MSR-VTT/IV2/ASR | .0005 | .0015 | ✓ |
| MultiVENT/mCLIP/noASR | .0035 | .0070 | ✓ |
| MSR-VTT/IV2/noASR | .0065 | .0097 | ✓ |
| MSR-VTT/mCLIP/ASR | .0115 | .0138 | ✓ |
| MSR-VTT/mCLIP/noASR | .2799 | .2799 | ✗ |

**5 of 6 survive** at FDR < 0.05. The only casualty is MSR-VTT/mCLIP/noASR, the lowest-heterogeneity
cell (sd(gain) = 7.95, the smallest), whose τ was already non-significant (+0.014) and whose bootstrap
CIs straddle zero. Correction removes exactly the cell the heterogeneity thesis predicts is dead, and
leaves every cell the thesis predicts is live. This is corroboration, not a loss.

## 3. Bootstrap confidence intervals

95% CIs from 2000 query-level bootstrap resamples (predictor fixed, evaluation resampled):

- **Kendall τ**: excludes 0 for all cells except MSR-VTT/mCLIP/noASR (τ CI [−0.036, +0.064]).
- **gap@0.5**: excludes 0 for the same 5 cells; MSR-VTT/mCLIP/noASR is [−0.02, +0.48].

The bootstrap and the FDR correction agree cell-for-cell, which is the point: the routing gain is
real and sized with error bars on 5 of 6 cells, and honestly null on the 6th.

## What this changes for the paper

- Report **APGR + CPT + oracle/random envelopes** as the primary frontier summary (RouterBench /
  RouteLLM vocabulary), not an ad-hoc "% captured".
- Lead with **BH-FDR-corrected** significance and **bootstrap CIs**; the uncorrected per-cell stars
  become supporting detail. Foreground that the result survives correction; most routing papers do
  not report it.
- Keep the in-sample-oracle caveat visible: APGR's denominator is conservative, and the gold-split
  (§4) is the reason we can say so.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_rigor.py
# -> results/ablations/router_rigor.json
```

Reads cached component tensors under `runs/<tag>/cache/`. τ column equals `router_hetero.py`'s.
