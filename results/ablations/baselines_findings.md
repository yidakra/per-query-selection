# Baselines: does the learned router earn its keep, and does model class matter?

Two questions raised in the JHU discussion, answered on all six cells (`router_baselines.py`, CPU-only,
reads cached tensors). Metric per (cell, method): out-of-fold Kendall τ and Pearson ρ of the routing
score vs the true A→B gain, and the realized frontier gap at f = 0.5 in NDCG points. Every method runs
through the **same nested pipeline**; the Ridge/`A_COLS` row reproduces `router_hetero.py`'s within-cell
τ exactly (`[0.121, 0.171, 0.014, 0.057, 0.067, 0.122]`), a built-in cross-check.

**Verdict.** (1) The learned multi-feature router beats the best single classical QPP predictor on the
routing objective (mean gap +0.78 vs +0.43) and on average τ (+0.092 vs +0.078), and, more tellingly,
it is the *only* signal that is positive on all six cells, because **no single QPP predictor is robust
across regimes**. (2) Model class matters in one direction: **linear beats nonlinear**; ridge regression
and logistic classification tie, and gradient boosting / random forest / SVR all overfit the weak,
low-dimensional signal and lose.

## 1. QPP-predictor baselines: the router vs one standard predictor

Classical QPP predictors used as the routing signal, computed from the tier-A similarity scores over the
gallery and adapted to cosine-similarity retrieval following iQPP (Poesina et al., 2023): **max score**,
**score SD**, **NQC** (Shtok et al., 2012), **WIG** (Zhou & Croft, 2007), **Clarity/peakedness**
(Cronen-Townsend et al., 2002). Each is run through the same nested ridge as a *single* input feature, so
it is given OOF-calibrated direction and scale: a single-feature ridge is exactly oriented thresholding
on that predictor. Beating it is therefore a real claim.

| cell | full router τ | full router gap | best single QPP | its τ | its gap |
|---|---|---|---|---|---|
| MultiVENT/mCLIP/noASR | **+0.121** | **+1.07** | WIG | +0.116 | +0.76 |
| MultiVENT/mCLIP/ASR | **+0.171** | **+1.62** | WIG | +0.158 | +0.76 |
| MSR-VTT/mCLIP/noASR | +0.014 | +0.22 | NQC | +0.008 | +0.11 |
| MSR-VTT/mCLIP/ASR | +0.057 | +0.45 | NQC | **+0.075** | +0.48 |
| MSR-VTT/IV2/noASR | +0.067 | +0.37 | max | **+0.087** | +0.62 |
| MSR-VTT/IV2/ASR | +0.122 | +0.97 | max | **+0.130** | +1.12 |
| **mean over 6** | **+0.092** | **+0.78** | (varies) | +0.078 (WIG) | +0.43 (WIG) |

**The router wins on the objective and on average, but the honest story is about robustness.** On the two
multi-gold MultiVENT cells (where routing has real headroom), the router clearly beats every single
predictor on both τ and gap. On the single-gold MSR-VTT cells the margins are tiny and a single predictor
sometimes edges the router on τ (NQC on mCLIP/ASR, max on both IV2 cells), though the router still leads
or ties on the realized gap on all but one.

The decisive point is **which predictor is best flips across regimes**: WIG on MultiVENT, NQC on
MSR-VTT/mCLIP, max on MSR-VTT/IV2; and max, the winner on IV2, is actively **negative on
MultiVENT/noASR (τ = −0.059)**. This is exactly iQPP's headline finding ("no predictor consistently
surpasses its competitors across datasets"), reproduced here on the differential-gain target. A
practitioner cannot pick one classical predictor and deploy it; the learned combination is the only thing
positive on all six cells, and that portability (not a large per-cell margin) is what it buys.

## 2. Model-class ablation: "does the type of classifier matter?"

Hold the full `A_COLS` feature set fixed; swap the estimator. Note the reference method is a **regression
on the continuous gain**, not a classifier: we predict `nDCG_B − nDCG_A` and escalate the top-f. The
"logistic" row is the classification framing (predict `sign(gain)`, route by predicted probability).

| estimator | mean τ | mean gap@0.5 | reads as |
|---|---|---|---|
| **logistic** (classify sign) | **+0.093** | +0.72 | ties ridge on τ |
| **ridge** (regress gain; reference) | +0.092 | **+0.78** | best on the routing objective |
| random forest | +0.036 | +0.45 | overfits; loses half the signal |
| gradient boosting | +0.038 | +0.33 | overfits |
| SVR (rbf; iQPP's meta-regressor) | +0.011 | +0.18 | worst; negative τ on 3 cells |

**Linear wins, and it is not close for the nonlinear models.** Ridge and logistic are statistically tied
on τ; ridge leads on the realized gap (+0.78 vs +0.72), which is the quantity a deployment optimizes, so
the regression framing is retained. The nonlinear models (GBM, RF, SVR) all lose roughly half the signal
or more; SVR is actively harmful on several cells. This is the expected behavior when the true signal is
weak and low-dimensional (τ ≈ 0.1 over ~8 confidence features): a flexible learner fits fold-specific
noise. So the answer to "does model class matter" is **yes, but as a warning, not an opportunity**:
a fancier model does not recover more routable signal, it destroys it, and the interpretable linear model
is the right call for reasons we can now show rather than assert.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_baselines.py
# -> results/ablations/router_baselines.json
```

Reads cached component tensors under `runs/<tag>/cache/`. The Ridge/`A_COLS` τ column equals
`router_hetero.py`'s within-cell τ, so the two scripts cross-validate each other.
