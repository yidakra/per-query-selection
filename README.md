# adaptive-q2e

**Per-query adaptive routing for zero-shot multilingual text-to-video retrieval.** Q2E decomposes
every query with an LLM and fuses all five similarity components at a fixed cost. This project muxes
Q2E's fusion tiers with an Adaptive-RAG-style complexity router that spends *per query* (visual-only
for easy queries, full event decomposition only where it pays), targeting the **accuracy–compute
frontier** rather than a single operating point.

The router is built on a **faithful reproduction of Q2E** (Dipta & Ferraro, IJCNLP-AACL 2025,
[arXiv:2506.10202](https://arxiv.org/abs/2506.10202)) as its validated base. Official code:
https://github.com/dipta007/Q2E (cloned to `external/q2e_official`, commit `a1c09da`).

> **Private, pre-publication.** Unpublished findings. Keep private until write-up.

Three tiers over Q2E's components, nested so escalation is free. A cascade pays only for what it
ends up scoring:

| tier | components | LLM calls / query |
|---|---|---|
| **A** | `query_vs_video` | 0 |
| **B** | `+ query_vs_captions` (= Q2E's `−Events` ablation) | 0 |
| **Full** | `+ {prequel,during,sequel}_vs_captions` | **~30** |

---

## The paper

The ECIR 2027 paper is written in a separate repository, `yidakra/project_a`. This one keeps the
experiments, the generated tables under `results/ablations/`, and the evidence documents in `reports/`
that those tables are verified against. Prose lives over there. Numbers are re-derived here.

The mapping from a claim in the paper to the artifact behind it is the evidence map in the paper
repository.

`reports/task_definition.md` states what this project evaluates and under which protocol: the
decision tasks, the collections and judgments, the query sets, the evaluation protocol and the
predictor families. It is written to be read on its own, and is the right starting point for anyone
who has not seen the project before.

---

## Findings

All on MultiVENT (259 queries, 2393 videos) and MSR-VTT-1kA, nDCG@10. Write-ups:
[`results/ablations/router_findings.md`](results/ablations/router_findings.md) and
[`results/ablations/tierC_findings.md`](results/ablations/tierC_findings.md).

### 1. The routing signal is not in the query text, but it is in the cheap tier's own confidence

A TF-IDF + LogReg router over query text scores **at or below the majority-class prior** and is
dominated by the constant Fixed-B policy. Adaptive-RAG's premise (complexity is predictable from the
question's surface form) does **not** transfer to text-to-video retrieval: whether event decomposition
helps depends on the query × gallery interaction, not the query alone.

Routing on the cheap tier's *retrieval confidence* works. Predict the per-query gain
`g_i = nDCG_B(i) − nDCG_A(i)`, escalate the top-`f`. The whole claim reduces to *"does the ranker
order queries by true gain?"*, exactly permutation-testable.

| | ρ(pred, true) | perm p | nested-CV gap vs cost-matched chord |
|---|---|---|---|
| MultiVENT noASR | +0.164 | .0065 | **+0.73 ± 0.22** |
| MultiVENT ASR | +0.233 | .0005 | **+1.68 ± 0.25** |

Significance is a broad plateau over `f ≈ 0.35–0.90`, not one lucky point. The operating point is
chosen by nested CV, so it carries no selection bias.

**Caveats, stated up front.** The router **never beats Fixed-B in absolute nDCG**. At cost 0.39 it
reaches 74.15 vs Fixed-B's 74.28. Its win lives *strictly between* Fixed-A and Fixed-B, where the only
fixed alternative is a cost-matched random mixture (the chord). This is an accuracy–compute frontier
claim, **not** "we beat Q2E".

**The Full tier is never purchased.** B→Full gain is far less predictable than A→B (ρ = +0.094,
p = .07 on noASR; +0.137, p = .011 on ASR). The ASR cell is significant, so the honest statement is
*weakly* predictable, not unpredictable. What kills the escalation is the size of the prize: the
oracle B→Full gap is only +2.55 / +2.56, and §5 shows that figure is itself an in-sample quantity
whose advantage does not survive a held-out label split (optimism 3.36 / 3.76). Event decomposition's
per-query benefit is not worth buying, which is itself a finding, and it bounds the approach.

### 2. The load-bearing claim: routing value scales with heterogeneity

Across six cells, `sd(per-query A→B gain)` predicts the achieved gap with **Spearman ρ = +0.943**
(n = 6; one-tailed critical value 0.829 at α = .05). The driver is heterogeneity, *not dataset
identity*: MSR-VTT/InternVideo2/ASR has sd = 13.81, close to MultiVENT's 15.22, and its gap lands
where the trend predicts despite 1.01 gold/query.

This is the differentiator from Adaptive-RAG. Relatedly, **17% of MultiVENT queries are actively
hurt** by adding captions (vs 2–4% on MSR-VTT), the quiet critique of Q2E's one-size-fits-all fusion.

### 3. The cost proxy is wrong, and wrong in the safe direction

The frontier plots use `cost = #components scored` (A = 0.2, B = 0.4, Full = 1.0), rating all five
similarity components at unit cost. Both halves of that assumption were measured, not assumed.

**The LLM half** ([`llm_cost_accounting.py`](src/evaluation/llm_cost_accounting.py)): the Full tier
issues mean 30.0 LLaMA-3.3-70B generations per query (median 21, max 96), ≈8k prompt / ≈940 generated
tokens. Tiers A and B issue **zero**.

**The similarity half** ([`cost_model_findings.md`](results/ablations/cost_model_findings.md)):
components differ by up to **68×**. Four of the five share one code path and differ only in how many
query-side strings they feed it: 259 for `query_vs_captions`, **7,770** for an event component
(`mx_q=30` padding). Cost is affine in that count: `E(N) = 3167 + 0.912·N` J per doc slot,
R² = 0.9994, confirmed by held-out extrapolation to N=7,770 within **1.3%**.

| tier | measured (marginal, per query) | normalised | proxy |
|---|---|---|---|
| A | 6.83 J | 0.0048 | 0.2 |
| B | 22.35 J | 0.0158 | 0.4 |
| Full | 1,418.36 J | 1.0 | 1.0 |

> **Scope:** this cost model was measured on the **original Q2E pipeline** (ViT-H similarity,
> `mx_q=30` padding, the paper's own corpora). It does **not** describe the MultiVENT 2.0 cascade,
> whose measured latency and energy are in [`reports/efficiency_metrics.md`](reports/efficiency_metrics.md);
> there, tiers A and B are CPU-only and cost ~0.01 J and ~1 J per query. Do not mix the two tables.

The proxy overstates A by 42× and B by 25× relative to Full, i.e. it **understates** Full, so the
reported savings are a **lower bound**. Escalating B→Full really costs **63.5× tier B**, not 2.5×,
which independently kills the Full escalation on price to go with §1's argument on prize.

**The routing gaps are invariant to all of this.** Cost of escalating a fraction `f` from A to B is
`cost(A) + f·(cost(B) − cost(A))`, affine in `f` under *any* per-component cost assignment, so
cost-matched is `f`-matched in either unit. Nothing in §1, §2 or §5 moves; only the x-axis label does.
The cascade's entire gain is obtained in the zero-LLM regime.

Incidental, found while measuring: tier A's query path runs the ViT-H **vision tower on a batch of
black images** and throws the output away (`vision_embedder.py:148`). Bypassing it makes tier A 8.3×
cheaper. We report the as-shipped number, since that is what the published nDCG paid.

### 4. Tier C does not exist: oracle headroom over Q2E is label noise

Q2E max-pools ~24 generated paraphrases per query. Because the pool is a `max`, a bad paraphrase can
never *lower* a video's score, only raise a wrong video's. So dropping the bad ones should be free.
An oracle agrees: **+2.00 nDCG** over Fixed-Full, keeping 0.9 of 24 paraphrases.

It is unachievable. Three independent estimates converge:

| how the subset was chosen | vs Fixed-Full |
|---|---|
| at random, cost-matched | −1.63 |
| by a learned model (nested CV, 11 features) | **−1.49**, 95% CI [−2.24, −0.74] |
| by an oracle denied the grading labels | −1.81 |

The decisive test: MultiVENT is multi-gold (≥4 relevant videos/query), so split each query's golds,
run the *identical* oracle on half A, grade on half B. In-sample **+3.07**, out-of-sample **−1.81**:
**optimism +4.89 nDCG, more than twice the headline it inflates.** The oracle was exploiting *which
videos are marked relevant*, not which paraphrases are good.

**Q2E's max-pool is vindicated**: dropping a paraphrase forfeits a chance to match a *right* video as
often as a wrong one. Neither tier-C design survives: not the learned selector (tested, loses), not
the iterative regeneration loop (its premise just failed).

### 5. Methodological: report oracle headroom against a held-out label split

We applied the gold split to our *own* routing ceilings
([`router_oracle_goldsplit.py`](src/evaluation/router_oracle_goldsplit.py)). It reproduces the
published +5.04 / +6.03 exactly, then:

| MultiVENT A→B ceiling | full golds (9.24/q) | half golds (4.62/q) | out-of-sample | optimism |
|---|---|---|---|---|
| noASR | +5.04 | +6.77 ± 0.20 | −1.58 ± 0.45 | 8.35 |
| ASR | +6.03 | +7.54 ± 0.19 | −1.17 ± 0.50 | 8.71 |

**Halving the gold set raises the ceiling.** This needs no transfer argument: deleting labels
strictly removes information, so a ceiling measuring recoverable headroom cannot rise. One measuring
an oracle's capacity to *fit* the labels must. It rises.

Consequences, carefully scoped:

- The **nested gaps (+0.73 / +1.68) are out-of-fold and unaffected.** Only the oracle *denominator*
  moves. The "% of oracle captured" framing is **retired**.
- A negative out-of-sample gap does **not** mean achievable gain is negative. That oracle estimates
  each query's gain from ~4 golds and is variance-dominated. The nested-CV router pools across
  training queries and wins. A pooled learner beating a per-query oracle fed noisy labels is no paradox.
- The ceiling is still a **valid bound** on a fixed label set, just a very loose one. Bound, not target.
- **MSR-VTT cannot be audited this way** (1.01 golds/query). Its ceilings stay labelled in-sample.
- Consequently the heterogeneity thesis (§2) is carried by the **achieved-gap** correlation, which is
  out-of-fold. The oracle-headroom correlation is corroborative at best.

A refuted hypothesis, recorded so it is not re-proposed: optimism does **not** grow with the size of
the oracle's choice space (3 tiers → optimism 10.55; 2^24 subsets → 4.89). It tracks the **spread in
option quality**, not the count of options.

---

## Reproduce

Paths are derived from `__file__` / `$0`, so the checkout can live anywhere.

```bash
# routing (§1, §2): CPU
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_diag.py             # query-text router fails
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_gain_curve.py       # permutation-tested curve
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_hetero.py           # all 6 cells
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_oracle_goldsplit.py # ceiling audit (§5)
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_figs.py             # figures

# cost model (§3): LLM accounting on CPU, component energy on GPU1
CUDA_VISIBLE_DEVICES="" python src/evaluation/llm_cost_accounting.py
CUDA_VISIBLE_DEVICES=1  python src/evaluation/component_energy_bench.py --repeats 2 --padded
CUDA_VISIBLE_DEVICES=1  python src/evaluation/component_energy_tierA.py

# tier C (§4): one GPU pass, then CPU
CUDA_VISIBLE_DEVICES=1  python src/evaluation/perparaphrase_scores.py --setting noASR --event all
CUDA_VISIBLE_DEVICES="" python src/evaluation/tierC_selection_oracle.py --setting noASR
CUDA_VISIBLE_DEVICES="" python src/evaluation/diag_zerorow_prior.py     --setting noASR
CUDA_VISIBLE_DEVICES="" python src/evaluation/tierC_learned_selector.py --setting noASR
CUDA_VISIBLE_DEVICES="" python src/evaluation/tierC_goldsplit_oracle.py --setting noASR
CUDA_VISIBLE_DEVICES="" python src/evaluation/tierC_optimism_curve.py   --setting noASR
```

The router and tier-C scripts read cached component tensors under `runs/<tag>/cache/`; only
`perparaphrase_scores.py` needs a GPU.

### Base reproduction (evaluation)

Data + models: see `repro_log.md` for exact commands (`runs/fetch_videos.sh`,
`src/evaluation/fetch_all_videos.py`). Then the official entry point:

```bash
cd external/q2e_official
WANDB_MODE=disabled python -m src.eval.infer \
  --note=<tag> \
  --dataset_dir=data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR \
  --aggregation_methods=inv_entropy \
  --t2v_encoder=multiclip
```

`src/evaluation/run_eval.py` wraps this across the {dataset × encoder × ASR} grid and writes
machine-readable metrics into `runs/` and `results/`.

---

## Experiment + energy tracking

Every experiment runs inside [`src/evaluation/tracking.py`](src/evaluation/tracking.py), which wraps
Weights & Biases (`natlang/adaptive-q2e`) and CodeCarbon.

```python
from tracking import track
with track("my-experiment", gpu_ids=[1], config={...}) as t:
    t.summary({"ndcg": 76.19})
```

W&B defaults to **offline** (`runs/wandb/`) unless credentials exist, so nothing leaves the box by
accident. `wandb login`, then `wandb sync runs/wandb/offline-*`.

Read before trusting any energy number:

- **CodeCarbon reads whole-device GPU power via NVML** and cannot attribute draw to a process. This
  box shares GPU0 with an unrelated service, so every tracked job **must** pass the GPU it actually
  uses. `gpu_ids` are *physical NVML indices*, **not** remapped by `CUDA_VISIBLE_DEVICES`: a job
  pinned to GPU1 sees `cuda:0` in torch but must tell CodeCarbon `[1]`. CPU-only jobs pass `[]`.
  `track()` refuses `gpu_ids=None` rather than guess.
- **CPU energy is a TDP-based estimate**, not a measurement (no RAPL powercap exposed). GPU energy is
  measured. We record which mode was used per run. Never sum them into one unqualified headline.
- Carbon intensity is CodeCarbon's bundled **annual** grid mix (NLD, 267.6 gCO2e/kWh), not live.

Per-run records land in `results/energy/{emissions.csv,runs.jsonl}`, which are append-only, so **merge
conflicts there must be resolved as a union**, never by taking one side.

---

## Hardware / tier

2× NVIDIA A2 (15 GB each), 16 vCPU, 31 GB RAM. See `environment/`.

The paper's *generation* models (Llama-3.3-70B, InternVL2.5-38B) do **not** fit. We execute a tiered
plan (`reports/reproduction_spec.md` §3, `reports/compute_cost_report.md`):

- **Tier A (executed)**: faithful reproduction of the *evaluation*, with real encoders (MultiCLIP,
  InternVideo2-1B), real ColBERT/PLAID-X text scorer, real inverse-entropy fusion, over the authors'
  released generation artifacts. Numbers are directly comparable to the paper's tables.
- **Tier B (optional)**: re-run generation with smaller open substitutes to measure model-size
  effects (the authors released 1B–70B LLM and 1B–38B VLM variants, so the *size ablations* are
  reproducible from released text without re-running generation).

### A reproducibility note

Q2E's cached component tensors are **not bitwise reproducible** across sessions (ColBERT runs in
fp16 and its scores depend on batch composition; 40% of `prequel` entries move by >1e-3, max 0.15 on
a 0–100 scale), yet its **reported metrics are** (Full-tier nDCG identical to 4 dp). Correctness
gates in this repo therefore assert on *metrics*, not float equality.

## Layout

- `external/q2e_official/`: upstream repo, unmodified. Its `data/` is a relative symlink to `../../data`.
- `data/`: HF datasets saved to disk, encoder checkpoints, videos, manifests.
- `src/evaluation/`: drivers, the router, the tier-C study, tracking.
- `configs/prompts/`: snapshot of the 14 upstream prompt templates.
- `runs/`: run logs, raw metrics JSON, shell drivers, cached component tensors.
- `results/`: comparison tables, `ablations/` (findings + JSON), `energy/`.
- `reports/`: spec, report, gap analysis, compute cost, figures.
- `environment/`: system/GPU/python/cuda manifests, pip freeze.

**Excluded from version control** (`.gitignore`): `data/` (HF datasets, checkpoints, videos;
re-fetch per `repro_log.md`), `external/` (clone `https://github.com/dipta007/Q2E` @ `a1c09da`),
`.venv-eval/`, and `*.pt` caches under `runs/`. All are large and/or regenerable.

## Environment

```bash
uv venv --python 3.10 .venv-eval
source .venv-eval/bin/activate
uv pip install -r env/requirements-eval.txt
uv pip install "langchain==0.3.18" "langchain-community==0.3.17"   # ragatouille dep
```

Eval-only env; upstream `requirements.txt` / `uv.lock` additionally pin vLLM etc. for generation,
which we skip.

## Status

`repro_log.md` is the running log; `reports/reproduction_report.md` has the reproduced-vs-reported
comparison. **MSVD is not part of this paper** and is out of scope.
