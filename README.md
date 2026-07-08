# adaptive-q2e

**Per-query adaptive routing for zero-shot multilingual text-to-video retrieval.** Q2E always
decomposes every query and fuses all five similarity components at fixed cost; this project muxes
Q2E's fusion tiers with an Adaptive-RAG-style complexity router that spends *per query* — visual-only
for easy queries, full event decomposition only where it pays — targeting the accuracy–compute
frontier rather than a single operating point. See `reports/` and the research-extension note below.

The router is built on a **faithful reproduction of Q2E** (Dipta & Ferraro, IJCNLP-AACL 2025,
arXiv:2506.10202) as its validated base — that reproduction is the bulk of what follows.

Official Q2E code: https://github.com/dipta007/Q2E (cloned to `external/q2e_official`, commit
`a1c09da`). We reproduce the paper's **evaluation** using the authors' released pre-generated
LLM/VLM/ASR artifacts (HF datasets) plus the actual retrieval encoders and fusion — the faithful,
reproducible path given the hardware.

## Hardware / tier
- 2× NVIDIA A2 (15 GB each), 16 vCPU, 31 GB RAM. See `environment/`.
- The paper's *generation* models (Llama-3.3-70B, InternVL2.5-38B) do **not** fit here.
  We execute a **tiered plan** (see `reports/reproduction_spec.md` §3 and
  `reports/compute_cost_report.md`):
  - **Tier A (executed):** faithful reproduction of the *evaluation* — real encoders
    (MultiCLIP, InternVideo2-1B), real ColBERT/PLAID-X text scorer, real inverse-entropy
    fusion, over the authors' released generation artifacts. Numbers are directly
    comparable to the paper's tables.
  - **Tier B (optional):** re-run generation with smaller open substitutes to measure the
    effect of model size (the authors also released 1B–70B LLM and 1B–38B VLM variants,
    so we can reproduce the *size ablations* from released text without re-running).

## Layout
- `external/q2e_official/` — upstream repo (unmodified). `data/` inside is a symlink to `./data`.
- `data/` — HF datasets saved to disk, encoder checkpoints, videos, manifests.
- `src/` — thin faithful drivers / helpers (fusion unit tests, batch runners, table builders).
- `configs/prompts/` — snapshot of the 14 upstream prompt templates.
- `runs/` — timestamped run logs + raw metrics JSON.
- `results/` — comparison tables (reproduced vs reported).
- `reports/` — spec, report, gap analysis, compute cost, recommendation memo.
- `environment/` — system/GPU/python/cuda manifests, pip freeze.

**Excluded from version control** (`.gitignore`): `data/` (HF datasets, checkpoints, videos —
re-fetch per `repro_log.md`), `external/` (clone `https://github.com/dipta007/Q2E` @ `a1c09da`),
and `.venv-eval/` (rebuild via the Environment section). All are large and/or regenerable.

### Research extension (WIP, private)
Beyond the reproduction, `src/evaluation/oracle_router_headroom{,_msrvtt}.py` and
`build_router_trainset.py` (→ `results/ablations/router_trainset.jsonl`,
`oracle_router_headroom*.json`) prototype a per-query complexity router that muxes Q2E's
fusion tiers with Adaptive-RAG-style routing. Unpublished — keep private until write-up.

## Environment
```bash
uv venv --python 3.10 .venv-eval
source .venv-eval/bin/activate
uv pip install -r env/requirements-eval.txt
uv pip install "langchain==0.3.18" "langchain-community==0.3.17"   # ragatouille dep
```
(Eval-only env; upstream `requirements.txt`/`uv.lock` additionally pin vLLM etc. for
generation, which we skip.)

## Reproduce (evaluation)
Data + models are fetched by `scripts/fetch_*.sh` (see `repro_log.md` for exact commands).
Then, from the upstream repo root, the official entry point:
```bash
cd external/q2e_official
WANDB_MODE=disabled python -m src.eval.infer \
  --note=<tag> \
  --dataset_dir=data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR \
  --aggregation_methods=inv_entropy \
  --t2v_encoder=multiclip
```
Our wrapper `src/evaluation/run_eval.py` calls this across the {dataset × encoder × ASR}
grid and writes machine-readable metrics into `runs/` and `results/`.

## Status
See `repro_log.md` for the running log and `reports/reproduction_report.md` for the
reproduced-vs-reported comparison. **MSVD is not part of this paper** and is out of scope.
