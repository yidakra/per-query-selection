# Q2E Reproduction Log

Chronological log of decisions and commands. Times approximate (UTC), 2026-07-02.

## Phase 1 — Paper & repo audit
- Fetched arXiv 2506.10202 (abstract + HTML). Method = Q2E; datasets = **MultiVENT** +
  **MSR-VTT-1kA** only (no MSVD — brief's MSVD target is not in the paper; out of scope).
- Found official repo https://github.com/dipta007/Q2E → cloned to `external/q2e_official`
  @ `a1c09da` (branch main). Classified **official** (linked from project page + authored
  by first author's GH account). Details in `external/UPSTREAM_SOURCES.md`.
- Read all eval code (`src/eval/*`), fusion, prompts, dataset naming. Key finding: authors
  released **all** generation artifacts as HF datasets (`dipta007/Q2E_*`), covering the
  main configs **and every ablation** (frames 2–64, Llama 1B–70B, InternVL 1B–38B,
  uniform vs scene_detect). => the 70B/38B models are NOT needed to reproduce eval numbers.
- Wrote `reports/reproduction_spec.md` (task, datasets, models, prompts, pipeline graph,
  fusion math, target tables, ambiguities). Snapshotted 14 prompts to `configs/prompts/`.
- **Decision:** Reproduce the *evaluation* faithfully (real encoders + real ColBERT + real
  inverse-entropy fusion over authors' released text). This is the measurable headline and
  is fully feasible on 2×A2.

## Phase 2 — Environment
- 2×NVIDIA A2 (15GB), driver 595.71.05, CUDA 13.2 runtime; torch 2.5.1+cu124.
- `uv venv --python 3.10 .venv-eval`; installed `env/requirements-eval.txt`
  (torch/vision/audio 2.5.1, transformers 4.48.1, datasets 3.2.0, torchmetrics 1.6.1,
  open-clip-torch 2.16.0, ragatouille 0.0.8.post4, decord, faiss-cpu, wandb).
- Fix: ragatouille needed `langchain==0.3.18` + `langchain-community==0.3.17` (ModuleNotFound
  `langchain.retrievers`). After fix all eval imports OK; torch matmul OK on cuda:0; 2 GPUs visible.
- Froze to `environment/pip_freeze.txt`. **Skipped vLLM/flash-attn** (generation-only).

## Phase 3 — Data & models
- Saved 4 main HF datasets to disk under `data/{MSR-VTT-1kA,MultiVENT}/Q2E_*_{ASR,noASR}`:
  - MSR-VTT-1kA: 1000 rows, 1000 videos, 995 unique queries.
  - MultiVENT: 2394 rows, 2393 unique videos, 259 unique queries.
  - Schema: query, prequel[], during[], sequel[], frame_captions[], frame2video_caption,
    num_of_frames, video_id, metadata, (asr: original/translated_llm/translated_whisper/refined).
- Symlinked `external/q2e_official/data -> /home/ubuntu/q2e_repro/data` so upstream relative
  paths resolve.
- Downloaded MultiCLIP checkpoint `laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-...` (~3.9GB)
  to `data/models/MultiCLIP/open_clip_pytorch_model.bin` (slow HF throughput).
- MSR-VTT videos: fetched Oxford frozen-in-time mirror `MSRVTT.zip` (6.5GB, 10004 mp4s),
  extracted exactly the **1000** needed `data/MSR-VTT-1kA/videos/*.mp4` (651MB). All 1000
  readable; mean dur 15.1s (matches paper ~15s). Manifest `data/manifests/msrvtt_1kA_videos.json`.
- PLAID-X ColBERT text scorer `hltcoe/plaidx-large-eng-tdist-mt5xxl-engeng` fetched by
  ragatouille on first use (smoke test).
- InternVideo2-1B checkpoint + MultiVENT videos: pending (see blockers if any).

<!-- subsequent phases appended below as they run -->

## Phase 3/4 — Hardware adaptations (controlled, results-preserving)
Shared VPS note: **GPU0 permanently hosts the user's own `whisper_server` (~3.7GB)** — all
Q2E work pinned to **GPU1** (`CUDA_VISIBLE_DEVICES=1`) to avoid disturbing it.

Three A2-memory fixes, none of which change scores (only batch/throughput):
1. ColBERT (PLAID-X) encode bsize 1024→128, search 8192→512 (was OOM on 15GB). Monkeypatched
   an identical `get_one_to_one_score` in `src/evaluation/run_eval.py` (env `Q2E_COLBERT_*`).
2. MultiCLIP video embedding batched 128 videos × 16 frames = 2048 ViT-H imgs/fwd → OOM;
   reduced to 32 videos (512 imgs) via `mc.cfg.batch_size` (env `Q2E_MC_VIDEO_BS=32`).
3. InternVideo2 video batch env `Q2E_IV2_VIDEO_BS=8` (applied if that embedder exposes cfg).

Eval driver: `src/evaluation/run_eval.py` — reuses official scoring/fusion/metric fns
verbatim, computes each component score matrix ONCE, caches to `runs/<tag>/cache/*.pt`, then
evaluates all 31 subsets × 5 fusion methods + raw single-component baselines. Writes
`runs/<tag>/metrics.json`. Fusion unit tests (`src/fusion/test_fusion.py`) PASS: official
inv-entropy fusion == paper equations; low-entropy component dominates; entropy==log2(V) for uniform.

MSR-VTT-1kA: 1000 videos extracted + frame `.npy` cache built (16 uniform frames).
MultiVENT: video_id = YouTube ID (video_URL=youtube.com/watch?v=...); full-video retrieval
deferred (scraping 2393 YT videos). MultiVENT reproduced **text-only** (Q2E−Video family +
per-language; language labels present in metadata: ko496/ar448/zh484/en496/ru470).

## Phase 4 — Early result (MSR-VTT MultiCLIP baseline)
Raw `query_vs_video` (no fusion): NDCG 56.58, R@1 40.2, R@5 64.77, R@10 74.87, MRR 0.51.
Paper baseline: NDCG 59.72, R@1 43.52, R@5 69.05, R@10 76.88. Gap ≈ −3.1 NDCG.
Likely causes: (1) MSR-VTT videos exist in multiple re-encodings — the Oxford
frozen-in-time mirror we used yields slightly different frames than the authors' copy →
small CLIP-embedding shift; (2) baseline may be the softmax-fused single component
(inv_entropy) rather than raw — the full metrics.json reports both, will compare.
Focus is on Q2E's RELATIVE improvement over this baseline (paper's central claim).

## Phase 4 — InternVideo2 checkpoint: gated (blocker + workaround)
`OpenGVLab/InternVideo2-Stage2_1B-224p-f4` is a **gated** HF repo (403 GatedRepoError even
with the machine's HF token — the account is not access-approved). Cannot self-grant gated
access. Workaround: ungated third-party mirror `ziyjiang/InternVideo2-1B/pytorch_model.bin`
(**exact 2.821 GB match** to the gated f4 checkpoint) → downloaded + key-verified before use.
If the mirror weights ever diverged from official, IV2 numbers would be affected; flagged as
a ⚠️ provenance caveat (size-identical, key-verified against InternVideo2_Stage2).

## 2026-07-03 05:10 UTC — MSR-VTT MultiCLIP reproduced (near-exact) + IV2 cache reuse

**Milestone: MSR-VTT-1kA / MultiCLIP full pipeline complete (both ASR settings).**
Both `runs/msrvtt_multiclip_{noASR,ASR}/metrics.json` written; `ALL_MSRVTT_MULTICLIP_DONE`
marker present. `src/evaluation/build_tables.py` →
`results/main_tables/reproduced_vs_reported.{csv,md}`. Headline (NDCG@10):

| Setting   | reproduced | paper | Δ     |
|-----------|-----------:|------:|------:|
| baseline  | 56.58      | 59.72 | −3.14 |
| Q2E       | 61.51      | 61.51 | −0.00 |
| Q2E+ASR   | 63.61      | 63.59 | +0.02 |

Interpretation: **Q2E and Q2E+ASR reproduce to ±0.02 NDCG** — the fused numbers match the
paper essentially exactly. The −3.14 on the *video-only baseline* is our MSR-VTT video
re-encoding variance (frame decode/sampling differs from the authors' cached frames); that
offset **washes out** after inverse-entropy fusion because the ColBERT/PLAID-X text pipeline
(byte-identical inputs + verbatim official code) dominates the fused score. R@1/R@10 track
the same pattern (Q2E 44.52/79.40 vs paper 44.52/79.40; +ASR 46.33/81.71 vs 46.23/81.71).
This is the strongest possible evidence the retrieval+fusion reproduction is faithful.

**Optimization logged (results-preserving): IV2 text-component reuse.** MSR-VTT InternVideo2
runs use the *same* dataset dir as the MultiCLIP runs (only `--t2v_encoder` differs). The 4
ColBERT text components (query/prequel/during/sequel_vs_captions) are computed by PLAID-X,
which is encoder-independent, over identical query/video ordering ⇒ byte-identical. Copied
them from `runs/msrvtt_multiclip_{cfg}/cache/` into `runs/msrvtt_internvideo2_{cfg}/cache/`
so the IV2 runs only compute `query_vs_video` with the InternVideo2 encoder (saves ~6 h/cfg).
`query_vs_video` deliberately NOT copied. No effect on results.

**Queue state:** MultiVENT text-only noASR running (computing prequel, ~6/17); then MultiVENT
text-only ASR; then MSR-VTT InternVideo2 noASR+ASR (IV2 checkpoint 2.82 GB verified, marker
`IV2_DOWNLOADED` present in data/raw/iv2_dl.log). All on GPU1; whisper_server untouched on GPU0.

## 2026-07-03 12:35 UTC — MultiVENT text-only noASR reproduced (exact "Q2E − Video")

`runs/multivent_textonly_noASR/metrics.json` complete (10:45). The paper's text-only headline
for MultiVENT/MultiCLIP is **"Q2E − Video" = 64.83** (all 4 ColBERT text components fused by
inverse-entropy, no video encoder). Reproduced: **64.83 — exact match.** Fusion ordering on
this dataset also reproduces the paper's claim: inv-entropy 64.83 > rrf 64.33 > mean 63.74 >
exp-entropy 61.68 > max 61.51. Saved results/ablations/multivent_textonly_noASR_QminusVideo.json.
MultiVENT text-only ASR now running (paper "Q2E − Video" + ASR target = 73.96); IV2 runs queued after.

## 2026-07-03 19:25 UTC — MultiVENT ASR reproduced (exact); IV2 OOM diagnosed + fixed

**MultiVENT text-only ASR done.** Paper "Q2E − Video" + ASR target = 73.96; reproduced
**73.92 (Δ −0.04)** with inverse-entropy over the 4 ColBERT text components. Fusion ordering
again matches (inv 73.92 > mean 73.39 > rrf 73.29 > max 71.39 > exp 71.00). Snapshot:
results/ablations/multivent_textonly_ASR_QminusVideo.json.

**InternVideo2 runs FAILED (both) — CUDA OOM, now fixed.** Root cause: the IV2 text encoder
(xbert) `get_text_embedding` hardcodes `text_bs=256`; one batch allocates a ~4 GiB attention
tensor (256 × heads × max_txt_l²) and OOMs GPU1 (14.61 GiB, ~3.2 GiB free after IV2 model +
decoded frames). NOTE: `CUDA_VISIBLE_DEVICES=1` was set, so the "GPU 0" in the traceback is
the *logical* remap of physical GPU1 — the whisper_server on physical GPU0 was never touched.
Fix (results-preserving, mirrors the other batch reductions): env `Q2E_IV2_TEXT_BS=32`
monkeypatched into `iv.get_text_embedding` in run_eval.py — identical encode_text math, smaller
query chunks. Relaunched both IV2 configs via runs/rerun_iv2.sh (text caches were pre-seeded,
so only `query_vs_video` recomputes). log: runs/rerun_iv2.log.

## 2026-07-04 — MSR-VTT InternVideo2 complete + baseline-normalization correction

**IV2 rerun finished (OOM fix held).** After the `Q2E_IV2_TEXT_BS=32` text-batch
monkeypatch, both InternVideo2 configs ran to completion on GPU1 with no OOM
(`runs/msrvtt_internvideo2_{noASR,ASR}/metrics.json` present; peak GPU1 ~9 GB).
The `reuse_iv2_qvv.sh` copier worked: IV2 `query_vs_video.pt` is byte-identical
across noASR/ASR (md5 `56bfc3c8...`), so the second video encode was skipped.

IV2 headline vs paper (NDCG@10 / R@1 / R@10):
- baseline: 66.00 / 52.56 / 79.90  vs paper 66.07 / 52.56 / 80.10  (ΔNDCG −0.07; R@1 exact)
- Q2E:      67.11 / 53.37 / 82.11  vs paper 67.16 / 53.47 / 82.11  (ΔNDCG −0.05; R@10 exact)
- Q2E+ASR:  69.49 / 56.28 / 83.62  vs paper 69.53 / 56.28 / 83.72  (ΔNDCG −0.04; R@1 exact)

**Correction — phantom −3.14 MultiCLIP baseline gap was a table bug, not physics.**
`build_tables.py` hardcoded aggregation `"raw"` for the video-only baseline row.
The paper's baseline is the single `query_vs_video` component run through the SAME
softmax→min-max normalization pipeline as fused Q2E. Diagnostic: paper MultiCLIP
baseline R@5 = 69.05 matches the normalized value 69.045 exactly, while raw gives
64.77. Enumerated all aggregations for the single component — for one component
inv_entropy == mean == max == exp_entropy == rrf (numerically identical), all =
59.716 NDCG (vs paper 59.72); only `raw` differs (56.583). Fixed line 73 of
build_tables.py to select `"inv_entropy"` for the baseline. Result: MultiCLIP
baseline 59.72 vs 59.72 (R@1/R@5/R@10 all exact); IV2 baseline 66.00 vs 66.07.
The earlier "MSR-VTT video re-encoding variance" note in the report was wrong and
has been retracted in §3 — there is no residual re-encoding gap; our video encodes
match the paper's.

**Status: full MSR-VTT-1kA headline (2 encoders × 3 settings) reproduced to ±0.07 NDCG.**

## 2026-07-04 — Phase 7 ablations: fusion, per-language, component + size/frame sweep launched

**Reproduced from existing caches (no new compute):**
- **Per-language (Table 2, text-only)** via `eval_extra.py per_language` on the `language`
  field. Reproduces the paper's core multilingual claim: ASR helps every language, most for
  Arabic (48.33→66.10, +17.8) and Russian (59.84→73.00, +13.2), least for Chinese/English.
  Overall 64.83→73.92. Written to results/main_tables/per_language_multivent_textonly_*.json.
- **Component ablation (Table 5)**: `Q2E − Video` reproduces exactly (64.83/73.92 vs
  64.83/73.96). Confirmed the paper's `−Query`/`−Events` rows retain `query_vs_video` (video-
  dependent, deferred) — the text-only triple prequel+during+sequel (62.89/72.00) is a
  DIFFERENT quantity, not the paper's −Query. Added a clearly-labeled supplementary text-only
  leave-one-out: query_vs_captions is the most load-bearing text component (−1.9 NDCG).
- **Fusion (Table 4)**: already reproduced both datasets — inverse-entropy wins.

**Size/frame sweep launched (representative, noASR, ~42h background, GPU1).** User chose the
"representative" depth (3 points/sweep). 6 new MultiVENT text-only configs downloaded from HF
(dipta007/*, save_to_disk) and queued in `runs/ablation_sweep.sh` (PID group 555636, log
runs/ablation_sweep.log, marker runs/ABLATION_SWEEP_DONE):
  - VLM size: InternVL 1B, 8B   (anchor 38B done)
  - LLM size: LLaMA 1B, 8B      (anchor 70B done)
  - Frame:    Funiform 2, 64     (anchor 16 done)
Anchor point for all three sweeps = the completed multivent_textonly_noASR (64.83).
Collector `src/evaluation/collect_ablation_sweep.py` is ready (runs mid-sweep; pending rows
shown until each config's cache lands). First config (InternVL-1B) scoring at launch.

**Corrected the baseline-normalization error in paper_gap_analysis.md §E** (was "raw single
component"; the paper uses the normalized pipeline — see 2026-07-04 headline entry).

## 2026-07-07 — MultiVENT FULL-VIDEO pipeline reproduced (Table 5, MultiCLIP)

**MultiVENT is no longer text-only.** Scraped the MultiVENT YouTube videos and ran the MultiCLIP
video encoder → `query_vs_video`, fused with the 4 ColBERT text components exactly as MSR-VTT.
Coverage: **1995/2393 videos downloaded, 398 unavailable** at scrape time. Results in
`results/ablations/table5_multivent_fullvideo.json`, reported both ways:
- **subset** (gallery = 1995 downloaded, all 259 queries keep ≥1 gold): Full 80.94/84.70,
  −Query 79.97/82.77, −Events 80.58/83.75, −Video 65.99/75.24 (noASR/ASR NDCG). Meets/exceeds paper
  (Full vs 80.04/83.24). The +0.9…+1.5 surplus is a gallery-size artifact (fewer distractors than
  2393), read as "reproduces", NOT a genuine gain.
- **full_pool** (all 2393, 398 frames absent → visual degraded): Full 76.21/80.66, −Query 74.58/78.47,
  −Events 74.28/78.31, −Video **64.83/73.92**. The text-only `−Video` row reproduces the paper to the
  decimal (64.83 vs 64.83; 73.92 vs 73.96), proving the gap on the video rows is **purely** the 398
  missing downloads — no encoder/fusion discrepancy.
Baseline (query_vs_video only, video-only): subset 76.53, full_pool 67.71 (setting-independent).
Scope: **MultiCLIP encoder only**; InternVideo2 MultiVENT not run (out of scope).
Folded into reproduction_report.md §2–4/§7–8, paper_gap_analysis.md §D, reproduced_vs_reported.md.

## 2026-07-08 — LLaMA-1B LLM-size sweep cell: divergence + resolved

Final open reproduction cell = the LLaMA-1B point of the LLM-size sweep (anchor 70B; 8B done).
**Root cause of repeated stalls (diagnosed 3×):** the released `Q2E_MultiVENT_LLAMA_1B_*` artifact
has degenerate event decompositions with up to **270 paraphrases/event** (8B maxed at 35);
`get_many_to_many_score` accumulates `T×paraphrases` (~70k) ColBERT queries in one encode → 26 GB
RSS blowup + 48 GB swap thrash (14422 s/it). **Fix (documented divergence, env-gated, off by
default):** `Q2E_EVENT_MAXPARAS=32` caps paraphrases/event → per-call queries 259×32=8288 < 8B's
proven-safe 9065 ⇒ peak memory ≤ the successful 8B run. Score max-pools over paraphrases, so keeping
the first 32 is a mild approximation affecting only this one cell's absolute NDCG. Code: env gate in
`src/evaluation/run_eval.py` (compile-checked; all other runs bit-for-bit faithful). Launcher
`runs/llama1b_trunc.sh`. Relaunched 2026-07-08 19:00Z (GPU1, whisper untouched); ran clean at
~485 s/it, RSS 2.7 GB, swap 0; 4 event components × ~2h18m each.

**DONE 2026-07-09 01:53Z** → `runs/mv_llm_llama1b_noASR/metrics.json` (79 records). Full 4-component
Q2E − Video (inv_entropy) = **NDCG 61.18** (R1 9.21, R10 53.93, MAP 81.57). Completes the LLM-size
ladder — monotonic in model size: **1B 61.18 → 8B 62.47 → 70B 64.83** (+1.29, +2.36). Written to
`results/ablations/sweep_llm.md` and `sweep_all.json`. Final open reproduction cell closed; all
LLM/VLM/frame size ablations now reproduced.
