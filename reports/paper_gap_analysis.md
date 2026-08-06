# Q2E Paper Gap Analysis

Status legend: ✅ exact match to paper method · 🟰 faithful via released artifact ·
⚠️ controlled deviation (results-preserving) · ⛔ deviation that may affect numbers ·
⏸️ deferred / not run. Numeric gaps are in `results/main_tables/reproduced_vs_reported.*`
and `reports/reproduction_report.md`.

## A. What matched the paper exactly (method + code)
- ✅ **Evaluation metrics**: used the repo's `evaluation.py` verbatim (torchmetrics
  RetrievalRecall/Precision/MRR/NDCG/MAP top_k=10; mean/median rank). Same definitions.
- ✅ **Inverse-entropy fusion**: repo `fusion_score.py` used verbatim; independently
  unit-tested to equal the paper's equation (pre-softmax over queries, `1/H·P`, min-max).
- ✅ **Text-text scorer**: PLAID-X ColBERT `hltcoe/plaidx-large-eng-tdist-mt5xxl-engeng`
  via ragatouille, the exact default in `infer.py`; max-aggregation many-to-many identical.
- ✅ **Encoders**: MultiCLIP (`CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B`) and
  InternVideo2-1B (`InternVideo2-Stage2_1B-224p-f4`), exact checkpoints from the repo.
- ✅ **Frame sampling**: uniform 16-frame mid-interval (MultiCLIP) / 4-frame (IV2), the
  repo's exact `uniform_sample_frames` / `get_frame_indices(middle)`.
- ✅ **Score components**: the 5 components + 31-subset sweep as enumerated in `infer.py`.

## B. Faithful via authors' released artifacts (🟰)
- 🟰 **Query decomposition / event refinement (Llama-3.3-70B)** and **frame+video
  captioning (InternVL2.5-38B)** and **ASR (Whisper-large-v3) + NLLB translation + LLM
  refinement**: not re-run (models don't fit 15 GB A2). We consumed the authors' released
  HF datasets containing these exact outputs. So the *text inputs* to retrieval are the
  paper's own, byte-for-byte. This is the intended "pre-generated data" path in the README.

## C. Controlled, results-preserving deviations (⚠️)
- ⚠️ **Batch sizes** reduced to fit 15 GB (ColBERT encode 1024→128 / search 8192→512;
  MultiCLIP video 128→32 videos; IV2 video 64→8). Batching changes only memory/throughput,
  not scores. Numerically inert (verified by identical scoring code + fusion unit tests).
- ⚠️ **GPU**: ran on one A2 (GPU1) vs the paper's A100. Same fp32 math.

## D. Deviations that could affect comparability (⛔ / ⏸️ / ⚠️)
- ⚠️ **MultiVENT video download coverage** (was ⏸️, now reproduced): the 2,393 source videos
  are YouTube-scraped (video_id = YouTube ID). **1995/2393 downloaded; 398 unavailable** at scrape
  time. The full MultiVENT video+text pipeline (MultiCLIP + Q2E) is now reproduced: on the
  downloadable **subset** it meets/exceeds the paper (Q2E 80.94 vs 80.04, +ASR 84.70 vs 83.24):
  the small surplus is a gallery-size artifact (1995 < 2393 → fewer distractors), read as "matches";
  on the **full 2393 gallery** the missing frames depress it (Q2E 76.21, +ASR 80.66), lower-bounding
  the coverage cost. The text-only `−Video` row is immune and reproduces to the decimal (64.83/73.92),
  isolating the gap entirely to downloads. **MultiVENT was run with MultiCLIP only**; the
  InternVideo2 MultiVENT rows (paper 50.43→76.10) are ⏸️ not run.
- ⚠️ **LLaMA-1B ablation cell, paraphrase cap (`Q2E_EVENT_MAXPARAS=32`)**: the released
  `Q2E_MultiVENT_LLAMA_1B_*` artifact contains degenerate event decompositions with up to **270
  paraphrases/event** (vs the 8B artifact's max 35); the many-to-many ColBERT scorer accumulates
  `T×paraphrases` queries and OOM/stalls at that width. We cap paraphrases-per-event to 32 (env-gated,
  **off by default** so every other run is bit-for-bit faithful), bounding peak memory to ≤ the
  proven-safe 8B run. Because the score max-pools over paraphrases, keeping the first 32 is a mild
  approximation affecting **only the single LLaMA-1B LLM-size sweep point**, and only its absolute
  NDCG (not any headline or other ablation). Documented in `runs/llama1b_trunc.sh` + code comment.
- ⏸️ **MSVD**: not in the paper (paper uses 2 datasets). Out of scope; no gap to report.
- ⏸️ **Generation-stage reproduction (Tier B)**: re-running decomposition/captioning with
  smaller open models to measure model-size effect is optional; the authors already
  released 1B–70B LLM and 1B–38B VLM artifact variants, so the *size ablation* is
  reproducible from released text without re-running generation.

## E. Row→subset mapping (inferred, verified against numbers)
- baseline encoder row = single component `[query_vs_video]` pushed through the **same
  softmax→min-max normalization pipeline** as fused Q2E (NOT raw dot-products). Verified:
  paper MultiCLIP baseline R@5 69.05 == normalized 69.045 (raw gives 64.77); paper IV2
  baseline R@5 73.07 == normalized 73.065 (raw 74.47). For a single component every fusion
  operator is numerically identical, so the label ("inv_entropy") is immaterial. An earlier
  `build_tables.py` used the raw scores here and produced a phantom −3.14 MultiCLIP-baseline
  gap; fixed. Corrected values: MultiCLIP baseline 59.72 vs 59.72, IV2 baseline 66.00 vs 66.07.
- "+Q2E" (noASR) = inv-entropy fusion of all 5 components on the noASR dataset.
- "+Q2E+ASR" = same 5 components on the ASR dataset (caption pool includes 3 ASR fields).
- Table 5: −Video = drop `query_vs_video`; −Events = `[video, query]`; −Query = drop
  query components. (Confirmed by matching reproduced NDCG to the paper's ablation values.)

## F. Open ambiguities resolved by code
- Pre-softmax is over **queries (dim=0)**, applied once per component (not in paper text).
- MSR-VTT-1kA exact 995-query split inherited from the released dataset.
- Per-language MultiVENT slicing uses the `language` field in row metadata.
