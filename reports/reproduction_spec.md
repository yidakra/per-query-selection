# Q2E Reproduction Spec

Derived from the paper (arXiv:2506.10202, HTML v1) and the official repo
(`external/q2e_official`, commit `a1c09da`). Every claim tagged:
**[PAPER]** confirmed in paper, **[REPO]** confirmed in code, **[INFER]** inferred,
**[UNKNOWN]** unresolved.

---

## 1. Research goal / formal task

**[PAPER]** Zero-shot **text-to-video retrieval** for complex, real-world *event*
queries, including **multilingual** queries. Given a text query `q` and a gallery of
videos `{v}`, rank videos by relevance. "Zero-shot" = no training/fine-tuning on the
target dataset; all components are frozen pretrained models and the fusion is
training-free.

Core idea: human queries are terse and under-specify the event. Q2E **decomposes the
query** into prequel / current ("during") / sequel sub-events using LLM world knowledge,
and **decomposes each video** into text (frame captions + a holistic video caption +
multilingual ASR transcript, translated & refined). Retrieval then becomes a set of
text-text and text-video similarity scores that are combined by **inverse-entropy rank
fusion**.

---

## 2. Datasets

> NOTE: The paper evaluates on **exactly two** datasets: **MultiVENT** and
> **MSR-VTT-1kA**. **MSVD is NOT used in this paper.** (The reproduction brief listed
> MSVD as an ideal target; it is out of scope for a *faithful* reproduction and is
> documented as such.)

### MultiVENT **[PAPER]**
- Multilingual event-centric news video retrieval benchmark (NeurIPS 2023 D&B).
- **259 queries** across **5 languages**: Arabic, Chinese, English, Korean, Russian.
- **2,394 videos**. Avg query length ~27 words; avg video length ~83 s.
- Multiple relevant videos per query → precision/NDCG/MAP are meaningful (not just R@1).
- Videos must be downloaded separately (redistribution restricted).

### MSR-VTT-1kA **[PAPER]**
- Standard "1k-A" test split of MSR-VTT (Yu et al. 2018 split; 1000 test videos).
- **995 queries**, **1,000 videos**. Avg query length ~9 words; avg video length ~15 s.
- One relevant video per query (standard MSR-VTT retrieval).
- **[INFER]** "1k-A" = the JSFusion/Yu et al. test split of 1000 clips, one caption
  sampled per test video (995 unique queries here).

### Metrics **[REPO]** (`src/eval/evaluation.py`)
- `R@{1,5,10}` = RetrievalRecall top-k (×100)
- `Pre@{1,5,10}` = RetrievalPrecision top-k (×100)
- `MRR` = RetrievalMRR (top_k=10), reported 0–1
- `NDCG` = RetrievalNormalizedDCG (top_k=10) ×100  ← **headline metric**
- `MAP` = RetrievalMAP (top_k=10) ×100
- `MeanRank`, `MedianRank` (1-based, lower better)
- All via `torchmetrics.retrieval`. Ranking flattens the QxV sim matrix with per-row
  query indices; targets are boolean QxV.

---

## 3. Model stack

| Role | Model | Source | Fits on 2×A2? |
|---|---|---|---|
| LLM: query decomp + refine + video summary | `meta-llama/Llama-3.3-70B-Instruct` | **[PAPER/REPO]** | ❌ (needs pre-gen data) |
| VLM: frame captioning | `OpenGVLab/InternVL2_5-38B` | **[PAPER/REPO]** | ❌ (needs pre-gen data) |
| ASR | `whisper-large-v3` | **[PAPER]** | ✅ (but pre-gen used) |
| Translation | NLLB | **[PAPER]** | ✅ (but pre-gen used) |
| Retrieval encoder A ("MultiCLIP") | `laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k` | **[REPO]** | ✅ (~1B, fp32 ok) |
| Retrieval encoder B | `OpenGVLab/InternVideo2-Stage2_1B-224p-f4` | **[REPO]** | ✅ (~1B) |
| Text–text similarity | `hltcoe/plaidx-large-eng-tdist-mt5xxl-engeng` (ColBERT/PLAID-X via ragatouille) | **[REPO]** | ✅ |
| Rank fusion | Inverse-entropy (training-free) | **[PAPER/REPO]** | ✅ |
| Reranking | none | **[REPO]** | — |

Generation decoding **[REPO]** (`src/data/utils.py` defaults): temperature=0.8,
top_p=0.95, max_tokens=2048, served by vLLM. Frame captioning uses
`gen_max_model_len=16384`.

---

## 4. Prompt inventory **[REPO]** (`src/eval/../src/data/prompts/*.jinja`)

| Stage | File |
|---|---|
| Main event extraction | `extract_event_info.jinja` |
| Spatial (place) extraction | `extract_spatial_info.jinja` |
| Temporal (time) extraction | `extract_temporal_info.jinja` |
| Prequel events | `prequel.jinja` |
| Current/"during" events | `during.jinja` |
| Sequel events | `sequel.jinja` |
| Refine event→natural query | `refine_event_query.jinja` |
| Frame caption (no ASR) | `frame_caption.jinja` |
| Frame caption (+ASR) | `frame_caption+ASR.jinja` |
| Contextualized frame caption (±ASR) | `contextualized_frame_caption[+ASR].jinja` |
| Holistic video summary (±ASR) | `summary_video_caption[+ASR].jinja` |
| ASR transcript refinement | `refine_translation.jinja` |

Prompts are preserved verbatim in the repo; copies snapshotted under
`q2e_repro/configs/prompts/` (see repro_log). Example (event extraction) asks the LLM to
output `EXPLANATION:` then `EVENTS:` numbered list; prequel/sequel prompts inject world
knowledge and require events "concrete enough to be visualized in a video"; the refine
prompt fuses base query + event + place + time into a natural search query.

---

## 5. Pipeline graph

```mermaid
flowchart TD
  Q[Raw query] --> EV[extract main event]
  Q --> SP[extract spatial]
  Q --> TM[extract temporal]
  Q --> PRE[prequel events]
  Q --> DUR[during events]
  Q --> SEQ[sequel events]
  EV & SP & TM --> RQ[refine -> natural query]
  V[Video] --> FR[uniform 16-frame sample]
  FR --> FC[frame captions - InternVL2.5-38B]
  FC --> VC[holistic video caption - Llama-3.3-70B]
  V --> ASR[Whisper-large-v3 ASR]
  ASR --> TR[NLLB translate + Whisper translate]
  TR --> RF[LLM refine transcript]
  subgraph Scores
    S1[query_vs_video: encoder(Q, frames)]
    S2[query_vs_captions]
    S3[prequel_vs_captions]
    S4[during_vs_captions]
    S5[sequel_vs_captions]
  end
  RQ --> S1
  Q & PRE & DUR & SEQ --> S2 & S3 & S4 & S5
  FC & VC & RF --> CAP[caption pool]
  CAP --> S2 & S3 & S4 & S5
  S1 & S2 & S3 & S4 & S5 --> FUSE[inverse-entropy fusion] --> RANK[ranked videos]
```

### Scoring details **[REPO]** (`src/eval/infer.py`, `text_embedder.py`, `*/vision_embedder.py`)
- **query_vs_video**: encoder (MultiCLIP or InternVideo2) text–video cosine ×100. Frames
  uniformly sampled from raw video (mid-interval of `num_of_frames` bins), mean-pooled.
  MultiCLIP normalizes cos to [0,1] then ×100.
- **{query,prequel,during,sequel}_vs_captions**: text–text ColBERT/PLAID-X max-sim.
  `get_many_to_many_score` = max over (query-variant × caption) pairs, i.e. best-matching
  event vs best-matching caption. Caption pool = frame captions (list) + holistic video
  caption; +ASR adds llm-translated, whisper-translated, and refined transcripts.
- The **with-ASR** run uses params `[vcaptions, ccaptions, llm_asr, whisper_asr, refined_asr]`
  as the caption pool; **no-ASR** uses `[vcaptions, ccaptions]`.

---

## 6. Inverse-entropy rank fusion **[REPO]** (`src/eval/fusion_score.py`, `infer.py`)

For each score matrix `S_i` (QxV):
1. **Pre-softmax** (default `--softmax pre`): `P_i = softmax(S_i, dim=0)` — softmax **over
   queries** (column-wise, dim=0), applied once per component when cached.
2. Row-wise entropy `H(P_i) = -Σ_v P_i log2(P_i + 1e-6)` (per query).
3. Fuse: `S_fused = Σ_i (1/(H(P_i)+1e-6)) · P_i`.
4. `min_max_normalize` per row (dim=-1) → final ranking scores.

`infer.py` evaluates **all 31 subsets** of the 5 score components under the chosen
aggregation; the "Q2E full" number = the all-5 subset. Baselines correspond to single
components (e.g. `[query_vs_video]` = raw encoder baseline).

> Note the fusion in code divides by entropy directly (`1/H`), matching the paper's
> "weight inversely to entropy → emphasize confident (low-entropy) components".
> Other fusion variants in code: `exp_entropy` (exp(-H)·P), `rrf` (reciprocal rank),
> plus `mean`/`max` aggregations — these map to the paper's Table 4 fusion ablation.

---

## 7. Main reported results (targets)

### Table 1 — MultiVENT (headline NDCG in **bold** context)
| Model | R@1 | R@5 | R@10 | P@10 | MRR | **NDCG** | MAP | MnR | MdR |
|---|---|---|---|---|---|---|---|---|---|
| MultiCLIP | 9.83 | 44.32 | 70.82 | 65.25 | 0.92 | **75.34** | 86.33 | 22.12 | 6 |
| MultiCLIP + ASR | 10.08 | 46.64 | 74.70 | 68.57 | 0.93 | **78.76** | 88.87 | 22.66 | 6 |
| MultiCLIP + Q2E | 10.24 | 46.71 | 75.76 | 69.73 | 0.95 | **80.04** | 89.42 | 21.13 | 6 |
| MultiCLIP + Q2E + ASR | 10.32 | 49.00 | 79.60 | 73.09 | 0.95 | **83.24** | 91.20 | 16.44 | 6 |
| InternVideo2-1B | 5.60 | 28.92 | 49.12 | 45.44 | 0.68 | **50.43** | 63.77 | 235.49 | 11 |
| InternVideo2-1B + ASR | 9.68 | 40.48 | 62.43 | 57.34 | 0.91 | **68.30** | 83.84 | 109.70 | 7 |
| InternVideo2-1B + Q2E | 9.54 | 40.88 | 63.40 | 58.73 | 0.92 | **69.15** | 83.96 | 53.84 | 7 |
| InternVideo2-1B + Q2E + ASR | 10.24 | 44.94 | 70.79 | 65.14 | 0.95 | **76.10** | 88.09 | 42.81 | 6 |

### Table 1 — MSR-VTT-1kA
| Model | R@1 | R@5 | R@10 | P@10 | MRR | **NDCG** | MAP | MnR | MdR |
|---|---|---|---|---|---|---|---|---|---|
| MultiCLIP | 43.52 | 69.05 | 76.88 | 7.71 | 0.54 | **59.72** | 54.27 | 20.29 | 2 |
| MultiCLIP + ASR | 45.43 | 70.75 | 77.19 | 7.74 | 0.56 | **61.22** | 56.11 | 31.31 | 2 |
| MultiCLIP + Q2E | 44.52 | 71.26 | 79.40 | 7.96 | 0.56 | **61.51** | 55.84 | 17.91 | 2 |
| MultiCLIP + Q2E + ASR | 46.23 | 73.37 | 81.71 | 8.19 | 0.58 | **63.59** | 57.83 | 18.73 | 2 |
| InternVideo2-1B | 52.56 | 73.07 | 80.10 | 8.03 | 0.62 | **66.07** | 61.62 | 25.12 | 1 |
| InternVideo2-1B + ASR | 53.17 | 73.27 | 81.61 | 8.18 | 0.62 | **67.08** | 62.49 | 23.41 | 1 |
| InternVideo2-1B + Q2E | 53.47 | 73.57 | 82.11 | 8.23 | 0.62 | **67.16** | 62.47 | 16.73 | 1 |
| InternVideo2-1B + Q2E + ASR | 56.28 | 76.58 | 83.72 | 8.39 | 0.65 | **69.53** | 65.06 | 15.85 | 1 |

### Table 2 — Per-language MultiVENT NDCG (MC = MultiCLIP)
| Lang | MC | +Q2E | +Q2E+ASR |
|---|---|---|---|
| Arabic | 76.10 | 78.09 | 82.07 |
| Chinese | 77.29 | 86.32 | 86.61 |
| English | 89.98 | 90.57 | 91.43 |
| Korean | 70.38 | 76.92 | 80.58 |
| Russian | 82.84 | 84.62 | 88.70 |

### Table 3 — LLM size (MultiVENT, NDCG): 1B 79.34/82.50 · 3B 79.78/83.03 · 8B 79.41/82.91 · 70B 80.04/83.24 (noAudio/Audio)
### Table 4 — Fusion (MultiVENT, NDCG noAudio/Audio): NegExpEnt 66.61/73.20 · RRF 69.74/76.29 · Max 76.60/80.04 · Mean 78.37/82.44 · **InvEnt 80.04/83.24**
### Table 5 — Component ablation (MultiVENT, NDCG noAudio/Audio): Full 80.04/83.24 · −Video 64.83/73.96 · −Query 78.78/81.54 · −Events 79.02/81.75

(VLM-size ablation datasets also exist on HF: InternVL 1B/2B/4B/8B/26B/38B.)

---

## 8. Repo audit summary

- **Implemented & runnable as-is** (given data): full evaluation pipeline
  (`src/eval/infer.py`) incl. all 5 score components, all 31 combinations, all fusion
  variants, all metrics. Encoders load from local checkpoints. Text-text via ragatouille.
- **Implemented but heavy**: generation pipeline (`src/data/*`) needs vLLM + 70B/38B GPUs.
  Bypassed via released HF datasets.
- **Prompts/configs match paper**: yes; prompts in `src/data/prompts`, decoding params in
  `src/data/utils.py`, dataset naming convention in `src/data/constants.py`.
- **Exact eval commands available**: yes — `scripts/eval_{msrvtt,multivent}.sh`.
- **wandb**: `infer.py` calls `wandb.init(entity="gcnssdvae")`; must run with
  `WANDB_MODE=disabled` (scripts already export it) or offline.

---

## 9. Ambiguities / inference points

- **[UNKNOWN]** Which of the 31 component-subsets the paper labels "Q2E" for each row.
  Strong inference: "+Q2E" (no ASR) = all-5 subset restricted to no-ASR caption pool
  (params `[query_vs_video, query_vs_captions, prequel, during, sequel]` with the
  no-ASR caption set); the plain baseline row = `[query_vs_video]`. Will confirm by
  matching numbers to specific subsets when running.
- **[INFER]** "MultiCLIP + ASR" (no Q2E) row = encoder score fused with ASR-caption
  text scores but without event decomposition — need to map to the exact subset.
- **[REPO]** Softmax is over dim=0 (queries), applied pre-fusion — confirmed in code,
  not stated in paper.
- **[INFER]** MSR-VTT-1kA exact query list (995) comes bundled inside the HF dataset;
  we inherit the authors' split rather than re-deriving it.
- **[UNKNOWN]** Whether per-language MultiVENT numbers come from slicing the same run by
  language metadata (likely) — will slice `video_id`/query language if present.
- **[INFER]** InternVideo2 frame handling (f4 model) differs from MultiCLIP; will read
  `src/eval/InternVideo2/vision_embedder.py` before that encoder's runs.

## 10. Confirmed vs inferred vs unknown (pre-implementation)
- **Confirmed from paper:** task, datasets/splits/sizes, model families, 16-frame default,
  metric set, fusion is inverse-entropy, headline numbers.
- **Confirmed from repo:** exact model IDs, exact metric implementation, exact fusion math
  + pre-softmax over queries, 31-subset sweep, prompt texts, decoding params, dataset dir
  naming, released pre-generated datasets covering all ablations.
- **Inferred:** mapping of paper table rows → specific component-subsets; MSR-VTT-1kA
  provenance; per-language slicing method.
- **Unknown until run:** exact subset→row mapping, per-language metadata availability.
