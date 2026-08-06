# Compute Cost Report

## Hardware
- 2× NVIDIA A2 (15.36 GB each; **GPU0 shares ~3.7 GB with a pre-existing user
  `whisper_server`, so all Q2E work runs on GPU1**), 16 vCPU, 31 GB RAM, ~480 GB disk.
- No external API calls used (all models run locally or via released artifacts). **$0 API cost.**

## What was actually run (Tier A: evaluation reproduction)
The paper's *generation* models (Llama-3.3-70B, InternVL2.5-38B; ~150 GB VRAM combined,
served via vLLM on A100s) do **not** fit on an A2. We instead consumed the authors'
released generation artifacts (HF datasets) and re-ran the *evaluation stack* locally:
encoders (MultiCLIP ViT-H/14 ~4.8 GB ckpt, InternVideo2-1B ~2.8 GB ckpt), ColBERT/PLAID-X
text scorer (~2.2 GB), inverse-entropy fusion, torchmetrics.

### Downloads (one-time)
| Item | Size |
|---|---|
| 4 main HF datasets (text artifacts) | < 0.2 GB |
| MSR-VTT videos (1000 needed, from 6.5 GB zip) | 0.65 GB kept |
| MultiCLIP checkpoint | 4.77 GB |
| InternVideo2-1B checkpoint | 2.82 GB |
| PLAID-X ColBERT model | ~2.2 GB |
| MSR-VTT MultiCLIP frame cache (.npy, 16f) | ~9.6 GB |

### Runtime (wall-clock, GPU1, filled as runs complete)
| Job | Components | Approx time |
|---|---|---|
| MSR-VTT MultiCLIP (video frame extract+embed, 1000 vids) | query_vs_video | ~8–9 min |
| MSR-VTT MultiCLIP text (per config, 4 ColBERT comps) | ColBERT max-sim | ~15–20 min |
| MSR-VTT InternVideo2 | query_vs_video + text | TBD |
| MultiVENT text-only (per config) | 4 ColBERT comps × 2393 vids | TBD |

Dominant cost = ColBERT re-encoding the video-caption gallery once per caption-column per
component (O(components × columns × encode(V))). Batch size was reduced from the upstream
default (encode bsize 1024→128, search 8192→512) to fit 15 GB; **this changes only
throughput, not scores** (verified: fusion unit tests + identical scoring code path).

## Cost-control measures
- Aggressive disk caching of every component score matrix (`runs/<tag>/cache/*.pt`) and of
  extracted CLIP frames (`.npy`) → re-runs and offline fusion/ablation are near-instant.
- All 31 component-subsets × 5 fusion methods are evaluated from the **same** cached
  component matrices (no recomputation), so the full ablation grid is nearly free once
  components exist.
- Generation entirely avoided by reusing released artifacts (would otherwise need multi-A100
  hours for 70B+38B inference over ~2400 news videos).

## Not run (documented deviations)
- **MultiVENT `query_vs_video`**: needs the 2,394 source videos, which are YouTube-scraped
  (video_id = YouTube ID). Full-video MultiVENT retrieval is deferred (fragile scraping,
  disk); MultiVENT is reproduced **text-only** (Q2E−Video family + per-language).
- **Generation pipeline** (query decomp / captioning / ASR) not re-run at paper scale;
  optionally validated at small scale with open substitutes (Tier B) if pursued.
