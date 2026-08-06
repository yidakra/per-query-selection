# Reproduced vs Reported (headline: NDCG@10, torchmetrics)

| Dataset | Encoder | Setting | rep NDCG | paper NDCG | ΔNDCG | rep R@1 | paper R@1 | rep R@10 | paper R@10 |
|---|---|---|---|---|---|---|---|---|---|
| MSR-VTT-1kA | multiclip | baseline | 59.72 | 59.72 | -0.00 | 43.52 | 43.52 | 76.88 | 76.88 |
| MSR-VTT-1kA | multiclip | Q2E | 61.51 | 61.51 | -0.00 | 44.52 | 44.52 | 79.40 | 79.40 |
| MSR-VTT-1kA | multiclip | Q2E+ASR | 63.61 | 63.59 | 0.02 | 46.33 | 46.23 | 81.71 | 81.71 |
| MSR-VTT-1kA | internvideo2 | baseline | 66.00 | 66.07 | -0.07 | 52.56 | 52.56 | 79.90 | 80.10 |
| MSR-VTT-1kA | internvideo2 | Q2E | 67.11 | 67.16 | -0.05 | 53.37 | 53.47 | 82.11 | 82.11 |
| MSR-VTT-1kA | internvideo2 | Q2E+ASR | 69.49 | 69.53 | -0.04 | 56.28 | 56.28 | 83.62 | 83.72 |
| MultiVENT † | multiclip | baseline | 76.53 | 75.34 | +1.19 | 12.67 | 9.83 | 73.63 | 70.82 |
| MultiVENT † | multiclip | Q2E | 80.94 | 80.04 | +0.90 | 12.99 | 10.24 | 78.14 | 75.76 |
| MultiVENT † | multiclip | Q2E+ASR | 84.70 | 83.24 | +1.46 | 13.39 | 10.32 | 82.26 | 79.60 |
| MultiVENT | internvideo2 | (all) | n/r | 50.43–76.10 | -- | n/r | 5.60–10.24 | n/r | 49.12–70.79 |

† **MultiVENT full-video reproduced on the downloadable-video *subset* (1995 of 2393 videos; 398
YouTube videos were unavailable at scrape time).** Numbers above are the subset (all 259 queries
retain ≥1 downloaded gold). The subset gallery is smaller than the paper's 2393 → fewer distractors,
so the consistent +0.9…+1.5 over the paper is a **gallery-size artifact, not a genuine gain**: the
faithful reading is "reproduces the paper." Same-gallery-size numbers with the 398 missing videos left
in the pool (their frames absent → visual component degraded) are lower (baseline 67.71 / Q2E 76.21 /
Q2E+ASR 80.66 NDCG), bounding the download-coverage cost. `n/r` = not run: **MultiVENT was reproduced
with the MultiCLIP video encoder only**; the InternVideo2 MultiVENT rows are out of scope (no such run).
The text-only path is exact independent of downloads; see the `−Video` row below (64.83 vs 64.83).

## Component ablation: MultiVENT / MultiCLIP (NDCG)

Reproduced on the **full 2393-video gallery** (same definition as the paper's Table 5), so the
398 missing-video frames depress every video-dependent row; the fully-text `−Video` row is unaffected
and reproduces to the decimal, cross-validating the text pipeline.

| Config | components | rep noASR | paper noASR | rep ASR | paper ASR |
|---|---|---:|---:|---:|---:|
| Q2E Full | video+query+prequel+during+sequel | 76.21 | 80.04 | 80.66 | 83.24 |
| Q2E −Video | query+prequel+during+sequel | **64.83** | **64.83** | **73.92** | **73.96** |
| Q2E −Query | video+prequel+during+sequel | 74.58 | 78.78 | 78.47 | 81.54 |
| Q2E −Events | video+query | 74.28 | 79.02 | 78.31 | 81.75 |

`−Video` (text-only, immune to missing downloads) reproduces exactly (−0.00 noASR / −0.04 ASR). The
three video-dependent rows sit ~3–4 NDCG low **purely** from the 398 undownloadable videos; on the
downloadable subset the same rows meet/exceed the paper (Full 80.94/84.70; see the † note above).
