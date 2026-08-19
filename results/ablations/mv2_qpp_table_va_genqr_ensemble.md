| Category | Method | ASRvOCR-genqr_ensemble nDCG@10 | τ |
|---|---|---|---|
| Original | visual only (cheap) | 0.2531 | -- |
| | uniform fusion (best w) | 0.1430 | -- |
| Pre-retrieval<br>(ASR text index) | IDF_avg | 0.2525 | +0.019 |
|  | IDF_max | 0.2531 | -0.019 |
|  | IDF_sum | 0.2479 | +0.018 |
|  | IDF_std | 0.2528 | -0.046 |
|  | SCQ_avg | 0.2531 | +0.033 |
|  | SCQ_max | 0.2531 | -0.054 |
|  | SCQ_sum | 0.2410 | +0.024 |
|  | avgICTF | 0.2502 | +0.030 |
|  | SCS_1 | 0.2526 | +0.018 |
|  | SCS_2 | 0.2476 | +0.037 |
|  | QL | 0.2441 | +0.013 |
| Post-retrieval<br>(score-only) | WIG_norm | 0.2527 | -0.068 |
|  | WIG | 0.2412 | -0.028 |
|  | NQC_norm | 0.2559 | -0.121 |
|  | NQC | 0.2516 | -0.125 |
|  | SMV_norm | 0.2511 | -0.096 |
|  | SMV | 0.2510 | -0.099 |
|  | RSD | 0.2474 | -0.069 |
|  | sigma_max | 0.2531 | -0.145 |
|  | sigma_x0.5 | 0.2533 | -0.073 |
|  | max | 0.2496 | -0.134 |
| Post-retrieval<br>(needs doc text) | clarity | n/a for visual | -- |
| **Ours** | **cheap-feature gain ridge** | **0.2512 | +0.158** |
| Oracle | route by true gain | 0.3013 | +1.000 |
