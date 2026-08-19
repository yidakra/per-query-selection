| Category | Method | ASRvOCR-genqr nDCG@10 | τ |
|---|---|---|---|
| Original | visual only (cheap) | 0.2554 | -- |
| | uniform fusion (best w) | 0.1460 | -- |
| Pre-retrieval<br>(ASR text index) | IDF_avg | 0.2546 | -0.001 |
|  | IDF_max | 0.2540 | -0.022 |
|  | IDF_sum | 0.2514 | -0.002 |
|  | IDF_std | 0.2492 | -0.030 |
|  | SCQ_avg | 0.2513 | +0.011 |
|  | SCQ_max | 0.2507 | -0.024 |
|  | SCQ_sum | 0.2471 | +0.005 |
|  | avgICTF | 0.2554 | +0.002 |
|  | SCS_1 | 0.2525 | +0.001 |
|  | SCS_2 | 0.2547 | +0.002 |
|  | QL | 0.2502 | -0.003 |
| Post-retrieval<br>(score-only) | WIG_norm | 0.2470 | -0.033 |
|  | WIG | 0.2528 | +0.001 |
|  | NQC_norm | 0.2539 | -0.118 |
|  | NQC | 0.2532 | -0.116 |
|  | SMV_norm | 0.2557 | -0.093 |
|  | SMV | 0.2545 | -0.090 |
|  | RSD | 0.2559 | -0.055 |
|  | sigma_max | 0.2533 | -0.136 |
|  | sigma_x0.5 | 0.2513 | -0.034 |
|  | max | 0.2467 | -0.103 |
| Post-retrieval<br>(needs doc text) | clarity | n/a for visual | -- |
| **Ours** | **cheap-feature gain ridge** | **0.2543 | +0.127** |
| Oracle | route by true gain | 0.3075 | +1.000 |
