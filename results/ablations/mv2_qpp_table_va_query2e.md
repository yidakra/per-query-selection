| Category | Method | ASRvOCR-query2e nDCG@10 | τ |
|---|---|---|---|
| Original | visual only (cheap) | 0.2610 | -- |
| | uniform fusion (best w) | 0.1427 | -- |
| Pre-retrieval<br>(ASR text index) | IDF_avg | 0.2592 | +0.012 |
|  | IDF_max | 0.2610 | -0.011 |
|  | IDF_sum | 0.2584 | -0.026 |
|  | IDF_std | 0.2610 | -0.020 |
|  | SCQ_avg | 0.2602 | +0.016 |
|  | SCQ_max | 0.2593 | -0.067 |
|  | SCQ_sum | 0.2568 | -0.020 |
|  | avgICTF | 0.2592 | +0.015 |
|  | SCS_1 | 0.2600 | +0.023 |
|  | SCS_2 | 0.2593 | +0.021 |
|  | QL | 0.2576 | -0.033 |
| Post-retrieval<br>(score-only) | WIG_norm | 0.2593 | -0.052 |
|  | WIG | 0.2610 | -0.003 |
|  | NQC_norm | 0.2565 | -0.112 |
|  | NQC | 0.2566 | -0.120 |
|  | SMV_norm | 0.2589 | -0.091 |
|  | SMV | 0.2598 | -0.099 |
|  | RSD | 0.2599 | -0.065 |
|  | sigma_max | 0.2564 | -0.134 |
|  | sigma_x0.5 | 0.2600 | -0.057 |
|  | max | 0.2571 | -0.129 |
| Post-retrieval<br>(needs doc text) | clarity | n/a for visual | -- |
| **Ours** | **cheap-feature gain ridge** | **0.2604 | +0.134** |
| Oracle | route by true gain | 0.3126 | +1.000 |
