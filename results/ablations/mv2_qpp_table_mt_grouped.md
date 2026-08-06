| Category | Method | ASR-translated nDCG@10 | τ |
|---|---|---|---|
| Original | visual only (cheap) | 0.3036 | -- |
| | uniform fusion (best w) | 0.3452 | -- |
| Pre-retrieval<br>(ASR text index) | IDF_avg | 0.3445 | -0.042 |
|  | IDF_max | 0.3452 | -0.016 |
|  | IDF_sum | 0.3468 | +0.063 |
|  | IDF_std | 0.3452 | -0.005 |
|  | SCQ_avg | 0.3452 | -0.002 |
|  | SCQ_max | 0.3447 | -0.007 |
|  | SCQ_sum | 0.3472 | +0.091 |
|  | avgICTF | 0.3451 | -0.029 |
|  | SCS_1 | 0.3452 | -0.040 |
|  | SCS_2 | 0.3455 | -0.044 |
|  | QL | 0.3471 | +0.100 |
| Post-retrieval<br>(score-only) | WIG_norm | 0.3606 | -0.159 |
|  | WIG | 0.3452 | -0.006 |
|  | NQC_norm | 0.3840 | -0.221 |
|  | NQC | 0.3834 | -0.238 |
|  | SMV_norm | 0.3792 | -0.209 |
|  | SMV | 0.3793 | -0.222 |
|  | RSD | 0.3431 | -0.092 |
|  | sigma_max | 0.3779 | -0.201 |
|  | sigma_x0.5 | 0.3679 | -0.180 |
|  | max | 0.3639 | -0.156 |
| Post-retrieval<br>(needs doc text) | clarity | n/a for visual | -- |
| **Ours** | **cheap-feature gain ridge** | **0.3920 | +0.240** |
| Oracle | route by true gain | 0.4565 | +1.000 |
