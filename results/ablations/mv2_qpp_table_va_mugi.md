| Category | Method | ASRvOCR-mugi nDCG@10 | τ |
|---|---|---|---|
| Original | visual only (cheap) | 0.3064 | -- |
| | uniform fusion (best w) | 0.1561 | -- |
| Pre-retrieval<br>(ASR text index) | IDF_avg | 0.3036 | +0.035 |
|  | IDF_max | 0.3064 | +0.048 |
|  | IDF_sum | 0.3034 | +0.050 |
|  | IDF_std | 0.3064 | +0.012 |
|  | SCQ_avg | 0.3048 | +0.004 |
|  | SCQ_max | 0.3061 | -0.038 |
|  | SCQ_sum | 0.3052 | +0.039 |
|  | avgICTF | 0.3045 | +0.034 |
|  | SCS_1 | 0.3064 | +0.016 |
|  | SCS_2 | 0.3046 | +0.034 |
|  | QL | 0.3051 | +0.040 |
| Post-retrieval<br>(score-only) | WIG_norm | 0.3064 | -0.058 |
|  | WIG | 0.3031 | -0.053 |
|  | NQC_norm | 0.3059 | -0.119 |
|  | NQC | 0.3067 | -0.123 |
|  | SMV_norm | 0.3054 | -0.089 |
|  | SMV | 0.3076 | -0.091 |
|  | RSD | 0.3056 | -0.033 |
|  | sigma_max | 0.3037 | -0.140 |
|  | sigma_x0.5 | 0.3046 | -0.065 |
|  | max | 0.3064 | -0.130 |
| Post-retrieval<br>(needs doc text) | clarity | n/a for visual | -- |
| **Ours** | **cheap-feature gain ridge** | **0.3062 | +0.158** |
| Oracle | route by true gain | 0.3580 | +1.000 |
