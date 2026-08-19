| Category | Method | ASRvOCR-original nDCG@10 | τ |
|---|---|---|---|
| Original | visual only (cheap) | 0.3133 | -- |
| | uniform fusion (best w) | 0.1330 | -- |
| Pre-retrieval<br>(ASR text index) | IDF_avg | 0.3133 | -0.017 |
|  | IDF_max | 0.3133 | -0.015 |
|  | IDF_sum | 0.3128 | -0.070 |
|  | IDF_std | 0.3133 | +0.007 |
|  | SCQ_avg | 0.3133 | -0.015 |
|  | SCQ_max | 0.3124 | -0.023 |
|  | SCQ_sum | 0.3133 | -0.061 |
|  | avgICTF | 0.3133 | -0.018 |
|  | SCS_1 | 0.3133 | -0.009 |
|  | SCS_2 | 0.3129 | -0.008 |
|  | QL | 0.3109 | -0.066 |
| Post-retrieval<br>(score-only) | WIG_norm | 0.3124 | -0.074 |
|  | WIG | 0.3126 | +0.068 |
|  | NQC_norm | 0.3122 | -0.201 |
|  | NQC | 0.3094 | -0.193 |
|  | SMV_norm | 0.3109 | -0.166 |
|  | SMV | 0.3112 | -0.152 |
|  | RSD | 0.3124 | -0.103 |
|  | sigma_max | 0.3104 | -0.204 |
|  | sigma_x0.5 | 0.3129 | -0.087 |
|  | max | 0.3140 | -0.143 |
| Post-retrieval<br>(needs doc text) | clarity | n/a for visual | -- |
| **Ours** | **cheap-feature gain ridge** | **0.3114 | +0.219** |
| Oracle | route by true gain | 0.3543 | +1.000 |
