| Category | Method | ASRvOCR-query2doc nDCG@10 | τ |
|---|---|---|---|
| Original | visual only (cheap) | 0.3106 | -- |
| | uniform fusion (best w) | 0.1581 | -- |
| Pre-retrieval<br>(ASR text index) | IDF_avg | 0.3106 | -0.003 |
|  | IDF_max | 0.3106 | +0.013 |
|  | IDF_sum | 0.3106 | -0.023 |
|  | IDF_std | 0.3091 | +0.012 |
|  | SCQ_avg | 0.3057 | -0.015 |
|  | SCQ_max | 0.3106 | -0.046 |
|  | SCQ_sum | 0.3106 | -0.023 |
|  | avgICTF | 0.3097 | -0.008 |
|  | SCS_1 | 0.3106 | +0.001 |
|  | SCS_2 | 0.3100 | +0.008 |
|  | QL | 0.3106 | -0.021 |
| Post-retrieval<br>(score-only) | WIG_norm | 0.3052 | -0.035 |
|  | WIG | 0.3093 | -0.003 |
|  | NQC_norm | 0.3076 | -0.120 |
|  | NQC | 0.3093 | -0.124 |
|  | SMV_norm | 0.3078 | -0.094 |
|  | SMV | 0.3101 | -0.096 |
|  | RSD | 0.3076 | -0.036 |
|  | sigma_max | 0.3064 | -0.141 |
|  | sigma_x0.5 | 0.3081 | -0.041 |
|  | max | 0.3119 | -0.129 |
| Post-retrieval<br>(needs doc text) | clarity | n/a for visual | -- |
| **Ours** | **cheap-feature gain ridge** | **0.3110 | +0.171** |
| Oracle | route by true gain | 0.3612 | +1.000 |
