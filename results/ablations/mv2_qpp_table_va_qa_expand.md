| Category | Method | ASRvOCR-qa_expand nDCG@10 | τ |
|---|---|---|---|
| Original | visual only (cheap) | 0.2949 | -- |
| | uniform fusion (best w) | 0.1502 | -- |
| Pre-retrieval<br>(ASR text index) | IDF_avg | 0.2946 | +0.005 |
|  | IDF_max | 0.2929 | -0.011 |
|  | IDF_sum | 0.2949 | -0.021 |
|  | IDF_std | 0.2949 | +0.003 |
|  | SCQ_avg | 0.2881 | -0.007 |
|  | SCQ_max | 0.2944 | -0.031 |
|  | SCQ_sum | 0.2947 | -0.022 |
|  | avgICTF | 0.2946 | -0.001 |
|  | SCS_1 | 0.2935 | +0.013 |
|  | SCS_2 | 0.2933 | +0.018 |
|  | QL | 0.2946 | -0.023 |
| Post-retrieval<br>(score-only) | WIG_norm | 0.2949 | -0.005 |
|  | WIG | 0.2945 | +0.015 |
|  | NQC_norm | 0.2947 | -0.125 |
|  | NQC | 0.2950 | -0.132 |
|  | SMV_norm | 0.2929 | -0.097 |
|  | SMV | 0.2915 | -0.103 |
|  | RSD | 0.2931 | -0.061 |
|  | sigma_max | 0.2929 | -0.142 |
|  | sigma_x0.5 | 0.2949 | -0.008 |
|  | max | 0.2903 | -0.137 |
| Post-retrieval<br>(needs doc text) | clarity | n/a for visual | -- |
| **Ours** | **cheap-feature gain ridge** | **0.2908 | +0.166** |
| Oracle | route by true gain | 0.3461 | +1.000 |
