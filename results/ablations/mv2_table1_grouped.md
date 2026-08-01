# Table 1 (paper layout)

| Category | Method | ASR-shipped nDCG@10 | τ | R@100 | ASR-dense nDCG@10 | τ | R@100 | OCR nDCG@10 | τ | R@100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Original | `best fixed policy (no selection)` | 0.3036 | — | 0.6027 | 0.3408 | — | 0.7268 | 0.3036 | — | 0.6027 |
|  | `visual only` | 0.3036 | — | 0.6027 | 0.3036 | — | 0.6027 | 0.3036 | — | 0.6027 |
|  | `uniform fusion (best w)` | 0.2795 | — | 0.6892 | 0.3408 | — | 0.7268 | 0.2445 | — | 0.6008 |
| Pre-retrieval | `IDF_avg` | 0.3031 | -0.033 | 0.6027 | 0.3408 | -0.008 | 0.7268 | 0.3036 | +0.022 | 0.6027 |
|  | `IDF_max` | 0.3036 | -0.008 | 0.6027 | 0.3408 | +0.024 | 0.7268 | 0.3036 | +0.006 | 0.6027 |
|  | `IDF_sum` | 0.3039 | +0.057 | 0.6065 | 0.3400 | +0.044 | 0.7258 | 0.3036 | +0.009 | 0.6027 |
|  | `IDF_std` | 0.3036 | +0.016 | 0.6027 | 0.3408 | +0.021 | 0.7268 | 0.3036 | -0.020 | 0.6027 |
|  | `ICTF_avg` | 0.3034 | -0.032 | 0.6027 | 0.3408 | -0.007 | 0.7268 | 0.3036 | +0.027 | 0.6027 |
|  | `SCQ_avg` | 0.3036 | -0.000 | 0.6027 | 0.3408 | -0.013 | 0.7268 | 0.3036 | +0.024 | 0.6027 |
|  | `SCQ_max` | 0.3033 | +0.064 | 0.6027 | 0.3408 | +0.037 | 0.7268 | 0.3036 | +0.034 | 0.6027 |
|  | `SCQ_sum` | 0.3029 | +0.067 | 0.6093 | 0.3408 | +0.040 | 0.7268 | 0.3036 | +0.004 | 0.6027 |
|  | `SCS_apx` | 0.3026 | -0.040 | 0.6028 | 0.3408 | -0.014 | 0.7268 | 0.3036 | +0.029 | 0.6027 |
|  | `SCS_full` | 0.3027 | -0.043 | 0.6028 | 0.3408 | -0.016 | 0.7268 | 0.3036 | +0.027 | 0.6027 |
|  | `QL` | 0.3009 | +0.079 | 0.6092 | 0.3408 | +0.052 | 0.7268 | 0.3036 | -0.004 | 0.6027 |
|  | `QSD_pre` | **<u>0.3137</u>** | +0.164 | 0.6390 | **<u>0.3466</u>** | +0.152 | 0.7074 | **0.3039** | +0.095 | 0.6028 |
|  | `DM` | *n.i.* | — | — | *n.i.* | — | — | *n.i.* | — | — |
| Post-retrieval | `RSD` | <u>0.3120</u> | -0.127 | 0.6383 | 0.3388 | -0.069 | 0.7202 | 0.3033 | -0.099 | 0.6026 |
|  | `clarity` | n/a | — | — | n/a | — | — | n/a | — | — |
|  | `NQC` | **<u>0.3205</u>** | -0.215 | 0.6570 | <u>0.3527</u> | -0.163 | 0.7198 | **0.3040** | -0.174 | 0.6019 |
|  | `NQC_norm` | <u>0.3172</u> | -0.204 | 0.6591 | **<u>0.3541</u>** | -0.154 | 0.7169 | 0.3035 | -0.162 | 0.6027 |
|  | `sigma_max` | <u>0.3168</u> | -0.196 | 0.6500 | <u>0.3507</u> | -0.144 | 0.7223 | 0.3036 | -0.149 | 0.6027 |
|  | `sigma_0.5` | <u>0.3152</u> | -0.182 | 0.6500 | <u>0.3446</u> | -0.122 | 0.7190 | 0.3036 | -0.100 | 0.6027 |
|  | `SMV` | <u>0.3197</u> | -0.208 | 0.6556 | <u>0.3509</u> | -0.151 | 0.7201 | 0.3038 | -0.169 | 0.6024 |
|  | `SMV_norm` | <u>0.3170</u> | -0.198 | 0.6560 | <u>0.3527</u> | -0.145 | 0.7174 | 0.3034 | -0.159 | 0.6026 |
|  | `WIG` | 0.3036 | +0.006 | 0.6026 | 0.3408 | +0.012 | 0.7268 | 0.3032 | +0.051 | 0.6025 |
|  | `WIG_norm` | <u>0.3154</u> | -0.168 | 0.6482 | 0.3411 | -0.106 | 0.7186 | 0.3036 | -0.100 | 0.6027 |
|  | `max` | <u>0.3070</u> | -0.124 | 0.6334 | <u>0.3430</u> | -0.112 | 0.7232 | 0.3036 | -0.087 | 0.6024 |
|  | `QSD_post` | *n.i.* | — | — | *n.i.* | — | — | *n.i.* | — | — |
|  | `BERTQPP` | 0.2795 | +0.237 | 0.6892 | 0.3408 | +0.229 | 0.7268 | 0.2445 | +0.167 | 0.6008 |
| Ours | `k-way channel selector` | <u>0.3193</u> | +0.211 | 0.6592 | <u>0.3531</u> | +0.160 | 0.7207 | 0.3036 | +0.154 | 0.6012 |
| Oracle | `route by true gain` | <u>0.3653</u> | +1.000 | 0.6196 | <u>0.3910</u> | +1.000 | 0.6286 | <u>0.3305</u> | +1.000 | 0.6038 |

underline = beats the Original row; bold = best in section; `n/a` = undefined for this channel; *n.i.* = no equivalent predictor implemented.

## Degenerate cells

Counted from each predictor's recorded escalation fraction, degenerate meaning exactly 0 or exactly 1: the same decision for all 2546 queries, so the number in the cell reports a fixed policy and not the predictor. A row can carry a healthy tau and still be degenerate, which is the gap between correlation and decision quality. Denominators cover the rows that have a fraction on record, so `clarity`, `DM` and `QSD_post` are excluded rather than counted as non-degenerate.

- Pre-retrieval / ASR-shipped: 3/12
- Pre-retrieval / ASR-dense: 6/12
- Pre-retrieval / OCR: 11/12
- Post-retrieval / ASR-shipped: 1/11
- Post-retrieval / ASR-dense: 2/11
- Post-retrieval / OCR: 2/11

Full list: IDF_avg/ASR-dense (always fuse); IDF_avg/OCR (never fuse); IDF_max/ASR-shipped (never fuse); IDF_max/OCR (never fuse); IDF_sum/OCR (never fuse); IDF_std/ASR-shipped (never fuse); IDF_std/ASR-dense (always fuse); IDF_std/OCR (never fuse); ICTF_avg/ASR-dense (always fuse); ICTF_avg/OCR (never fuse); SCQ_avg/ASR-shipped (never fuse); SCQ_avg/ASR-dense (always fuse); SCQ_avg/OCR (never fuse); SCQ_max/OCR (never fuse); SCQ_sum/OCR (never fuse); SCS_apx/ASR-dense (always fuse); SCS_apx/OCR (never fuse); SCS_full/ASR-dense (always fuse); SCS_full/OCR (never fuse); QL/OCR (never fuse); sigma_0.5/OCR (never fuse); WIG/ASR-dense (always fuse); BERTQPP/ASR-shipped (always fuse); BERTQPP/ASR-dense (always fuse); BERTQPP/OCR (always fuse).
