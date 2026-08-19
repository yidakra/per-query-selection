# Query-variant selection on MultiVENT 2.0 (Task B, 6 methods x 1 sample)

All 2546 test queries. Pool per query: the original plus one variant from each of the
source study's six QueryGym methods (self-hosted qwen2.5-7b, temperature 0.6). Pipeline: dense
speech channel (bge-m3). Selection means the predictor ranks the pool and its top candidate is
executed. The full 6x5 pool rerun replaces this table when generation completes.

| Policy | nDCG@10 | vs original |
|---|---:|---:|
| Original query | 0.3133 | -- |
| Concatenate all expansions into one query | 0.2823 | -0.0310 |
| Fuse all candidates' result lists (RRF) | 0.3149 | +0.0016 |
| Best single method (query2doc) | 0.3106 | -0.0027 |
| Best corpus-statistic selection (QL, tau +0.03) | 0.3026 | -0.0107 |
| Best score-based selection (NQC_norm, tau +0.24) | 0.3306 | +0.0172 |
| Per-query oracle over the pool | 0.4079 | +0.0946 |

Family counts: corpus-statistic predictors select above the original 0 of 11 (within-need tau
between -0.10 and +0.03); score-based predictors 7 of 10. Per-method means (each below or at the
original): query2doc 0.3106, mugi 0.3064, qa_expand 0.2949, query2e 0.2610, genqr 0.2554, genqr_ensemble 0.2530.

Secondary pipeline (speech+OCR fusion): original 0.2142, concatenation 0.1591, fusion of all
candidates 0.2364, oracle 0.2563; the same family contrast holds.
