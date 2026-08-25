# Query-variant selection on MultiVENT 2.0 (Task B, full 6 methods x 5 samples)

All 2546 test queries, 31 candidates per query (the source study's pool size),
generated with QueryGym's six methods (self-hosted qwen2.5-7b, temperature 0.6). Pipeline: dense
speech channel (bge-m3). The generation-axis companion (nugget metrics under the qwen2.5-14b
judge) is in `rag/metrics_n400_all.json`; the pre-selector's picks lose on nDCG yet beat the
original on every nugget metric, replicating the source study's generation-axis claim.

| Policy | nDCG@10 | vs original |
|---|---:|---:|
| Original query | 0.3133 | -- |
| Concatenate all expansions into one query | 0.2823 | -0.0310 |
| Fuse all candidates' result lists (RRF) | 0.3073 | -0.0060 |
| Best single method (query2doc) | 0.3099 | -0.0034 |
| Best corpus-statistic selection (QL, tau +0.07) | 0.2980 | -0.0153 |
| Best score-based selection (NQC_norm, tau +0.21) | 0.3320 | +0.0187 |
| Per-query oracle over the pool | 0.4375 | +0.1242 |

Family counts: corpus-statistic predictors select above the original 0 of 11; score-based
predictors 7 of 10. Per-method means, all at or below the original: query2doc 0.3099, mugi 0.3072, qa_expand 0.2923, query2e 0.2622, genqr 0.2550, genqr_ensemble 0.2522.
