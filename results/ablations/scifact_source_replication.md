# SciFact evidence-source selection replication

BM25 retrieves each of 1,109 labeled scientific claims independently against paper titles and abstracts. Queries sharing a relevant paper stay in the same fold. Every selector is repeated five-fold cross-fitted over five randomized group partitions; there is no evaluation-label threshold sweep.

| fixed / selector | nDCG@10 | vs best fixed | chooses abstract | tau |
|---|---:|---:|---:|---:|
| title only | 0.4116 | -- | 0.0% | -- |
| abstract only | 0.6526 | -- | 100.0% | -- |
| corpus-stat multi-feature control | 0.6520 | -0.06 | 99.9% | +0.066 |
| score-distribution multi-feature control | 0.6486 | -0.40 | 91.6% | +0.270 |
| oracle | 0.6876 | +3.50 | 45.0% | +1.000 |

Analytic corpus-statistic predictors above the best fixed source by >0.0005: **0/11** (degenerate 11/11). Score-only predictors: **0/10** (degenerate 10/10).

The score control's paired gain is -0.40 nDCG (group bootstrap 95% CI -1.10 to +0.21; two-sided group sign-flip p=0.2499).

This uses train and test labels together as a cross-fitted analysis set, not as a BEIR leaderboard submission. The source-aware differences are deliberately generous to both predictor families: each scalar sees its value on both fields before choosing.
