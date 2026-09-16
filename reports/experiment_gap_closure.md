# Experiment gap closure, 3 Aug 2026

This note records the experiments added after the first complete draft. It is deliberately separate
from the second paper's scope note, which lives in the paper repository. Every item here tests the
boundary-condition paper.

## Shared protocol

- MultiVENT learned baselines use five outer folds over 536 event groups and 2,546 queries.
- Within each outer training fold, a group-disjoint 20% calibration subset chooses an escalation
  fraction from 0 to 1 in increments of 0.02. The model is then fitted on the full outer training fold
  and that fixed fraction is applied to the outer test fold. No test label selects an operating point.
- QSD_post receives five epochs. BERT-QPP cross and bi receive three epochs on bert-base with AMP. QSD_pre
  is closed-form but receives the identical nested fraction choice for its matched comparison.
- Every artifact stores out-of-fold predictions and decision bits. All predictions are finite, all
  decision strings contain 2,546 bits, their one-counts reproduce the recorded escalation fractions,
  and the decisions reproduce routed nDCG@10 and Recall@100.

## Results

### Stronger translated MultiVENT channel

The fixed binary baseline rises to 0.3452. Three of eleven corpus-statistic rows clear the source
study's 0.0005 margin, reaching only 0.3468--0.3472; eight of ten score-only rows clear it and the
multifeature control reaches 0.3920. This qualifies the literal 0/33 result while preserving the
family-level reliability contrast. Artifact: `mv2_qpp_table_mt_grouped.{json,md}`.

### Second collection

On the 995-query MSR-VTT-1kA split, two encoders and ASR/no-ASR evidence define four conditions for
direct visual-versus-caption choice and four for visual-to-fusion escalation. Direct choice has
5.57--10.37 nDCG points of oracle headroom, and the control beats fixed in all four conditions by
0.10--3.26 points. Corpus-statistic rows clear the margin in 0/88 tests; score-only rows do so in 5/80.
The boundary replicates, while universal score-only transfer does not. Artifact:
`mv2_msrvtt_source_replication.{json,md}`.

### Matched QSD

| cell | QSD_pre nDCG / τ | QSD_post nDCG / τ | post − pre nDCG / τ |
|---|---|---|---|
| ASR-shipped | 0.3102 / +0.164 | 0.3104 / +0.122 | +0.0001 / −0.042 |
| ASR-dense | 0.3469 / +0.152 | 0.3486 / +0.142 | +0.0017 / −0.010 |
| OCR | 0.3037 / +0.095 | 0.3014 / +0.100 | −0.0023 / +0.005 |

The earlier one-epoch claim that adding document text makes QSD worse does not survive the matched
comparison. The supported
result is no consistent benefit or harm. Artifacts: `mv2_qsd_pre_nested_grouped.json` and
`mv2_qsd_post_5ep_nested_grouped.json`.

### BERT-QPP and operating-point calibration

| variant / cell | τ | routed fraction | raw-zero nDCG | nested nDCG |
|---|---:|---:|---:|---:|
| cross / ASR-shipped | +0.223 | 0.524 | 0.2795 | 0.3204 |
| cross / ASR-dense | +0.210 | 0.652 | 0.3408 | 0.3560 |
| cross / OCR | +0.180 | 0.120 | 0.2445 | 0.3042 |
| bi / ASR-shipped | −0.017 | 0.160 | 0.2922 | 0.3006 |
| bi / ASR-dense | −0.029 | 0.992 | 0.3136 | 0.3403 |
| bi / OCR | −0.015 | 0.008 | 0.2838 | 0.3028 |

The cross-encoder's all-positive zero crossing is a scale failure: nested calibration makes it the
strongest learned QPP row. The bi-encoder remains below fixed in every cell. Artifacts:
`mv2_bertqpp_cross_3ep_nested_grouped.json` and `mv2_bertqpp_bi_3ep_nested_grouped.json`.

### Correlation versus utility

Using the nested learned decisions, Kendall τ against utility over the best fixed policy is −0.099 over
78 rows (p=0.213, Pearson −0.182), rather than the old zero-crossing estimate of −0.213. It is −0.189
over 56 non-degenerate rows (p=0.040), −0.244 over the 35 rows escalating 5--95% (p=0.040), and −0.046
against cheap-only (p=0.558). The anti-correlation headline is retracted. The supported claim is that
rank correlation does not supply a deployable operating point. Artifact:
`mv2_qpp_utility_nested_grouped.json`.

### Second nugget assignment judge

The 7B assignment pass holds the 14B-produced gold nuggets and reports fixed, changing only the
checkpoint that labels support. This isolates assignment sensitivity. It does not test nuggetization or
generation sensitivity. Both checkpoints are Qwen2.5, so this is not a cross-family judge test.

| grounding | assigner | routed vital | fixed vital | delta | paired p |
|---|---|---:|---:|---:|---:|
| all retrieved-document text | 14B | 0.5007 | 0.4648 | +0.0359 | 0.0391 |
| all retrieved-document text | 7B | 0.4781 | 0.4381 | +0.0399 | 0.0384 |
| selected-channel text only | 14B | 0.4822 | 0.4746 | +0.0076 | 0.6814 |
| selected-channel text only | 7B | 0.4583 | 0.4328 | +0.0255 | 0.2202 |

The all-text gain and selected-channel non-result reproduce. Exact assignment-label agreement is 0.712
and 0.720 (Cohen's κ 0.525 and 0.535) for all and own evidence. Artifacts:
`mv2_rag_judge_comparison.{json,md}`, `rag/assigned_n400_{all,own}_qwen7b.jsonl`, and
`rag/metrics_n400_{all,own}_qwen7b.json`.

## Integrity record

SHA-256 hashes of the learned artifacts:

- QSD_pre: `0750df4e46d5f2ea61e1cc5cd0f5681a38d22c4b76aa1a99f1a1e294f1402c25`
- QSD_post: `1796516b41643922587bfd1926a57fbe188d41d069204341d24073001953a0f1`
- BERT-QPP cross: `206b1bcd0c409d5a1d50997de910a9da2aa021afd4db7fae756ea95a413da6f9`
- BERT-QPP bi: `3f6cf65bc7eb421270ba47f36b86f7aecd6a350c4d633ed8063ece30597c8dad`

The nested full table and its per-query nugget mixes are
`mv2_table1_nested_grouped.md` and `mv2_table1_nuggets_nested_grouped.json`. The judged A/B endpoints
reproduce the direct judged runs exactly before any predictor row is mixed. Run
`python src/multivent2/mv2_nested_audit.py` to repeat the prediction, metric and event-overlap checks
without loading a model.
