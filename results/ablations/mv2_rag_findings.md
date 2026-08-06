# Does the routing gain survive a change of objective?

Every number this project reports is nDCG@10. A reviewer is entitled to ask whether picking a better
channel per query actually produces a better *answer*, or only a better ranked list. The two can come
apart. Nugget coverage saturates where nDCG keeps counting, since a report needs one good document to
cover a fact and gains nothing from the second.

So the ranked lists go through the QPP-4-RAG nuggetizer protocol end to end (create, score, generate,
assign, measure) with a local `qwen2.5:14b-instruct` as the judge. `mv2_rag_nuggets.py`,
`mv2_routed_run.py`.

## Setup

400 queries sampled stratified from the test set; 395 produced nuggets. Gold nuggets are built from up
to 8 *judged-relevant* documents per query, reading all three channel texts, so no policy has a hand in
writing the questions it is scored against. 2,131 nuggets, median 5 per query, 914 vital and 1,217 okay.

Five policies. Three single channels, the best fixed policy the selector could have chosen on training
folds (`asr+visual`), and the routed system, which reuses the same event-grouped out-of-fold decisions
as the main table, materialised as ranked lists so the generator can consume them.

Two arms, and the difference between them turns out to be the finding.

`--evidence all` writes every policy's report from all three channel texts of the documents *it*
retrieved, so the only thing that varies is which documents each policy found. That isolates retrieval
quality.

`--evidence own` holds each policy to the text of the channels it actually selected. It is the
deployment-realistic version: a system that routes to the visual channel has frames, and frames do not
hand a generator anything to read.

## Result, `--evidence all`

| policy | nDCG@10 | vital | strict vital | all | strict all |
|---|---|---|---|---|---|
| **routed** | **0.4068** | **0.5007** | **0.3902** | **0.4072** | **0.3048** |
| best fixed (`asr+visual`) | 0.3331 | 0.4648 | 0.3432 | 0.3834 | 0.2724 |
| dense ASR | 0.3310 | 0.4477 | 0.3433 | 0.3605 | 0.2599 |
| visual | 0.2917 | 0.3995 | 0.2997 | 0.3235 | 0.2256 |
| OCR | 0.1232 | 0.2524 | 0.1617 | 0.2033 | 0.1261 |

Paired permutation test, 10,000 samples, routed against the best fixed policy:

| metric | Δ | p | win / tie / loss |
|---|---|---|---|
| vital | +0.0359 | .037 | 105 / 206 / 84 |
| strict vital | +0.0469 | .014 | 79 / 257 / 59 |
| strict all | +0.0324 | .026 | 110 / 198 / 87 |
| all | +0.0238 | .093 | 139 / 137 / 119 |

Against dense ASR alone all four are significant, p ≤ .012. Against visual and OCR, p < .0001
everywhere.

**The gain carries, and it shrinks.** +7.4 nDCG on this subset becomes +3.6 points of vital-nugget
coverage, about half as much in absolute terms. That is the saturation effect showing up where it
should. Routing pulls a usable document into the top-10 for queries that had none, and it also improves
ranks 2 through 10 for queries that were already answerable. The second kind of improvement earns nDCG
and earns nothing here.

The large tie counts are the mechanism rather than a weakness of the test. The router selects
`asr+visual`, the best fixed policy, for 88 of the 395 queries, and on those the two runs are identical
by construction. It picks a single channel for 292 (`asr` 165, `visual` 123, `ocr` 4) and something
else for the remaining 15.

Under this arm the ordering under nugget coverage is the ordering under nDCG: routed > best fixed >
dense ASR > visual > OCR. `strict_vital` looks like it swaps the middle pair. But 0.34329 against
0.34324 is five parts in a hundred thousand, so that is a tie, and calling it a flip would be dishonest.

So when retrieval quality is isolated, nDCG is a faithful stand-in for answer quality.

## Result, `--evidence own`

Same queries, same gold nuggets, same judge. Each policy now generates only from the channels it chose.

| policy | nDCG@10 | vital | strict vital | all | strict all |
|---|---|---|---|---|---|
| routed | **0.4068** | **0.4822** | **0.3674** | 0.3776 | 0.2665 |
| best fixed (`asr+visual`) | 0.3331 | 0.4746 | 0.3473 | **0.3808** | **0.2718** |
| dense ASR | 0.3310 | 0.4438 | 0.3137 | 0.3383 | 0.2306 |
| visual | 0.2917 | 0.3854 | 0.2578 | 0.3043 | 0.1917 |
| OCR | 0.1232 | 0.2328 | 0.1436 | 0.1749 | 0.0945 |

Routed against the best fixed policy, same test:

| metric | Δ | p | win / tie / loss |
|---|---|---|---|
| vital | +0.0076 | .67 | 108 / 183 / 104 |
| strict vital | +0.0201 | .29 | 80 / 242 / 73 |
| all | −0.0032 | .83 | 128 / 131 / 136 |
| strict all | −0.0054 | .71 | 94 / 191 / 110 |

**The gain does not survive.** Routed is still ahead of every single channel on all four metrics, worst
case p = .025 against dense ASR, so the selector is doing something. Against the best fixed policy it is indistinguishable, in
either direction, while holding a +7.4 nDCG lead over that same policy. That is the utility gap: a large
ranking advantage that buys nothing measurable in answer quality.

**Where it went.** Compare the two arms policy by policy. The best fixed policy is unmoved by the
restriction (vital 0.4648 → 0.4746, strict vital 0.3432 → 0.3473) because it always had `asr+visual`,
which is two text sources either way. The routed system loses 2 to 4 points on every metric (vital
0.5007 → 0.4822, strict all 0.3048 → 0.2665) because it picks a single channel for 73% of these queries:
`asr` alone for 165, `visual` alone for 123, `ocr` alone for 4, and both for only 88.

The reports are not shorter for it: 868 characters on average against the best fixed policy's 867, over
the same five documents. What changes is what is in them. A query routed to the visual channel is
written from captions describing the video instead of from what was said in it.

**The design conclusion is the useful part.** Selecting a channel for *retrieval* is worth +7.4 nDCG.
Letting that selection also restrict what the generator may read gives the gain back. A cascade should
route retrieval and then ground on everything it can reach for the documents it found, which is the
`all` arm, and the `all` arm is the one that converts.

## What this arm also pays for: Table 1's nugget columns

Three policies were added to the `all` arm (`cellB_asr_shipped`, `cellB_asr_dense`, `cellB_ocr`), each
the fused run of one Table 1 cell, restricted to these 395 queries and verified against that cell's
stored nDCG before being written (`mv2_cell_runs.py`). With `visual` already judged as run A and shared
across all three cells, four judged runs cover the entire table.

Every Table 1 row is then arithmetic rather than a judge pass. A predictor executes run A or run B per
query; a report's nugget coverage is a property of the ranked list it was written from, not of the
predictor that chose it; so the row's coverage is the per-query mix of the two under that predictor's
recorded decisions. `mv2_table1_nuggets.py` does the mix and refuses to write unless both endpoints
reproduce the judged runs exactly, which is what would catch a decision vector misaligned to `qids`.
That turned roughly thirty judge passes into three.

Two things it makes visible that the nDCG column only implied. BERT-QPP at its raw zero crossing
escalates every query, so its nugget row *is* the uniform-fusion row in all three cells: a selector
that never selects, under a
generation metric. And in ASR-shipped our selector is the only row whose coverage beats both endpoints
(0.3411 against 0.3235 visual and 0.3167 fused), so routing there is better than either fixed policy on
answer quality and not only on ranking.

## Caveats

- One judge model, 14B, running locally. The judge never ranks anything, so it cannot prefer its own
  retrieval, but a stronger judge would give tighter labels.
- 395 queries, not 2,546. The subset's nDCG spread (0.4068 / 0.3331) tracks the full set's
  (0.4131 / 0.3372) closely enough that the sample is not obviously unrepresentative.
- Nugget creation reads only 8 relevant documents per query. Queries with many relevant videos have
  their gold list built from a subset of them.
- We predicted the `own` arm would *widen* the gap, on the reasoning that single-channel policies would
  fall hardest. They did fall, and so did the routed system, which is itself single-channel on 73% of
  queries. The prediction was wrong in a way worth keeping on the record.
- The two arms disagree, so neither can be quoted alone. "Routing improves answer quality" is true of
  the `all` arm and false of the `own` arm, and the difference is a design choice about grounding rather
  than a fact about routing.
- `strict_all` and `all` reverse order in the `own` arm, by 0.005 and 0.003 at p = .71 and .83. The
  script flags any reversal above a 0.001 tie threshold, which these clear. They are not evidence that
  the fixed policy generates better; they are evidence that the two are indistinguishable.
