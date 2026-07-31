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

This arm is `--evidence all`: every policy's report is written from all three channel texts of the
documents *it* retrieved. The only thing that varies between policies is which documents they found.
The `--evidence own` arm, where each policy may read only the channels it selected, is still running.

## Result

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

## No utility gap

The ordering under nugget coverage is the ordering under nDCG: routed > best fixed > dense ASR > visual
> OCR. `strict_vital` looks like it swaps the middle pair. But 0.34329 against 0.34324 is five parts in
a hundred thousand, so that is a tie, and calling it a flip would be dishonest.

Worth stating plainly, because it did not have to come out this way and QPP-4-RAG exists because it
often doesn't. For this cascade, retrieval nDCG is a faithful stand-in for downstream answer quality at
the policy level, which licenses every other table in the paper that reports nDCG alone.

## Caveats

- One judge model, 14B, running locally. The judge never ranks anything, so it cannot prefer its own
  retrieval, but a stronger judge would give tighter labels.
- 395 queries, not 2,546. The subset's nDCG spread (0.4068 / 0.3331) tracks the full set's
  (0.4131 / 0.3372) closely enough that the sample is not obviously unrepresentative.
- Nugget creation reads only 8 relevant documents per query. Queries with many relevant videos have
  their gold list built from a subset of them.
- The `--evidence own` arm is pending. It asks a harder question: whether the routed system still wins
  when each policy is held to the evidence it actually selected. The single-channel policies should
  fall and the gap should widen. Some of that widening will be the restricted evidence rather than
  better retrieval, and the write-up will have to say so.
