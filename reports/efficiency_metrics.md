# Measured efficiency: latency, throughput, energy, and risk–coverage

The companion to `efficiency_metrics_review.md`, which surveyed what routing and IR efficiency papers
report and listed what we were missing. These are the measured numbers.

Hardware: NVIDIA A2 (15 GB) for the LLM stage, 16-core CPU host, 8 torch threads. MultiVENT 2.0 test
queries, 1,000 candidates/query, MiniLM-L6-v2 caption embeddings, qwen2.5:14b-instruct as the
expansion model. Latency measured warm, one query at a time (the serving regime), on an otherwise idle
machine. `src/multivent2/mv2_efficiency.py`, `src/multivent2/mv2_riskcov.py`.

## 1. Per-stage latency (ms)

| stage | what it is | mean | p50 | p95 | p99 |
|---|---|---|---|---|---|
| `rank_A` | sort the visual channel's candidate scores | 0.17 | 0.17 | 0.22 | 0.24 |
| `qenc` | encode the query (MiniLM, CPU) | 9.36 | 7.95 | 12.03 | 20.51 |
| `capsim` | caption-embedding dot product over 1,000 candidates | 3.82 | 3.67 | 5.67 | 9.01 |
| `fuse_B` | RRF over two rank lists + sort | 1.56 | 1.62 | 1.71 | 1.87 |
| `route` | confidence features + ridge predict — **the router itself** | 1.04 | 1.02 | 1.14 | 1.73 |
| `evenc` | encode the 3 event descriptions | 13.15 | 11.63 | 19.55 | 26.58 |
| `evfuse` | max-pool event similarities + weighted RRF + sort | 2.80 | 2.56 | 4.70 | 5.72 |
| `llm_gen` | **the expansion call on GPU** | **9,396.74** | 9,648.19 | 11,454.02 | 11,505.86 |

The router costs 1.04 ms — 0.01% of the 9.4 s decision it is making. (An earlier measurement of the
decision alone, without feature extraction, put it at 345 µs.)

## 2. Per-tier latency and throughput

| tier | mean | p50 | p95 | p99 | serial throughput |
|---|---|---|---|---|---|
| A (ranking only) | 0.17 ms | 0.17 | 0.22 | 0.24 | 5,883 q/s |
| B (+captions) | **14.9 ms** | 14.1 | 18.1 | 26.5 | **67.1 q/s** |
| Full (+LLM expansion) | **9,425 ms** | 9,728 | 11,534 | 11,543 | **0.11 q/s** |

**Tier B to Full is a 633× increase in mean latency and a 610× drop in throughput.** In joules the same
step is 22.35 J → 282 J. Latency is the harsher axis of the two, which the energy-only framing hid.

Concurrency does not rescue it: the Ollama server is compute-bound on one A2, so throughput is flat at
0.111 / 0.117 / 0.114 q/s for 1 / 2 / 4 concurrent workers. Escalation cost cannot be batched away on
this hardware.

## 3. The tail is the real argument for routing

Latency percentiles of the served population as the escalation fraction *f* sweeps (mixture of tier-B
and Full queries):

| f | mean | p95 | p99 | throughput |
|---|---|---|---|---|
| 0.00 | 14.9 ms | 18.1 ms | 26.5 ms | 67.1 q/s |
| 0.10 | 1,003 ms | 10,083 ms | 11,532 ms | 1.00 q/s |
| 0.25 | 2,425 ms | 11,430 ms | 11,543 ms | 0.41 q/s |
| 0.50 | 4,801 ms | 11,438 ms | 11,543 ms | 0.21 q/s |
| 1.00 | 9,425 ms | 11,534 ms | 11,543 ms | 0.11 q/s |

Escalating just **10%** of queries multiplies p99 by **435×** (26.5 ms → 11.5 s) while the mean rises
only 67×. Under any p99 SLO the escalation budget is set by the tail, not the average — so a router
paper on this problem should report the percentile curve, not a mean-cost number. This is the single
most useful thing the efficiency work added.

## 4. Energy, carbon, and cost per useful result

| metric | tier B | Full |
|---|---|---|
| energy / query | 22.35 J | 282 J (259.7 J of it the LLM call) |
| energy / query | — | 72.1 mWh |
| gCO₂e / query | — | 0.034 |
| gCO₂e over the 2,546-query test set | — | 87.2 |
| relevant items in top-10 (mean) | 1.78 | 1.78 |
| **joules per relevant item retrieved@10** | **12.6** | **158.5** |

Carbon uses an assumed grid intensity of 0.475 kg CO₂e/kWh (IEA world average) — documented, not
measured. Cost-per-correct-answer is the cleanest statement of the Full tier's problem: **12.6× the
energy per relevant item surfaced, for no measurable change in how many are surfaced.**

## 5. Risk–coverage and AURC (routing as selective prediction)

Coverage = fraction of queries answered by the cheap tier; risk = mean regret on those queries. Lower
AURC is better. `mv2_riskcov.json`.

| cell | AURC router | AURC oracle | AURC random | share of excess risk the router closes |
|---|---|---|---|---|
| A → B (visual → +captions) | 0.835 | −14.242 | 5.093 | **22.0%** |
| B → Full (+LLM expansion) | 0.277 | −7.643 | 0.808 | **6.3%** |

The A→B router removes 22% of the risk a perfect router would remove; the B→Full router removes 6.3%.
Same conclusion as the frontier and τ analyses, in the vocabulary selective-prediction readers expect.

## Caveats

- Tier A excludes the CLIP text encode and the search over 218K videos: the benchmark ships a
  precomputed run, so tier A latency is a **lower bound** on a deployed tier A.
- Corpus-side caption embedding is an offline one-off and is excluded from every tier; all tiers pay it
  equally.
- The LLM stage is measured on one A2. A larger GPU would shrink the absolute latency but not the
  structure of the tail argument.
- Carbon intensity is assumed, not measured at this location.

## What is still missing

FLOPs per tier in a hardware-independent unit, index/memory footprint of the retrieval side, and a
QPS-vs-load curve. All are secondary for a router paper; the first three sections cover what the
efficiency review flagged as expected.
