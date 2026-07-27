# Adaptive Q2E — one-page summary

**Per-query routing over sources of retrieval evidence that differ in cost and reliability.** One
claim: pay for an expensive evidence source only on the queries it helps, decided from what the cheap
source already produced. That decision improves accuracy, not only cost.

All numbers on MultiVENT 2.0 (2,546 test queries, graded judgments, nDCG@10) unless stated.

## Research question

**When a retrieval system can draw on several sources of evidence that differ wildly in cost and
reliability, can it decide _per query_ which ones to pay for, using only what the cheapest source has
already produced? And does deciding beat both using the cheap source alone and using everything?**

## Motivation, in one table

Multimodal video retrieval scores a query against several channels: frames, speech transcripts,
on-screen text, captions, LLM expansions. Nearly every system fuses them the same way for every query.
That assumes every channel is worth having for every query. It isn't, and this holds whether the channel
is strong or weak.

Two channels added to the visual baseline, same benchmark, same router:

| channel added to visual (0.3036) | uniform fusion, best weights | routed per query | oracle |
|---|---|---|---|
| speech, a strong dense ASR retriever (0.3134 alone) | 0.3408 | **0.3545** @ f=0.70 | 0.3910 |
| on-screen text (OCR, 0.1223 alone) | 0.2445 | 0.3036 @ f=0.00 (declines) | 0.3305 |

Speech is one of the best single channels on this benchmark. Fuse it everywhere and you still leave
**+1.37** on the table that routing recovers. OCR is weak enough that uniform fusion *loses* 5.9 points,
so the router refuses it. The two channels look opposite until you measure the spread: per-query gain
has sd 23.1 for speech and 19.4 for OCR, five to ten times each channel's mean effect. What a per-query
policy turns into accuracy is that spread, not the sign of the average.

So the motivation is not that routing saves compute. For heterogeneous evidence a single global fusion
weight is the wrong object. It leaves a strong channel under-used and makes a weak one actively harmful,
and only a per-query decision fixes both.

## Research questions and evidence

| RQ | Question | Evidence | Status |
|---|---|---|---|
| **RQ1** | Where does the routing signal live — in the **query text** (Adaptive-RAG's premise) or in the query×corpus **retrieval interaction**? | TF-IDF+LogReg over the query string scores at/below the majority-class prior. Signal is in the cheap channel's score distribution (top-1, margins, entropy, softmax mass, disagreement), which is free. | supported, **partly refutes** Adaptive-RAG |
| **RQ2** | Is per-query escalation gain predictable from those cheap features, and does it beat classical QPP? | Out-of-fold ridge orders queries by true gain: τ = **+0.217** (shipped speech list), **+0.170** (strong dense speech channel), **+0.127** (captions), all p=.0005. Predictability holds even on the strong channel. Beats Clarity/WIG/NQC as routing baselines. Router overhead 1.04 ms. | supported |
| **RQ3** | What does routing buy on the **accuracy–cost** plane? | Nested-CV gap vs cost-matched random: **+3.07 ± 0.38** (shipped speech), **+2.45 ± 0.32** (dense speech), **+2.23 ± 0.18** (captions). Equal-accuracy cost cuts of 24–58% across cells; +0.24 to +1.92 nDCG at equal cost. | supported |
| **RQ4** | Does per-query gain **heterogeneity** govern how much routing wins? | `sd(per-query gain)` predicts the achieved gap at Spearman **ρ = +0.943** (n=6 cells). Tells you whether routing will pay *before* building it. | supported — the differentiator from Adaptive-RAG |
| **RQ5** | Does query **extension + concatenation** reduce the heterogeneity routing exploits — substitute or complement? | Concatenation beats decompose-and-fuse (**+1.57 vs +0.81** at 14B, same weight) and is lower-variance at every matched weight, but only ~5–10% lower, so the router keeps its signal (τ +0.042 vs +0.035). | supported — **complement**, not substitute |

## Results

**Routing across modality channels (the headline).** The benchmark ships speech and on-screen-text
ranked lists next to the CLIP run, so weighted RRF over them costs nothing extra. It also ships the raw
transcripts, so a real dense retriever can be run over them. Both the shipped speech list and a dense
retriever built over the same transcripts are tested.

| channel / cell | nDCG@10 | help / hurt | τ | routed vs uniform |
|---|---|---|---|---|
| visual | 0.3036 | — | — | — |
| speech, shipped CLIP-text list | 0.2667 | — | — | — |
| speech, dense retriever (bge-m3) over raw transcripts | **0.3134** | — | — | — |
| on-screen text (OCR) | 0.1223 | — | — | — |
| visual → +speech (shipped), routed | 0.3221 @ f=0.55 | 27% / 35% | +0.217 | +4.25 |
| visual → +speech (dense), routed | **0.3545** @ f=0.70 | 34% / 28% | +0.170 | +1.37 |
| visual → +OCR, routed | 0.3036 @ f=0.00 | 14% / 42% | +0.162 | +5.92 (declines) |
| oracle best-single-channel per query (dense) | 0.4777 | — | — | — |

The dense speech channel is the strongest single channel here, above visual, and routing still adds
**+5.09** over visual-only and **+1.37** over the best fixed fusion weight. Its nested-CV gap against
cost-matched random is **+2.45 ± 0.32**, and CPT50/CPT80 come out at 0.21/0.36, defined for the first
time because the strong channel's escalation endpoint finally beats the cheap one. In the risk–coverage
view its router closes **31.6%** of the excess risk a perfect router would remove, against 6.3% for the
LLM tier. Routing is not a
crutch for weak channels, then. A genuinely good channel still has per-query structure worth exploiting.
The gain over uniform fusion does shrink as the channel improves, +4.25 on the shipped list down to
+1.37 on the dense one. That is what the spread framing predicts: a better channel helps more queries,
so there is less left for selectivity to recover.

OCR is the opposite pole. Uniform fusion loses 5.9 points and the router declines it at f=0.00, the same
call it makes on the LLM expansion tier. A router that only ever found reasons to spend would be
suspicious. This one refuses two channels outright.

**Tiers and cost.** Nested, so escalation is free.

| tier | components | LLM calls/q | energy | mean latency | p99 | throughput |
|---|---|---|---|---|---|---|
| A (visual) | `query_vs_video` | 0 | 0.011 J *(est.)* | 0.17 ms | 0.24 ms | 5,883 q/s |
| B (+captions) | `+ query_vs_captions` | 0 | 1.01 J *(est.)* | 14.9 ms | 26.5 ms | 67.1 q/s |
| Full (+events) | `+ {prequel,during,sequel}` | ~30 | **260.7 J** | 9,425 ms | 11,543 ms | 0.11 q/s |

Latency is measured throughout. Energy is measured for the LLM stage (NVML, net of a model-loaded idle
baseline); tiers A and B are CPU-only and this host has no RAPL counters, so their energy is a TDP-based
estimate and is marked as such.

**Efficiency — the tail is the argument.** Escalating just **10%** of queries multiplies p99 by **435×**
(26.5 ms → 11.5 s) while the mean rises 67×, so under any p99 SLO the escalation budget is set by the
tail rather than the average. Concurrency does not help: throughput is flat at ~0.11 q/s for 1/2/4
workers, compute-bound on one A2. Cost per useful result: **0.57 J vs 146.5 J per relevant item@10**, at
identical mean relevant@10 (1.78). Risk–coverage/AURC: the A→B router closes **22.0%** of the excess
risk a perfect router would remove, B→Full only **6.3%**.

**Paired negatives — the *shape* of the frontier, not failures.**

- **The LLM expansion tier is correctly declined.** B→Full τ = +0.002 (p=.44), gap +0.15 (ns); price is
  ~260× tier B in energy and 633× in latency. On MSR-VTT/IV2/noASR it is literally Pareto-dominated
  (67.11 < Fixed-B 67.52). The decomposer size curve is
  monotone in both axes (3B +0.50 nDCG / 72.5 J, 7B +0.56 / 143.5 J, 14B +0.81 / 259.7 J), so a 70B
  would not rescue it.
- **"Tier C" (paraphrase selection) does not exist.** Oracle shows +2.00 headroom; three independent
  achievable estimates all lose (−1.49 to −1.81); a gold-split proves the headroom is label noise
  (optimism +4.89).

## Positioning vs SOTA

Our tiers are a deliberately cheap visual/caption cascade: A = 0.304 (matches the benchmark's mCLIP
baseline to the last digit), B = 0.361. The strong systems (**MMMORRF 0.586**, **CLaMR 0.585**,
**OmniEmbed 0.753**) get their lift from dense retrieval over speech and on-screen text, at far higher
cost. The honest framing is not that we beat SOTA. The router is a **method-agnostic frontier layer**,
and the channel result puts it on the path those systems already take: *how* to weight the channels, per
query, rather than globally. MMMORRF's weights are global. Ours says they should not be. We have now
built the dense channel that argument needs. A bge-m3 retriever over the raw transcripts scores
**0.3134**, above the shipped CLIP-text list (0.2667) and the visual channel, and routing on top of it
still pays (τ +0.170, +1.37 over the best fixed weight). It is not MMMORRF's translate-distill retriever,
so it does not compete on absolute score, but it moves the routing claim off the benchmark's toy lists
onto a real one.

## What we borrowed vs. what is ours

**Borrowed:** Q2E decomposition + inverse-entropy fusion + tier structure (Dipta & Ferraro, AACL 2025,
[arXiv:2506.10202](https://arxiv.org/abs/2506.10202)); the query-complexity-routing premise
(Adaptive-RAG), which we test and partly refute; classical QPP predictors (Clarity/WIG/NQC) as routing
baselines; router cost-quality metrics APGR, CPT₅₀/₈₀ (RouteLLM/RouterBench); risk–coverage/AURC from
selective prediction; MultiVENT 2.0 and its shipped modality channels; measured GPU joules via
NVML/CodeCarbon.

**Ours:** (1) **cheap-features-only escalation-gain regression**, where the expensive source is never
invoked to decide, which no prior routing work does; (2) the finding that **a single global fusion
weight is the wrong object for heterogeneous channels**, since uniform fusion can make a weak channel
actively harmful (OCR) and still under-use a strong one (dense speech), and per-query routing recovers
both; (3) the **heterogeneity → routing-value law** (ρ=0.943); (4) a **measured joules-and-latency**
cost model with the tail argument for routing; (5) a set of **paired negative results** established with
nested-CV and gold-split rigor.

## What the paper would claim

1. Channel usefulness in multimodal retrieval is strongly query-dependent. A single global fusion weight
   is the wrong object: it can make a weak channel actively harmful and still under-use a strong one, and
   the per-query spread that causes this is large whether the channel's average effect is positive or
   negative.
2. Which queries benefit is predictable from the cheap channel's own score distribution, at negligible
   cost (1.04 ms/query, 0.01% of the decision being made).
3. The size of that predictability, and therefore the value of routing, is governed by the spread of
   per-query gain, which can be estimated in advance.

## Decisions I would like your opinion on

**Should we pivot from expansion-tier routing to channel routing?** Three reasons to:

- The effect is much larger, +3.07 nested gap against +2.23 for the caption tier and ~0 for the
  expansion tier.
- The channels ship with the benchmark as ranked lists, so it reproduces on CPU at no extra
  retrieval cost.
- It is what the strong systems on this benchmark already do, so the contribution sits on their path
  rather than beside it.

Honest caveat: absolute numbers are still moderate (0.30–0.48) because these are cheap channels, not
MMMORRF's retrievers. But the dense channel closes part of that gap (0.3134 for speech, above visual)
and routing survives on it, so the claim no longer rests only on the benchmark's toy lists. The routing
claim is about the policy, not the leaderboard.

**Which efficiency metrics should lead?** We now have latency distributions, throughput under
concurrency, energy per query, cost per relevant item, and risk–coverage/AURC. My instinct is that the
p99 tail curve is the one that carries the argument and the rest are supporting; tell me if a different
one lands better for this venue.
