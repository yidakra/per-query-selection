# Is the component-count cost proxy right? Measured joules per component

> **Scope:** this cost model was measured on the **original Q2E pipeline** (ViT-H similarity,
> `mx_q=30` padding, the paper's own corpora). It does **not** describe the MultiVENT 2.0 cascade,
> whose measured latency and energy are in [`legacy/reports/efficiency_metrics.md`](../../legacy/reports/efficiency_metrics.md);
> there, tiers A and B are CPU-only and cost ~0.01 J and ~1 J per query. Do not mix the two tables.


Every frontier figure in this repo plots **cost = number of similarity components scored**,
normalised so Full = 1.0 (A = 0.2, B = 0.4). That rates all five components at unit cost. This
is the check that the assumption deserved, and it does not survive it.

`llm_cost_accounting.py` already showed the proxy ignores the ~30 LLaMA-3.3-70B generations the
Full tier issues per query. This document attacks the other half: **are the similarity components
themselves equal cost?** They are not: they differ by up to **68×**.

Measured on MultiVENT noASR (T=259 queries, V=2393 videos, 17 caption slots per video), NVML power
sampled at 5 Hz on physical GPU1 and integrated, minus a warm idle baseline. Scripts:
`component_energy_bench.py` (the four ColBERT caption components) and `component_energy_tierA.py`
(the MultiCLIP video component).

## 1. The cost law

Four of the five components run through **one code path** (`run_eval.py:221-228` →
`get_many_to_many_score`), differing only in how many query-side strings they feed it. Upstream
(`text_embedder.py:96-102`) pads every query to `mx_queries` paraphrase slots and concatenates:

| component | `mx_q` | strings scored per doc slot |
|---|---|---|
| `query_vs_captions` | 1 | 259 |
| `{prequel,during,sequel}_vs_captions` | 30 | **7,770** (2,076 non-empty) |

For a fixed doc slot, cost is affine in the number of query-side strings `N`:

```
E(N) = a + b·N        a = 3167 J    b = 0.9124 J/string     R² = 0.9994   (net, per doc slot)
                      a =  83.6 s   b = 0.0493 s/string     R² = 0.9959
```

Fitted on `N ∈ {259, 519, 1038}` only. Two held-out points confirm it:

| held out | predicted | observed | error |
|---|---|---|---|
| N = 2,076 | 5,061 J | 5,077 J | **−0.33%** |
| N = 7,770 (upstream padding) | 10,256 J | 10,392 J | **−1.31%** |

A 7.5× extrapolation lands within 1.3%. The law is not fitted noise.

**Per-string cost is length-independent.** The real queries (mean 172.7 chars) cost 3,444 J at
N=259; event paraphrases (mean 116.5 chars) cost 3,398 J at the same N, a ratio of 1.014. ColBERT
pads to `query_maxlen`, so what you pay for is *how many strings*, not how long they are. The cost
axis is therefore a **string count**, exactly the quantity the proxy assumes is constant.

## 2. The intercept dominates a single call, and that is a red herring

`a = 3167 J` is **93.1%** of a `query_vs_captions` call. It is the doc side: `_patched_one_to_one`
calls `RAG.encode(seqs2)` on all 2,393 captions **every call**, then `clear_encoded_docs`.

Charge that intercept to every component and the components look nearly equal (`event/qvc =
1.49×`) and the 1:1 proxy looks fine. **That accounting is wrong for routing.** A caption index is
built once, offline, and is paid identically by every tier; it is not a per-query cost. What a
router pays when it escalates one query is the **marginal** cost `b·N`. The intercept is an artefact
of the evaluation harness re-encoding the gallery on every call, not a property of the method.

So there are two defensible numbers, and they differ by 5×:

| accounting | event / `query_vs_captions` |
|---|---|
| amortised, index rebuilt every call (what the code does) | 1.49× |
| **marginal, index built once (what a deployment pays)** | **8.02×** unpadded |
| marginal, with upstream's `mx_q=30` padding (what was actually run) | **30.0×** |

## 3. Tier A pushes black images through a ViT-H

`vision_embedder.get_text_embedding` (line ~148) does:

```python
video = torch.zeros((bsz * 1, 3, 224, 224))
outputs = model(video, input_ids)
sequence_output = outputs["text_features"]      # image features thrown away
```

Every text batch drags a batch of **black images** through the ViT-H vision tower, then discards
the result. Measured:

| tier-A marginal | J / query |
|---|---|
| as shipped | **6.83** |
| vision tower bypassed (`model.encode_text`) | **0.83** |

**8.3× of tier A's query-side cost is a no-op.** We report the as-shipped number below, because it
is what the published nDCG figures actually paid. It is worth knowing the floor is 8× lower.

## 4. What the five components actually cost

Marginal energy per query, across all 17 doc slots (net of idle, GPU1):

| component | measured | proxy |
|---|---|---|
| `query_vs_video` | 6.83 J | 1 unit |
| `query_vs_captions` | 15.51 J | 1 unit |
| one event component, unpadded | 124.40 J | 1 unit |
| one event component, **as published** (`mx_q=30`) | **465.34 J** | 1 unit |

Spread between the cheapest and most expensive "unit": **68×**.

Tier totals, as published:

| tier | measured | normalised (Full=1) | proxy |
|---|---|---|---|
| A | 6.83 J | 0.0048 | 0.2 |
| B | 22.35 J | 0.0158 | 0.4 |
| Full | 1,418.36 J | 1.0 | 1.0 |

The proxy **overstates A by 42× and B by 25×** relative to Full. Equivalently: escalating B→Full
costs **63.5× tier B**, where the proxy claims 2.5×. Even with the padding removed it is 17.7×.

And this counts only GPU similarity cost. Full alone additionally issues ~30 generations on a 70B
model per query (`llm_cost_accounting.py`), which is not in the 1,418 J at all.

## 5. What this changes, and what it does not

**The routing results are invariant.** This is not a hedge; it follows from the ladder being
nested. A router escalating a fraction `f` of queries from A to B pays

```
cost(f) = (1−f)·cost(A) + f·cost(B) = cost(A) + f·(cost(B) − cost(A))
```

This is affine and strictly increasing in `f` under *any* per-component cost assignment. Proxy:
`0.2 + 0.2f`. Joules: `6.83 + 15.51f`. A cost-matched baseline is therefore an **`f`-matched**
baseline in either unit, so every gap in `router_findings.md` §2 (nested-CV `+0.73` / `+1.68`,
permutation-tested) is unchanged. Only the axis labels move.

**The "never buy Full" conclusion gets stronger, and for a second, independent reason.**
`router_findings.md` §2 killed the B→Full escalation on the size of the prize (oracle gap ≈ +2.55,
itself in-sample per §4). We can now also kill it on price: Full costs **63.5× tier B**, not the
2.5× the figures imply. Two independent arguments, one about the numerator and one about the
denominator.

**The proxy is conservative, in the direction that matters.** It understates Full's cost, so every
saving this work reports against Full is a *lower bound*. Nothing needs to be walked back. But the
frontier figures should not be read as though the x-axis were energy: it is a component count, and
a component is not a unit of anything.

### Caveats

- Similarity/GPU energy only. Excludes the ~30 LLM generations per query that **only Full** pays,
  and excludes CPU (no RAPL on this host; CodeCarbon's CPU figure is a TDP estimate, not a
  measurement).
- Absolute joules are A2-specific. The **string-count scaling law** (§1) and the **ratios** (§4) are
  not: they follow from `mx_q` and the code path, not the hardware.
- Gallery/index encoding is excluded by design as amortised and tier-independent. §2 states the
  alternative accounting explicitly rather than burying the choice.
- Extrapolating the single benchmarked doc slot (slot 0) to all 17 overpredicts the full
  `during_vs_captions` component by +19% energy / +14% wall vs the 35.62 Wh independently recorded
  by `perparaphrase_scores.py`. Cause is known and checked: slot 0 holds the video-level caption
  (1,021 chars) while slots 1-16 hold frame captions (~460 chars), so slot 0 carries **2.13×** the
  average doc-encode load. The *ratios* in §4 depend only on the within-slot slope `b`, which this
  bias does not touch.

## 6. End-to-end: the LLM decomposition Full pays (estimate)

Everything above is **measured** GPU similarity energy. It excludes the one thing only the Full tier
buys: ~30 LLaMA-3.3-70B generations per query for the event decomposition (`llm_cost_accounting.py`:
mean 30.0 calls, **8,046 prompt + 936 generated tokens/query**). The 70B ran offline, so this can only
be **estimated**, but the estimate is worth having, because it is the largest cost in the pipeline.

Two independent methods (`llm_cost_accounting.py`):

- **FLOPs (primary).** A dense decoder forward is ~`2·P` FLOP per token processed (prefill and cached
  decode alike), so `FLOP/query = 2·70e9·(8046+936) = 1.26e15`. Divide by effective delivered
  efficiency (accelerator peak × MFU; peaks: A100 BF16 0.78, H100 BF16 1.41, H100 FP8 2.83 TFLOP/J).
  The workload is prefill-dominated (8.6:1 prompt:gen), so MFU sits at the compute-bound end (~25–40%):
  **1,111 J (H100 FP8) – 6,449 J (A100 BF16), central 2,548 J.**
- **Per-output-token (cross-check).** Published measured energy per *generated* token: 0.39 J
  (Llama-3-70B FP8, 8×H100 vLLM, batched) to 3.5 J (Llama-65B, A100/V100, Samsi et al. 2023) →
  365–3,277 J. This brackets the FLOPs band **from below**, because it amortises little prefill while
  we prefill 8× more than we generate.

The estimate spans ~6×, but the conclusion does not:

| | similarity only (measured) | + LLM decomposition (est.) |
|---|---|---|
| Full, J/query | 1,418 | **~3,967 central** (1,783–7,867) |
| B → Full price | 63× tier B | **~177× tier B** (80–352×) |

**The LLM decomposition is comparable to or larger than Full's entire similarity cost** (central
1.8×), and tiers A and B pay **none** of it: the router never triggers a single generation. So the
already-conservative "savings vs Full" from §5 understate the real gap by roughly another 2–5×, and
the "never buy Full" verdict holds on end-to-end energy, not just similarity FLOPs. This is an
estimate and labelled as one throughout; the measured axis (§1–5) is unchanged.

## 7. The router's own cost is negligible: the frontier is router-inclusive

The frontier charges tier-A/B similarity energy but not the cost of the routing *decision* itself. The
obvious reviewer question: at a 15.52 J A→B escalation, does the router's overhead eat the saving? It
does not, by ~3½ orders of magnitude. `router_overhead.py`, measured on the real MultiVENT noASR score
vectors (T=259, V=2393):

| per-query decision step | wall-clock |
|---|---|
| feature extraction (`conf_feats`: softmax/sort/std over the V-dim gallery vector) | 343.0 µs |
| ridge inference (standardize + dot, raw) | 2.1 µs |
| -- ridge inference via sklearn `.predict`, as-implemented upper bound | 185.6 µs |
| **per-query router decision** | **345.1 µs** |

At a single-core CPU-TDP estimate of 5–25 W (no RAPL on this host, same caveat as the CPU figures
above), that is **~5.2 mJ/query** (1.7–8.6 mJ). The router pays it on *every* query, escalation only on
the top-f, so charge it to all and compare:

| router overhead vs | ratio |
|---|---|
| A→B marginal (15.52 J) | **0.033%**: the escalation is **~3,000×** the router |
| one tier-A evaluation (6.83 J) | 0.076% |
| Full (1,418 J) | 0.00036% |

So the **router-inclusive** cost, `cost(A) + E_router + f·(cost(B) − cost(A))`, differs from the axis
we plot by a constant ~5 mJ (0.08% of tier A, invisible at figure resolution) and the *gap* is exactly
unchanged, since a constant added to every operating point cancels (same invariance as §5). The feature
extraction dominates (the ridge is ~2 µs of genuine FLOPs; the 186 µs sklearn number is Python dispatch,
not intrinsic cost), and even charging the full as-measured `conf_feats` time is conservative: it is
work a deployment could fuse into tier-A scoring. The routing decision is free relative to what it saves.

## Reproduce

```
CUDA_VISIBLE_DEVICES=1 python src/evaluation/component_energy_bench.py --repeats 2 --padded
CUDA_VISIBLE_DEVICES=1 python src/evaluation/component_energy_tierA.py
CUDA_VISIBLE_DEVICES="" python src/evaluation/llm_cost_accounting.py
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_overhead.py
```

GPU1 only: GPU0 hosts an unrelated whisper server whose idle draw (22.3 W, measured) would
otherwise be billed to these runs. See `tracking.py`.
