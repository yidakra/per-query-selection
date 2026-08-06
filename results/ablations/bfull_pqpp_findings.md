# Can a PQPP-style signal resurrect the B→Full escalation? No: it is unpredictable *ex ante*

**Verdict.** The B→Full escalation stays not-worth-buying, now for a **third independent reason** to add
to the small prize (`router_findings.md` §4) and the 177× price (`cost_model_findings.md` §6): **whether
the LLM event-decomposition helps a query cannot be predicted before you run it.** The only
routing-legal signal (the query as a generation *prompt*, in the sense of PQPP (Poesina et al., 2025))
carries **nothing** (Kendall τ = −0.05 / −0.07, negative nested gap, fails the gold-split acid test on
both cells). Whatever real B→Full signal exists survives the acid test only in features computed *after*
the generation and scoring have been paid for, and even then it is a fraction of an NDCG point. You
cannot know if the decomposition will help until you have already bought it.

`router_bfull_pqpp.py`, MultiVENT noASR + ASR (the only cells where event decomposition is meaningful and
multi-gold enough to split). CPU-only. Target: per-query B→Full gain `h = nDCG_Full − nDCG_B`
(mean +1.93 / +2.34, sd 7.91 / 7.54: a small, heavy-tailed prize).

## 1. Three feature sets, ranked by cascade legality

The routing decision at tier B must be made *before* paying for the ~30 LLaMA-70B generations. That cost
is the dominant part of Full (~2548 J of ~3967 J, `cost_model_findings.md` §6), so **only the prompt is
truly routing-legal**; anything derived from the generation or its scoring has already spent what the
router is deciding whether to spend.

| feature set | legality | computed from |
|---|---|---|
| **prompt** | **legal** (pre-generation) | the query text alone: length, digits/year, capitalized-token count, a small news-event lexicon, temporal markers. PQPP proper. |
| generation | illegal (paid the LLM) | the generated prequel/during/sequel paraphrases, before scoring: count, diversity, query-overlap |
| postscore | illegal (paid it all) | Full-tier retrieval confidence + how much events moved the ranking vs B |

Two evaluations per set: (a) OOF nested gap + Kendall τ / permutation on full golds; (b) **the acid test**,
the gold-split from §4 that killed the oracle: order queries by a ridge trained on one gold half's
B→Full gain, grade the realized gap on the *other* half. A signal is real only if it survives (b).

## 2. The routing-legal prompt signal is absent

| cell | prompt τ (p) | prompt nested gap | prompt acid OOS | oracle acid OOS |
|---|---|---|---|---|
| MultiVENT noASR | −0.050 (.86) | −0.20 ± 0.29 | −0.00 ± 0.23 | −0.37 |
| MultiVENT ASR | −0.068 (.95) | −0.30 ± 0.16 | −0.09 ± 0.21 | −0.72 |

The prompt predicts B→Full gain no better than chance: τ is *negative* on both cells, the nested gap is
negative, and the gold-split acid test dies. Predicting decomposability from the query alone does not
work here.

**This is a real divergence from PQPP, and the reason is instructive.** PQPP predicts, from a text-to-image
prompt, how *good the generation itself* will be, an intrinsic property of the prompt. We would need to
predict whether the generation is *useful for retrieval in this gallery*, an **interaction** between the
query and the video collection, which is not a property of the prompt. This is the same lesson as
`router_findings.md` §1 (the A→B signal is not in the query text either): across both escalations, the
query surface form never carries the routing signal; the retrieval interaction does.

## 3. Real signal exists, but only after you have already paid, and it is tiny

| cell | feature set | τ (p) | nested gap | acid OOS | verdict |
|---|---|---|---|---|---|
| noASR | generation | +0.096 (.012) | +0.55 ± 0.26 | +0.15 ± 0.15 | survives (noASR only) |
| noASR | postscore | +0.023 (.29) | +0.30 ± 0.29 | +0.21 ± 0.26 | survives, weak |
| noASR | all | +0.115 (.003) | +0.45 ± 0.17 | **+0.32 ± 0.19** | survives, strongest |
| ASR | generation | −0.003 (.53) | −0.06 ± 0.19 | −0.08 ± 0.18 | dies |
| ASR | postscore | +0.042 (.15) | +0.28 ± 0.19 | +0.13 ± 0.24 | survives, weak |
| ASR | all | +0.065 (.06) | +0.22 ± 0.25 | +0.18 ± 0.19 | survives, weak |

Three things, all pointing the same way:

- **The signal that survives is post-payment.** Every feature set that beats the oracle's negative
  out-of-sample gap (−0.37 / −0.72) requires having run the generation (and usually the scoring). None is
  available at the routing decision point.
- **It is fragile.** The generation-content signal is significant on noASR (τ +0.096) and *absent* on ASR
  (τ −0.003). Only the illegal post-scoring / all sets are consistently positive, and their τ is not
  significant.
- **It is tiny.** The best out-of-sample acid gap is +0.32 NDCG (noASR, `all`), against an in-sample
  oracle of +2.95, i.e. ~90% of the apparent prize was label noise, exactly as §4 found. What is left is
  a fraction of a point, reachable only after paying 177× tier B.

## 4. What this means for the paper

- **B→Full is dead for a third, independent reason: ex-ante unpredictability.** §4 killed it on the size
  of the prize (small, mostly label noise); §6 of the cost model killed it on price (177×); this kills it
  on *timing*: the only signal that is real cannot be seen until the expensive step is already done. Any
  one of the three suffices; together they are conclusive.
- **The PQPP direction was worth testing and is a clean negative.** Giving the escalation its best shot
  (prompt features in the exact PQPP spirit, plus generation and post-scoring features as an upper bound,
  all under the gold-split acid test) makes the "never buy Full" verdict far stronger than the original
  retrieval-confidence-only check. It also yields a citable contrast: prompt-side performance prediction,
  which works for generation *quality* (PQPP), fails for generation *utility to retrieval*.
- **Efficiency framing intact.** This is a negative result reported as one; it does not claim to beat Full,
  it shows there is no ex-ante-visible reason to invoke it.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_bfull_pqpp.py
# -> results/ablations/router_bfull_pqpp.json
```

Reads cached component tensors under `runs/<tag>/cache/` and the dataset's generated event fields.
