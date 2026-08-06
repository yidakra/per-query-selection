# Tier C: paraphrase selection over Q2E's event decomposition

**Verdict: tier C does not exist. Q2E's max-pool over all generated paraphrases is already the
right thing to do, and the apparent headroom above it is label noise.**

This is a negative result. It is reported here in full because it also invalidates a class of
"oracle headroom" numbers, including two of our own.

§§1–7 below are MultiVENT, **noASR**, T = 259 queries, V = 2393 videos, nDCG@10, exact unfrozen
`fuse()`. **§8 replicates every panel on the ASR setting**, the one cell where the fixed-tier
B→Full prize is *largest*, and therefore the hardest case for this verdict. It survives.

---

## 1. Setup

Q2E decomposes a query into prequel / during / sequel events, generates ~8 paraphrases per event
type (~24 per query), scores each against every caption, and **max-pools** over the
paraphrase × caption grid (`text_embedder.get_many_to_many_score`, line 114).

Because the pool is a `max`, a bad paraphrase can never lower a video's score; it can only
**raise a wrong video's score**. That asymmetry motivated tier C: if we could drop the bad
paraphrases, we should win for free.

`perparaphrase_scores.py` rebuilds the three event components retaining the paraphrase axis,
`P[query, paraphrase, video]`, such that `P.max(dim=1)` reproduces Q2E's cached component.

## 2. The oracle says there is a lot on the table

`tierC_selection_oracle.py`, greedy forward selection per query against the true labels:

| arm | nDCG@10 | vs Full |
|---|---|---|
| Fixed-B (no events) | 74.28 | |
| Fixed-Full (all paraphrases, Q2E) | 76.19 | |
| oracle all-or-nothing (routing only) | 76.87 | +0.68 |
| **oracle subset (routing + selection)** | **78.19** | **+2.00** |

It keeps **0.9 of 24** paraphrases; 45% of queries keep none. Queries hurt relative to Fixed-B
fall from 25% to 3%. Selection, not routing, is 66% of the gain, and selection costs **zero LLM
calls**, reusing the ~30 generations/query Q2E already paid for (`llm_cost_accounting.py`).

Read on its own, this says: build a learned selector.

## 3. Two things that could have made it an artifact. Neither did.

**Baseline contamination.** ColBERT runs in fp16, and its scores depend on batch composition, so
the rebuilt components shift Full-tier nDCG by 0.0246. Scoring the selected arm from the paracache
while baselining against the *published* cache would fold that shift straight into a gain that is
itself ~1 nDCG. `load_cell` now returns the reconstruction for **both** arms, so it cancels.

**Zeros-row artifact.** An empty subset writes a zero row, which is *not* the same as dropping the
component: `fuse()` softmaxes across queries, so a zero row still contributes
`exp(0 − D[v]) = 1/D[v]`, which varies across videos, a query-independent video prior the oracle
could switch on and off. `diag_zerorow_prior.py`: omitting the component and zeroing it give
identical nDCG to **2.4e-07**, and the prior alone reproduces Fixed-B **exactly**. Not an artifact.

So the +2.00 is a correctly computed oracle. It is still meaningless.

## 4. No selector can reach it

`tierC_learned_selector.py`. Gradient-boosted classifier over 11 label-free features (agreement with
the cheap B-tier ranking, own max score, margin, rank, entropy, whether the paraphrase wins the
max-pool for any video, …). Outer 5-fold over queries, inner 3-fold picks the threshold; the model
never sees a test query's label and the threshold is never chosen on the data it is scored on.

| arm | nDCG@10 | vs Full |
|---|---|---|
| Fixed-Full | 76.19 | |
| random prune, cost-matched | 74.56 ± 0.12 | −1.63 |
| heuristic: top-k by own max score | 75.99 | −0.20 |
| **learned selector (nested CV)** | **74.70** | **−1.49**, 95% CI [−2.24, −0.74] |
| oracle subset | 78.19 | +2.00 |

The learned selector is **worse than doing nothing**, and beats random pruning by +0.14 (1.2 sd),
i.e. not at all. Realised fraction of the oracle gain: **−74%**.

## 5. Why: the oracle does not transfer across the label

Every query has ≥ 4 relevant videos (median 10). `tierC_goldsplit_oracle.py` splits each query's
golds into halves A and B, runs the *identical* greedy oracle using only half A, and grades it on
half B. A genuinely better subset does not know which half it will be graded on, so its advantage
must survive.

Graded on held-out half B, averaged over 5 splits:

| arm | nDCG@10 | vs Full |
|---|---|---|
| Fixed-B | 58.42 ± 0.65 | |
| Fixed-Full | 59.87 ± 0.39 | |
| oracle(B) on B, in-sample | 62.94 ± 0.60 | **+3.07** |
| oracle(A) on B, out-of-sample | 58.06 ± 0.49 | **−1.81** |

**Oracle optimism: +4.89 nDCG, more than twice the entire +2.00 headline.**

Three independent estimates now agree, and they agree on a number close to −1.6:

| how the subset was chosen | vs Fixed-Full |
|---|---|
| at random (cost-matched) | −1.63 |
| by a learned model (nested CV) | −1.49 |
| by an oracle forbidden from seeing the grading labels | −1.81 |

Any subset selection that is not fitted to the labels it is graded on **loses to keeping every
paraphrase**. The information the oracle exploits is which specific videos are marked relevant for
this query, not which paraphrases are good. There is nothing to learn.

**Q2E's max-pool is vindicated.** Dropping a paraphrase removes a chance to match a *right* video
as often as it removes a chance to match a wrong one.

## 6. A hypothesis we wrote down, then refuted

We expected optimism to grow with the size of the oracle's choice space: 3 tiers → 8 masks → 2^24
subsets. `tierC_optimism_curve.py`, same gold-split protocol:

| oracle | log2\|space\| | in-sample | out-of-sample | optimism |
|---|---|---|---|---|
| tiers (A/B/Full) | 1.6 | +4.91 | **−5.64** | **10.55** |
| all-or-nothing | 3.0 | +2.47 | −1.73 | 4.20 |
| subset | 24.0 | +3.07 | −1.81 | 4.89 |

**Refuted.** The *smallest* space has the *largest* optimism. Space size does not order optimism.
What does: the **spread in option quality**. Fixed-A trails Fixed-Full by ~8.5 nDCG, so a
noise-induced mis-pick between tiers is catastrophic, while swapping one paraphrase for another is
nearly free.

No oracle here has a positive out-of-sample gain. Every one of them loses to Fixed-Full.

### What this does and does not bound

`oracle(A) on B` decides each query independently from that query's ~5 remaining golds, a
high-variance estimator. It bounds oracles that use **only this query's own labels**. It does *not*
bound a model that pools across training queries; that model is tested separately in §4, pools, and
also loses.

It therefore does **not** contradict the routing study, whose nested-CV cascade achieves a positive
frontier gain (+0.73 noASR / +1.68 ASR) **at lower cost than Full**, and which never claimed to beat
Full in absolute nDCG. The routing claim lives strictly below the Full tier on the cost axis. Tier C
claimed to beat Full, and cannot.

## 7. Consequence for the paper

- **Do not build tier C.** Neither the learned selector (H2) nor the iterative regeneration loop
  (H1) is warranted: the premise both rest on (that a better paraphrase subset exists and is
  identifiable) is false for H2 and untested-but-now-implausible for H1.
- The Adaptive-RAG ladder over Q2E ends at **B → Full routing**, on the cost axis, as already
  reported.
- **Report oracle headroom with a held-out label split.** Our own +5.04 / +6.03 routing ceilings
  are in-sample quantities and should be labelled as such. The gold-split protocol here costs one
  extra evaluation and converts an uninterpretable ceiling into an interpretable one.

  **Now done**: see `router_findings.md` §4 (`router_oracle_goldsplit.py`). The routing ceilings
  behave exactly as this study predicts: halving MultiVENT's gold set *raises* the A→B ceiling
  (+5.04 → +6.77, +6.03 → +7.54), which recoverable headroom cannot do, and the oracle's ordering
  fails to transfer across gold halves (optimism 8.35 / 8.71). The nested-CV gaps (+0.73 / +1.68)
  are out-of-fold and unaffected; the "% of oracle captured" column is retired.

## 8. Replication on ASR: the harder case, same verdict

Everything above is noASR. The ASR cell is the one where fixed-tier Full pulls furthest ahead of B
(`router_hetero.json`: Full − B = **+2.34** on ASR vs +1.93 on noASR), so if a selection tier were
ever going to earn its keep, it would be here. It does not. Every panel replicates.

**§2 oracle: the mirage is identical.** The realizable greedy-frozen estimate again *exceeds the
oracle it is frozen from*, the label-noise signature, on both settings:

| | Fixed-B | Fixed-Full | oracle all-or-nothing | oracle subset | greedy-frozen | frozen > oracle? |
|---|---|---|---|---|---|---|
| noASR | 74.28 | 76.19 | 76.87 (+0.68) | 78.19 (+2.00) | **79.37** | **yes → mirage** |
| ASR | 78.31 | 80.63 | 81.36 (+0.73) | 82.33 (+1.70) | **83.25** | **yes → mirage** |

The load-bearing observation: **ASR's routable headroom is *smaller* (+1.70 vs +2.00), even though
its fixed-tier prize is *larger* (+2.34 vs +1.93).** A stronger base tier (ASR Fixed-Full 80.63 vs
76.19) leaves less for per-query selection to recover, not more. The apparent prize and the
recoverable prize move in opposite directions, which is exactly what "the headroom is label noise,
not signal" predicts. (Fraction of queries the oracle keeps none of is **44.79% on both**; it is a
query-side property, unchanged by the doc-side ASR transcripts.)

**§3 zerorow: not an artifact, again.** ASR omit vs zero-row: identical nDCG, `artifact_share = 0.0%`;
prior-only reproduces Fixed-B to 78.313 vs 78.313.

**§4 no selector reaches it: again worse than doing nothing.**

| arm | noASR vs Full | ASR vs Full |
|---|---|---|
| random prune, cost-matched | −1.63 | −1.88 (78.75 ± 0.14) |
| heuristic top-k by own max | −0.20 | −0.09 (80.54) |
| **learned selector (nested CV)** | **−1.49** [−2.24, −0.74] | **−1.72** [−2.43, −1.00] |
| oracle subset | +2.00 | +1.70 |

The learned selector beats random pruning by **+0.16** (≈1 sd) on ASR (i.e. not at all) and
realizes **−101.5%** of the oracle gain (noASR: −74%): it moves the wrong way.

**§5 the oracle does not transfer across the label, again.** Gold-split, graded on held-out half B,
5 splits:

| arm | noASR | ASR |
|---|---|---|
| Fixed-Full | 59.87 | 62.79 |
| oracle(B) on B, in-sample | 62.94 (+3.07) | 65.42 (+2.63) |
| oracle(A) on B, out-of-sample | 58.06 (−1.81) | **60.64 (−2.15)** |
| optimism | 4.89 | 4.78 |

The out-of-sample oracle again lands **below Fixed-Full**: choosing a subset from a query's own
labels and applying it to that query's *other* labels loses to keeping everything.

**§6 optimism is not ordered by choice-space size, again.** ASR gold-split optimism by oracle:
tiers (log₂ 1.58) **10.85** > all-or-nothing (log₂ 3.0) 5.04 > subset (log₂ 24.1) 4.78. The
*smallest* space has the *largest* optimism, refuting the size hypothesis on both settings; what
orders optimism is the spread in option quality (A trails Full by ~8 nDCG), not the space size.

**One code fix was required for this cell.** ASR's source data has one query (row 1306, "EAST
tokamak") whose event paraphrases are empty where noASR's has five, so it has zero valid paraphrases
in every event. `tierC_learned_selector.build_features` assumed every (query, event) had ≥1 and
crashed on the empty `torch.stack`; it now skips such (query, event) pairs, matching `allsel`, which
already enumerated them identically. The paracache correctly marks those paraphrases invalid; the
emptiness is in the ASR dataset, not the reconstruction.

**Bottom line.** Tier C is dead on the full dataset, not half of it. The verdict was not an artifact
of the noASR setting, and the setting with the most fixed-tier headroom to give back gives back the
least under honest, label-split evaluation.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=1 python src/evaluation/perparaphrase_scores.py --setting noASR --event all
CUDA_VISIBLE_DEVICES="" python src/evaluation/tierC_selection_oracle.py --setting noASR
CUDA_VISIBLE_DEVICES="" python src/evaluation/diag_zerorow_prior.py    --setting noASR
CUDA_VISIBLE_DEVICES="" python src/evaluation/tierC_decompose.py       --setting noASR
CUDA_VISIBLE_DEVICES="" python src/evaluation/tierC_learned_selector.py --setting noASR
CUDA_VISIBLE_DEVICES="" python src/evaluation/tierC_goldsplit_oracle.py --setting noASR
CUDA_VISIBLE_DEVICES="" python src/evaluation/tierC_optimism_curve.py  --setting noASR
```

§8 (ASR) reruns the same ladder with `--setting ASR`; the paracache is regenerated once on GPU1
(GPU0 hosts an unrelated whisper server; keep GPU work off it):

```bash
CUDA_VISIBLE_DEVICES=1 python src/evaluation/perparaphrase_scores.py --setting ASR --event all
for s in tierC_selection_oracle diag_zerorow_prior tierC_decompose \
         tierC_learned_selector tierC_goldsplit_oracle tierC_optimism_curve; do
  CUDA_VISIBLE_DEVICES="" python src/evaluation/$s.py --setting ASR    # selection_oracle must run first
done
```
