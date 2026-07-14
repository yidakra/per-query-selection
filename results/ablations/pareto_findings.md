# The effectiveness hook: the router moves the efficiency–effectiveness Pareto frontier

The efficiency framing ("+gap nDCG at a fixed budget") is the vertical reading of one fact. Read
horizontally it is an **effectiveness** claim a full-track IR reviewer will accept: **to reach a given
accuracy, the router costs less than the honest baseline.** `router_pareto.py`, measured joules,
existing A→B caches — no new retrieval. Predicted gain is out-of-fold (same protocol as
`router_hetero.py`).

**Headline.** On the accuracy–compute plane the router lifts the random-escalation chord by `gap(f)`
across the operating region, so it **reaches the accuracy of escalating 50% of queries at random for
24–58% less escalation cost** (1.9–4.5 J/query), equivalently **+0.24 to +1.92 nDCG at equal cost**.
And the frontier stops at tier B *by design*: Full is a dominated escalation the router correctly
never buys — on one cell Full is literally Pareto-dominated (lower nDCG than B at 63× the cost).

## 1. Two readings of the same frontier

On the plane, `cost(f) = J_A + f·(J_B − J_A)`, the **random-f chord** (escalate a random fraction) is
`ndA + f·(ndB − ndA)`, and the **router** is that chord lifted by `gap(f) ≥ 0`. So:

| cell | Fixed-A | Fixed-B | ISO-COST: +nDCG at equal cost | ISO-ACCURACY: budget to match random@50% | saving |
|---|---|---|---|---|---|
| MultiVENT/mCLIP/noASR | 67.71 | 74.28 | **+1.23** (f*=0.42) | f=0.34 | **32%** (2.48 J/q) |
| MultiVENT/mCLIP/ASR | 67.71 | 78.31 | **+1.92** (f*=0.58) | f=0.38 | **24%** (1.86 J/q) |
| MSR-VTT/mCLIP/noASR | 59.72 | 60.94 | +0.24 (f*=0.52) | f=0.34 | 32% (2.48 J/q) |
| MSR-VTT/mCLIP/ASR | 59.72 | 61.95 | +0.51 (f*=0.43) | f=0.32 | 36% (2.79 J/q) |
| MSR-VTT/IV2/noASR | 66.00 | 67.52 | +0.54 (f*=0.54) | f=0.33 | 34% (2.64 J/q) |
| MSR-VTT/IV2/ASR | 66.00 | 68.86 | **+1.06** (f*=0.53) | f=0.21 | **58%** (4.50 J/q) |

- **ISO-COST** (vertical): at a budget the router beats the only fixed alternative at that budget (a
  cost-matched random A/B mixture) by up to +1.92 nDCG. This is `router_findings.md` §2's gap, restated
  on the accuracy–compute plane.
- **ISO-ACCURACY** (horizontal, the new positive framing): random escalation needs 50% of the budget
  to reach its 50%-mix accuracy; the router reaches the *same* accuracy escalating only **21–38%** of
  queries — a **24–58% cut in escalation cost at equal effectiveness.** This is the "same nDCG, lower
  cost" statement, and it is what turns an efficiency result into an effectiveness one.

## 2. The router frontier dominates the chord across the operating region

`router_ndcg(f) ≥ chord_ndcg(f)` wherever `gap(f) ≥ 0`, which holds across the whole significant
plateau (`f ≈ 0.1–0.95`). The only sub-chord excursions are **≤ 0.12 nDCG and confined to the extreme
tails** (`f < 0.06` or `f > 0.99`), where routing is moot because you escalate almost nobody or almost
everybody. So the honest statement is *weak Pareto dominance over the operating region*, not a
literal "dominates at every f" — the tail dips are predictor noise at fractions no deployment uses.

## 3. The negative results are the frontier's *shape*, not a failure

A Pareto-optimal router escalates only where the accuracy gained outweighs the cost. The paper's
negatives are exactly the escalations that fail that test, and the router prunes them:

- **B→Full is a dominated escalation.** Full costs **177× tier B** end-to-end
  (`cost_model_findings.md` §6) for a prize that is small and mostly label noise (`router_findings.md`
  §4) and unpredictable *ex ante* (`bfull_pqpp_findings.md`). On **MSR-VTT/IV2/noASR the domination is
  literal**: Fixed-Full 67.11 < Fixed-B 67.52 — Full buys **−0.41 nDCG for 63× the similarity cost.**
  On the other cells Full clears B by only +0.56…+2.34 nDCG at 63–177× the cost, i.e. dominated on any
  cost-normalized axis.
- **Tier C (paraphrase selection) is a dominated escalation** — its headroom is label noise
  (`tierC_findings.md`), so it adds cost for no reliable accuracy.

So "we tried the expensive tiers and they don't help" is not a null result — it is *why the frontier
turns over at B*. The router is Pareto-efficient **because** it declines the dominated escalations,
and the negatives are the evidence that those escalations are dominated.

## 4. What this changes for the paper

- **Lead with the Pareto frontier**, not "efficiency only". Primary claim: the router moves the
  efficiency–effectiveness frontier — **same accuracy for 24–58% less escalation cost**, or +nDCG at
  equal cost, in the sub-B budget regime. This is the effectiveness hook the venue analysis flagged as
  required (`paper/related_work.md`).
- **Demote the negatives to the frontier's shape** (§3): they explain where the frontier stops, they
  are not the thesis. This is the RLT / cutoff framing (fixed-k hard to beat, upper tiers pruned).
- **Keep the scope honest.** The router does **not** beat Fixed-B in absolute nDCG; the claim is
  frontier dominance in the `[A, B]` budget region, where the only fixed alternative is a cost-matched
  random mixture. Stated plainly this is a legitimate accuracy–compute-frontier result, not "we beat
  Q2E".

See `reports/figures/router_pareto.png` — router (solid) above the random chord (dashed) in the A→B
region, Full 60–177× off-scale to the right.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES="" python src/evaluation/router_pareto.py
# -> results/ablations/router_pareto.json, reports/figures/router_pareto.{pdf,png}
```

Reads cached component tensors under `runs/<tag>/cache/`; costs from `tier_cost.py`.
