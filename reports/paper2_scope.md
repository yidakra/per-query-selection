# Paper 2: the efficiency material, a scope note

Decided 31 Jul 2026: the efficiency and cascade work becomes its own paper rather than an appendix to
the boundary-condition paper. This note exists so the material stops being homeless. It is a scope
sketch, not an outline: the framing question below is not settled and should be settled before anyone
allocates sections.

## The material

All measured, all in the repo.

| what | where |
|---|---|
| Per-stage latency, per-tier throughput, concurrency | `efficiency_metrics.md` §1–2 |
| Percentile curve vs escalation fraction; p99 ×435 at f = 0.10 | `efficiency_metrics.md` §3 |
| Measured GPU energy, carbon, joules per relevant item (0.57 vs 146.5) | `efficiency_metrics.md` §4 |
| Risk–coverage and AURC against oracle and random | `efficiency_metrics.md` §5 |
| Measured component energy; the affine cost model, R² = 0.9994 | `router_findings.md` §5 |
| sd(per-query gain) predicts achieved routing gap, ρ = 0.943 over 6 cells | `router_findings.md` §3 |
| Router transfer across encoders and settings; sign flip across benchmarks | `transfer_findings.md` |
| Oracle ceilings are in-sample and mostly label noise | `router_findings.md` §4 |
| The A→B expected-gain cascade and its closed-form gap | `router_findings.md` §2 |

## The framing problem, which is real

The obvious title is "per-query routing on an accuracy–compute frontier," and it does not survive
contact with our own numbers. **Routing among modality channels saves nothing.** The three channels cost
about the same, so the +7.59 nDCG is an accuracy result with no compute story attached. The compute
story lives one tier up, at the LLM expansion step, and there the router's honest finding is that the
escalation is not worth making: oracle prize +2.55 that does not survive a held-out gold split, against
a price of 63.5× tier B.

So paper 2 cannot be "routing saves compute." The two halves would sit next to each other without
touching, which is exactly the objection a reviewer would raise first.

The framing that does hold, and my suggestion:

> **When is a cost cascade worth building?** Escalation is usually assumed worthwhile and rarely
> measured. Here it is measured: the price is 633× latency and 610× throughput for tier B → Full, and
> escalating a tenth of queries multiplies p99 by 435×, so the tail sets the budget rather than the
> mean. Against that bar the prize has to be large, and the way to know in advance is the spread of
> per-query gain, which predicts the achieved routing gap at ρ = 0.943 across six cells. A system
> designer can compute that spread before building the expensive tier.

That makes the negative result the contribution and the heterogeneity correlation the tool. The
accuracy-routing work then appears as the mechanism being priced, cited to paper 1 rather than
re-argued.

## Overlap with paper 1, which needs managing

Both papers touch the +7.59 selector. In paper 1 it is the positive control that makes the pre-retrieval
null interpretable; in paper 2 it is the mechanism whose cost is being assessed. That is a defensible
split, but it is the kind of thing a PC checks, so:

- Paper 1 goes first and paper 2 cites it. The reverse ordering would make paper 1 look like a follow-up.
- Paper 2 does not re-report the +7.59 as a contribution. One table, cited, framed as prior.
- If they end up under review at the same venue in the same cycle, disclose the companion.

## What paper 2 still needs

- **A coherent system boundary.** `efficiency_metrics.md` measures the MultiVENT 2.0 cascade; the six
  heterogeneity cells are MultiVENT v1 and MSR-VTT under the original Q2E pipeline; the component
  energy model is the original pipeline too. Three systems in one argument. Either the paper is
  explicitly a cross-system study or the numbers need re-measuring on one.
- **FLOPs in a hardware-independent unit**, index/memory footprint, and a QPS-vs-load curve. The
  efficiency review flagged these and they are still missing.
- **A venue.** Undecided. The negative-result framing is a poor fit for a venue that wants systems
  wins, and a good fit for a reproducibility or evaluation track.
- Tier A energy is a TDP estimate, not a measurement, since the host exposes no RAPL counters. Fine to
  report as an order of magnitude; not fine as a headline number.

## Not yet decided

Whether paper 2 is worth writing before paper 1 is submitted. The material is measured and will not
rot, and paper 1 is the one with a deadline and a live conversation attached to it.
