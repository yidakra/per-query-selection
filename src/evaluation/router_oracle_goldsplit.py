#!/usr/bin/env python
"""How much of the router's oracle ceiling (+5.04 / +6.03) is in-sample optimism?

router_gain_curve.py reports an "oracle" column computed as

    gap_at(f, ghat=g, g=g)          # rank queries by the TRUE per-query gain

i.e. the oracle orders queries by a gain measured on the SAME relevance judgments used to score
the result. That is in-sample selection, exactly the pattern tierC_goldsplit_oracle.py showed can
manufacture several nDCG points of pure optimism.

The tier-C study measured optimism for a per-query oracle over {A, B, Full} vs Fixed-Full. This is
a different quantity from the router's oracle, which is the max achievable ESCALATION GAP over the
cost-matched A-B chord. So the caveat has to be measured here directly, not imported.

PROTOCOL. MultiVENT gives 9.24 relevant videos per query. Split each query's golds into halves A
and B, then:

  in-sample     order queries by gain measured on B, pick f on B, evaluate the gap on B
  out-of-sample order queries by gain measured on A, pick f on A, evaluate the gap on B
  OPTIMISM      in-sample - out-of-sample

Both escalate the same number of queries; only the information used to choose them differs. An
oracle whose ordering reflects real per-query structure keeps its advantage when the grading
labels change. One that has memorised which videos are marked relevant does not.

A SECOND, SHARPER TEST, which needs no transfer argument at all. Recompute the in-sample ceiling on
HALF the golds. Halving the gold set strictly REMOVES information. A ceiling that measures real
recoverable headroom cannot rise when information is deleted; a ceiling that measures how well an
oracle can fit the labels must. We report both, so the reader can see which it does.

MSR-VTT CANNOT BE TESTED THIS WAY: 1.01 golds per query, so there is no second half to grade on.
Its ceilings (+1.46 to +3.09) are reported in-sample and must stay labelled as such.

WHAT THIS DOES NOT SHOW, AND MUST NOT BE READ AS SHOWING. The out-of-sample gap is NEGATIVE, but
that does not mean the achievable gain is negative, and it does not contradict the nested-CV
result. The out-of-sample oracle orders queries by a per-query gain estimated from ~4 golds -- a
high-variance estimator that lands on queries whose gain is noisily large. The nested-CV router
instead POOLS across training queries, trading that variance for bias, and achieves a
permutation-significant +0.73 / +1.68. A pooled learner can beat a per-query oracle fed noisy
labels; there is no paradox.

Nor is the ceiling "wrong". gap(f) is maximised by ordering on the true gain, so on a FIXED label
set the in-sample ceiling is a valid upper bound on any router's achieved gap. The point is that it
is a very loose one: a large share of the per-query gain on any one label set is irreducible label
noise, which no feature can predict. The ceiling is a bound, not a target, and "% of oracle
captured" is therefore not a measure of how much room is left.

CPU-only.
"""
import os, sys, json, argparse, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
from tracking import track                                            # noqa: E402
import tierC_selection_oracle as O                                    # noqa: E402
from tierC_goldsplit_oracle import split_golds                        # noqa: E402
from router_hetero import comps_multivent, gap_at                     # noqa: E402

FS = np.linspace(0.05, 1.0, 20)


def per_q(comps, tier, tgt):
    return O.per_query(O.fuse(comps, O.LADDER[tier]), tgt)[0].numpy()


def best_gap(ghat, g):
    """Choose f to maximise the gap, using ghat for BOTH the ordering and the f sweep."""
    scores = [gap_at(f, ghat, ghat) for f in FS]
    f_star = FS[int(np.argmax(scores))]
    return f_star, gap_at(f_star, ghat, g) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    with track("router-oracle-goldsplit", gpu_ids=[], tags=["router", "methodology"],
               config={"seeds": a.seeds}) as tr:
        out = {}
        for setting in ["noASR", "ASR"]:
            queries, target, comps = comps_multivent(setting)
            T = len(queries)
            ins, oos, ins_bf, oos_bf = [], [], [], []

            # reference: the ceiling exactly as router_findings.md reports it, on the FULL gold set
            g_full = per_q(comps, "B_noevents", target) - per_q(comps, "A_visual", target)
            _, ceil_full = best_gap(g_full, g_full)
            golds_full = float(target.sum(1).float().mean())

            for s in range(a.seeds):
                rs = np.random.RandomState(500 + s)
                A, B = split_golds(target, rs)

                gA = per_q(comps, "B_noevents", A) - per_q(comps, "A_visual", A)
                gB = per_q(comps, "B_noevents", B) - per_q(comps, "A_visual", B)
                # in-sample: order and f from B, graded on B
                _, i_gap = best_gap(gB, gB)
                # out-of-sample: order and f from A, graded on B
                fA, _ = best_gap(gA, gA)
                o_gap = gap_at(fA, gA, gB) * 100
                ins.append(i_gap); oos.append(o_gap)

                # the B -> Full escalation, the one the paper says is never purchasable
                hA = per_q(comps, "C_full", A) - per_q(comps, "B_noevents", A)
                hB = per_q(comps, "C_full", B) - per_q(comps, "B_noevents", B)
                _, i2 = best_gap(hB, hB)
                f2, _ = best_gap(hA, hA)
                ins_bf.append(i2); oos_bf.append(gap_at(f2, hA, hB) * 100)

            m = lambda x: (float(np.mean(x)), float(np.std(x)))
            (i, isd), (o, osd) = m(ins), m(oos)
            (i2, i2sd), (o2, o2sd) = m(ins_bf), m(oos_bf)
            out[setting] = {"T": T, "golds_per_query_full": golds_full,
                            "ceiling_full_golds": ceil_full,
                            "A_to_B": {"in_sample": i, "in_sd": isd, "out_of_sample": o,
                                       "out_sd": osd, "optimism": i - o,
                                       "inflation_from_halving_golds": i - ceil_full},
                            "B_to_Full": {"in_sample": i2, "in_sd": i2sd, "out_of_sample": o2,
                                          "out_sd": o2sd, "optimism": i2 - o2}}
            print(f"\n### MultiVENT {setting} (T={T}, {a.seeds} gold splits), graded on held-out half\n")
            print(f"  A->B escalation oracle")
            print(f"    ceiling on FULL golds ({golds_full:.2f}/q) {ceil_full:+.2f}"
                  f"   <- the number router_findings.md reports")
            print(f"    in-sample, HALF golds ({golds_full/2:.2f}/q) {i:+.2f} +- {isd:.2f}"
                  f"   <- deleting labels RAISED it by {i-ceil_full:.2f}")
            print(f"    out-of-sample                       {o:+.2f} +- {osd:.2f}")
            print(f"    optimism                            {i-o:.2f} nDCG")
            print(f"  B->Full escalation oracle")
            print(f"    in-sample, half golds  {i2:+.2f} +- {i2sd:.2f}")
            print(f"    out-of-sample          {o2:+.2f} +- {o2sd:.2f}")
            print(f"    optimism               {i2-o2:.2f} nDCG")

        print(f"\n  The ceiling RISES when golds are halved:")
        for s in out:
            ab = out[s]["A_to_B"]
            print(f"    {s:6} {out[s]['ceiling_full_golds']:+.2f} (full golds) -> "
                  f"{ab['in_sample']:+.2f} (half golds), i.e. "
                  f"{ab['inflation_from_halving_golds']:+.2f} for strictly LESS information.")
        print(f"  Real recoverable headroom cannot grow when labels are deleted. Capacity to fit")
        print(f"  the labels can. The ceiling is measuring the latter.")
        print(f"\n  Its ordering also fails to transfer: escalating the queries chosen on one gold")
        print(f"  half LOSES to escalating none of them when graded on the other half.")
        print(f"  Do NOT read that as 'achievable gain is negative' -- the out-of-sample oracle is a")
        print(f"  per-query estimator over ~4 golds and is variance-dominated. The nested-CV router")
        print(f"  pools across queries and achieves +0.73 / +1.68, permutation-significant.")

        of = f"{_ROOT}/results/ablations/router_oracle_goldsplit.json"
        json.dump({"seeds": a.seeds, "cells": out,
                   "msrvtt": "not testable: 1.01 golds/query, no second half to grade on",
                   "note": "nested-CV achieved gaps are out-of-fold and unaffected; only the "
                           "oracle denominator (the 'captured %' column) moves"},
                  open(of, "w"), indent=2)
        print(f"\nwrote {of}")
        tr.summary({f"{s}_AtoB_optimism": out[s]["A_to_B"]["optimism"] for s in out})


if __name__ == "__main__":
    main()
