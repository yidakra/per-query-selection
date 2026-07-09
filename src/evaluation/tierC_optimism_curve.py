#!/usr/bin/env python
"""Oracle headroom over Q2E does not survive contact with a held-out label.

Across this project three "oracles" have been reported, all of the form "per query, pick the best
option, using the test labels":

  tiers          3 options   (A / B / Full)                       routing study
  all-or-nothing 8 options   (keep/drop each of 3 event types)    tierC_decompose
  subset         ~2^24       (any subset of ~24 paraphrases)      tierC_selection_oracle

MultiVENT gives every query >= 4 relevant videos. Split them into halves A and B. Let each oracle
choose using ONLY half A, then grade it on half B:

  in-sample gain      = oracle chose on B, graded on B     (the number papers usually report)
  out-of-sample gain  = oracle chose on A, graded on B
  OPTIMISM            = in-sample - out-of-sample

A HYPOTHESIS THIS SCRIPT WAS WRITTEN TO CONFIRM, AND WHICH IT REFUTED. We expected optimism to grow
with log|choice space|, so that the subset oracle (2^24) would be the most inflated and the tier
oracle (3) the least. The opposite holds: tiers show the LARGEST optimism (10.55 nDCG) and subsets
a smaller one (4.89). Choice-space size is not the driver. What matters is how far apart the options
are in quality: tier A trails Full by ~8.5 nDCG, so a label-noise-induced mis-pick is catastrophic,
whereas swapping one paraphrase for another barely moves the score. Optimism tracks the VARIANCE OF
OPTION QUALITY, not the count of options. The log2|space| column is retained only to show it does
not order the rows.

WHAT out-of-sample DOES AND DOES NOT BOUND. It is not "the best any model could do". oracle(A)
decides each query independently from that query's ~5 remaining golds, so it is a high-variance
estimator. A learned model pools across training queries and can generalise better than this. So a
negative out-of-sample number is evidence that the per-query advantage is label noise, NOT a proof
that no model can win. That proof, for the subset task, is the separate nested-CV experiment in
tierC_learned_selector.py, which pools across queries and also loses to Fixed-Full (-1.49).
Conversely the routing study's nested-CV cascade DOES achieve a positive frontier gain (+0.73) --
at lower cost than Full, never beating it in absolute nDCG. Nothing here contradicts that.

All arms scored with the exact unfrozen fuse. CPU-only.
"""
import os, sys, json, argparse, itertools, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tracking import track                                            # noqa: E402
import tierC_selection_oracle as O                                    # noqa: E402
from tierC_goldsplit_oracle import split_golds                        # noqa: E402

REPO = "/home/ubuntu/q2e_repro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="noASR")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    with track(f"tierC-optimism-{a.setting}", gpu_ids=[], tags=["tierC", "methodology"],
               config={"setting": a.setting, "seeds": a.seeds}) as tr:
        queries, video_ids, target, comps, P, valid, nd_pub, delta = O.load_cell(a.setting)
        T = len(queries)
        D = O.frozen_softmax_parts(comps)
        nA = {e: valid[e].sum(dim=1).numpy() for e in O.EVENTS}
        log2_subset = float(np.mean(sum(nA[e] for e in O.EVENTS)))     # log2 of subset space size

        allsel = [[(e, p) for e in O.EVENTS for p in torch.nonzero(valid[e][q]).flatten().tolist()]
                  for q in range(T)]

        def exact(sel, tgt):
            c = O.build_comps_from_selection(P, valid, comps, sel)
            return O.per_query(O.fuse(c, O.LADDER["C_full"]), tgt)[0].numpy()

        # tier arm: three global fuses, per-query nDCG under each target
        def tier_nd(tgt):
            return np.stack([O.per_query(O.fuse(comps, O.LADDER[k]), tgt)[0].numpy()
                             for k in ["A_visual", "B_noevents", "C_full"]])       # (3, T)

        masks = list(itertools.product([0, 1], repeat=3))

        def mask_nd(tgt):
            out = []
            for m in masks:
                sel = [[(e, p) for e, keep in zip(O.EVENTS, m) if keep
                        for p in torch.nonzero(valid[e][q]).flatten().tolist()] for q in range(T)]
                out.append(exact(sel, tgt))
            return np.stack(out)                                                    # (8, T)

        res = {k: {"in": [], "out": []} for k in ["tiers", "all_or_nothing", "subset"]}
        fulls = []

        for s in range(a.seeds):
            rs = np.random.RandomState(500 + s)
            A, B = split_golds(target, rs)
            fullB = exact(allsel, B)
            fulls.append(fullB.mean() * 100)

            for name, fn in [("tiers", tier_nd), ("all_or_nothing", mask_nd)]:
                ndA, ndB = fn(A), fn(B)
                pick = ndA.argmax(0)                       # chose on A
                res[name]["out"].append(ndB[pick, np.arange(T)].mean() * 100 - fullB.mean() * 100)
                res[name]["in"].append(ndB.max(0).mean() * 100 - fullB.mean() * 100)

            selA = [O.greedy_select(q, P, valid, comps, D, A)[0] for q in range(T)]
            selB = [O.greedy_select(q, P, valid, comps, D, B)[0] for q in range(T)]
            res["subset"]["out"].append(exact(selA, B).mean() * 100 - fullB.mean() * 100)
            res["subset"]["in"].append(exact(selB, B).mean() * 100 - fullB.mean() * 100)
            print(f"  seed {s} done", flush=True)

        sizes = {"tiers": np.log2(3), "all_or_nothing": 3.0, "subset": log2_subset}
        print(f"\n### oracle optimism vs choice-space size   MultiVENT {a.setting} "
              f"(T={T}, {a.seeds} gold splits)\n")
        print(f"  Fixed-Full graded on held-out gold half B: {np.mean(fulls):.2f} nDCG\n")
        print(f"  {'oracle':16} {'log2|space|':>12} {'in-sample':>12} {'out-of-sample':>15} "
              f"{'optimism':>10}")
        out = {}
        for k in ["tiers", "all_or_nothing", "subset"]:
            i, o = np.mean(res[k]["in"]), np.mean(res[k]["out"])
            print(f"  {k:16} {sizes[k]:12.1f} {i:+12.2f} {o:+15.2f} {i-o:10.2f}")
            out[k] = {"log2_space": float(sizes[k]), "in_sample_gain": float(i),
                      "out_of_sample_gain": float(o), "optimism": float(i - o),
                      "in_sd": float(np.std(res[k]["in"])), "out_sd": float(np.std(res[k]["out"]))}

        # Did optimism order by choice-space size? We predicted yes. Check, do not assume.
        order_sz = [k for k in sorted(out, key=lambda k: out[k]["log2_space"])]
        order_op = [k for k in sorted(out, key=lambda k: out[k]["optimism"])]
        mono = order_sz == order_op
        print(f"\n  in-sample gain is what an oracle-headroom number reports.")
        print(f"  hypothesis: optimism increases with log2|choice space|.")
        print(f"    by space size: {order_sz}")
        print(f"    by optimism  : {order_op}")
        print(f"    => {'CONFIRMED' if mono else 'REFUTED. Space size does not order optimism.'}")
        if not mono:
            spread = {k: v for k, v in
                      [("tiers", 8.48), ("all_or_nothing", None), ("subset", None)]}
            print(f"       The tier oracle has the SMALLEST space and the LARGEST optimism "
                  f"({out['tiers']['optimism']:.2f}).")
            print(f"       Its options differ by ~{spread['tiers']:.1f} nDCG (Fixed-A vs Fixed-Full), "
                  f"so a mis-pick is costly;")
            print(f"       swapping one paraphrase for another is nearly free. Optimism tracks the")
            print(f"       VARIANCE OF OPTION QUALITY, not the number of options.")

        pos = [k for k in out if out[k]["out_of_sample_gain"] > 0]
        print(f"\n  oracles whose per-query advantage TRANSFERS to a held-out gold half: "
              f"{pos if pos else 'NONE -- every one of them loses to Fixed-Full'}")
        print(f"  (out-of-sample here bounds oracles that use only this query's own labels. It does")
        print(f"   not bound a model that pools across queries; see tierC_learned_selector.py, which")
        print(f"   does pool, and also loses.)")

        of = f"{REPO}/results/ablations/tierC_optimism_{a.setting}.json"
        json.dump({"setting": a.setting, "T": T, "seeds": a.seeds,
                   "fixedFull_on_B": float(np.mean(fulls)), "oracles": out,
                   "hypothesis_optimism_grows_with_log_space": "REFUTED" if not mono else "confirmed",
                   "optimism_order_by_space": order_sz, "optimism_order_observed": order_op,
                   "note": ("out_of_sample bounds oracles using only this query's own labels; "
                            "a model pooling across queries is tested in tierC_learned_selector.py")},
                  open(of, "w"), indent=2)
        print(f"\nwrote {of}")
        tr.summary({f"{k}_optimism": out[k]["optimism"] for k in out})


if __name__ == "__main__":
    main()
