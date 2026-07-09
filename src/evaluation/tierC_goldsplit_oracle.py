#!/usr/bin/env python
"""Is the tier-C subset oracle real signal, or is it fitting the labels?

The subset oracle gains +2.00 nDCG over Fixed-Full. A learned selector under nested CV gets
-1.49 -- below Fixed-Full, and statistically indistinguishable from pruning at random. Two
readings of that:

  H2a  A genuinely good subset exists for each query; our features simply cannot identify it.
  H2b  No such subset exists. Greedy searches 2^24 subsets while maximising nDCG@10 against the
       SAME relevance judgments it is scored on, so it is free to fit label noise. The oracle is
       an artifact and there is nothing for any selector to learn.

These are distinguishable, because MultiVENT is multi-gold: every query here has >= 4 relevant
videos (median 10). Split each query's golds into halves A and B. Run the identical greedy oracle
using ONLY half A as the relevance signal, then score the resulting subset on half B.

  If a subset is genuinely better at retrieving this query's videos, it was not told which half
  it would be graded on, so its advantage must transfer:  oracle(A) on B  >  Fixed-Full on B.
  If greedy is merely memorising which specific videos are marked relevant, the advantage evaporates
  the moment the grading videos change:  oracle(A) on B  ~=  Fixed-Full on B, or worse.

oracle(B) scored on B is reported too, as the in-sample number. The gap between it and oracle(A)
on B IS the optimism of the oracle -- the quantity the +2.00 headline silently contains.

Averaged over SEEDS random gold splits. Scored with the exact unfrozen fuse throughout.
CPU-only.
"""
import os, sys, json, argparse, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tracking import track                                            # noqa: E402
import tierC_selection_oracle as O                                    # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
REPO = _ROOT
SEEDS = 5


def split_golds(target, rs):
    """Per query, split relevant videos into two disjoint halves."""
    T, V = target.shape
    A = torch.zeros_like(target)
    B = torch.zeros_like(target)
    for q in range(T):
        g = torch.nonzero(target[q]).flatten().numpy()
        rs.shuffle(g)
        h = len(g) // 2
        A[q, g[:h]] = True
        B[q, g[h:]] = True
    return A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="noASR")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()

    with track(f"tierC-goldsplit-{a.setting}", gpu_ids=[], tags=["tierC", "diagnostic"],
               config={"setting": a.setting, "seeds": a.seeds}) as tr:
        queries, video_ids, target, comps, P, valid, nd_pub, delta = O.load_cell(a.setting)
        T = len(queries)
        D = O.frozen_softmax_parts(comps)

        allsel = [[(e, p) for e in O.EVENTS for p in torch.nonzero(valid[e][q]).flatten().tolist()]
                  for q in range(T)]

        def exact(sel, tgt):
            c = O.build_comps_from_selection(P, valid, comps, sel)
            return O.per_query(O.fuse(c, O.LADDER["C_full"]), tgt)[0]

        rows = []
        for s in range(a.seeds):
            rs = np.random.RandomState(500 + s)
            A, B = split_golds(target, rs)

            selA = [O.greedy_select(q, P, valid, comps, D, A)[0] for q in range(T)]
            selB = [O.greedy_select(q, P, valid, comps, D, B)[0] for q in range(T)]

            fullB = exact(allsel, B).mean().item() * 100
            bB = exact([[] for _ in range(T)], B).mean().item() * 100
            oAB = exact(selA, B).mean().item() * 100      # out-of-sample: selected on A, graded on B
            oBB = exact(selB, B).mean().item() * 100      # in-sample:     selected on B, graded on B
            rows.append((bB, fullB, oAB, oBB))
            print(f"  seed {s}: Fixed-B {bB:.2f} | Fixed-Full {fullB:.2f} | "
                  f"oracle(A) on B {oAB:.2f} | oracle(B) on B {oBB:.2f}", flush=True)

        r = np.array(rows)
        bB, fullB, oAB, oBB = r.mean(0)
        sd = r.std(0)

        print(f"\n### gold-split oracle   MultiVENT {a.setting}  (T={T}, {a.seeds} splits)\n")
        print(f"  graded on gold half B throughout:")
        print(f"    Fixed-B                        {bB:.2f} +- {sd[0]:.2f}")
        print(f"    Fixed-Full (all paraphrases)   {fullB:.2f} +- {sd[1]:.2f}")
        print(f"    oracle(A) on B  [out-of-sample] {oAB:.2f} +- {sd[2]:.2f}"
              f"   [vs Full {oAB-fullB:+.2f}]")
        print(f"    oracle(B) on B  [in-sample]     {oBB:.2f} +- {sd[3]:.2f}"
              f"   [vs Full {oBB-fullB:+.2f}]")
        print(f"\n  oracle optimism (in-sample minus out-of-sample): {oBB - oAB:+.2f} nDCG")
        verdict = ("TRANSFERS: a genuinely better subset exists (H2a). The learned selector's "
                   "failure is a feature problem." if oAB - fullB > 0.5 else
                   "DOES NOT TRANSFER: the oracle is fitting label noise (H2b). There is no "
                   "recoverable subset, and no selector can win.")
        print(f"  => {verdict}")

        out = {"setting": a.setting, "T": T, "seeds": a.seeds,
               "graded_on": "held-out gold half B",
               "fixedB": bB, "fixedFull": fullB,
               "oracle_A_on_B": oAB, "oracle_B_on_B": oBB,
               "sd": {"fixedB": sd[0], "fixedFull": sd[1], "oracle_A_on_B": sd[2],
                      "oracle_B_on_B": sd[3]},
               "transfer_gain_vs_full": float(oAB - fullB),
               "oracle_optimism": float(oBB - oAB),
               "verdict": verdict}
        out = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in out.items()}
        out["sd"] = {k: float(v) for k, v in out["sd"].items()}
        of = f"{REPO}/results/ablations/tierC_goldsplit_{a.setting}.json"
        json.dump(out, open(of, "w"), indent=2)
        print(f"\nwrote {of}")
        tr.summary({k: v for k, v in out.items() if isinstance(v, (int, float))})


if __name__ == "__main__":
    main()
