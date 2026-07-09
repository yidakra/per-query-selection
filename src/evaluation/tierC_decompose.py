#!/usr/bin/env python
"""Decompose the tier-C subset oracle into ROUTING gain and SELECTION gain.

The subset oracle keeps ~0.9 of 24 available paraphrases and keeps NONE for 45% of queries.
"Keep none for all three event types" is exactly Fixed-B for that query. So the subset oracle is
silently doing two different jobs at once:

  ROUTING   -- per event type, keep ALL paraphrases or NONE. This is per-query tier selection
               (B vs Full, at event-type granularity). We already know from the routing study that
               oracle routing gains are largely unachievable by a learned router.
  SELECTION -- choose WHICH paraphrases to keep. This is the genuinely new tier-C capability, and
               the one a learned selector could plausibly capture, since it needs no LLM call.

If the all-or-nothing oracle already captures most of the subset oracle's gain, tier C is just
routing under another name and adds nothing. If subset >> all-or-nothing, selection is real.

Both arms are scored with the EXACT unfrozen fuse (build_comps_from_selection + fuse), the same
metric Q2E reports, so the numbers are directly comparable to Fixed-Full.

NOTE ON WHAT AN ORACLE IS. Both oracles here choose using the test labels. They upper-bound any
selector, including one that cheats. They are NOT achievable performance. The routing study showed
an oracle ceiling of +5.04 collapsing to +0.73 once a real router had to predict the choice under
nested CV. Expect the same collapse here; that experiment is the next one, not this one.

CPU-only.
"""
import os, sys, json, argparse, itertools, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tracking import track                                            # noqa: E402
import tierC_selection_oracle as O                                    # noqa: E402

REPO = "/home/ubuntu/q2e_repro"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="noASR")
    a = ap.parse_args()

    with track(f"tierC-decompose-{a.setting}", gpu_ids=[], tags=["tierC"],
               config={"setting": a.setting}) as tr:
        queries, video_ids, target, comps, P, valid, nd_pub, delta = O.load_cell(a.setting)
        T = len(queries)
        D = O.frozen_softmax_parts(comps)

        ndB = O.per_query(O.fuse(comps, O.LADDER["B_noevents"]), target)[0]
        ndF = O.per_query(O.fuse(comps, O.LADDER["C_full"]), target)[0]

        base = ["query_vs_video", "query_vs_captions"]
        zero = torch.zeros_like(comps["query_vs_captions"][0])

        # ---- all-or-nothing oracle: per query, pick the best of 2^3 keep/drop masks ----
        masks = list(itertools.product([0, 1], repeat=3))
        an_sel, mask_hist = [], {m: 0 for m in masks}
        for q in range(T):
            rel = target[q].numpy().astype(np.float64)
            rows0 = {p: comps[p][q] for p in base}
            best, bm = -1.0, None
            for m in masks:
                rows = dict(rows0)
                for e, keep in zip(O.EVENTS, m):
                    rows[e] = comps[e][q] if keep else zero
                s = O.ndcg10(O.fused_row(rows, D).numpy(), rel)
                if s > best:
                    best, bm = s, m
            mask_hist[bm] += 1
            # express the mask as an explicit paraphrase selection
            sel_q = []
            for e, keep in zip(O.EVENTS, bm):
                if keep:
                    sel_q += [(e, p) for p in torch.nonzero(valid[e][q]).flatten().tolist()]
            an_sel.append(sel_q)

        an_comps = O.build_comps_from_selection(P, valid, comps, an_sel)
        ndAN = O.per_query(O.fuse(an_comps, O.LADDER["C_full"]), target)[0]

        # ---- subset oracle, as computed by tierC_selection_oracle ----
        sub = json.load(open(f"{REPO}/results/ablations/tierC_selection_oracle_{a.setting}.json"))
        sub_sel = [[tuple(x) for x in s] for s in sub["selection"]]
        sub_comps = O.build_comps_from_selection(P, valid, comps, sub_sel)
        ndSUB = O.per_query(O.fuse(sub_comps, O.LADDER["C_full"]), target)[0]

        f = lambda x: float(x.mean()) * 100
        gB, gF, gAN, gSUB = f(ndB), f(ndF), f(ndAN), f(ndSUB)
        route_gain = gAN - gF
        sel_gain = gSUB - gAN

        print(f"\n### tier-C oracle decomposition  MultiVENT {a.setting}  (T={T})\n")
        print(f"  Fixed-B                                    {gB:.2f}")
        print(f"  Fixed-Full (all paraphrases)               {gF:.2f}")
        print(f"  ORACLE all-or-nothing (routing only)       {gAN:.2f}   [vs Full {route_gain:+.2f}]")
        print(f"  ORACLE subset        (routing + selection) {gSUB:.2f}   [vs Full {gSUB-gF:+.2f}]")
        print(f"\n  gain decomposition vs Fixed-Full:")
        print(f"    routing   (keep-all / keep-none)  {route_gain:+.2f}")
        print(f"    selection (which paraphrases)     {sel_gain:+.2f}")
        tot = gSUB - gF
        if abs(tot) > 1e-9:
            print(f"    selection is {100*sel_gain/tot:.0f}% of the total oracle gain")

        print(f"\n  best keep/drop mask per query (prequel, during, sequel):")
        for m, c in sorted(mask_hist.items(), key=lambda kv: -kv[1]):
            lab = ",".join(e.split("_")[0] for e, k in zip(O.EVENTS, m) if k) or "NONE (= Fixed-B)"
            print(f"    {m}  {c:4d} ({100*c/T:4.1f}%)  keep: {lab}")

        out = {"setting": a.setting, "T": T, "fixedB": gB, "fixedFull": gF,
               "oracle_all_or_nothing": gAN, "oracle_subset": gSUB,
               "routing_gain": route_gain, "selection_gain": sel_gain,
               "selection_share_pct": float(100 * sel_gain / tot) if abs(tot) > 1e-9 else None,
               "mask_histogram": {str(k): v for k, v in mask_hist.items()},
               "caveat": "both arms are label-fitted oracles; upper bounds, not achievable"}
        of = f"{REPO}/results/ablations/tierC_decompose_{a.setting}.json"
        json.dump(out, open(of, "w"), indent=2)
        print(f"\nwrote {of}")
        tr.summary(out)


if __name__ == "__main__":
    main()
