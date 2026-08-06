"""Where in the cheap ranking does the expensive channel do its work?

A channel can only improve nDCG@10 by pulling a relevant document into the top ten that the cheap
channel did not already have there. Call those documents the cell's *carriers*. This measures two things
about them: how many there are, and how far down the visual ranking they were sitting.

The answer to the second is narrow enough to be worth stating as a property of multimodal fusion rather
than of this dataset. Carriers sit just below the cut -- essentially all of them within the visual top
fifty, none past a hundred, none missing from the visual list altogether. The expensive channel does
short-range rescue. It re-orders documents the cheap channel had already found and nearly ranked; it
does not discover documents the cheap channel missed.

That bounds what channel fusion can be expected to do, and it explains the shape of the gains in the RQ2
table. It also disposes of an attractive but wrong intuition about extraction budgets, that the way to
spend less is to extract only for documents the cheap channel ranks poorly, on the grounds that those
are the ones needing help. They are not the ones needing help. They are past rescuing.

The count of documents fusion DROPS out of the top ten is reported beside the count it adds, because in
two of the three cells it drops more than it adds, and that is where the negative mean gains come from.

CPU only, but it loads two full run files per cell and holds a rank index over the visual run, so it
wants a few GB.

  python src/multivent2/mv2_rescue_depth.py
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run              # noqa: E402
from mv2_channels import rank_map                    # noqa: E402
from mv2_recall_sidecar import CELLS, VISUAL         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

BUCKETS = ((1, 10), (11, 50), (51, 100), (101, 500), (501, 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10, help="rank cutoff the metric is measured at")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_rescue_depth.json"))
    a = ap.parse_args()

    qrels, meta = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    visual = load_run(os.path.join(DATA, VISUAL))
    print(f"visual run: {len(visual)} queries", flush=True)

    vis_rrf, vrank, vtop = {}, {}, {}
    for q, s in visual.items():
        order = sorted(s, key=lambda d: -s[d])
        vis_rrf[q] = rank_map(s)
        vrank[q] = {d: i + 1 for i, d in enumerate(order)}
        vtop[q] = set(order[:a.k])

    out = {}
    for cell, (chan_fn, w, cell_fn) in CELLS.items():
        ref = json.load(open(os.path.join(ABL, cell_fn)))
        qids = [q for q in ref["per_query"] if q in visual]
        chan = load_run(os.path.join(DATA, chan_fn))

        carriers, losers = collections.Counter(), collections.Counter()
        pair_ranks = []
        for q in qids:
            rel = {d for d, r in qrels.get(q, {}).items() if r > 0}
            acc = dict(vis_rrf[q])
            if q in chan:
                for d, c in rank_map(chan[q]).items():
                    acc[d] = acc.get(d, 0.0) + w * c
            topB = {d for d, _ in sorted(acc.items(), key=lambda kv: -kv[1])[:a.k]}
            for d in (topB - vtop[q]) & rel:
                carriers[d] += 1
                pair_ranks.append(vrank[q].get(d, 10_000))
            for d in (vtop[q] - topB) & rel:
                losers[d] += 1

        ndA = np.array([ref["per_query"][q]["ndA"] for q in qids])
        ndB = np.array([ref["per_query"][q]["ndB"] for q in qids])
        gain = ndB - ndA
        pr = np.array(pair_ranks)

        print(f"\n=== {cell}  (fusing {chan_fn} at w={w})")
        print(f"  {len(qids)} queries, mean gain {gain.mean():+.4f}  "
              f"({100*(gain>0).mean():.1f}% helped, {100*(gain<0).mean():.1f}% hurt)")
        print(f"  documents pulled INTO the top-{a.k}: {len(carriers):5d} distinct, "
              f"{len(pr)} (query, document) pairs")
        print(f"  documents pushed OUT of the top-{a.k}: {len(losers):5d} distinct")
        print(f"  visual rank of a carrier, for the query it carried:")
        dist = {}
        for lo, hi in BUCKETS:
            n = int(((pr >= lo) & (pr <= hi)).sum())
            dist[f"{lo}-{hi}"] = n
            print(f"    {lo:>4}-{hi:<5}{n:>7}  ({100*n/max(1,len(pr)):5.1f}%)")
        miss = int((pr >= 10_000).sum())
        dist["absent"] = miss
        print(f"    absent   {miss:>7}  ({100*miss/max(1,len(pr)):5.1f}%)")
        print(f"    median {np.median(pr):.0f}, 90th percentile {np.percentile(pr, 90):.0f}")

        out[cell] = {"n_queries": len(qids), "mean_gain": float(gain.mean()),
                     "n_carriers": len(carriers), "n_losers": len(losers),
                     "n_pairs": len(pr), "rank_buckets": dist,
                     "median_rank": float(np.median(pr)),
                     "p90_rank": float(np.percentile(pr, 90))}
        del chan

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
