"""Materialise the router's decision as an ordinary ranked-list file, so downstream tasks can consume it.

Everywhere else in this repo routing is scored, not written out: `mv2_channel_select.py` reports the
selection gap directly from per-query nDCG. The RAG arm needs the actual ranked lists, because a
generator reads documents, not scores. This writes two runs over the same channels:

  routed_<tag>.json        per query, the ranked list of whichever policy the selector picked for it
  routed_<tag>_picks.json  qid -> the policy picked, so a consumer can also restrict itself to the
                           evidence that policy actually reads (the RAG arm's --evidence own)
  bestfixed_<tag>.json     the best fixed policy applied to every query, i.e. what routing beats. Fixed
                           is chosen per fold on the training split, the same best-fixed-on-train rule
                           as the nested gap, so this baseline peeks at no test label either.

All three are out-of-fold and event-grouped by default: the pick for query q, routed or fixed, comes
from data that never saw q's event, so feeding these to the RAG evaluation cannot smuggle in a label
the nDCG numbers do not already allow. Without --group-cv the split is by query and the routed run is
optimistic; the flag is on by default here for that reason.

  python src/multivent2/mv2_routed_run.py --channel asr=asr_dense_bge-m3.json --tag dense_m3
"""
import os
import sys
import json
import argparse

import numpy as np
from sklearn.model_selection import cross_val_predict, GroupKFold, KFold

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_channels import fuse  # noqa: E402
from mv2_channel_select import load_cell, mk  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", action="append", default=[], metavar="NAME=FILE",
                    help="add or override a channel, as in mv2_channel_select.py")
    ap.add_argument("--tag", default="", help="suffix for the two output run files")
    ap.add_argument("--query-cv", action="store_true",
                    help="split by query instead of by event group. Optimistic -- see module docstring")
    ap.add_argument("--outdir", default=DATA)
    a = ap.parse_args()

    channels, names, policies, qids, X, Y, _ = load_cell(a.channel)
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    print(f"channels: {names}  queries: {len(qids)}  policies: {len(policies)}")

    if a.query_cv:
        splits = list(KFold(5, shuffle=True, random_state=0).split(X))
    else:
        from mv2_qsd import event_groups
        grp = event_groups(qids, qrels)
        print(f"grouped CV: {len(set(grp))} event groups")
        splits = list(GroupKFold(5).split(X, groups=grp))

    sel = np.argmax(cross_val_predict(mk(), X, Y, cv=splits), axis=1)
    # the fixed baseline follows the same rule as the nested gap: chosen on each fold's training
    # split, applied to its held-out queries, so neither side of the comparison reads a test label
    fixed_sel = np.zeros(len(qids), dtype=int)
    for tr, te in splits:
        fixed_sel[te] = int(np.argmax(Y[tr].mean(axis=0)))
    fixed_names = sorted({policies[j] for j in set(fixed_sel)})

    # every policy's fused list, once; the routed run then just picks a row from each
    runs = {n: load_run(os.path.join(DATA, fn)) for n, fn in channels.items()}
    fused = {}
    for pol in policies:
        w = {n: (1.0 if n in pol.split("+") else 0.0) for n in names}
        fused[pol] = fuse(runs, w, qids)

    routed = {q: fused[policies[sel[i]]][q] for i, q in enumerate(qids)}
    bestfixed = {q: fused[policies[fixed_sel[i]]][q] for i, q in enumerate(qids)}

    # sanity: the written run must score what the selector said it would
    nd_routed = float(np.mean(list(per_query_ndcg(qrels, routed).values())))
    nd_fixed = float(np.mean(list(per_query_ndcg(qrels, bestfixed).values())))
    print(f"routed run     {nd_routed:.5f}   (selector's out-of-fold selection)")
    print(f"best fixed     {nd_fixed:.5f}   (on-train picks: {', '.join(fixed_names)})")
    print(f"gap            {100 * (nd_routed - nd_fixed):+.2f} nDCG")

    tag = f"_{a.tag}" if a.tag else ""
    for name, run in (("routed", routed), ("bestfixed", bestfixed)):
        # trim to a sane depth: RRF over three channels unions thousands of candidates per query and
        # nothing downstream reads past the top of the list
        trimmed = {q: dict(sorted(s.items(), key=lambda kv: -kv[1])[:100]) for q, s in run.items()}
        p = os.path.join(a.outdir, f"{name}{tag}.json")
        json.dump(trimmed, open(p, "w"))
        print(f"wrote {p}")

    picks = {q: policies[sel[i]] for i, q in enumerate(qids)}
    p = os.path.join(a.outdir, f"routed{tag}_picks.json")
    json.dump(picks, open(p, "w"))
    print(f"wrote {p}")

    hist = {policies[j]: int((sel == j).sum()) for j in range(len(policies))}
    print("picks: " + "  ".join(f"{k}={v}" for k, v in sorted(hist.items(), key=lambda x: -x[1]) if v))


if __name__ == "__main__":
    main()
