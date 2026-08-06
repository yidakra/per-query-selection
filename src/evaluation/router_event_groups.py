#!/usr/bin/env python
"""Does the MultiVENT 2.0 duplicate-topic leakage apply to the original six router cells?

On MultiVENT 2.0 several queries phrase the same event and share a relevant document, so a query-level
CV split puts near-duplicates on both sides of a fold boundary. Any predictor that learns from other
queries' labels then reads its answer off a duplicate (`src/multivent2/mv2_qsd.py`). Everything in that
benchmark is therefore evaluated on 536 event groups instead of 2,546 queries.

The heterogeneity law of `router_findings.md` section 3 (Spearman rho = +0.943 between sd(per-query
gain) and the achieved nested gap) was measured on six cells drawn from MultiVENT v1 and MSR-VTT, under
a plain KFold split, so the same question has to be asked of it.

This script asks it directly: link two queries when they share any relevant video and count the
connected components. If every query is its own component, there are no duplicates to leak and grouped
CV is identical to KFold.

  python src/evaluation/router_event_groups.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # before torch: GPU0 runs an unrelated service

import numpy as np                                # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_hetero import comps_multivent, comps_msrvtt  # noqa: E402

CELLS = [("MultiVENT/mCLIP/noASR", lambda: comps_multivent("noASR")),
         ("MultiVENT/mCLIP/ASR", lambda: comps_multivent("ASR")),
         ("MSR-VTT/mCLIP/noASR", lambda: comps_msrvtt("multiclip", "noASR")),
         ("MSR-VTT/mCLIP/ASR", lambda: comps_msrvtt("multiclip", "ASR")),
         ("MSR-VTT/IV2/noASR", lambda: comps_msrvtt("internvideo2", "noASR")),
         ("MSR-VTT/IV2/ASR", lambda: comps_msrvtt("internvideo2", "ASR"))]


def event_groups(target):
    """Connected components of the queries-share-a-relevant-video graph. Same construction as
    mv2_qsd.event_groups, over a dense relevance matrix instead of a qrels dict."""
    T = target.numpy() if hasattr(target, "numpy") else np.asarray(target)
    n = T.shape[0]
    doc2q = {}
    for i in range(n):
        for d in np.nonzero(T[i] > 0)[0]:
            doc2q.setdefault(int(d), []).append(i)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for qs in doc2q.values():
        for other in qs[1:]:
            a, b = find(qs[0]), find(other)
            if a != b:
                parent[b] = a
    return len({find(i) for i in range(n)}), n


def main():
    print(f"  {'cell':24} {'queries':>8} {'gold/q':>7} {'event groups':>13} {'grouped CV differs?':>20}")
    any_dup = False
    for name, fn in CELLS:
        _, target, _ = fn()
        ng, n = event_groups(target)
        gpq = float((target.numpy() if hasattr(target, "numpy") else target).sum(1).mean())
        differs = ng < n
        any_dup |= differs
        print(f"  {name:24} {n:8d} {gpq:7.2f} {ng:13d} {'yes' if differs else 'no':>20}")
    print()
    if any_dup:
        print("At least one cell has queries sharing a relevant video: rerun those under GroupKFold.")
    else:
        print("Every query is its own event group in every cell, so GroupKFold reduces to KFold and the\n"
              "section 3 numbers stand as measured. The leakage is specific to MultiVENT 2.0, which is\n"
              "the only corpus here carrying several phrasings of one event.")


if __name__ == "__main__":
    main()
