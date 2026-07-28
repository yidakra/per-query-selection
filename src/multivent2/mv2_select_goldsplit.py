"""Gold-split audit of the K-way channel-selection oracle (the section-5 test, applied to
mv2_channel_select.py's numbers before anyone leans on them).

The selection cell reports an oracle -- per-query argmax of nDCG over the policy set (0.4669 shipped,
0.5181 dense-m3). That argmax is taken on the same relevance judgments used to grade it, exactly the
in-sample pattern router_oracle_goldsplit.py showed can manufacture several points of optimism. The
K-way version has 7 options per query, and the tierC study's refuted-hypothesis note says optimism
tracks the spread in option quality, not the option count -- channel nDCGs spread wide here, so the
concern is live.

PROTOCOL, per seed: split each query's rel>0 docs into halves A and B (queries with a single gold are
excluded and counted). In-sample: pick each query's policy by nDCG on B, grade on B. Out-of-sample:
pick on A, grade on B. Optimism = in - out. The fixed-policy baseline is chosen on A and graded on B,
so the comparison stays legal. The learned selector's own numbers are NOT audited here -- its
predictions are out-of-fold ridge over label-free features, so the gold-split leak has no path in;
this audit scopes the ORACLE (the "% captured" denominator), as in section 5.

CPU-only.

  python src/multivent2/mv2_select_goldsplit.py --channel asr=asr_dense_bge-m3.json --cell-tag _dense_m3
"""
import os
import sys
import json
import argparse
import itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_channels import CHANNELS, fuse  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")


def split_golds(qrels, rs):
    """Halve each query's rel>0 docs (grades kept). Queries with <2 golds are dropped."""
    A, B = {}, {}
    for q, docs in qrels.items():
        golds = [d for d, r in docs.items() if r > 0]
        if len(golds) < 2:
            continue
        rs.shuffle(golds)
        h = len(golds) // 2
        A[q] = {d: docs[d] for d in golds[:h]}
        B[q] = {d: docs[d] for d in golds[h:]}
    return A, B


def nd_matrix(qrels_half, pol_runs, qids):
    """[n_queries, n_policies] per-query nDCG@10 for one gold half; absent query -> 0."""
    out = np.zeros((len(qids), len(pol_runs)))
    for j, (_, run) in enumerate(pol_runs):
        pq = per_query_ndcg(qrels_half, run)
        out[:, j] = [pq.get(q, 0.0) for q in qids]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--cell-tag", default="")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    channels = dict(CHANNELS)
    for spec in a.channel:
        name, _, fn = spec.partition("=")
        channels[name] = fn

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    runs = {n: load_run(os.path.join(DATA, fn)) for n, fn in channels.items()}
    all_q = sorted(set.intersection(set(qrels), *[set(r) for r in runs.values()]))
    names = sorted(runs)
    pol_runs = []
    for k in range(1, len(names) + 1):
        for c in itertools.combinations(names, k):
            w = {n: (1.0 if n in c else 0.0) for n in names}
            pol_runs.append(("+".join(c), fuse(runs, w, all_q)))
    policies = [p for p, _ in pol_runs]

    # reference: the oracle exactly as the selection cell reports it, full golds
    nd_full = nd_matrix(qrels, pol_runs, all_q)
    keep = nd_full.sum(axis=1) >= 0  # all queries; filtering happens per split below
    oracle_full = float(nd_full.max(axis=1).mean())
    fixed_full = float(nd_full.mean(axis=0).max())
    golds = np.array([sum(r > 0 for r in qrels[q].values()) for q in all_q])
    splittable = int((golds >= 2).sum())
    print(f"queries {len(all_q)}, splittable (>=2 golds) {splittable}, "
          f"golds/query mean {golds.mean():.2f}")
    print(f"full-gold oracle {oracle_full:.5f}  best fixed {fixed_full:.5f}  "
          f"(the numbers mv2_channel_select{a.cell_tag}.json reports)")

    ins, oos, fixed_b = [], [], []
    for s in range(a.seeds):
        rs = np.random.RandomState(500 + s)
        qA, qB = split_golds(qrels, rs)
        qids = sorted(set(qA) & set(all_q))
        ndA = nd_matrix(qA, pol_runs, qids)
        ndB = nd_matrix(qB, pol_runs, qids)
        ins.append(ndB[np.arange(len(qids)), ndB.argmax(axis=1)].mean())
        oos.append(ndB[np.arange(len(qids)), ndA.argmax(axis=1)].mean())
        fixed_b.append(ndB[:, ndA.mean(axis=0).argmax()].mean())

    m = lambda x: (float(np.mean(x)), float(np.std(x)))
    (i, isd), (o, osd), (fb, fbsd) = m(ins), m(oos), m(fixed_b)
    print(f"\nper-seed means over {a.seeds} splits, graded on held-out half B:")
    print(f"  in-sample oracle (pick on B, grade on B)   {i:.5f} +- {isd:.5f}")
    print(f"  out-of-sample oracle (pick on A, grade B)  {o:.5f} +- {osd:.5f}")
    print(f"  best fixed policy (chosen on A)            {fb:.5f} +- {fbsd:.5f}")
    print(f"  OPTIMISM (in - out)                        {100*(i-o):+.2f} nDCG")
    print(f"  out-of-sample oracle vs fixed              {100*(o-fb):+.2f} nDCG")

    out_path = os.path.join(ABL, f"mv2_select_goldsplit{a.cell_tag}.json")
    json.dump({"channels": channels, "policies": policies, "seeds": a.seeds,
               "n_queries": len(all_q), "splittable": splittable,
               "oracle_full": oracle_full, "fixed_full": fixed_full,
               "in_sample": i, "in_sd": isd, "out_of_sample": o, "out_sd": osd,
               "fixed_on_A": fb, "fixed_sd": fbsd,
               "optimism": 100 * (i - o), "oos_vs_fixed": 100 * (o - fb)},
              open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
