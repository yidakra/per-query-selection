"""Gold-split audit of a selection oracle on any axis.

The oracle numbers quoted for the language and query-variant axes are argmaxes taken on the same
relevance judgments that then grade them. `mv2_select_goldsplit.py` measured what that costs on the
channel axis (it removed 15 of the oracle's 16 points). This script runs the identical protocol on
an arbitrary option set, so the query-side axes stop being quoted with a caveat and no number.

PROTOCOL, per seed: split each query's rel>0 docs into halves A and B (queries with a single gold
are dropped and counted). In-sample: pick each query's option by nDCG@10 on B and grade on B.
Out-of-sample: pick on A, grade on B. Optimism is in minus out. The baselines are chosen on A and
graded on B, so every comparison is legal. A named default option is graded on B as well, because
the axis tables report gains against a default rather than against the best fixed option.

Learned selectors are not audited here. Their picks come from out-of-fold predictions over
label-free features, so the gold-split leak has no path in; this audit scopes the ORACLE.

CPU-only.

  python src/multivent2/mv2_axis_goldsplit.py --tag language --default en \
      --run en=asr_dense_bge-m3.json --run zh=asr_dense_bge-m3_qzh.json
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_select_goldsplit import split_golds, nd_matrix  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")


def audit(qrels, pol_runs, qids, default, seeds=5):
    """The protocol, shared by every axis. `pol_runs` is a list of (label, run) in a fixed order and
    `default` names the option the axis reports gains against. Returns the record written to JSON."""
    labels = [label for label, _ in pol_runs]
    d = labels.index(default)

    nd_full = nd_matrix(qrels, pol_runs, qids)
    oracle_full = float(nd_full.max(axis=1).mean())
    default_full = float(nd_full[:, d].mean())
    golds = np.array([sum(r > 0 for r in qrels[q].values()) for q in qids])
    print(f"{len(labels)} options, {len(qids)} queries, "
          f"splittable (>=2 golds) {int((golds >= 2).sum())}, golds/query mean {golds.mean():.2f}")
    print(f"full-gold oracle {oracle_full:.5f} vs default {default} {default_full:.5f} "
          f"({100 * (oracle_full - default_full):+.2f} nDCG, the number the axis table reports)")

    ins, oos, fixed_b, deflt_b = [], [], [], []
    for s in range(seeds):
        rs = np.random.RandomState(500 + s)
        qA, qB = split_golds(qrels, rs)
        sub = sorted(set(qA) & set(qids))
        ndA = nd_matrix(qA, pol_runs, sub)
        ndB = nd_matrix(qB, pol_runs, sub)
        rows = np.arange(len(sub))
        ins.append(ndB[rows, ndB.argmax(axis=1)].mean())
        oos.append(ndB[rows, ndA.argmax(axis=1)].mean())
        fixed_b.append(ndB[:, ndA.mean(axis=0).argmax()].mean())
        deflt_b.append(ndB[:, d].mean())

    m = lambda x: (float(np.mean(x)), float(np.std(x)))          # noqa: E731
    (i, isd), (o, osd), (fb, fbsd), (db, dbsd) = m(ins), m(oos), m(fixed_b), m(deflt_b)
    print(f"\nper-seed means over {seeds} splits, graded on the held-out half B:")
    print(f"  in-sample oracle (pick on B, grade on B)   {i:.5f} +- {isd:.5f}")
    print(f"  out-of-sample oracle (pick on A, grade B)  {o:.5f} +- {osd:.5f}")
    print(f"  best fixed option (chosen on A)            {fb:.5f} +- {fbsd:.5f}")
    print(f"  default option {default:<12}              {db:.5f} +- {dbsd:.5f}")
    print(f"  OPTIMISM (in - out)                        {100 * (i - o):+.2f} nDCG")
    print(f"  honest oracle vs default                   {100 * (o - db):+.2f} nDCG")
    print(f"  honest oracle vs best fixed                {100 * (o - fb):+.2f} nDCG")

    return {"default": default, "seeds": seeds, "n_queries": len(qids),
            "splittable": int((golds >= 2).sum()), "n_options": len(labels),
            "oracle_full": oracle_full, "default_full": default_full,
            "oracle_full_vs_default": 100 * (oracle_full - default_full),
            "in_sample": i, "in_sd": isd, "out_of_sample": o, "out_sd": osd,
            "fixed_on_A": fb, "fixed_sd": fbsd, "default_on_B": db, "default_sd": dbsd,
            "optimism": 100 * (i - o), "honest_vs_default": 100 * (o - db),
            "honest_vs_fixed": 100 * (o - fb)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=FILE",
                    help="one option of the axis; repeat once per option")
    ap.add_argument("--default", required=True, help="the option the axis reports gains against")
    ap.add_argument("--tag", required=True, help="axis name, used in the output filename")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    options = {}
    for spec in a.run:
        label, _, fn = spec.partition("=")
        options[label] = fn if os.path.isabs(fn) else os.path.join(DATA, fn)
    if a.default not in options:
        raise SystemExit(f"--default {a.default} is not one of the options: {sorted(options)}")

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    runs = {label: load_run(fn) for label, fn in options.items()}
    qids = sorted(set(qrels).intersection(*[set(r) for r in runs.values()]))
    pol_runs = [(label, {q: runs[label][q] for q in qids}) for label in sorted(runs)]

    print(f"axis {a.tag}")
    rec = audit(qrels, pol_runs, qids, a.default, a.seeds)
    rec = {"axis": a.tag, "options": options, **rec}

    out_path = os.path.join(ABL, f"mv2_axis_goldsplit_{a.tag}.json")
    json.dump(rec, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
