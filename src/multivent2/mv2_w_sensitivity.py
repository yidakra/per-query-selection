"""Fusion-weight sensitivity for the Table 1 family counts.

The pairwise cells' fused endpoints use the RRF weight the full-test sweep chose (mv2_channels.py
--sweep), which is a test-set choice sitting under a table about calibration hygiene. The predictors
themselves never see the weight: post-retrieval rows read the visual channel's scores and
pre-retrieval rows read the query against a lexical index, so w moves only the routing targets
(per-query ndB) and the Original baseline. This recomputes the cell at every weight on the sweep grid
and reports the family underline counts per weight, under the same symmetric nested protocol as the
main table. If the counts do not move across the grid, the sweep cannot be doing the work.

At the recorded weight the rebuilt per-query ndB must reproduce the cell artifact exactly, or the
script aborts: same-fusion verification, not a re-implementation.

  python src/multivent2/mv2_w_sensitivity.py --target asr --channel asr=asr_dense_bge-m3.json \
      --cell mv2_chan_visual_to_asr_dense_m3.json --recorded-w 1.0
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run, load_queries, load_qrels  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_channels import CHANNELS, fuse  # noqa: E402
from mv2_qpp_predictors import (score_only_suite, pre_retrieval_suite, Index,  # noqa: E402
                                SCORE_ONLY, PRE_RETRIEVAL)
from mv2_qpp_table import route_nested  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
MARGIN = 5e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["asr", "ocr"])
    ap.add_argument("--channel", action="append", default=[], metavar="NAME=FILE")
    ap.add_argument("--cell", required=True, help="cell artifact under results/ablations/")
    ap.add_argument("--recorded-w", type=float, required=True,
                    help="the weight the sweep chose; rebuilt ndB must reproduce the cell exactly")
    ap.add_argument("--weights", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--index-text", default="asr_text.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(ABL, f"mv2_w_sensitivity_{a.target}.json")

    from sklearn.model_selection import GroupKFold

    channels = dict(CHANNELS)
    for spec in a.channel:
        name, _, fn = spec.partition("=")
        channels[name] = fn

    cell = json.load(open(os.path.join(ABL, a.cell)))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    visual = load_run(os.path.join(DATA, channels["visual"]))
    target = load_run(os.path.join(DATA, channels[a.target]))
    qids = [q for q in cell["per_query"] if q in visual and q in queries]
    ndA = np.array([cell["per_query"][q]["ndA"] for q in qids])
    ndB_cell = np.array([cell["per_query"][q]["ndB"] for q in qids])

    grp = event_groups(qids, qrels)
    splits = list(GroupKFold(5).split(np.arange(len(qids)), groups=grp))
    print(f"{a.cell}: {len(qids)} queries, {len(set(grp))} groups", flush=True)

    post = {n: [] for n in SCORE_ONLY}
    for q in qids:
        nq = len(queries[q].split())
        for n, v in score_only_suite(list(visual[q].values()), nq).items():
            post[n].append(v)

    texts = []
    with open(os.path.join(DATA, a.index_text)) as f:
        for line in f:
            t = json.loads(line).get("text", "")
            if t.strip():
                texts.append(t)
    idx = Index(texts)
    print(f"index: {idx.n_docs} docs", flush=True)
    pre = {n: [] for n in PRE_RETRIEVAL}
    for q in qids:
        s = pre_retrieval_suite(queries[q].lower().split(), idx)
        for n in PRE_RETRIEVAL:
            pre[n].append(s[n])

    runs = {"visual": visual, a.target: target}
    results = {}
    for w in [float(x) for x in a.weights.split(",")]:
        fused = fuse(runs, {"visual": 1.0, a.target: w}, qids)
        nd = per_query_ndcg(qrels, fused)
        ndB = np.array([nd[q] for q in qids])
        if abs(w - a.recorded_w) < 1e-9:
            if not np.allclose(ndB, ndB_cell, atol=1e-9):
                sys.exit(f"rebuilt ndB at recorded w={w} does not reproduce the cell artifact")
            print(f"w={w}: reproduces the cell artifact exactly", flush=True)
        g = ndB - ndA
        orig = max(ndA.mean(), ndB.mean())
        row = {"ndB_mean": float(ndB.mean()), "original": float(orig)}
        for fam, values, names in (("pre", pre, PRE_RETRIEVAL), ("post", post, SCORE_ONLY)):
            und = deg = 0
            per = {}
            for n in names:
                (rnd, tau, _, frac, _), _cal = route_nested(
                    values[n], g, ndA, ndB, splits, grp, seed=a.seed)
                u = rnd > orig + MARGIN
                und += u
                deg += frac in (0.0, 1.0)
                per[n] = {"routed": rnd, "tau": tau, "frac": frac, "underlined": bool(u)}
            row[fam] = {"underlined": int(und), "of": len(names), "degenerate": int(deg),
                        "rows": per}
        results[f"{w}"] = row
        print(f"w={w}: ndB={ndB.mean():.4f} orig={orig:.4f}  "
              f"pre {row['pre']['underlined']}/{row['pre']['of']} underlined  "
              f"post {row['post']['underlined']}/{row['post']['of']} underlined", flush=True)

    json.dump({"cell": a.cell, "target": a.target, "recorded_w": a.recorded_w,
               "margin": MARGIN, "weights": results}, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
