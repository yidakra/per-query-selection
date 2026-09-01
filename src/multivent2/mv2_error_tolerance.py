"""How accurate must a selector be before acting on it beats the fixed policy?

The centrepiece of the supervision research plan's RQ2: build simulated selectors with controlled
error, sweep the error rate, and find where the delivered gain crosses zero. One curve per axis,
because the axes price mistakes differently: the channel decision escalates a fraction and degrades
gracefully, while the language and variant decisions commit to one option whose alternatives differ
by several points, so errors there are expensive.

Error model, version 1 and stated as such: with probability (1 - accuracy) the simulated selector
replaces the oracle's pick with an option drawn uniformly from the others. Real selectors' observed
oracle-agreement rates then place them on the same curve.

  python src/multivent2/mv2_error_tolerance.py
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_variant_selection import rrf  # noqa: E402

SUFFIX = {"en": "", "zh": "_qzh", "ko": "_qko", "ru": "_qru", "ar": "_qar"}


def axis_matrices():
    """Per-axis (Y, default_column_index, axis_name)."""
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    out = {}

    # language: five options, default = English
    runs = {L: load_run(os.path.join(DATA, f"asr_dense_bge-m3{SUFFIX[L]}.json")) for L in SUFFIX}
    qids = sorted(set(qrels) & set.intersection(*[set(r) for r in runs.values()]))
    cols = []
    for L in SUFFIX:
        pq = per_query_ndcg(qrels, {q: runs[L][q] for q in qids})
        cols.append([pq.get(q, 0.0) for q in qids])
    out["language"] = (np.array(cols).T, 0)

    # channel: three options on the same dense pair, default = speech
    asr = runs["en"]
    ocr = load_run(os.path.join(DATA, "ocr_dense_bge-m3.json"))
    both = {q: rrf(asr[q], ocr[q]) for q in qids}
    cols = []
    for run in (asr, ocr, both):
        pq = per_query_ndcg(qrels, {q: run[q] for q in qids})
        cols.append([pq.get(q, 0.0) for q in qids])
    out["channel_2ch"] = (np.array(cols).T, 0)

    # variant: 31 options from the feature dump, default = original
    d = json.load(open(os.path.join(ABL, "mv2_variant_features_full.json")))
    by_q = collections.defaultdict(dict)
    for r in d["rows"]:
        by_q[r["qid"]][r["label"]] = r["ndcg10"]
    labels = d["labels"]
    vq = sorted(by_q)
    out["variant"] = (np.array([[by_q[q][lab] for lab in labels] for q in vq]),
                      labels.index("original"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accuracies", default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_error_tolerance.json"))
    a = ap.parse_args()

    res = {"error_model": "uniform substitution of the oracle pick", "axes": {}}
    for axis, (Y, dflt) in axis_matrices().items():
        n, k = Y.shape
        oracle = Y.argmax(axis=1)
        base = Y[:, dflt].mean()
        curve = []
        rng = np.random.default_rng(0)
        for acc in [float(x) for x in a.accuracies.split(",")]:
            gains = []
            for _ in range(a.seeds):
                pick = oracle.copy()
                wrong = rng.random(n) > acc
                # draw a non-oracle option uniformly for the wrong picks
                rand = rng.integers(0, k - 1, size=n)
                rand = rand + (rand >= oracle)
                pick[wrong] = rand[wrong]
                gains.append(100 * (Y[np.arange(n), pick].mean() - base))
            curve.append({"accuracy": acc, "gain_mean": float(np.mean(gains)),
                          "gain_sd": float(np.std(gains))})
        # linear interpolation for the break-even accuracy
        be = None
        for lo, hi in zip(curve[::-1], curve[::-1][1:]):
            pass
        xs = [c["accuracy"] for c in curve][::-1]
        ys = [c["gain_mean"] for c in curve][::-1]
        for i in range(len(xs) - 1):
            if ys[i] <= 0 <= ys[i + 1]:
                be = xs[i] + (xs[i + 1] - xs[i]) * (0 - ys[i]) / (ys[i + 1] - ys[i])
                break
        res["axes"][axis] = {"n_options": int(k), "default_mean": float(base),
                             "oracle_gain": curve[0]["gain_mean"],
                             "curve": curve, "break_even_accuracy": be}
        print(f"{axis:12s} k={k:2d} oracle {curve[0]['gain_mean']:+.2f} | "
              f"break-even accuracy {be if be is None else round(be, 3)} | "
              f"random-pick gain {curve[-1]['gain_mean']:+.2f}", flush=True)

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
