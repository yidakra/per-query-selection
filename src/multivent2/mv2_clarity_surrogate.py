"""Clarity over the caption surrogate: the concession row the review panel asked for.

Clarity (Cronen-Townsend et al.) needs a language model over the retrieved documents, and the visual
channel's documents are frames, so Table 1 reports it n/a. But the per-channel index repair and
BERT-QPP's document side were both granted the shipped captions as a text surrogate for those same
documents, and granting the surrogate to every predictor except Clarity is an asymmetry. This computes
Clarity over the captions of the visual channel's top-ranked documents, so the asymmetry is gone and
the n/a becomes a measured concession row.

Definition kept classical: relevance model p(w|R) = sum_d p(d|q) p_mle(w|d) over the top-K retrieved
documents that have captions, with p(d|q) a softmax over the channel's own retrieval scores; clarity is
KL(p(w|R) || p(w|C)) in bits against the caption-collection unigram model. The decision protocol is the
identical one every analytic row gets in the symmetric table: single-feature out-of-fold ridge, with
the nested escalation-fraction choice on a group-disjoint calibration split.

  python src/multivent2/mv2_clarity_surrogate.py --group-cv --nested-calibration
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run, load_queries, load_qrels  # noqa: E402
from mv2_bertqpp import load_captions  # noqa: E402
from mv2_qpp_table import CELLS, route, route_nested, bits  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")


def tok(text):
    return [w for w in text.lower().split() if w.isalnum() or any(c.isalnum() for c in w)]


def clarity_score(scores_docs, captions, cf, total_terms, topk=100):
    """KL(relevance model || collection model) in bits, over the top-K captioned documents."""
    ranked = sorted(scores_docs.items(), key=lambda kv: -kv[1])
    docs = [(d, s) for d, s in ranked if d in captions][:topk]
    if not docs:
        return 0.0
    s = np.array([v for _, v in docs], dtype=np.float64)
    pdq = np.exp(s - s.max())
    pdq /= pdq.sum()
    pwr = collections.defaultdict(float)
    for (d, _), w_d in zip(docs, pdq):
        words = tok(captions[d])
        if not words:
            continue
        inv = w_d / len(words)
        for w in words:
            pwr[w] += inv
    z = sum(pwr.values())
    if z <= 0:
        return 0.0
    kl = 0.0
    floor = 0.5 / total_terms
    for w, p in pwr.items():
        p /= z
        pc = max(cf.get(w, 0) / total_terms, floor)
        kl += p * np.log2(p / pc)
    return float(kl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--group-cv", action="store_true")
    ap.add_argument("--nested-calibration", action="store_true")
    ap.add_argument("--calibration-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_clarity_surrogate_grouped.json"))
    a = ap.parse_args()
    if a.nested_calibration and not a.group_cv:
        ap.error("--nested-calibration requires --group-cv")

    from sklearn.model_selection import GroupKFold
    from scipy.stats import kendalltau

    visual = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    captions = load_captions()

    cf = collections.Counter()
    for text in captions.values():
        cf.update(tok(text))
    total_terms = sum(cf.values())
    print(f"caption collection: {len(captions)} docs, {total_terms} tokens, {len(cf)} vocab",
          flush=True)

    out = {"topk": a.topk, "group_cv": bool(a.group_cv),
           "decision_rule": "nested_fraction" if a.nested_calibration else "zero"}
    for label, fn in CELLS:
        d = json.load(open(os.path.join(ABL, fn)))
        qids = [q for q in d["per_query"] if q in visual and q in queries]
        ndA = np.array([d["per_query"][q]["ndA"] for q in qids])
        ndB = np.array([d["per_query"][q]["ndB"] for q in qids])
        g = ndB - ndA

        raw = np.array([clarity_score(visual[q], captions, cf, total_terms, a.topk) for q in qids])

        recA = recB = None
        sc_path = os.path.join(ABL, "mv2_recall_sidecar.json")
        if os.path.exists(sc_path):
            sc = json.load(open(sc_path)).get(label)
            if sc and all(q in sc["per_query"] for q in qids):
                recA = np.array([sc["per_query"][q][0] for q in qids])
                recB = np.array([sc["per_query"][q][1] for q in qids])

        if a.group_cv:
            grp = event_groups(qids, qrels)
            cv = list(GroupKFold(5).split(np.arange(len(qids)), groups=grp))
        else:
            grp, cv = None, None

        if a.nested_calibration:
            (nd, tau, rec, frac, dec), cal = route_nested(
                raw, g, ndA, ndB, cv, grp, recA, recB, seed=a.seed, cal_size=a.calibration_size)
        else:
            nd, tau, rec, frac, dec = route(raw, g, ndA, ndB, cv, recA, recB)
            cal = []
        out[label] = {"routed_ndcg10": nd, "tau": tau, "routed_recall100": rec,
                      "frac_escalated": frac, "cheap": float(ndA.mean()),
                      "uniform": float(ndB.mean()), "n": len(qids),
                      "raw_tau_check": float(kendalltau(raw, g).statistic),
                      "nested_calibration": cal, "qids": qids, "decisions": dec,
                      "clarity": {q: float(v) for q, v in zip(qids, raw)}}
        print(f"{label}: clarity(surrogate) routed={nd:.4f} tau={tau:+.3f} esc={frac:.2f}"
              + (f" R@100={rec:.4f}" if rec is not None else ""), flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
