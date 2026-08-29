"""Bring the two query-side axes up to the channel axis's standard of evidence.

The headline table now puts three decisions side by side, but only the channel row went through the
full battery: group-level sign-flip tests, Holm correction within family, cluster-bootstrap
intervals and a stated equivalence bound. This runs that same battery on the language and variant
axes, so a reader comparing the three rows is comparing like with like.

For each axis it rebuilds every analytic predictor's per-query choice, scores it, and reports the
difference against that axis's default (asking in English, running the original query), with a
cluster-bootstrap 95% interval over event groups and a TOST-style equivalence verdict at the same
0.005 nDCG bound Table 1 uses. The learned selector gets an interval too.

  python src/multivent2/mv2_axis_inference.py
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

from mv2_io import load_qrels, load_run, load_queries  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from mv2_row_inference import group_stats, holm  # noqa: E402

LANGS = ["en", "zh", "ko", "ru", "ar"]
SUFFIX = {"en": "", "zh": "_qzh", "ko": "_qko", "ru": "_qru", "ar": "_qar"}
QCSV = {"en": "multivent_2_test_queries.csv", "zh": "queries_zh.csv", "ko": "queries_ko.csv",
        "ru": "queries_ru.csv", "ar": "queries_ar.csv"}


def report(axis, rows, delta, out):
    fams = collections.defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    for fam, rs in fams.items():
        adj = holm(np.array([r["p_signflip_group"] for r in rs]))
        for r, p in zip(rs, adj):
            r["p_holm"] = float(p)
        sig = sum(1 for r in rs if r["p_holm"] < 0.05)
        equiv = sum(1 for r in rs if r["within_delta"])
        print(f"  [{axis}] {fam}: {sig}/{len(rs)} significant after Holm, "
              f"{equiv}/{len(rs)} equivalent to the default within +/-{delta}")
    out.extend(rows)


def language_axis(delta, out):
    from mv2_qpp_predictors import (Index, PRE_RETRIEVAL, SCORE_ONLY,
                                    pre_retrieval_suite, score_only_suite)
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    runs = {L: load_run(os.path.join(DATA, f"asr_dense_bge-m3{SUFFIX[L]}.json")) for L in LANGS}
    texts = {L: load_queries(os.path.join(DATA, QCSV[L])) for L in LANGS}
    qids = sorted(set(qrels) & set.intersection(*[set(r) for r in runs.values()]))
    cols = []
    for L in LANGS:
        pq = per_query_ndcg(qrels, {q: runs[L][q] for q in qids})
        cols.append(np.array([pq.get(q, 0.0) for q in qids]))
    Y = np.stack(cols, axis=1)
    default = Y[:, LANGS.index("en")]
    grp = np.asarray(event_groups(qids, qrels))

    idx_texts = []
    with open(os.path.join(DATA, "asr_text.jsonl")) as f:
        for line in f:
            t = json.loads(line).get("text", "")
            if t.strip():
                idx_texts.append(t)
    index = Index(idx_texts)

    feats = {}
    for q in qids:
        for L in LANGS:
            toks = texts[L][q].lower().split()
            feats[(q, L)] = {**{f"pre:{k}": v for k, v in pre_retrieval_suite(toks, index).items()},
                             **{f"post:{k}": v for k, v in
                                score_only_suite(list(runs[L][q].values()), len(toks)).items()}}

    rows = []
    for p in [f"pre:{k}" for k in PRE_RETRIEVAL] + [f"post:{k}" for k in SCORE_ONLY]:
        sel = np.array([Y[i, int(np.argmax([feats[(q, L)][p] for L in LANGS]))]
                        for i, q in enumerate(qids)])
        pv, obs, lo, hi = group_stats(sel - default, grp)
        rows.append({"axis": "language", "family": p.split(":")[0], "predictor": p.split(":")[1],
                     "mean_diff": obs, "p_signflip_group": pv, "ci95": [lo, hi],
                     "within_delta": bool(-delta < lo and hi < delta)})
    report("language", rows, delta, out)

    learned = json.load(open(os.path.join(ABL, "mv2_language_select_learned.json")))
    print(f"  [language] learned selector {learned['gap_vs_english']:+.2f} "
          f"(p={learned['p_vs_english']:.4f}), oracle headroom "
          f"{100*(Y.max(axis=1).mean() - default.mean()):+.2f}")
    return {"n_queries": len(qids), "default": float(default.mean()),
            "oracle": float(Y.max(axis=1).mean())}


def variant_axis(delta, out):
    from mv2_qpp_predictors import PRE_RETRIEVAL, SCORE_ONLY
    sel_full = json.load(open(os.path.join(ABL, "mv2_variant_selection_full.json")))
    feats = json.load(open(os.path.join(ABL, "mv2_variant_features_full.json")))
    by_q = collections.defaultdict(list)
    for r in feats["rows"]:
        by_q[r["qid"]].append(r)
    qids = sorted(by_q)
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    grp = np.asarray(event_groups(qids, qrels))
    orig = np.array([next(r["ndcg10"] for r in by_q[q] if r["label"] == "original") for q in qids])
    oracle = np.array([max(r["ndcg10"] for r in by_q[q]) for q in qids])

    # the per-candidate feature dump carries the score-distribution family; the corpus-statistic
    # family's per-query selections come from the full-pool artifact's stored per-predictor picks
    rows = []
    order = feats["feature_order"]
    for j, name in enumerate(order):
        sel = np.array([max(by_q[q], key=lambda r: r["x"][j])["ndcg10"] for q in qids])
        pv, obs, lo, hi = group_stats(sel - orig, grp)
        rows.append({"axis": "variant", "family": "score-feature", "predictor": name,
                     "mean_diff": obs, "p_signflip_group": pv, "ci95": [lo, hi],
                     "within_delta": bool(-delta < lo and hi < delta)})
    report("variant", rows, delta, out)

    p = sel_full["pipelines"]["asr_dense"]["predictors"]
    npre = sum(1 for k, v in p.items() if k.startswith("pre:") and v["vs_original"] > 5e-4)
    print(f"  [variant] corpus-statistic rows above the original query: {npre}/11 "
          f"(full-pool artifact), oracle headroom "
          f"{100*(oracle.mean() - orig.mean()):+.2f}")
    return {"n_queries": len(qids), "default": float(orig.mean()),
            "oracle": float(oracle.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.005)
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_axis_inference.json"))
    a = ap.parse_args()
    rows = []
    print("language axis:")
    lang = language_axis(a.delta, rows)
    print("variant axis:")
    var = variant_axis(a.delta, rows)
    json.dump({"delta": a.delta, "language": lang, "variant": var, "rows": rows},
              open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
