"""Task A of the pre-registered variant experiments: the channel null, re-tested per formulation.

For each QueryGym formulation (and the original), the same two re-runnable channels are retrieved
with that formulation's query text, and the speech-vs-OCR cell goes through the identical symmetric
protocol as Table 1. This script builds the per-formulation ingredients: the variant query CSVs, the
cheap-channel run files (score features read them), the cell artifacts (per-query nDCG for both
options plus the selector's confidence features), and the k-way {speech, OCR, both} ridge selection
with event-grouped folds, variants inheriting the parent query's group. The analytic families then
run through mv2_qpp_table --cell/--queries/--score-run and mv2_row_inference --cell per formulation
(the driver loop lives in the shell, one call per formulation).

  python src/multivent2/mv2_variant_task_a.py --sample 0
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

from mv2_io import load_qrels, load_queries  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from mv2_variant_selection import load_pool, encode_and_search, rrf, METHODS  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402


def overlap(a, b, depth):
    ta = set(sorted(a, key=a.get, reverse=True)[:depth])
    tb = set(sorted(b, key=b.get, reverse=True)[:depth])
    return len(ta & tb) / max(1, len(ta | tb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=os.path.join(DATA, "query_variants.jsonl"))
    ap.add_argument("--sample", type=int, default=0, help="which sample index supplies the text")
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    from mv2_channel_select import mk, group_signflip_p

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    pool = load_pool(a.variants, a.sample + 1)

    qids = sorted(q for q in queries if q in qrels)
    complete = [q for q in qids if all((q, m, a.sample) in pool for m in METHODS)]
    print(f"{len(complete)}/{len(qids)} queries complete at sample {a.sample}", flush=True)

    forms = {"original": {q: queries[q] for q in complete}}
    for m in METHODS:
        forms[m] = {q: pool[(q, m, a.sample)] for q in complete}

    texts, owners = [], []
    for f, qmap in forms.items():
        for q in complete:
            texts.append(qmap[q]); owners.append((f, q))

    print(f"scoring {len(texts)} texts on both channels", flush=True)
    asr_runs = encode_and_search(texts, os.path.join(_ROOT, "runs", "embcache", "asr_bge-m3.npz"),
                                 "BAAI/bge-m3")
    ocr_runs = encode_and_search(texts, os.path.join(_ROOT, "runs", "embcache", "ocr_bge-m3.npz"),
                                 "BAAI/bge-m3")

    grp = np.asarray(event_groups(complete, qrels))
    splits = list(GroupKFold(5).split(np.arange(len(complete)), groups=grp))

    out = {"sample": a.sample, "n_queries": len(complete), "formulations": {}}
    for fi, f in enumerate(forms):
        rows = [i for i, (ff, _) in enumerate(owners) if ff == f]
        asr = {complete[j]: asr_runs[rows[j]] for j in range(len(complete))}
        ocr = {complete[j]: ocr_runs[rows[j]] for j in range(len(complete))}
        both = {q: rrf(asr[q], ocr[q]) for q in complete}

        nd = {}
        for name, run in (("asr", asr), ("ocr", ocr), ("both", both)):
            pq = per_query_ndcg(qrels, run)
            nd[name] = np.array([pq.get(q, 0.0) for q in complete])

        # variant query CSV (pre-retrieval features read this text)
        import csv
        qcsv = os.path.join(DATA, f"variant_queries_{f}.csv")
        with open(qcsv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["Query_id", "query"])
            for q in complete:
                w.writerow([q, forms[f][q]])

        # cheap-channel run file (score-only features read it)
        run_path = os.path.join(DATA, f"asr_dense_bge-m3_var_{f}.json")
        json.dump(asr, open(run_path, "w"))

        # the speech-vs-OCR cell artifact for mv2_qpp_table
        cell = {"scorer": f"variant_asr_to_ocr_{f}", "feature_order": FEATURE_ORDER,
                "ndcgA": float(nd["asr"].mean()), "ndcgB": float(nd["ocr"].mean()),
                "per_query": {q: {"ndA": float(nd["asr"][j]), "ndB": float(nd["ocr"][j])}
                              for j, q in enumerate(complete)},
                "features": {q: [conf_features(list(asr[q].values()))[k] for k in FEATURE_ORDER]
                             for q in complete}}
        cell_path = os.path.join(ABL, f"mv2_cell_asr_v_ocr_{f}.json")
        json.dump(cell, open(cell_path, "w"))

        # k-way {asr, ocr, both} ridge with event-grouped folds
        X = []
        for j, q in enumerate(complete):
            fa = conf_features(list(asr[q].values()))
            fo = conf_features(list(ocr[q].values()))
            X.append([fa[k] for k in FEATURE_ORDER] + [fo[k] for k in FEATURE_ORDER]
                     + [overlap(asr[q], ocr[q], 10), overlap(asr[q], ocr[q], 100)])
        X = np.asarray(X)
        Y = np.stack([nd["asr"], nd["ocr"], nd["both"]], axis=1)
        routed = np.zeros(len(complete)); fixed = np.zeros(len(complete))
        for tr, te in splits:
            m = mk().fit(X[tr], Y[tr])
            sel = np.argmax(m.predict(X[te]), axis=1)
            fixed_j = int(np.argmax(Y[tr].mean(axis=0)))
            routed[te] = Y[te, sel]; fixed[te] = Y[te, fixed_j]
        gap = 100 * (routed.mean() - fixed.mean())
        p, _ = group_signflip_p(routed - fixed, grp)

        out["formulations"][f] = {
            "asr": float(nd["asr"].mean()), "ocr": float(nd["ocr"].mean()),
            "both": float(nd["both"].mean()),
            "kway_fixed": float(fixed.mean()), "kway_routed": float(routed.mean()),
            "kway_gap": float(gap), "kway_p_group_signflip": float(p),
            "cell": cell_path, "queries_csv": qcsv, "score_run": run_path}
        print(f"{f:16s} asr {nd['asr'].mean():.4f} ocr {nd['ocr'].mean():.4f} "
              f"both {nd['both'].mean():.4f} | k-way {fixed.mean():.4f} -> {routed.mean():.4f} "
              f"(+{gap:.2f}, p={p:.4f})", flush=True)

    path = os.path.join(ABL, f"mv2_variant_task_a_s{a.sample}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
