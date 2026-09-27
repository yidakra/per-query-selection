"""Corpus-statistic features for every rewriting candidate, in the learned selector's format.

The rewriting selector's positive control reads each candidate's retrieval-confidence features. A
reviewer's objection is that this compares a multi-feature learner with single-feature analytic
predictors, so "reads an outcome" is not separated from "has more features". This writes the matched
control: for every candidate, the eleven pre-retrieval corpus-statistic features over the same
lexical index the analytic predictors use, in exactly the format mv2_variant_select_learned.py reads.
Run the learner on it with the same flags and only the features differ.

  python src/multivent2/mv2_variant_corpus_features.py
  python src/multivent2/mv2_variant_select_learned.py --center --no-method-feature \\
      --features results/ablations/mv2_variant_features_corpus.json \\
      --out results/ablations/mv2_variant_select_learned_corpus.json
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome-features", default=os.path.join(ABL, "mv2_variant_features_full.json"),
                    help="the outcome-feature file; its rows fix the candidate set and targets")
    ap.add_argument("--variants", default=os.path.join(DATA, "query_variants.jsonl"))
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_variant_features_corpus.json"))
    a = ap.parse_args()

    from mv2_io import load_queries
    from mv2_variant_selection import load_pool
    from mv2_qpp_predictors import Index, PRE_RETRIEVAL, pre_retrieval_suite

    src = json.load(open(a.outcome_features))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    pool = load_pool(a.variants, a.samples)

    idx_texts = []
    with open(os.path.join(DATA, "asr_text.jsonl")) as f:
        for line in f:
            t = json.loads(line).get("text", "")
            if t.strip():
                idx_texts.append(t)
    index = Index(idx_texts)
    print(f"lexical index over {len(idx_texts)} transcripts; {len(src['rows'])} candidates",
          flush=True)

    rows = []
    for i, r in enumerate(src["rows"]):
        if r["label"] == "original":
            text = queries[r["qid"]]
        else:
            m, s = r["label"].rsplit("#", 1)
            text = pool[(r["qid"], m, int(s))]
        pre = pre_retrieval_suite(text.lower().split(), index)
        rows.append({"qid": r["qid"], "label": r["label"], "method": r["method"],
                     "x": [pre[k] for k in PRE_RETRIEVAL], "ndcg10": r["ndcg10"]})
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(src['rows'])}", flush=True)

    json.dump({"feature_order": list(PRE_RETRIEVAL), "labels": src["labels"], "rows": rows},
              open(a.out, "w"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
