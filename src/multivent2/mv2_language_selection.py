"""A third selection axis: which language should the query be asked in?

The boundary claim says corpus statistics carry selection signal when the options are different
query texts and none when the options are different document sources. Query language is a
query-side option: the five versions of a query are five different texts against one fixed
corpus, exactly the shape the source study's rewrites have. So the claim makes a falsifiable
prediction here, and this measures it.

The five options are the English query and its NLLB translations into the four largest corpus
languages, all already retrieved against the same dense speech channel. Every predictor ranks the
five, its top choice is executed, and the result is scored against asking in English (the default),
the best fixed language, and the per-query oracle. Pre-retrieval features read the original
mixed-language transcript index, the index a deployed system would actually have.

  python src/multivent2/mv2_language_selection.py
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

from mv2_io import load_qrels, load_run, load_queries  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402

LANGS = ["en", "zh", "ko", "ru", "ar"]
SUFFIX = {"en": "", "zh": "_qzh", "ko": "_qko", "ru": "_qru", "ar": "_qar"}
QCSV = {"en": "multivent_2_test_queries.csv", "zh": "queries_zh.csv", "ko": "queries_ko.csv",
        "ru": "queries_ru.csv", "ar": "queries_ar.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="asr", choices=["asr", "ocr"])
    ap.add_argument("--drop", default="",
                    help="comma-separated languages to exclude; use for zh, whose whitespace "
                         "tokenization is degenerate (1.08 tokens per query, 99% out of vocabulary) "
                         "and would hand one option broken pre-retrieval features")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_language_selection.json"))
    a = ap.parse_args()

    global LANGS
    if a.drop:
        LANGS = [L for L in LANGS if L not in {x.strip() for x in a.drop.split(",")}]
        print(f"languages: {LANGS}")

    from scipy.stats import kendalltau
    from mv2_qpp_predictors import (Index, PRE_RETRIEVAL, SCORE_ONLY,
                                    pre_retrieval_suite, score_only_suite)
    from mv2_channel_select import group_signflip_p

    qrels, meta = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    runs, texts = {}, {}
    for L in LANGS:
        runs[L] = load_run(os.path.join(DATA, f"{a.channel}_dense_bge-m3{SUFFIX[L]}.json"))
        texts[L] = load_queries(os.path.join(DATA, QCSV[L]))
    qids = sorted(set(qrels) & set.intersection(*[set(r) for r in runs.values()]))
    print(f"{len(qids)} queries with all five language versions on the {a.channel} channel",
          flush=True)

    nd = {}
    for L in LANGS:
        pq = per_query_ndcg(qrels, {q: runs[L][q] for q in qids})
        nd[L] = np.array([pq.get(q, 0.0) for q in qids])
    Y = np.stack([nd[L] for L in LANGS], axis=1)

    print("building the lexical index over the original transcripts", flush=True)
    idx_texts = []
    with open(os.path.join(DATA, "asr_text.jsonl")) as f:
        for line in f:
            t = json.loads(line).get("text", "")
            if t.strip():
                idx_texts.append(t)
    index = Index(idx_texts)

    feats = {}
    for i, q in enumerate(qids):
        for j, L in enumerate(LANGS):
            toks = texts[L][q].lower().split()
            pre = pre_retrieval_suite(toks, index)
            post = score_only_suite(list(runs[L][q].values()), len(toks))
            feats[(q, L)] = {**{f"pre:{k}": v for k, v in pre.items()},
                             **{f"post:{k}": v for k, v in post.items()}}
        if (i + 1) % 500 == 0:
            print(f"  features {i+1}/{len(qids)}", flush=True)

    grp = np.asarray(event_groups(qids, qrels))
    oracle = Y.max(axis=1)
    picks = Y.argmax(axis=1)
    best_fixed = LANGS[int(np.argmax(Y.mean(axis=0)))]
    out = {"channel": a.channel, "n_queries": len(qids),
           "fixed": {L: float(nd[L].mean()) for L in LANGS},
           "best_fixed": best_fixed,
           "english_default": float(nd["en"].mean()),
           "oracle": float(oracle.mean()),
           "oracle_minus_english": float(oracle.mean() - nd["en"].mean()),
           "oracle_picks": {L: int((picks == j).sum()) for j, L in enumerate(LANGS)},
           "predictors": {}}
    print(f"fixed: " + "  ".join(f"{L} {nd[L].mean():.4f}" for L in LANGS))
    print(f"oracle {oracle.mean():.4f} (+{100*(oracle.mean()-nd['en'].mean()):.2f} over English), "
          f"picks {out['oracle_picks']}", flush=True)

    predictors = [f"pre:{k}" for k in PRE_RETRIEVAL] + [f"post:{k}" for k in SCORE_ONLY]
    for p in predictors:
        taus, sel = [], []
        for i, q in enumerate(qids):
            vals = np.array([feats[(q, L)][p] for L in LANGS])
            if vals.std() > 0 and Y[i].std() > 0:
                t = kendalltau(vals, Y[i]).statistic
                if not np.isnan(t):
                    taus.append(t)
            sel.append(Y[i, int(np.argmax(vals))])
        sel = np.array(sel)
        pval, _ = group_signflip_p(sel - nd["en"], grp)
        out["predictors"][p] = {
            "mean_within_query_tau": float(np.mean(taus)) if taus else 0.0,
            "selected_ndcg10": float(sel.mean()),
            "vs_english": float(sel.mean() - nd["en"].mean()),
            "p_group_signflip_vs_english": float(pval)}

    for fam, pref in (("pre", "pre:"), ("post", "post:")):
        rows = {k: v for k, v in out["predictors"].items() if k.startswith(pref)}
        above = sum(1 for v in rows.values() if v["vs_english"] > 5e-4)
        sig = sum(1 for v in rows.values() if v["p_group_signflip_vs_english"] < 0.05)
        best = max(rows.items(), key=lambda kv: kv[1]["selected_ndcg10"])
        out[f"{fam}_summary"] = {"above_english": above, "n": len(rows), "significant": sig,
                                 "best": best[0], "best_ndcg10": best[1]["selected_ndcg10"]}
        print(f"{fam}: {above}/{len(rows)} above English, {sig} significant; best {best[0]} "
              f"{best[1]['selected_ndcg10']:.4f} ({best[1]['vs_english']:+.4f}, "
              f"tau {best[1]['mean_within_query_tau']:+.3f})", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
