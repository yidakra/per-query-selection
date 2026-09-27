"""The positive control for the language axis: can a trained selector pick the query language?

mv2_language_selection.py showed the headroom (oracle +10.4 nDCG over always-English) and that no
analytic predictor converts it. That is half an answer: everywhere else in this study the analytic
families fail and a small ridge over score-distribution features succeeds, which is the whole
boundary argument. This runs that same selector on the language axis, so the third axis gets the
same treatment as the first.

Features per query: each language version's own confidence features over its ranked list, plus the
pairwise top-10/top-100 overlap between language result lists (how much the languages disagree).
Targets: per-query nDCG for each language option. Protocol is the channel-selection protocol,
event-grouped folds with the fixed baseline chosen on the training fold, so neither side peeks.

  python src/multivent2/mv2_language_select_learned.py --drop zh
"""
import os
import sys
import json
import argparse
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402

LANGS = ["en", "zh", "ko", "ru", "ar"]
SUFFIX = {"en": "", "zh": "_qzh", "ko": "_qko", "ru": "_qru", "ar": "_qar"}


def overlap(top_a, top_b, depth):
    """Jaccard over the top-`depth` prefixes of two already-sorted document lists."""
    ta, tb = set(top_a[:depth]), set(top_b[:depth])
    return len(ta & tb) / max(1, len(ta | tb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="asr", choices=["asr", "ocr"])
    ap.add_argument("--drop", default="", help="languages to exclude, e.g. zh")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_language_select_learned.json"))
    ap.add_argument("--feature-set", default="outcome", choices=["outcome", "corpus"],
                    help="outcome: each version's score-distribution features and result overlaps "
                         "(the positive control); corpus: each version's eleven pre-retrieval "
                         "corpus-statistic features instead, same learner, folds and test")
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    from mv2_channel_select import mk, group_signflip_p

    langs = [L for L in LANGS if L not in {x.strip() for x in a.drop.split(",") if x.strip()}]
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    runs = {L: load_run(os.path.join(DATA, f"{a.channel}_dense_bge-m3{SUFFIX[L]}.json"))
            for L in langs}
    qids = sorted(set(qrels) & set.intersection(*[set(r) for r in runs.values()]))
    print(f"languages {langs}, {len(qids)} queries on the {a.channel} channel", flush=True)

    cols = []
    for L in langs:
        pq = per_query_ndcg(qrels, {q: runs[L][q] for q in qids})   # scored once per language
        cols.append(np.array([pq.get(q, 0.0) for q in qids]))
    Y = np.stack(cols, axis=1)

    if a.feature_set == "corpus":
        # the matched control for the family claim: the same multi-feature learner, reading only
        # what a pre-retrieval predictor may read, the query text and corpus term statistics
        from mv2_io import load_queries
        from mv2_language_selection import QCSV
        from mv2_qpp_predictors import Index, PRE_RETRIEVAL, pre_retrieval_suite
        qtext = {L: load_queries(os.path.join(DATA, QCSV[L])) for L in langs}
        idx_texts = []
        with open(os.path.join(DATA, "asr_text.jsonl")) as f:
            for line in f:
                t = json.loads(line).get("text", "")
                if t.strip():
                    idx_texts.append(t)
        index = Index(idx_texts)
        print(f"corpus features over a lexical index of {len(idx_texts)} transcripts", flush=True)

    X = []
    for i, q in enumerate(qids):
        row = []
        tops = {}
        for L in langs:
            if a.feature_set == "corpus":
                pre = pre_retrieval_suite(qtext[L][q].lower().split(), index)
                row += [pre[k] for k in PRE_RETRIEVAL]
                continue
            r = runs[L][q]
            tops[L] = sorted(r, key=r.get, reverse=True)[:100]
            cf = conf_features(list(r.values()))
            row += [cf[k] for k in FEATURE_ORDER]
        if a.feature_set == "outcome":
            for L1, L2 in itertools.combinations(langs, 2):
                row += [overlap(tops[L1], tops[L2], 10), overlap(tops[L1], tops[L2], 100)]
        X.append(row)
        if (i + 1) % 500 == 0:
            print(f"  features {i+1}/{len(qids)}", flush=True)
    X = np.asarray(X)
    print(f"{X.shape[1]} features per query", flush=True)

    grp = np.asarray(event_groups(qids, qrels))
    splits = list(GroupKFold(5).split(X, groups=grp))
    routed = np.zeros(len(qids)); fixed = np.zeros(len(qids)); pick = np.zeros(len(qids), int)
    for tr, te in splits:
        m = mk().fit(X[tr], Y[tr])
        sel = np.argmax(m.predict(X[te]), axis=1)
        fixed_j = int(np.argmax(Y[tr].mean(axis=0)))
        routed[te] = Y[te, sel]; fixed[te] = Y[te, fixed_j]; pick[te] = sel
    gap = 100 * (routed.mean() - fixed.mean())
    p, _ = group_signflip_p(routed - fixed, grp)
    en_i = langs.index("en")
    p_en, _ = group_signflip_p(routed - Y[:, en_i], grp)
    oracle = Y.max(axis=1)

    out = {"channel": a.channel, "languages": langs, "n_queries": len(qids),
           "feature_set": a.feature_set,
           "n_features": int(X.shape[1]),
           "fixed_per_language": {L: float(Y[:, j].mean()) for j, L in enumerate(langs)},
           "fixed_nested": float(fixed.mean()), "routed": float(routed.mean()),
           "gap_vs_fixed": float(gap), "p_vs_fixed": float(p),
           "english_default": float(Y[:, en_i].mean()),
           "gap_vs_english": float(100 * (routed.mean() - Y[:, en_i].mean())),
           "p_vs_english": float(p_en),
           "oracle": float(oracle.mean()),
           "fraction_of_oracle_headroom": float(
               (routed.mean() - Y[:, en_i].mean()) / (oracle.mean() - Y[:, en_i].mean())),
           "picks": {L: int((pick == j).sum()) for j, L in enumerate(langs)}}
    print(f"fixed(nested) {fixed.mean():.4f} -> routed {routed.mean():.4f} "
          f"({gap:+.2f}, p={p:.4f})")
    print(f"vs always-English {Y[:, en_i].mean():.4f}: {out['gap_vs_english']:+.2f} "
          f"(p={p_en:.4f}), oracle {oracle.mean():.4f}, "
          f"captures {100*out['fraction_of_oracle_headroom']:.0f}% of the headroom")
    print(f"picks: {out['picks']}")
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
