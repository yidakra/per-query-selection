"""Do the axes compose? Choosing the channel and the query language in one decision.

The three axes were each measured against their own default, which leaves the question a deployer
actually asks: if the system decides both which evidence to search and which language to ask in,
does it collect both gains, or is one of them the other wearing a different hat?

The option space is every (language, channel policy) pair, fifteen in all, scored per query from
runs that already exist. The selector is the same ridge over confidence features, the protocol the
same event-grouped folds with the fixed baseline chosen on the training fold. Reported against
three references: the best fixed pair, the channel-only selector's ceiling (English fixed, channel
chosen) and the language-only selector's ceiling (speech fixed, language chosen).

  python src/multivent2/mv2_joint_selection.py
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
from mv2_variant_selection import rrf  # noqa: E402

LANGS = ["en", "zh", "ko", "ru", "ar"]
SUFFIX = {"en": "", "zh": "_qzh", "ko": "_qko", "ru": "_qru", "ar": "_qar"}
CHANNELS = ["asr", "ocr", "both"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_joint_selection.json"))
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    from mv2_channel_select import mk, group_signflip_p
    from mv2_row_inference import group_stats

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    raw = {}
    for L in LANGS:
        raw[(L, "asr")] = load_run(os.path.join(DATA, f"asr_dense_bge-m3{SUFFIX[L]}.json"))
        raw[(L, "ocr")] = load_run(os.path.join(DATA, f"ocr_dense_bge-m3{SUFFIX[L]}.json"))
    qids = sorted(set(qrels) & set.intersection(*[set(r) for r in raw.values()]))
    for L in LANGS:
        raw[(L, "both")] = {q: rrf(raw[(L, "asr")][q], raw[(L, "ocr")][q]) for q in qids}
    policies = [(L, c) for L in LANGS for c in CHANNELS]
    print(f"{len(policies)} policies, {len(qids)} queries", flush=True)

    cols = []
    for pol in policies:
        pq = per_query_ndcg(qrels, {q: raw[pol][q] for q in qids})
        cols.append(np.array([pq.get(q, 0.0) for q in qids]))
    Y = np.stack(cols, axis=1)

    X = []
    for q in qids:
        row, tops = [], {}
        for L in LANGS:
            for c in ("asr", "ocr"):
                r = raw[(L, c)][q]
                tops[(L, c)] = sorted(r, key=r.get, reverse=True)[:100]
                cf = conf_features(list(r.values()))
                row += [cf[k] for k in FEATURE_ORDER]
        for k1, k2 in itertools.combinations(sorted(tops), 2):
            row.append(len(set(tops[k1][:10]) & set(tops[k2][:10])) / 10.0)
        X.append(row)
    X = np.asarray(X)
    print(f"{X.shape[1]} features per query", flush=True)

    grp = np.asarray(event_groups(qids, qrels))
    splits = list(GroupKFold(5).split(X, groups=grp))

    def run_selector(cols_idx):
        """out-of-fold selection restricted to a subset of the policy columns"""
        sub = Y[:, cols_idx]
        routed = np.zeros(len(qids)); fixed = np.zeros(len(qids)); picks = np.zeros(len(qids), int)
        for tr, te in splits:
            m = mk().fit(X[tr], sub[tr])
            sel = np.argmax(m.predict(X[te]), axis=1)
            fixed_j = int(np.argmax(sub[tr].mean(axis=0)))
            routed[te] = sub[te, sel]; fixed[te] = sub[te, fixed_j]; picks[te] = sel
        return routed, fixed, picks

    en_asr = policies.index(("en", "asr"))
    chan_only = [policies.index(("en", c)) for c in CHANNELS]
    lang_only = [policies.index((L, "asr")) for L in LANGS]
    all_idx = list(range(len(policies)))

    res = {}
    for name, idx in (("channel_only", chan_only), ("language_only", lang_only),
                      ("joint", all_idx)):
        routed, fixed, picks = run_selector(idx)
        p_def, obs, lo, hi = group_stats(routed - Y[:, en_asr], grp)
        res[name] = {"routed": float(routed.mean()), "fixed_nested": float(fixed.mean()),
                     "vs_default_en_asr": float(100 * (routed.mean() - Y[:, en_asr].mean())),
                     "p_vs_default": float(p_def), "ci95_vs_default": [100 * lo, 100 * hi],
                     "n_options": len(idx),
                     "picks_off_default": int((np.array(idx)[picks] != en_asr).sum())}
        print(f"{name:14s} ({len(idx):2d} options): routed {routed.mean():.4f}, "
              f"{res[name]['vs_default_en_asr']:+.2f} vs English-speech "
              f"(p={p_def:.4f}, CI [{100*lo:+.2f},{100*hi:+.2f}])", flush=True)

    oracle_all = Y.max(axis=1)
    oracle_chan = Y[:, chan_only].max(axis=1)
    oracle_lang = Y[:, lang_only].max(axis=1)
    base = Y[:, en_asr].mean()
    add = res["channel_only"]["vs_default_en_asr"] + res["language_only"]["vs_default_en_asr"]
    res["reference"] = {
        "default_en_asr": float(base),
        "best_fixed_pair": float(Y.mean(axis=0).max()),
        "best_fixed_pair_name": "|".join(policies[int(np.argmax(Y.mean(axis=0)))]),
        "oracle_joint": float(oracle_all.mean()),
        "oracle_channel_only": float(oracle_chan.mean()),
        "oracle_language_only": float(oracle_lang.mean()),
        "sum_of_separate_gains": float(add),
        "joint_gain": res["joint"]["vs_default_en_asr"],
        "composition_ratio": float(res["joint"]["vs_default_en_asr"] / add) if add else None}
    print(f"\nsum of the two separate gains {add:+.2f}, joint selector "
          f"{res['joint']['vs_default_en_asr']:+.2f} "
          f"({100*res['reference']['composition_ratio']:.0f}% of additive)")
    print(f"oracles: channel-only {oracle_chan.mean():.4f}, language-only {oracle_lang.mean():.4f}, "
          f"joint {oracle_all.mean():.4f} (default {base:.4f})")
    json.dump({"policies": ["|".join(p) for p in policies], "n_queries": len(qids), **res},
              open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
