"""Three additions from the supervision research plan, on runs that already exist.

One: the fixed-fusion baseline the language axis was missing (RRF over all five language lists),
and the same for the two-channel decision. Two: query-conditioned fusion, the plan's fourth
strategy, where out-of-fold predicted per-option gains become per-query fusion weights instead of a
hard pick. Three: the composed-selectors arm of the joint experiment (each axis's selector picks
independently, the two picks are applied together) and every learned selector's oracle-agreement
rate, so the real selectors sit on the error-tolerance curves.

Query-conditioned fusion, version 1 and stated as such: weights are a softmax over the ridge's
z-scored predicted nDCG per option, temperature 1, applied as weighted reciprocal-rank fusion.

  python src/multivent2/mv2_plan_additions.py
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


def rrf_weighted(runs_list, weights, k=60, depth=1000):
    out = {}
    for r, w in zip(runs_list, weights):
        for i, d in enumerate(sorted(r, key=r.get, reverse=True)[:depth]):
            out[d] = out.get(d, 0.0) + w / (k + i + 1)
    return out


def features_for(options_runs, qids):
    """Per-query features: each option's confidence stats plus pairwise top-10/100 overlap."""
    X = []
    for q in qids:
        row, tops = [], []
        for r in options_runs:
            lst = sorted(r[q], key=r[q].get, reverse=True)[:100]
            tops.append(lst)
            cf = conf_features(list(r[q].values()))
            row += [cf[k] for k in FEATURE_ORDER]
        for a, b in itertools.combinations(range(len(options_runs)), 2):
            row += [len(set(tops[a][:10]) & set(tops[b][:10])) / 10.0,
                    len(set(tops[a]) & set(tops[b])) / 100.0]
        X.append(row)
    return np.asarray(X)


def axis_block(name, options_runs, option_names, default_i, qids, qrels, grp, splits, mk,
               group_stats, out, fusion_idx=None):
    """`fusion_idx`: indices of the options that both fusion strategies combine. Defaults to every
    option, which is right when all options are raw lists (the language axis); pass the raw
    channels' indices when an option is itself a fusion of the others, so nothing is counted twice.
    Query-conditioned weights are softmaxed over those same options' predictions."""
    Y = np.stack([np.array([per_query_ndcg(qrels, {q: r[q] for q in qids}).get(q, 0.0)
                            for q in qids]) for r in options_runs], axis=1)
    default = Y[:, default_i]
    X = features_for(options_runs, qids)

    # both fusion strategies combine the raw members only
    fidx = list(fusion_idx) if fusion_idx is not None else list(range(len(options_runs)))
    members = [options_runs[i] for i in fidx]
    fused = {q: rrf_weighted([r[q] for r in members], [1.0] * len(members)) for q in qids}
    pq = per_query_ndcg(qrels, fused)
    fixed_fusion = np.array([pq.get(q, 0.0) for q in qids])

    # out-of-fold predictions drive both the hard pick and the per-query fusion weights
    sel = np.zeros(len(qids)); agree = np.zeros(len(qids), bool)
    qcond_fused_run = {}
    oracle_pick = Y.argmax(axis=1)
    for tr, te in splits:
        m = mk().fit(X[tr], Y[tr])
        pred = m.predict(X[te])
        pick = np.argmax(pred, axis=1)
        sel[te] = Y[te, pick]
        agree[te] = pick == oracle_pick[te]
        pm = pred[:, fidx]
        z = (pm - pm.mean(axis=1, keepdims=True)) / (pm.std(axis=1, keepdims=True) + 1e-9)
        W = np.exp(z); W /= W.sum(axis=1, keepdims=True)
        for row_i, qi in enumerate(te):
            q = qids[qi]
            qcond_fused_run[q] = rrf_weighted([r[q] for r in members], W[row_i])
    pq2 = per_query_ndcg(qrels, qcond_fused_run)
    qcond_fusion = np.array([pq2.get(q, 0.0) for q in qids])

    rows = {}
    for label, vals in (("fixed_fusion", fixed_fusion), ("hard_selection", sel),
                        ("qcond_fusion", qcond_fusion)):
        p, obs, lo, hi = group_stats(vals - default, grp)
        rows[label] = {"ndcg10": float(vals.mean()),
                       "vs_default": float(100 * (vals.mean() - default.mean())),
                       "p": float(p), "ci95": [float(100 * lo), float(100 * hi)]}
    rows["default"] = {"ndcg10": float(default.mean())}
    rows["oracle"] = {"ndcg10": float(Y.max(axis=1).mean())}
    rows["selector_oracle_agreement"] = float(agree.mean())
    out[name] = rows
    print(f"[{name}] default {default.mean():.4f} | fixed-fusion "
          f"{rows['fixed_fusion']['vs_default']:+.2f} (p={rows['fixed_fusion']['p']:.4f}) | "
          f"hard selection {rows['hard_selection']['vs_default']:+.2f} | "
          f"q-cond fusion {rows['qcond_fusion']['vs_default']:+.2f} "
          f"(p={rows['qcond_fusion']['p']:.4f}) | selector agrees with oracle "
          f"{100*agree.mean():.1f}%", flush=True)
    return Y, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_plan_additions.json"))
    a = ap.parse_args()
    from sklearn.model_selection import GroupKFold
    from mv2_channel_select import mk
    from mv2_row_inference import group_stats

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    lang_runs = {L: load_run(os.path.join(DATA, f"asr_dense_bge-m3{SUFFIX[L]}.json"))
                 for L in LANGS}
    ocr_runs = {L: load_run(os.path.join(DATA, f"ocr_dense_bge-m3{SUFFIX[L]}.json"))
                for L in LANGS}
    ocr = ocr_runs["en"]
    # every run that any policy will index is intersected into the query set, so no policy can
    # meet a query it never retrieved for
    qids = sorted(set(qrels)
                  & set.intersection(*[set(r) for r in lang_runs.values()])
                  & set.intersection(*[set(r) for r in ocr_runs.values()]))
    grp = np.asarray(event_groups(qids, qrels))
    splits = list(GroupKFold(5).split(np.arange(len(qids)), groups=grp))
    out = {}

    lang_opts = [lang_runs[L] for L in LANGS]
    Yl, _ = axis_block("language", lang_opts, LANGS, 0, qids, qrels, grp, splits, mk,
                       group_stats, out)

    both = {q: rrf(lang_runs["en"][q], ocr[q]) for q in qids}
    chan_opts = [lang_runs["en"], ocr, both]
    Yc, _ = axis_block("channel_2ch", chan_opts, ["asr", "ocr", "both"], 0, qids, qrels, grp,
                       splits, mk, group_stats, out, fusion_idx=[0, 1])

    # composed selectors: each axis's own out-of-fold pick, applied together as a (language,
    # channel) pair, scored on the joint policy grid
    pol_runs = {}
    for L in LANGS:
        pol_runs[(L, "asr")] = lang_runs[L]
        pol_runs[(L, "ocr")] = ocr_runs[L]
    for L in LANGS:
        pol_runs[(L, "both")] = {q: rrf(pol_runs[(L, "asr")][q], pol_runs[(L, "ocr")][q])
                                 for q in qids}
    policies = [(L, c) for L in LANGS for c in ("asr", "ocr", "both")]
    Yj = np.stack([np.array([per_query_ndcg(qrels, {q: pol_runs[p][q] for q in qids}).get(q, 0.0)
                             for q in qids]) for p in policies], axis=1)
    default_j = policies.index(("en", "asr"))

    lang_pick = np.zeros(len(qids), int); chan_pick = np.zeros(len(qids), int)
    Xl = features_for(lang_opts, qids); Xc = features_for(chan_opts, qids)
    for tr, te in splits:
        lang_pick[te] = np.argmax(mk().fit(Xl[tr], Yl[tr]).predict(Xl[te]), axis=1)
        chan_pick[te] = np.argmax(mk().fit(Xc[tr], Yc[tr]).predict(Xc[te]), axis=1)
    chan_names = ["asr", "ocr", "both"]
    composed = np.array([Yj[i, policies.index((LANGS[lang_pick[i]], chan_names[chan_pick[i]]))]
                         for i in range(len(qids))])
    p, obs, lo, hi = group_stats(composed - Yj[:, default_j], grp)
    out["composed_selectors"] = {"ndcg10": float(composed.mean()),
                                 "vs_default": float(100 * (composed.mean() - Yj[:, default_j].mean())),
                                 "p": float(p), "ci95": [float(100 * lo), float(100 * hi)]}
    print(f"[composed] two independent selectors applied together: {composed.mean():.4f} "
          f"({out['composed_selectors']['vs_default']:+.2f} vs English-speech, p={p:.4f})",
          flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
