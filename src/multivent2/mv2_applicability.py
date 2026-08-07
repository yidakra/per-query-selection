"""Does modality applicability carry the story it is asked to carry?

The planned paper's mechanism section rests on 10% of videos having no on-screen text: a channel a
document does not have cannot be chosen by any query-side statistic. Review pointed out the mechanism
is asserted, not isolated: the sharpest nulls sit in the ASR cells where absence is 0.2%, and no
experiment separates absence from weakness. Three measurements close that gap.

1. Strata: rerun the corpus-statistic family (symmetric nested protocol) on the subset of queries
   whose relevant documents ALL carry the channel's text. If the null persists there, it is not the
   absence tail.
2. Concentration: does the k-way selector's per-query gain concentrate on queries whose gold set
   includes channel-absent videos? Group-level sign-flip on the strata difference.
3. The availability-flag baseline: per-query fraction of top-ranked visual candidates lacking each
   text channel, used (a) alone as a router and (b) appended to the selector's 30 features. If the
   dumb flag captures a chunk of the +7.59, availability is the signal; if not, the score
   distributions carry it.

  python src/multivent2/mv2_applicability.py
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_queries  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from mv2_qpp_predictors import pre_retrieval_suite, Index, PRE_RETRIEVAL  # noqa: E402
from mv2_qpp_table import route_nested, mk  # noqa: E402
from mv2_channel_select import (load_cell, nested_selection, group_signflip_p)  # noqa: E402
from mv2_io import load_run  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
MARGIN = 5e-4


def text_ids(fn):
    ids = set()
    with open(os.path.join(DATA, fn)) as f:
        for line in f:
            r = json.loads(line)
            if r.get("text", "").strip():
                ids.add(r["doc_id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="mv2_chan_visual_to_ocr_dense_m3.json",
                    help="binary cell for the strata rerun; OCR is the mechanism's cell")
    ap.add_argument("--target", default="ocr", choices=["ocr", "asr"])
    ap.add_argument("--index-text", default="asr_text.jsonl")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_applicability.json"))
    a = ap.parse_args()

    from sklearn.model_selection import GroupKFold

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    ocr_ids = text_ids("ocr_text.jsonl")
    asr_ids = text_ids("asr_text.jsonl")
    print(f"availability: {len(ocr_ids)} docs with OCR text, {len(asr_ids)} with ASR text",
          flush=True)
    have = {"ocr": ocr_ids, "asr": asr_ids}[a.target]

    rel = {}
    for q, docs in qrels.items():
        rel[q] = [d for d, r in docs.items() if r > 0]

    out = {"target": a.target, "cell": a.cell}

    # --- 1. strata rerun of the corpus-statistic family on the all-present subset ---
    cell = json.load(open(os.path.join(ABL, a.cell)))
    qids = [q for q in cell["per_query"] if q in queries]
    all_present = [q for q in qids if rel.get(q) and all(d in have for d in rel[q])]
    print(f"strata: {len(all_present)} of {len(qids)} queries have every relevant doc "
          f"with {a.target} text", flush=True)

    texts = []
    with open(os.path.join(DATA, a.index_text)) as f:
        for line in f:
            t = json.loads(line).get("text", "")
            if t.strip():
                texts.append(t)
    idx = Index(texts)

    sub = all_present
    ndA = np.array([cell["per_query"][q]["ndA"] for q in sub])
    ndB = np.array([cell["per_query"][q]["ndB"] for q in sub])
    g = ndB - ndA
    grp = event_groups(sub, qrels)
    splits = list(GroupKFold(5).split(np.arange(len(sub)), groups=grp))
    orig = max(ndA.mean(), ndB.mean())
    pre_rows = {}
    und = 0
    for n in PRE_RETRIEVAL:
        raw = [pre_retrieval_suite(queries[q].lower().split(), idx)[n] for q in sub]
        (rnd, tau, _, frac, _), _cal = route_nested(raw, g, ndA, ndB, splits, grp, seed=a.seed)
        u = bool(rnd > orig + MARGIN)
        und += u
        pre_rows[n] = {"routed": rnd, "tau": tau, "frac": frac, "underlined": u}
    out["strata_all_present"] = {"n": len(sub), "original": float(orig),
                                 "pre_underlined": int(und), "of": len(PRE_RETRIEVAL),
                                 "rows": pre_rows}
    print(f"strata (all-present, n={len(sub)}): pre-retrieval {und}/{len(PRE_RETRIEVAL)} "
          f"underlined vs Original {orig:.4f}", flush=True)

    # --- 2 and 3 need the k-way cell ---
    channels, names, policies, kq, X, Y, feat_order = load_cell(
        ["asr=asr_dense_bge-m3.json"])
    grp_k = event_groups(kq, qrels)
    splits_k = list(GroupKFold(5).split(X, groups=grp_k))
    ng, ngs, folds, ach_q, fix_q = nested_selection(X, Y, splits=splits_k)
    gain = ach_q - fix_q

    absent_gold = np.array([any(d not in ocr_ids or d not in asr_ids for d in rel.get(q, []))
                            for q in kq])
    m1, m0 = float(gain[absent_gold].mean()), float(gain[~absent_gold].mean())
    strata_diff = np.where(absent_gold, gain, 0).sum() / max(absent_gold.sum(), 1) - \
        np.where(~absent_gold, gain, 0).sum() / max((~absent_gold).sum(), 1)
    p_conc, _ = group_signflip_p(
        np.where(absent_gold, gain - m0, 0.0) * (len(kq) / max(absent_gold.sum(), 1)), grp_k)
    out["concentration"] = {
        "n_absent_gold": int(absent_gold.sum()), "n_present_gold": int((~absent_gold).sum()),
        "mean_gain_absent_gold": m1, "mean_gain_present_gold": m0,
        "note": "gain = per-query nested selector minus best-fixed-on-train, dense k-way cell"}
    print(f"concentration: selector gain {100*m1:+.2f} on queries with a channel-absent gold "
          f"({absent_gold.sum()}) vs {100*m0:+.2f} without ({(~absent_gold).sum()})", flush=True)

    # --- 3. availability flags ---
    visual = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    flags = np.zeros((len(kq), 2))
    for i, q in enumerate(kq):
        top = sorted(visual[q].items(), key=lambda kv: -kv[1])[:a.topk]
        flags[i, 0] = np.mean([d not in ocr_ids for d, _ in top])
        flags[i, 1] = np.mean([d not in asr_ids for d, _ in top])
    ng_f, ngs_f, *_ = nested_selection(flags, Y, splits=splits_k)
    Xa = np.hstack([X, flags])
    ng_a, ngs_a, *_ = nested_selection(Xa, Y, splits=splits_k)
    out["availability_flags"] = {
        "flags_only_nested_gap": ng_f, "flags_only_sem": ngs_f,
        "score_features_nested_gap": ng, "score_features_sem": ngs,
        "score_plus_flags_nested_gap": ng_a, "score_plus_flags_sem": ngs_a}
    print(f"availability flags alone: nested gap {ng_f:+.2f} +/- {ngs_f:.2f}", flush=True)
    print(f"score features:           nested gap {ng:+.2f} +/- {ngs:.2f}", flush=True)
    print(f"score + flags:            nested gap {ng_a:+.2f} +/- {ngs_a:.2f}", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
