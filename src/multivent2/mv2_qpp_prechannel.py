"""Can pre-retrieval QPP select a *channel*? The per-channel index test.

Arabzadeh et al. (2026, arXiv:2604.22661) find cheap pre-retrieval predictors competitive for choosing
among query variants. Our RQ2 table finds the same family at tau ~ 0. The related-work section explains
the difference structurally: in variant selection the options ARE different query texts, so a query-side
feature moves as the choice moves; in channel selection the query is fixed, so IDF or SCS is one number
per query and takes the same value whichever channel is under consideration. Our own table is built that
way -- one lexical index, over ASR (`mv2_qpp_table.py`) -- so it cannot distinguish "pre-retrieval QPP
has no signal here" from "we only gave it one index".

This script gives it every index the benchmark allows, which is the strongest form of the repair:

  asr   lexical index over the speech transcripts        (the speech channel's own documents)
  ocr   lexical index over the on-screen text            (the OCR channel's own documents)
  cap   lexical index over the shipped qwen captions     (a SURROGATE for the visual channel)

The caption index is the interesting one and it needs stating plainly. The visual channel retrieves by
CLIP similarity over frames; it has no term index of its own. The captions describe the same videos in
text, so an index over them is a proxy for what the visual channel searches, not the thing itself. If
per-channel pre-retrieval features work once that proxy is supplied, the structural argument is wrong
and the honest finding is that pre-retrieval QPP reaches a non-text channel through a text surrogate. If
they still fail, the argument holds having been tested at full strength.

Decision metric is the one the paper reports: nested selection gap over the 7-policy k-way cell, folds
grouped by event, best fixed policy chosen on the training fold.

  python src/multivent2/mv2_qpp_prechannel.py
"""
import os
import sys
import json
import argparse

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_queries  # noqa: E402
from mv2_qpp_predictors import pre_retrieval_suite, Index  # noqa: E402
from mv2_channel_select import load_cell, nested_selection  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

# channel key -> the text file whose documents that channel retrieves over.
# "cap" is a surrogate: the visual channel searches frame embeddings, not these captions.
TEXT = {"asr": "asr_text.jsonl",
        "ocr": "ocr_text.jsonl",
        "cap": "qwen_captions_test.jsonl"}


def build_index(fn):
    texts = []
    with open(os.path.join(DATA, fn)) as f:
        for line in f:
            t = json.loads(line).get("text", "").strip()
            if t:
                texts.append(t)
    idx = Index(texts)
    print(f"  {fn}: {idx.n_docs} docs, {idx.total_terms} tokens, {len(idx.df)} terms", flush=True)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", action="append", default=["asr=asr_dense_bge-m3.json"],
                    metavar="NAME=FILE")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_qpp_prechannel.json"))
    ap.add_argument("--single", action="store_true",
                    help="also run every single predictor alone through the same 7-policy learner, "
                         "with group sign-flip tests (Holm within the pre and the post family)")
    a = ap.parse_args()

    channels, names, policies, qids, Xconf, Y, feat_order = load_cell(a.channel)
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    print(f"cell: {names}  queries: {len(qids)}  policies: {len(policies)}  conf features: {Xconf.shape[1]}")

    print("building lexical indices...")
    idxs = {k: build_index(fn) for k, fn in TEXT.items()}

    # pre-retrieval features, one block per index
    blocks, pre_names = {}, None
    for k, idx in idxs.items():
        rows = []
        for q in qids:
            s = pre_retrieval_suite(queries[q].lower().split(), idx)
            if pre_names is None:
                pre_names = sorted(s)
            rows.append([s[n] for n in pre_names])
        blocks[k] = np.asarray(rows, dtype=np.float64)
        print(f"  {k}: {blocks[k].shape[1]} predictors x {len(qids)} queries")

    from mv2_qsd import event_groups
    grp = event_groups(qids, qrels)
    print(f"grouped CV: {len(set(grp))} event groups over {len(qids)} queries")
    splits = list(GroupKFold(5).split(Xconf, groups=grp))

    # how much does a query-side feature actually move across channels? If the answer is "not at all"
    # for the ASR-only setup, that is the structural claim measured rather than asserted.
    spread = {}
    for j, n in enumerate(pre_names):
        vals = np.stack([blocks[k][:, j] for k in ("asr", "ocr", "cap")], axis=1)
        rng = vals.max(axis=1) - vals.min(axis=1)
        denom = np.abs(vals).mean(axis=1) + 1e-9
        spread[n] = float(np.mean(rng / denom))

    sets = {
        "conf30 (paper baseline)":            Xconf,
        "pre @ asr only (current table)":     blocks["asr"],
        "pre @ asr+ocr (text channels)":      np.hstack([blocks["asr"], blocks["ocr"]]),
        "pre @ asr+ocr+cap (all, surrogate)": np.hstack([blocks[k] for k in ("asr", "ocr", "cap")]),
        "conf30 + pre @ all":                 np.hstack([Xconf] + [blocks[k] for k in ("asr", "ocr", "cap")]),
    }

    out = {"n_queries": len(qids), "policies": policies, "channels": channels,
           "n_event_groups": len(set(grp)), "pre_predictors": pre_names,
           "cross_channel_spread": spread, "sets": {}}
    print(f"\n{'feature set':<38} {'nested gap':>12} {'sem':>7}  {'features':>9}")
    print("-" * 72)
    from mv2_row_inference import group_stats, holm
    grp_arr = np.asarray(grp)
    for label, X in sets.items():
        gap, sem, folds, ach_q, fix_q = nested_selection(X, Y, splits=splits)
        p, _, lo, hi = group_stats(ach_q - fix_q, grp_arr)
        out["sets"][label] = {"nested_gap": gap, "nested_sem": sem, "n_features": int(X.shape[1]),
                             "achieved": float(ach_q.mean()), "fixed": float(fix_q.mean()),
                             "p_signflip_group": p, "ci95": [lo, hi], "folds": folds}
        print(f"{label:<38} {gap:>+11.2f} {sem:>7.2f}  {X.shape[1]:>9}  p={p:.4f}")

    if a.single:
        # One predictor at a time, given to the same learner as its value on each channel: three
        # columns per predictor (one per channel index for pre-retrieval, one per channel ranking
        # for post-retrieval), so the single-predictor rows differ from the ridge rows only in
        # how many signals the learner sees.
        fams = {"pre": {n: np.stack([blocks[k][:, j] for k in ("asr", "ocr", "cap")], axis=1)
                        for j, n in enumerate(pre_names)},
                "post": {}}
        for k in sorted({lab.split(":", 1)[1] for lab in feat_order if not lab.startswith("overlap")}):
            cols = [feat_order.index(f"{n}:{k}") for n in names]
            fams["post"][k] = Xconf[:, cols]
        for kk in (10, 100):
            cols = [i for i, lab in enumerate(feat_order) if lab.startswith(f"overlap{kk}:")]
            fams["post"][f"overlap{kk}"] = Xconf[:, cols]
        out["single"] = {}
        for fam, preds in fams.items():
            rows = {}
            for n, X in preds.items():
                gap, sem, _, ach_q, fix_q = nested_selection(X, Y, splits=splits)
                p, _, lo, hi = group_stats(ach_q - fix_q, grp_arr)
                rows[n] = {"nested_gap": gap, "nested_sem": sem, "achieved": float(ach_q.mean()),
                           "fixed": float(fix_q.mean()), "p_signflip_group": p, "ci95": [lo, hi]}
            adj = holm(np.array([r["p_signflip_group"] for r in rows.values()]))
            for r, ap_ in zip(rows.values(), adj):
                r["p_holm"] = float(ap_)
            out["single"][fam] = rows
            best = max(rows, key=lambda n: rows[n]["nested_gap"])
            print(f"\nsingle {fam}-retrieval predictors ({len(rows)}), best = {best}:")
            for n in sorted(rows, key=lambda n: -rows[n]["nested_gap"]):
                r = rows[n]
                print(f"  {n:<14} gap {r['nested_gap']:+6.2f}  achieved {r['achieved']:.4f}  "
                      f"p {r['p_signflip_group']:.4f}  holm {r['p_holm']:.4f}")

    print("\ncross-channel spread of each pre-retrieval predictor "
          "(mean relative range across the three indices; 0 = same value whichever channel):")
    for n in sorted(spread, key=lambda x: -spread[x]):
        print(f"  {n:<12} {spread[n]:.3f}")

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
