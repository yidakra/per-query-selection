"""Route-by-language: does the channel choice depend on the query language?

A question from the 2026-08-13 review. The same 2,546 queries exist in English and in the
four largest corpus languages (NLLB translations); each version was retrieved against the same two
re-runnable channels (dense speech, dense OCR). This measures, per query language: each channel's
nDCG@10, the best fixed channel, the per-query oracle over {speech, OCR, both}, and the oracle's
pick distribution. It then splits by video language: for judgments restricted to videos in language
L, does asking in L beat asking in English, and does the winning channel move?

  python src/multivent2/mv2_route_by_language.py
"""
import os
import sys
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

QLANGS = {"en": "", "zh": "_qzh", "ko": "_qko", "ru": "_qru", "ar": "_qar"}
VLANGS = {"english": "en", "chinese": "zh", "korean": "ko", "russian": "ru", "arabic": "ar"}


def rrf(a, b, k=60):
    ra = {d: i + 1 for i, d in enumerate(sorted(a, key=a.get, reverse=True))}
    rb = {d: i + 1 for i, d in enumerate(sorted(b, key=b.get, reverse=True))}
    return {d: 1.0 / (k + ra.get(d, 10**9)) + 1.0 / (k + rb.get(d, 10**9))
            for d in set(ra) | set(rb)}


def main():
    qrels, meta = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    out = {"query_language": {}, "video_language_interaction": {}}

    pq = {}   # (qlang, policy) -> {qid: ndcg}
    for ql, suffix in QLANGS.items():
        asr = load_run(os.path.join(DATA, f"asr_dense_bge-m3{suffix}.json"))
        ocr = load_run(os.path.join(DATA, f"ocr_dense_bge-m3{suffix}.json"))
        qids = sorted(set(asr) & set(ocr) & set(qrels))
        runs = {"asr": asr, "ocr": ocr,
                "both": {q: rrf(asr[q], ocr[q]) for q in qids}}
        for pol, run in runs.items():
            pq[(ql, pol)] = per_query_ndcg(qrels, {q: run[q] for q in qids})

        nd = {pol: np.array([pq[(ql, pol)].get(q, 0.0) for q in qids]) for pol in runs}
        stack = np.stack([nd[p] for p in ("asr", "ocr", "both")])
        oracle = stack.max(axis=0)
        picks = np.argmax(stack, axis=0)
        best_fixed = max(("asr", "ocr", "both"), key=lambda p: nd[p].mean())
        out["query_language"][ql] = {
            "n": len(qids),
            "asr": float(nd["asr"].mean()), "ocr": float(nd["ocr"].mean()),
            "both": float(nd["both"].mean()),
            "best_fixed": best_fixed,
            "oracle": float(oracle.mean()),
            "oracle_minus_best_fixed": float(oracle.mean() - nd[best_fixed].mean()),
            "oracle_picks": {p: int((picks == i).sum())
                             for i, p in enumerate(("asr", "ocr", "both"))}}
        r = out["query_language"][ql]
        print(f"q-lang {ql}: asr {r['asr']:.4f} ocr {r['ocr']:.4f} both {r['both']:.4f} "
              f"| best {best_fixed} | oracle {r['oracle']:.4f} (+{r['oracle_minus_best_fixed']:.4f}) "
              f"| picks {r['oracle_picks']}", flush=True)

    # interaction: judgments restricted to videos in language V, queries asked in English vs in V
    for vname, vshort in VLANGS.items():
        if vshort == "en" or (vshort, "asr") not in pq:
            continue
        sub = {}
        for q in qrels:
            kept = {d: r for d, r in qrels[q].items()
                    if meta[q][d]["video_language"] == vname}
            if any(r > 0 for r in kept.values()):
                sub[q] = kept
        if len(sub) < 25:
            continue
        row = {"n_queries": len(sub)}
        for ql in ("en", vshort):
            asr = load_run(os.path.join(DATA, f"asr_dense_bge-m3{QLANGS[ql]}.json"))
            ocr = load_run(os.path.join(DATA, f"ocr_dense_bge-m3{QLANGS[ql]}.json"))
            qs = sorted(set(sub) & set(asr) & set(ocr))
            a = per_query_ndcg(sub, {q: asr[q] for q in qs})
            o = per_query_ndcg(sub, {q: ocr[q] for q in qs})
            row[f"asr_q{ql}"] = float(np.mean([a.get(q, 0.0) for q in qs]))
            row[f"ocr_q{ql}"] = float(np.mean([o.get(q, 0.0) for q in qs]))
        row["match_delta_asr"] = row[f"asr_q{vshort}"] - row["asr_qen"]
        row["match_delta_ocr"] = row[f"ocr_q{vshort}"] - row["ocr_qen"]
        out["video_language_interaction"][vname] = row
        print(f"videos in {vname:8s} (n={row['n_queries']:4d}): "
              f"asr en {row['asr_qen']:.4f} vs {vshort} {row[f'asr_q{vshort}']:.4f} "
              f"({row['match_delta_asr']:+.4f}) | "
              f"ocr en {row['ocr_qen']:.4f} vs {vshort} {row[f'ocr_q{vshort}']:.4f} "
              f"({row['match_delta_ocr']:+.4f})", flush=True)

    path = os.path.join(ABL, "mv2_route_by_language.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
