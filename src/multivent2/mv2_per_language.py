"""Per-language nDCG@10, to test whether translating the speech channel closes the gap against MMMORRF.

MMMORRF and OmniEmbed buy their absolute lead partly with translate-distill dense retrieval per channel,
and their reported advantage over an untranslated dense baseline is largest on the non-Latin-script
languages. Our speech channel was retrieved in the original language with a multilingual encoder, so the
obvious question is how much of that gap is the translation step alone. All 109,488 transcripts were
NLLB-translated to English and re-encoded with the same bge-m3 (`mv2_translate_corpus.py`), which holds
everything else fixed: same encoder, same fusion, same queries.

Per-language reading. Queries are English; the language axis belongs to the *videos*, and 22.5% of
queries have relevant videos in more than one language, so assigning each query a single language would
be lossy. Instead the judgments are restricted to one language at a time: for language L, keep only the
judgments whose `video_language` is L, keep the queries that still have a relevant document, and score
the unchanged ranked lists against that reduced qrels. This is the "how well does the system surface
L-language video" subtask, and it is the reading the per-language tables in this benchmark's literature
report. A query can therefore appear in several languages' rows, which is correct: it genuinely has
relevant material in each.

  python src/multivent2/mv2_per_language.py
"""
import os
import sys
import json
import argparse

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_channels import fuse  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

# the channel runs the comparison needs: the visual cheap tier, the speech channel before and after
# translation, and the two fusions built from them
RUNS = {
    "visual":   "10pyscene_clip.json",
    "asr_orig": "asr_dense_bge-m3.json",
    "asr_mt":   "asr_dense_bge-m3-mt.json",
    "ocr":      "ocr_dense_bge-m3.json",
}
MIN_QUERIES = 25          # languages thinner than this are reported but not read as evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_per_language.json"))
    a = ap.parse_args()

    qrels, meta = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    runs = {k: load_run(os.path.join(DATA, v)) for k, v in RUNS.items()}
    qids = sorted(set.intersection(set(qrels), *[set(r) for r in runs.values()]))
    print(f"{len(qids)} queries with all channels present")

    # the two fusions, at the weights each cell's own sweep chose (visual fixed at 1.0)
    policies = dict(runs)
    policies["fuse_orig"] = fuse({"visual": runs["visual"], "asr": runs["asr_orig"]},
                                 {"visual": 1.0, "asr": 1.0}, qids)
    policies["fuse_mt"] = fuse({"visual": runs["visual"], "asr": runs["asr_mt"]},
                               {"visual": 1.0, "asr": 2.0}, qids)

    langs = sorted({m["video_language"] for q in qids for m in meta[q].values() if m["video_language"]})
    out = {"n_queries": len(qids), "languages": {}}

    rows = []
    for lang in langs:
        # restrict the judgments to this language, then keep the queries that still have something to find
        sub = {}
        for q in qids:
            kept = {d: r for d, r in qrels[q].items() if meta[q][d]["video_language"] == lang}
            if any(r > 0 for r in kept.values()):
                sub[q] = kept
        if not sub:
            continue
        cell = {"n_queries": len(sub),
                "n_relevant": int(sum(1 for q in sub for r in sub[q].values() if r > 0))}
        for name, run in policies.items():
            pq = per_query_ndcg(sub, {q: run[q] for q in sub})
            cell[name] = float(np.mean([pq.get(q, 0.0) for q in sub]))
        cell["mt_delta_channel"] = cell["asr_mt"] - cell["asr_orig"]
        cell["mt_delta_fused"] = cell["fuse_mt"] - cell["fuse_orig"]
        out["languages"][lang] = cell
        rows.append((lang, cell))

    rows.sort(key=lambda r: -r[1]["n_queries"])
    hdr = (f"{'language':<12} {'queries':>8} {'visual':>8} {'ASR orig':>9} {'ASR +MT':>9} "
           f"{'Δ chan':>8} {'fuse orig':>10} {'fuse +MT':>9} {'Δ fused':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for lang, c in rows:
        flag = "" if c["n_queries"] >= MIN_QUERIES else "  (thin)"
        print(f"{lang:<12} {c['n_queries']:>8} {c['visual']:>8.4f} {c['asr_orig']:>9.4f} "
              f"{c['asr_mt']:>9.4f} {c['mt_delta_channel']:>+8.4f} {c['fuse_orig']:>10.4f} "
              f"{c['fuse_mt']:>9.4f} {c['mt_delta_fused']:>+8.4f}{flag}")

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
