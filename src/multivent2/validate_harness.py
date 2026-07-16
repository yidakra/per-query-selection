"""Validate our eval harness against the official evaluator.

Loads the test qrels and the provided `10pyscene_clip` baseline run, computes nDCG@10, and checks it
equals the number the official `evaluate_multivent_test.py` prints for "Overall" (0.3036377966...).
Matching to the last digit proves our qrels parsing, id matching, and metric are identical to the
benchmark's, so anything we build on top is graded the same way.

Data (gitignored) under data/multivent2/. Run:
  CUDA_VISIBLE_DEVICES="" python src/multivent2/validate_harness.py
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_eval import evaluate, evaluate_splits  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
OFFICIAL_NDCG10 = 0.30363779663917656          # official evaluate_multivent_test.py "Overall"


def main():
    qrels, meta = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    run = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    print(f"queries in qrels {len(qrels)} | queries in run {len(run)}")

    m = evaluate(qrels, run)
    for k, v in m.items():
        print(f"  {k:12s} {v:.6f}")

    got = m["nDCG@10"]
    delta = abs(got - OFFICIAL_NDCG10)
    print(f"\nnDCG@10 ours   {got:.17f}")
    print(f"nDCG@10 official {OFFICIAL_NDCG10:.17f}")
    print(f"|delta| = {delta:.2e}  ->  {'MATCH' if delta < 1e-9 else 'MISMATCH'}")

    langs = evaluate_splits(qrels, meta, run, "video_language")
    print("\nper-language nDCG@10 (sanity vs official Russian 0.24734 / Korean 0.07042):")
    for k in ("russian", "korean", "english", "chinese"):
        if k in langs:
            print(f"  {k:9s} {langs[k]:.5f}")

    if delta >= 1e-9:
        raise SystemExit("harness does NOT match official metric")
    print("\nOK: harness matches the official evaluator.")


if __name__ == "__main__":
    main()
