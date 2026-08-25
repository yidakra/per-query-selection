"""Paired tests the generation-axis claim was waiting for: each variant policy against the original.

The metrics phase tests routed against every policy; the replication claim needs each variant
policy against the original query (the asr_dense policy) on the nugget metrics. Same per-query
scoring as the metrics phase, paired sign-flip permutation over the shared judged queries.

  python src/multivent2/mv2_variant_nugget_tests.py
"""
import os
import sys
import json
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_rag_nuggets import score  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
RAG = os.path.join(_ROOT, "results", "ablations", "rag")

BASELINE = "asr_dense"
POLICIES = ["varpol_sel_pre", "varpol_sel_post", "varpol_oracle", "varpol_concat", "varpol_fuse"]
METRICS = ["strict_vital", "vital", "strict_all", "all"]
PERM = 10000


def main():
    per = collections.defaultdict(dict)
    with open(os.path.join(RAG, "assigned_n400_all.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            per[r["policy"]][r["qid"]] = score(r["nuggets"])

    rng = np.random.default_rng(0)
    out = {"baseline": BASELINE, "perm": PERM, "rows": []}
    print(f"paired sign-flip, {PERM} samples, each policy vs {BASELINE}")
    print(f"{'policy':<18}{'metric':<14}{'delta':>9}{'p':>9}  win/tie/loss")
    for pol in POLICIES:
        shared = sorted(set(per[pol]) & set(per[BASELINE]))
        for met in METRICS:
            d = np.array([per[pol][q][met] - per[BASELINE][q][met] for q in shared])
            obs = float(d.mean())
            flips = rng.choice([-1.0, 1.0], size=(PERM, len(d)))
            null = (flips * d).mean(axis=1)
            p = float((1 + np.sum(np.abs(null) >= abs(obs))) / (PERM + 1))
            win = int((d > 1e-9).sum()); loss = int((d < -1e-9).sum())
            out["rows"].append({"policy": pol, "metric": met, "n": len(shared),
                                "delta": obs, "p_twosided": p,
                                "win": win, "tie": len(d) - win - loss, "loss": loss})
            print(f"{pol:<18}{met:<14}{obs:>+9.4f}{p:>9.4f}  "
                  f"{win}/{len(d)-win-loss}/{loss}")

    path = os.path.join(RAG, "variant_vs_original_tests.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
