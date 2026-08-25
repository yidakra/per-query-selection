"""Convert the Task B picks file into run JSONs the nugget pipeline can read.

mv2_variant_selection --save-picks records each query's top-100 documents under each key policy;
this writes one run file per policy (scores are reciprocal ranks, which preserves the ordering,
the only thing the nugget phases use).

  python src/multivent2/mv2_variant_picks_to_runs.py
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")

POLICY_FILES = {
    "sel_pre_QL": "varpol_sel_pre_QL.json",
    "sel_post_NQC_norm": "varpol_sel_post_NQC_norm.json",
    "oracle": "varpol_oracle.json",
    "concat_all": "varpol_concat_all.json",
    "fuse_all": "varpol_fuse_all.json",
}


def main():
    picks = os.path.join(DATA, "variant_picks_full.jsonl")
    runs = {k: {} for k in POLICY_FILES}
    n = 0
    with open(picks) as f:
        for line in f:
            row = json.loads(line)
            for pol in POLICY_FILES:
                docs = row[pol]["docs"]
                runs[pol][row["qid"]] = {d: 1.0 / (i + 1) for i, d in enumerate(docs)}
            n += 1
    for pol, fn in POLICY_FILES.items():
        path = os.path.join(DATA, fn)
        json.dump(runs[pol], open(path, "w"))
        print(f"wrote {path} ({n} queries)")


if __name__ == "__main__":
    main()
