"""Audit the nested-calibration QSD and BERT-QPP artifacts without training models.

Checks finite predictions, decision-vector alignment, exact nDCG/Recall reconstruction, and disjoint
event groups across model-fit, calibration, and outer-test partitions.

  python src/multivent2/mv2_nested_audit.py
"""
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mv2_io import load_qrels  # noqa: E402
from mv2_nested_calibration import calibration_split  # noqa: E402
from mv2_qsd import ABL, CELLS, DATA, event_groups  # noqa: E402
from mv2_recall_sidecar import load_cell_recall  # noqa: E402

ARTIFACTS = {
    "QSD_pre": "mv2_qsd_pre_nested_grouped.json",
    "QSD_post": "mv2_qsd_post_5ep_nested_grouped.json",
    "BERT_cross": "mv2_bertqpp_cross_3ep_nested_grouped.json",
    "BERT_bi": "mv2_bertqpp_bi_3ep_nested_grouped.json",
}


def main():
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    for method, filename in ARTIFACTS.items():
        path = os.path.join(ABL, filename)
        raw = open(path, "rb").read()
        artifact = json.loads(raw)
        print(f"{method:<11} sha256={hashlib.sha256(raw).hexdigest()}")
        for cell, cell_filename in CELLS.items():
            source = json.load(open(os.path.join(ABL, cell_filename)))
            outer = artifact[cell]
            qids = outer["qids"]
            row = outer["k"]["100_inv_dist"] if method == "QSD_pre" else outer
            pred = np.asarray([row["pred"][qid] for qid in qids], dtype=float)
            decisions = np.asarray([bit == "1" for bit in row["decisions"]])
            if len(pred) != len(qids) or len(decisions) != len(qids) or not np.isfinite(pred).all():
                raise RuntimeError(f"{method}/{cell}: invalid prediction or decision vector")

            nd_a = np.asarray([source["per_query"][q]["ndA"] for q in qids])
            nd_b = np.asarray([source["per_query"][q]["ndB"] for q in qids])
            routed = float(np.where(decisions, nd_b, nd_a).mean())
            rec_a, rec_b = load_cell_recall(cell, qids)
            recall = float(np.where(decisions, rec_b, rec_a).mean())
            if abs(routed - row["routed_ndcg10"]) > 1e-12:
                raise RuntimeError(f"{method}/{cell}: routed nDCG does not reproduce")
            if abs(recall - row["routed_recall100"]) > 1e-12:
                raise RuntimeError(f"{method}/{cell}: routed recall does not reproduce")

            groups = event_groups(qids, qrels)
            splits = list(GroupKFold(5).split(np.arange(len(qids)), groups=groups))
            for fold, (train, test) in enumerate(splits):
                fit, calibration = calibration_split(train, groups, 0.2, fold)
                group_sets = [set(groups[x]) for x in (fit, calibration, test)]
                if any(group_sets[i] & group_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                    raise RuntimeError(f"{method}/{cell}/fold{fold}: event-group overlap")
                metadata = row["nested_calibration"][fold]
                if metadata["n_fit"] != len(fit) or metadata["n_calibration"] != len(calibration):
                    raise RuntimeError(f"{method}/{cell}/fold{fold}: split-size metadata mismatch")
            print(f"  {cell:<12} exact nDCG/recall; finite; 5 disjoint nested folds")


if __name__ == "__main__":
    main()
