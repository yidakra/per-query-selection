"""nDCG@10 on MultiVENT 2.0, matching the official `evaluate_multivent_test.py` "Overall".

The official script runs `ir_measures.calc_aggregate([... nDCG@10 ...], qrels, run)` on the full
qrels and raw run. We compute the same, plus the recall/Judged numbers, and expose the per-split
breakdown (video_type / language / event_type) the benchmark reports.
"""
import ir_measures
from ir_measures import nDCG, R, AP, RR, Judged

METRICS = [nDCG @ 10, Judged @ 10, R @ 10, R @ 100, R @ 1000, AP, RR]


def evaluate(qrels, run):
    res = ir_measures.calc_aggregate(METRICS, qrels, run)
    return {str(m): float(res[m]) for m in METRICS}


def ndcg10(qrels, run):
    return float(ir_measures.calc_aggregate([nDCG @ 10], qrels, run)[nDCG @ 10])


def evaluate_splits(qrels, meta, run, field):
    """Per-split nDCG@10 for field in {video_language, video_type, video_modality, query_event_type}.

    Replicates the official split protocol exactly: (1) keep only judgments whose field == value;
    (2) for each query kept, find its videos that are relevant (rel>0) but belong to a *different*
    field value, and DELETE those from the run before scoring -- otherwise a video relevant under
    another split would rank high and unfairly depress this split's nDCG.
    """
    names = {meta[qid][did][field] for qid in meta for did in meta[qid] if meta[qid][did][field]}
    out = {}
    for name in sorted(names):
        sub = {qid: {did: rel for did, rel in qrels[qid].items() if meta[qid][did][field] == name}
               for qid in qrels}
        sub = {qid: d for qid, d in sub.items() if d}
        irrelevant = {qid: {did for did, rel in qrels[qid].items()
                            if did not in sub.get(qid, {}) and rel > 0}
                      for qid in sub}
        pruned = {qid: {did: s for did, s in run.get(qid, {}).items()
                        if did not in irrelevant.get(qid, set())}
                  for qid in sub}
        out[name] = ndcg10(sub, pruned)
    return out
