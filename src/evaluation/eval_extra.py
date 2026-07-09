"""Per-language (MultiVENT Table 2) and fusion-method (Table 4) reproductions from
cached component score matrices produced by run_eval.py.

Usage:
  python src/evaluation/eval_extra.py per_language --tag multivent_textonly_ASR \
      --dataset data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR
  python src/evaluation/eval_extra.py fusion --tag msrvtt_multiclip_noASR
"""
import argparse, json, os, sys
from collections import defaultdict
import torch, torch.nn.functional as F

OFFICIAL = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "q2e_official"))
sys.path.insert(0, OFFICIAL)
from datasets import load_from_disk  # noqa: E402
os.chdir(OFFICIAL)
from src.eval.evaluation import retrieval_score  # noqa: E402
from src.eval.fusion_score import fusion_inverse_entropy, fusion_exp_entropy, fusion_reciprocal_rank  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
RUNS = f"{_ROOT}/runs"
ALL5 = ["query_vs_video", "query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"]


def load_components(tag):
    cdir = os.path.join(RUNS, tag, "cache")
    comps = {}
    for p in ALL5:
        f = os.path.join(cdir, f"{p}.pt")
        if os.path.exists(f):
            comps[p] = torch.load(f)
    return comps


def rebuild_order(ds_path):
    ds = load_from_disk(ds_path if os.path.isabs(ds_path) else os.path.join(OFFICIAL, ds_path))
    queries, video_ids, q_lang = [], [], {}
    seen_q, seen_v = set(), set()
    for row in ds:
        q = row["query"]
        if q not in seen_q:
            seen_q.add(q); queries.append(q)
            lang = row["metadata"].get("language", "?") if isinstance(row["metadata"], dict) else "?"
            q_lang[q] = lang
        v = row["video_id"]
        if v not in seen_v:
            seen_v.add(v); video_ids.append(v)
    q2vi = defaultdict(list)
    vindex = {v: i for i, v in enumerate(video_ids)}
    for row in ds:
        q2vi[row["query"]].append(vindex[row["video_id"]])
    target = torch.zeros((len(queries), len(video_ids)))
    for i, q in enumerate(queries):
        target[i, q2vi[q]] = 1
    return queries, video_ids, [q_lang[q] for q in queries], target.bool()


def fuse(comps, params, agg):
    res = [F.softmax(comps[p], dim=0) for p in params]
    if agg == "mean": return torch.stack(res, 0).mean(0)
    if agg == "max": return torch.stack(res, 0).max(0).values
    Q, V = res[0].shape
    if agg == "inv_entropy": return fusion_inverse_entropy(None, res, params, range(Q), range(V))
    if agg == "exp_entropy": return fusion_exp_entropy(None, res, params, range(Q), range(V))
    if agg == "rrf": return fusion_reciprocal_rank(None, res, params, range(Q), range(V))
    raise ValueError(agg)


def cmd_per_language(a):
    comps = load_components(a.tag)
    available = [p for p in ALL5 if p in comps]
    queries, video_ids, langs, target = rebuild_order(a.dataset)
    langs_t = langs
    sm = fuse(comps, available, "inv_entropy")  # full Q2E (available comps)
    out = {"tag": a.tag, "components": available, "overall": retrieval_score(sm, target)}
    per = {}
    for L in sorted(set(langs_t)):
        idx = [i for i, l in enumerate(langs_t) if l == L]
        rows = torch.tensor(idx)
        met = retrieval_score(sm[rows], target[rows])
        per[L] = {"n": len(idx), "NDCG": met["NDCG"], "R10": met["R10"], "MRR": met["MRR"], "MAP": met["MAP"]}
    out["per_language"] = per
    os.makedirs(f"{_ROOT}/results/main_tables", exist_ok=True)
    p = f"{_ROOT}/results/main_tables/per_language_{a.tag}.json"
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps(out, indent=2)); print("[written]", p)


def cmd_fusion(a):
    comps = load_components(a.tag)
    available = [p for p in ALL5 if p in comps]
    # need target: reload from the run's metrics meta or dataset. Use dataset if given.
    if a.dataset:
        _, _, _, target = rebuild_order(a.dataset)
    else:
        raise SystemExit("--dataset required for fusion target")
    res = {}
    for agg in ["inv_entropy", "mean", "max", "exp_entropy", "rrf"]:
        sm = fuse(comps, available, agg)
        res[agg] = retrieval_score(sm, target)["NDCG"]
    out = {"tag": a.tag, "components": available, "NDCG_by_fusion": res}
    p = f"{_ROOT}/results/ablations/fusion_{a.tag}.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"), indent=2)
    print(json.dumps(out, indent=2)); print("[written]", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["per_language", "fusion"]:
        s = sub.add_parser(name); s.add_argument("--tag", required=True); s.add_argument("--dataset", default="")
    a = ap.parse_args()
    if a.cmd == "per_language": cmd_per_language(a)
    else: cmd_fusion(a)
