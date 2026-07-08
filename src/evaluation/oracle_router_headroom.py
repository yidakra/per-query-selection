"""Oracle-router headroom for the Q2E x Adaptive-RAG mux.

Framing-independent go/no-go: from the ALREADY-cached MultiVENT component scores, build the
per-query fused ranking under each effort tier, then let a PERFECT per-query router pick the tier
that ranks each query's gold video best. The resulting oracle NDCG/R@k is the CEILING of the whole
adaptive-routing idea (a trained classifier can only approach it). CPU-only; reads caches; does not
touch the GPU or the running LLaMA-1B job.

Effort ladder (Adaptive-RAG A/B/C ported to Q2E):
  A  visual-only      : [query_vs_video]
  B  no-events (-Events): [query_vs_video, query_vs_captions]
  C  full Q2E         : [query_vs_video, query_vs_captions, prequel, during, sequel]
Also reports an oracle over ALL paper ablation tiers (Full/-Query/-Events/-Video) for context.

Reuses the exact fusion path (softmax dim=0 -> inverse-entropy -> min-max) from
fuse_multivent_fullvideo.py. Sanity-asserts that mean per-query NDCG == retrieval_score's aggregate
NDCG for the Full tier before reporting any oracle number.
"""
import os, sys, json
from collections import defaultdict

REPO = "/home/ubuntu/q2e_repro/external/q2e_official"
RUNS = "/home/ubuntu/q2e_repro/runs"
VID_RUN = os.path.join(RUNS, "multivent_video_multiclip")
OUT = "/home/ubuntu/q2e_repro/results/ablations/oracle_router_headroom.json"

sys.path.insert(0, REPO)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from torchmetrics.functional.retrieval import (  # noqa: E402
    retrieval_normalized_dcg, retrieval_recall)
os.chdir(REPO)
from src.eval.evaluation import retrieval_score  # noqa: E402
from src.eval.fusion_score import fusion_inverse_entropy  # noqa: E402

TEXT4 = ["query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"]
# Adaptive-RAG effort ladder (the router chooses among these)
LADDER = {
    "A_visual":  ["query_vs_video"],
    "B_noevents": ["query_vs_video", "query_vs_captions"],
    "C_full":    ["query_vs_video", "query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"],
}
# all paper ablation tiers (for an upper-upper oracle)
ABLATION = {
    "Full":    ["query_vs_video", "query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"],
    "-Query":  ["query_vs_video", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"],
    "-Events": ["query_vs_video", "query_vs_captions"],
    "-Video":  ["query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"],
}
SETTINGS = {
    "noASR": ("multivent_textonly_noASR", "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"),
    "ASR":   ("multivent_textonly_ASR",   "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR"),
}


def canonical_order(ds):
    queries, video_ids, sq, sv = [], [], set(), set()
    q2v = defaultdict(list)
    for row in ds:
        q, v = row["query"], row["video_id"]
        if q not in sq: sq.add(q); queries.append(q)
        if v not in sv: sv.add(v); video_ids.append(v)
    vidx = {v: i for i, v in enumerate(video_ids)}
    for row in ds:
        q2v[row["query"]].append(vidx[row["video_id"]])
    target = torch.zeros((len(queries), len(video_ids)))
    for i, q in enumerate(queries):
        target[i, q2v[q]] = 1
    return queries, video_ids, target.bool()


def fuse(comps, params):
    res = [F.softmax(comps[p], dim=0) for p in params]
    Q, V = res[0].shape
    return fusion_inverse_entropy(None, res, params, range(Q), range(V))


def per_query(fused, target):
    """Return per-query (ndcg10, recall1, recall10) tensors over queries."""
    Q = fused.shape[0]
    nd = torch.zeros(Q); r1 = torch.zeros(Q); r10 = torch.zeros(Q)
    for i in range(Q):
        p, t = fused[i], target[i]
        nd[i] = retrieval_normalized_dcg(p, t, top_k=10)
        r1[i] = retrieval_recall(p, t, top_k=1)
        r10[i] = retrieval_recall(p, t, top_k=10)
    return nd, r1, r10


def tier_matrices(comps, tiers, target):
    """per-tier fused matrix + per-query metric tensors."""
    pm = {}
    for name, params in tiers.items():
        fused = fuse(comps, params)
        nd, r1, r10 = per_query(fused, target)
        pm[name] = {"ndcg": nd, "r1": r1, "r10": r10}
    return pm


def oracle_over(pm, names, sel="ndcg"):
    """Perfect router: per query pick the tier maximizing `sel`; report mean metrics + pick share."""
    stack_nd = torch.stack([pm[n]["ndcg"] for n in names])   # T x Q
    stack_r1 = torch.stack([pm[n]["r1"] for n in names])
    stack_r10 = torch.stack([pm[n]["r10"] for n in names])
    crit = stack_nd if sel == "ndcg" else (stack_r1 if sel == "r1" else stack_r10)
    pick = crit.argmax(dim=0)                                 # Q  (index into names)
    Q = pick.shape[0]
    nd = stack_nd[pick, torch.arange(Q)].mean().item() * 100
    r1 = stack_r1[pick, torch.arange(Q)].mean().item() * 100
    r10 = stack_r10[pick, torch.arange(Q)].mean().item() * 100
    share = {names[t]: round((pick == t).float().mean().item() * 100, 1) for t in range(len(names))}
    return {"NDCG": round(nd, 2), "R1": round(r1, 2), "R10": round(r10, 2), "pick_share_%": share}


def report_setting(comps, target, avail_idx, subset=False):
    if subset:
        keep_cols = avail_idx
        comps = {k: v[:, keep_cols] for k, v in comps.items()}
        target = target[:, keep_cols]
        keep_rows = target.any(dim=1)
        comps = {k: v[keep_rows] for k, v in comps.items()}
        target = target[keep_rows]
    all_tiers = {**ABLATION, "A_visual": LADDER["A_visual"]}  # ABLATION already has -Events(=B) & Full(=C)
    pm = tier_matrices(comps, all_tiers, target)
    # sanity: mean per-query NDCG for Full ~= retrieval_score aggregate
    agg = retrieval_score(fuse(comps, ABLATION["Full"]), target)["NDCG"]
    mine = pm["Full"]["ndcg"].mean().item() * 100
    fixed = {n: {"NDCG": round(pm[n]["ndcg"].mean().item() * 100, 2),
                 "R1": round(pm[n]["r1"].mean().item() * 100, 2),
                 "R10": round(pm[n]["r10"].mean().item() * 100, 2)}
             for n in ["A_visual", "-Events", "Full", "-Query", "-Video"]}
    out = {
        "n_queries": int(target.shape[0]), "n_videos": int(target.shape[1]),
        "sanity_Full_NDCG_agg_vs_perquery": [round(agg, 3), round(mine, 3)],
        "fixed_tiers": fixed,
        "oracle_ABC_ladder": oracle_over(pm, ["A_visual", "-Events", "Full"]),
        "oracle_all_ablation_tiers": oracle_over(pm, ["Full", "-Query", "-Events", "-Video"]),
    }
    return out


def main():
    qv = torch.load(os.path.join(VID_RUN, "cache", "query_vs_video.pt"))
    order = json.load(open(os.path.join(VID_RUN, "video_order.json")))
    qv_vids = {v: i for i, v in enumerate(order["video_ids"])}
    qv_qs = {q: i for i, q in enumerate(order["queries"])}
    missing = set(order["missing"])
    result = {"missing_videos": len(missing), "settings": {}}

    for setting, (text_tag, ds_name) in SETTINGS.items():
        ds = load_from_disk(f"data/MultiVENT/{ds_name}")
        queries, video_ids, target = canonical_order(ds)
        rmap = torch.tensor([qv_qs[q] for q in queries])
        cmap = torch.tensor([qv_vids[v] for v in video_ids])
        qv_aligned = qv[rmap][:, cmap]
        cdir = os.path.join(RUNS, text_tag, "cache")
        comps = {"query_vs_video": qv_aligned}
        for p in TEXT4:
            comps[p] = torch.load(os.path.join(cdir, f"{p}.pt"))
        avail_idx = [i for i, v in enumerate(video_ids) if v not in missing]
        result["settings"][setting] = {
            "full_pool": report_setting(dict(comps), target.clone(), avail_idx, subset=False),
            "subset":    report_setting(dict(comps), target.clone(), avail_idx, subset=True),
        }
        fp = result["settings"][setting]["full_pool"]
        print(f"[{setting}] full-pool  fixed-Full={fp['fixed_tiers']['Full']['NDCG']}  "
              f"oracle-ABC={fp['oracle_ABC_ladder']['NDCG']}  "
              f"oracle-all={fp['oracle_all_ablation_tiers']['NDCG']}  "
              f"(sanity {fp['sanity_Full_NDCG_agg_vs_perquery']})")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print("\n" + json.dumps(result, indent=2))
    print("[written]", OUT)


if __name__ == "__main__":
    main()
