"""Oracle-router headroom on MSR-VTT-1kA (both encoders) — broadens the frontier claim to a second,
more single-event dataset. Same method as oracle_router_headroom.py (MultiVENT): perfect per-query
router over the A/B/C effort ladder vs fixed-Full. CPU-only; reads caches; no GPU.

MSR-VTT is simpler than MultiVENT: no missing videos, all 5 components cached in one run dir (already
in the same canonical order), so no video alignment / pool policy needed.
"""
import os, sys, json
from collections import defaultdict

REPO = "/home/ubuntu/q2e_repro/external/q2e_official"
RUNS = "/home/ubuntu/q2e_repro/runs"
OUT = "/home/ubuntu/q2e_repro/results/ablations/oracle_router_headroom_msrvtt.json"

sys.path.insert(0, REPO)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from torchmetrics.functional.retrieval import retrieval_normalized_dcg, retrieval_recall  # noqa: E402
os.chdir(REPO)
from src.eval.evaluation import retrieval_score  # noqa: E402
from src.eval.fusion_score import fusion_inverse_entropy  # noqa: E402

ALL5 = ["query_vs_video", "query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"]
LADDER = {
    "A_visual":  ["query_vs_video"],
    "-Events":   ["query_vs_video", "query_vs_captions"],
    "Full":      ALL5,
}
ABLATION = {
    "Full":    ALL5,
    "-Query":  ["query_vs_video", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"],
    "-Events": ["query_vs_video", "query_vs_captions"],
    "-Video":  ["query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"],
}
ENCODERS = ["multiclip", "internvideo2"]
SETTINGS = {"noASR": "noASR", "ASR": "ASR"}
DS = "data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_{s}"


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
    Q = fused.shape[0]
    nd = torch.zeros(Q); r1 = torch.zeros(Q); r10 = torch.zeros(Q)
    for i in range(Q):
        p, t = fused[i], target[i]
        nd[i] = retrieval_normalized_dcg(p, t, top_k=10)
        r1[i] = retrieval_recall(p, t, top_k=1)
        r10[i] = retrieval_recall(p, t, top_k=10)
    return nd, r1, r10


def tier_metrics(comps, tiers, target):
    pm = {}
    for name, params in tiers.items():
        nd, r1, r10 = per_query(fuse(comps, params), target)
        pm[name] = {"ndcg": nd, "r1": r1, "r10": r10}
    return pm


def oracle_over(pm, names):
    stack_nd = torch.stack([pm[n]["ndcg"] for n in names])
    stack_r1 = torch.stack([pm[n]["r1"] for n in names])
    stack_r10 = torch.stack([pm[n]["r10"] for n in names])
    pick = stack_nd.argmax(dim=0)
    Q = pick.shape[0]
    idx = torch.arange(Q)
    share = {names[t]: round((pick == t).float().mean().item() * 100, 1) for t in range(len(names))}
    return {"NDCG": round(stack_nd[pick, idx].mean().item() * 100, 2),
            "R1": round(stack_r1[pick, idx].mean().item() * 100, 2),
            "R10": round(stack_r10[pick, idx].mean().item() * 100, 2),
            "pick_share_%": share}


def main():
    result = {}
    for enc in ENCODERS:
        result[enc] = {}
        for setting, s in SETTINGS.items():
            ds = load_from_disk(DS.format(s=s))
            queries, video_ids, target = canonical_order(ds)
            cdir = os.path.join(RUNS, f"msrvtt_{enc}_{s}", "cache")
            comps = {p: torch.load(os.path.join(cdir, f"{p}.pt")) for p in ALL5}
            tiers = {**ABLATION, "A_visual": LADDER["A_visual"]}
            pm = tier_metrics(comps, tiers, target)
            agg = retrieval_score(fuse(comps, ALL5), target)["NDCG"]
            mine = pm["Full"]["ndcg"].mean().item() * 100
            fixed = {n: round(pm[n]["ndcg"].mean().item() * 100, 2)
                     for n in ["A_visual", "-Events", "Full", "-Query", "-Video"]}
            result[enc][setting] = {
                "n_queries": int(target.shape[0]), "n_videos": int(target.shape[1]),
                "sanity_Full_NDCG_agg_vs_perquery": [round(agg, 3), round(mine, 3)],
                "fixed_tiers_NDCG": fixed,
                "oracle_ABC_ladder": oracle_over(pm, ["A_visual", "-Events", "Full"]),
                "oracle_all_ablation_tiers": oracle_over(pm, ["Full", "-Query", "-Events", "-Video"]),
            }
            r = result[enc][setting]
            print(f"[{enc}/{setting}] fixed-Full={fixed['Full']}  "
                  f"oracle-ABC={r['oracle_ABC_ladder']['NDCG']} "
                  f"(+{round(r['oracle_ABC_ladder']['NDCG']-fixed['Full'],2)})  "
                  f"pick={r['oracle_ABC_ladder']['pick_share_%']}  sanity={r['sanity_Full_NDCG_agg_vs_perquery']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(result, open(OUT, "w"), indent=2)
    print("[written]", OUT)


if __name__ == "__main__":
    main()
