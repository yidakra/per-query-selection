"""Fuse the MultiVENT `query_vs_video` component with the cached text components to produce the
full-video MultiVENT numbers: the Table-5 component ablation (Full / -Query / -Events / -Video)
for noASR and ASR, under two missing-video policies:

  (b) full pool  : all 2,393 videos; the ~400 unrecoverable ones carry a null (zeroed) video
                   column -> uniform after the dim=0 softmax. Honest pool size (PRIMARY).
  (a) subset     : restrict to the recovered videos, and drop queries whose gold target is gone.
                   Smaller candidate pool (easier); reported alongside for context.

Video signal is shared across noASR/ASR; text components come from each setting's cached run.
Alignment between the video component and each text run is by video_id / query string (robust to
row-order differences between the noASR and ASR datasets).

Writes results/ablations/table5_multivent_fullvideo.json.
"""
import os, sys, json
from collections import defaultdict

REPO = "/home/ubuntu/q2e_repro/external/q2e_official"
RUNS = "/home/ubuntu/q2e_repro/runs"
VID_RUN = os.path.join(RUNS, "multivent_video_multiclip")
OUT = "/home/ubuntu/q2e_repro/results/ablations/table5_multivent_fullvideo.json"

sys.path.insert(0, REPO)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_from_disk  # noqa: E402
os.chdir(REPO)
from src.eval.evaluation import retrieval_score  # noqa: E402
from src.eval.fusion_score import fusion_inverse_entropy  # noqa: E402

TEXT4 = ["query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"]
SUBSETS = {  # name -> component list
    "Full":     ["query_vs_video", "query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"],
    "-Query":   ["query_vs_video", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"],
    "-Events":  ["query_vs_video", "query_vs_captions"],
    "-Video":   ["query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"],
}
PAPER = {  # MultiCLIP MultiVENT Table 5 (noASR, ASR)
    "Full":    (80.04, 83.24), "-Query": (78.78, 81.54),
    "-Events": (79.02, 81.75), "-Video": (64.83, 73.96),
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


def metrics_for(comps, params, target):
    return retrieval_score(fuse(comps, params), target)


def main():
    qv = torch.load(os.path.join(VID_RUN, "cache", "query_vs_video.pt"))
    order = json.load(open(os.path.join(VID_RUN, "video_order.json")))
    qv_vids = {v: i for i, v in enumerate(order["video_ids"])}
    qv_qs = {q: i for i, q in enumerate(order["queries"])}
    missing = set(order["missing"])
    out = {"missing_videos": len(missing), "settings": {}}

    for setting, (text_tag, ds_name) in SETTINGS.items():
        ds = load_from_disk(f"data/MultiVENT/{ds_name}")
        queries, video_ids, target = canonical_order(ds)
        # sanity: same video/query sets as the video component
        assert set(video_ids) == set(qv_vids), f"{setting}: video set mismatch"
        assert set(queries) == set(qv_qs), f"{setting}: query set mismatch"
        # align query_vs_video rows/cols to this run's order
        rmap = torch.tensor([qv_qs[q] for q in queries])
        cmap = torch.tensor([qv_vids[v] for v in video_ids])
        qv_aligned = qv[rmap][:, cmap]
        # load the 4 cached text components (already in this run's canonical order)
        cdir = os.path.join(RUNS, text_tag, "cache")
        comps = {"query_vs_video": qv_aligned}
        for p in TEXT4:
            comps[p] = torch.load(os.path.join(cdir, f"{p}.pt"))
        avail_idx = [i for i, v in enumerate(video_ids) if v not in missing]

        res = {"full_pool": {}, "subset": {}, "n_videos_full": len(video_ids),
               "n_videos_subset": len(avail_idx)}
        for name, params in SUBSETS.items():
            # (b) full pool
            m = metrics_for(comps, params, target)
            res["full_pool"][name] = {"NDCG": round(m["NDCG"], 2), "R1": round(m["R1"], 2),
                                      "R10": round(m["R10"], 2), "MRR": round(m["MRR"], 2),
                                      "paper_NDCG": PAPER[name][0 if setting == "noASR" else 1]}
            # (a) subset: restrict cols to available, drop queries with no available gold
            sub = {p: comps[p][:, avail_idx] for p in params}
            tgt_sub = target[:, avail_idx]
            keep = tgt_sub.any(dim=1)
            sub = {p: sub[p][keep] for p in params}
            ms = retrieval_score(fuse(sub, params), tgt_sub[keep])
            res["subset"][name] = {"NDCG": round(ms["NDCG"], 2), "R1": round(ms["R1"], 2),
                                   "R10": round(ms["R10"], 2), "n_queries": int(keep.sum())}
        out["settings"][setting] = res
        print(f"[{setting}] full-pool Full NDCG={res['full_pool']['Full']['NDCG']} "
              f"(paper {PAPER['Full'][0 if setting=='noASR' else 1]}) | "
              f"subset Full NDCG={res['subset']['Full']['NDCG']} over {res['n_videos_subset']} vids")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print("\n" + json.dumps(out, indent=2))
    print("[written]", OUT)


if __name__ == "__main__":
    main()
