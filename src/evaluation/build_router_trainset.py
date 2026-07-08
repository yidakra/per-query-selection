"""Build the per-query router training set for the Q2E x Adaptive-RAG mux.

The oracle_router_headroom*.py scripts report only AGGREGATE headroom (mean NDCG, pick_share_%);
the per-query (query -> best tier) labels — the actual training signal for an Adaptive-RAG-style
complexity classifier — were computed transiently and discarded. This script rebuilds them and
persists one row per query with:
  - identity: dataset, encoder, setting, pool, query text, gold video id(s)
  - per-tier metrics: ndcg/r1/r10 for A_visual, -Events(B), Full(C), -Query, -Video
  - LABEL VARIANTS (user chose "emit all, decide later"):
      ndcg_argmax_ABC        : argmax NDCG over the A<B<C ladder (== the reported headroom rule)
      ndcg_argmax_all        : argmax NDCG over all 5 ablation tiers
      cheapest_hit@1         : cheapest ladder tier ranking gold@1   (Adaptive-RAG "cheapest that works")
      cheapest_hit@10        : cheapest ladder tier ranking gold@10
      cheapest_within_eps    : cheapest ladder tier with NDCG >= max_ladder_NDCG - EPS (EPS=0.01)
    ("none" when no tier satisfies the rule -> an unroutable query, recorded honestly.)
  - cheap classifier features: char/word length, dominant unicode script (MultiVENT multilinguality
    is central to the "routing value scales with heterogeneity" thesis).

CPU-only; reads cached component tensors; reuses oracle_router_headroom's EXACT fusion/alignment so
labels are numerically consistent with the published headroom. Does not touch the GPU or the
running LLaMA-1B job. Output: results/ablations/router_trainset.jsonl (+ a printed summary).
"""
import os, sys, json, unicodedata
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# importing this runs its module-level sys.path/chdir(REPO) + heavy imports, but NOT main() (guarded)
from oracle_router_headroom import (  # noqa: E402
    canonical_order, fuse, per_query, TEXT4, ABLATION, LADDER, SETTINGS as MV_SETTINGS)
import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402

RUNS = "/home/ubuntu/q2e_repro/runs"
OUT = "/home/ubuntu/q2e_repro/results/ablations/router_trainset.jsonl"
EPS = 0.01

# tier name in pm  <->  role.  Ladder cost order (cheapest first): A(no captions,no events) <
# B(-Events: captions, no event decomp) < C(Full: captions + event decomp).
LADDER_COST = [("A_visual", "A_visual"), ("B_noevents", "-Events"), ("C_full", "Full")]
ALL_TIER_NAMES = ["A_visual", "-Events", "Full", "-Query", "-Video"]


def dominant_script(s):
    """Cheap language-family proxy: most common unicode script among the query's letters."""
    c = Counter()
    for ch in s:
        if ch.isalpha():
            try:
                name = unicodedata.name(ch)
            except ValueError:
                continue
            fam = name.split(" ")[0]  # LATIN / CYRILLIC / ARABIC / CJK / HANGUL / DEVANAGARI ...
            c[fam] += 1
    return c.most_common(1)[0][0] if c else "NONE"


def label_query(tiers_metric, i):
    """All label variants for query i. tiers_metric[name] = {'ndcg','r1','r10'} tensors over queries."""
    nd = {n: tiers_metric[n]["ndcg"][i].item() for n in ALL_TIER_NAMES}
    r1 = {n: tiers_metric[n]["r1"][i].item() for n in ALL_TIER_NAMES}
    r10 = {n: tiers_metric[n]["r10"][i].item() for n in ALL_TIER_NAMES}
    ladder = [(role, pm_name) for role, pm_name in LADDER_COST]           # cheapest -> dearest
    ladder_nd = {role: nd[pm_name] for role, pm_name in ladder}

    def argmax_over(names):
        return max(names, key=lambda n: nd[n])                            # ties -> first listed

    def cheapest(pred):
        for role, pm_name in ladder:
            if pred(pm_name):
                return role
        return "none"

    max_ladder_nd = max(ladder_nd.values())
    return {
        "ndcg_argmax_ABC": {"A_visual": "A_visual", "-Events": "B_noevents", "Full": "C_full"}[
            argmax_over(["A_visual", "-Events", "Full"])],
        "ndcg_argmax_all": argmax_over(ALL_TIER_NAMES),
        # "hit" = ANY gold in top-k (hit-rate = recall>0). MultiVENT queries are often multi-gold,
        # so recall@1 <= 1/num_gold; requiring full recall would mislabel every multi-gold query a
        # miss. Adaptive-RAG's success signal is "answer found", i.e. at least one relevant retrieved.
        "cheapest_hit@1": cheapest(lambda n: r1[n] > 1e-9),
        "cheapest_hit@10": cheapest(lambda n: r10[n] > 1e-9),
        "cheapest_within_eps": cheapest(lambda n: nd[n] >= max_ladder_nd - EPS),
        "_ndcg": {k: round(v, 4) for k, v in nd.items()},
        "_r1": {k: int(v) for k, v in r1.items()},
        "_r10": {k: int(v) for k, v in r10.items()},
    }


def emit_rows(f, meta, queries, video_ids, target, comps, keep_rows=None, keep_cols=None):
    if keep_cols is not None:
        comps = {k: v[:, keep_cols] for k, v in comps.items()}
        target = target[:, keep_cols]
        video_ids = [video_ids[j] for j in keep_cols]
    if keep_rows is not None:
        comps = {k: v[keep_rows] for k, v in comps.items()}
        target = target[keep_rows]
        queries = [queries[i] for i in range(len(queries)) if keep_rows[i]]
    tiers = {**ABLATION, "A_visual": LADDER["A_visual"]}
    pm = {n: dict(zip(("ndcg", "r1", "r10"), per_query(fuse(comps, p), target)))
          for n, p in tiers.items()}
    n = 0
    for i, q in enumerate(queries):
        gold = [video_ids[j] for j in torch.nonzero(target[i]).flatten().tolist()]
        row = {**meta, "query": q, "gold_video_ids": gold,
               "feat_charlen": len(q), "feat_wordlen": len(q.split()),
               "feat_script": dominant_script(q),
               **label_query(pm, i)}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    return n, pm


def main():
    rows_by = Counter()
    f = open(OUT, "w")

    # ---------- MultiVENT (both pools, both settings), multiclip encoder ----------
    VID_RUN = os.path.join(RUNS, "multivent_video_multiclip")
    qv = torch.load(os.path.join(VID_RUN, "cache", "query_vs_video.pt"))
    order = json.load(open(os.path.join(VID_RUN, "video_order.json")))
    qv_vids = {v: i for i, v in enumerate(order["video_ids"])}
    qv_qs = {q: i for i, q in enumerate(order["queries"])}
    missing = set(order["missing"])
    for setting, (text_tag, ds_name) in MV_SETTINGS.items():
        ds = load_from_disk(f"data/MultiVENT/{ds_name}")
        queries, video_ids, target = canonical_order(ds)
        rmap = torch.tensor([qv_qs[q] for q in queries])
        cmap = torch.tensor([qv_vids[v] for v in video_ids])
        comps = {"query_vs_video": qv[rmap][:, cmap]}
        cdir = os.path.join(RUNS, text_tag, "cache")
        for p in TEXT4:
            comps[p] = torch.load(os.path.join(cdir, f"{p}.pt"))
        avail = [i for i, v in enumerate(video_ids) if v not in missing]
        # full_pool: all queries, full gallery (missing videos absent from gallery already? kept as cols)
        n, _ = emit_rows(f, {"dataset": "MultiVENT", "encoder": "multiclip", "setting": setting,
                             "pool": "full_pool"},
                         list(queries), list(video_ids), target.clone(), dict(comps))
        rows_by[f"MultiVENT/multiclip/{setting}/full_pool"] += n
        # subset: gallery restricted to downloaded videos, drop queries whose gold got removed
        tgt = target.clone()
        keep_cols = avail
        sub_tgt = tgt[:, keep_cols]
        keep_rows = sub_tgt.any(dim=1)
        n, _ = emit_rows(f, {"dataset": "MultiVENT", "encoder": "multiclip", "setting": setting,
                             "pool": "subset"},
                         list(queries), list(video_ids), tgt, dict(comps),
                         keep_rows=keep_rows, keep_cols=keep_cols)
        rows_by[f"MultiVENT/multiclip/{setting}/subset"] += n

    # ---------- MSR-VTT-1kA (both encoders, both settings) ----------
    DS = "data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_{s}"
    for enc in ["multiclip", "internvideo2"]:
        for setting in ["noASR", "ASR"]:
            ds = load_from_disk(DS.format(s=setting))
            queries, video_ids, target = canonical_order(ds)
            cdir = os.path.join(RUNS, f"msrvtt_{enc}_{setting}", "cache")
            comps = {p: torch.load(os.path.join(cdir, f"{p}.pt"))
                     for p in ["query_vs_video"] + TEXT4}
            n, _ = emit_rows(f, {"dataset": "MSR-VTT", "encoder": enc, "setting": setting,
                                 "pool": "full"},
                             list(queries), list(video_ids), target.clone(), dict(comps))
            rows_by[f"MSR-VTT/{enc}/{setting}/full"] += n

    f.close()
    total = sum(rows_by.values())
    print(f"[written] {OUT}  ({total} rows)")
    for k in sorted(rows_by):
        print(f"  {rows_by[k]:5d}  {k}")

    # quick label-distribution sanity on the flagship cell
    print("\n[label distributions] MultiVENT/multiclip/noASR/subset:")
    dist = defaultdict(Counter)
    for line in open(OUT):
        r = json.loads(line)
        if (r["dataset"], r["encoder"], r["setting"], r["pool"]) == ("MultiVENT", "multiclip", "noASR", "subset"):
            for lab in ["ndcg_argmax_ABC", "cheapest_hit@1", "cheapest_hit@10", "cheapest_within_eps"]:
                dist[lab][r[lab]] += 1
    for lab, c in dist.items():
        print(f"  {lab:22s} {dict(c)}")


if __name__ == "__main__":
    main()
