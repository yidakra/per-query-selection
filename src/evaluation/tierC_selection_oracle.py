#!/usr/bin/env python
"""Tier C, part 1: is GENERATION or SELECTION the bottleneck?

Q2E scores an event component by max-pooling similarity over the LLM's paraphrases.  A max-pool
means a bad paraphrase can only ever RAISE a wrong video's score -- paraphrases are monotone
noise injection.  So there are two rival explanations for why the Full tier underperforms and
why its per-query gain is unpredictable:

  (H1) GENERATION is the bottleneck -- the events are wrong; we need evidence-conditioned
       regeneration (an iterative retrieve->refine->re-retrieve loop, which can DRIFT).
  (H2) SELECTION is the bottleneck -- the right events are already in the generated set, but
       max-pool drowns them in noise.  Fix = prune, no LLM, no drift.

This script measures the ceiling of H2 exactly: the per-query oracle over SUBSETS of the
already-generated paraphrases.  If that ceiling is large, tier C should be a paraphrase
SELECTOR, not a regeneration loop.

Method.  `fuse` applies softmax over dim=0 (across QUERIES, per video column), so changing one
query's paraphrase subset perturbs every other query's scores by O(1/T).  Greedy search is run
with the softmax denominators FROZEN at their full-paraphrase values -- which makes per-query
nDCG exactly separable and each candidate evaluation O(V) -- and the reported number is then
recomputed EXACTLY with the unfrozen fusion over all selected subsets.  Approximate in the
search, exact in the result.

min_max_normalize is a global monotone map, so it cannot change any per-row ranking and is
irrelevant to nDCG; we skip it inside the search and apply the real `fuse` at the end.

CPU-only.  Requires runs/<tag>/paracache/*.pt from perparaphrase_scores.py.
"""
import os, sys, json, argparse, numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oracle_router_headroom import canonical_order, fuse, per_query, LADDER  # noqa: E402
from tracking import track  # noqa: E402
from datasets import load_from_disk  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
REPO = _ROOT
RUNS = f"{REPO}/runs"
EVENTS = ["prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"]
ALL5 = ["query_vs_video", "query_vs_captions"] + EVENTS
SETTINGS = {
    "noASR": ("multivent_textonly_noASR", "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"),
    "ASR": ("multivent_textonly_ASR", "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR"),
}


def ndcg10(scores, rel):
    """nDCG@10 for one query. scores, rel: (V,) numpy. rel is 0/1. Matches torchmetrics."""
    k = min(10, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    gains = rel[top]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains * disc).sum())
    n = int(rel.sum())
    idcg = float(disc[: min(k, n)].sum()) if n else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def load_cell(setting):
    tag, ds_name = SETTINGS[setting]
    ds = load_from_disk(f"{REPO}/data/MultiVENT/{ds_name}")
    queries, video_ids, target = canonical_order(ds)

    comps = {}
    cdir = f"{RUNS}/{tag}/cache"
    comps["query_vs_captions"] = torch.load(f"{cdir}/query_vs_captions.pt").float()
    for e in EVENTS:
        comps[e] = torch.load(f"{cdir}/{e}.pt").float()

    VID = f"{RUNS}/multivent_video_multiclip"
    qv = torch.load(f"{VID}/cache/query_vs_video.pt").float()
    order = json.load(open(f"{VID}/video_order.json"))
    qi = {q: i for i, q in enumerate(order["queries"])}
    vi = {v: i for i, v in enumerate(order["video_ids"])}
    rmap = torch.tensor([qi[q] for q in queries]); cmap = torch.tensor([vi[v] for v in video_ids])
    comps["query_vs_video"] = qv[rmap][:, cmap]

    P, valid = {}, {}
    pdir = f"{RUNS}/{tag}/paracache"
    recon = dict(comps)
    for e in EVENTS:
        d = torch.load(f"{pdir}/{e}.pt")
        assert d["queries"] == list(queries), f"query order mismatch in {e}"
        assert d["video_ids"] == list(video_ids), f"video order mismatch in {e}"
        P[e] = d["P"].float(); valid[e] = d["valid"]
        recon[e] = P[e].max(dim=1).values
        md = (recon[e] - comps[e]).abs().max().item()
        print(f"  [{e}] max|paracache.max - cached| = {md:.3e}")

    # ColBERT is fp16; re-scoring under a different batch composition perturbs scores at ~1e-1 on a
    # 0-100 scale, so `recon` is not bitwise equal to the published `comps`. The gate that matters
    # is not float equality but METRIC equality.
    nd_cached = per_query(fuse(comps, LADDER["C_full"]), target)[0].mean() * 100
    nd_recon = per_query(fuse(recon, LADDER["C_full"]), target)[0].mean() * 100
    delta = abs(float(nd_cached - nd_recon))
    print(f"[{setting}] Full-tier nDCG: published {nd_cached:.4f} | from paracache {nd_recon:.4f} "
          f"| delta {delta:.4f}")
    # A LOGIC error (wrong axis, wrong pooling, misaligned queries) would move nDCG by whole points.
    # fp16 batch noise moves it by hundredths. 0.10 separates those regimes by an order of magnitude
    # on either side.
    if delta > 0.10:
        raise SystemExit(f"paracache changes the published Full-tier nDCG by {delta:.4f} "
                         f"(> 0.10). That is too large to be fp16 batch noise. Refusing to build "
                         f"an oracle on it.")

    # CRITICAL: return `recon`, not `comps`, as the event components. The oracle reports a
    # DIFFERENCE, nDCG(best subset) - nDCG(all paraphrases). Selecting every paraphrase reproduces
    # `recon` exactly, so `recon` is the true "all paraphrases" arm. Baselining against the
    # published `comps` instead would fold the {delta:.4f} fp16 reconstruction shift into the
    # reported gain -- a real contaminant when the effect we are trying to measure may itself be
    # under a point of nDCG. Using `recon` on both arms makes that noise cancel identically.
    # We keep `nd_cached` only to report the reconstruction delta.
    for e in EVENTS:
        comps[e] = recon[e]
    print(f"[{setting}] using paracache-reconstructed event components on BOTH arms "
          f"(fp16 delta {delta:.4f} cancels in the gain)")
    return queries, video_ids, target, comps, P, valid, nd_cached, delta


def frozen_softmax_parts(comps):
    """Return per-component log-denominators D[p][v] = logsumexp over queries, frozen."""
    return {p: torch.logsumexp(comps[p], dim=0) for p in ALL5}   # (V,)


def fused_row(rows, D):
    """Replicate fuse() for a SINGLE query with frozen softmax denominators.
    rows: dict comp -> (V,) raw score row. Returns (V,) fused score row (pre min-max)."""
    out = None
    for p, x in rows.items():
        s = torch.exp(x - D[p])                       # softmax(dim=0) for this row
        ent = -(s * torch.log2(s + 1e-6)).sum()       # entropy(), dim=-1
        w = 1.0 / (ent + 1e-6)
        out = w * s if out is None else out + w * s
    return out


def greedy_select(q, P, valid, comps, D, target, max_steps=None):
    """Greedy forward selection of paraphrases (across all 3 event types) for query q.

    An empty subset for event type e means that component's row is ZERO -- exactly what upstream
    produces for a query with no paraphrases (`scores` is zero-initialised and padded slots are
    masked to -inf before the max).  It does NOT mean the component is dropped from the fusion:
    fuse() softmaxes over dim=0, so a zero row still contributes exp(0 - D[v]) = 1/D[v], which is
    non-uniform across videos.  The greedy objective must use the same zero-row semantics as
    build_comps_from_selection(), or it would be optimising a different function than we score.
    """
    rel = target[q].numpy().astype(np.float64)
    base_rows = {p: comps[p][q] for p in ["query_vs_video", "query_vs_captions"]}
    zero = torch.zeros_like(comps["query_vs_captions"][q])

    cands = [(e, p) for e in EVENTS for p in torch.nonzero(valid[e][q]).flatten().tolist()]
    cur = {e: None for e in EVENTS}          # running max row per event type (None = empty set)
    chosen = []

    def score_of(cur):
        rows = dict(base_rows)
        for e in EVENTS:
            rows[e] = zero if cur[e] is None else cur[e]
        return ndcg10(fused_row(rows, D).numpy(), rel)

    best = score_of(cur)
    steps = max_steps or len(cands)
    for _ in range(steps):
        gain, pick, newrow = 0.0, None, None
        for (e, p) in cands:
            if (e, p) in chosen:
                continue
            cand = P[e][q, p]
            merged = cand if cur[e] is None else torch.maximum(cur[e], cand)
            trial = dict(cur); trial[e] = merged
            s = score_of(trial)
            if s > best + 1e-9 and s - best > gain:
                gain, pick, newrow = s - best, (e, p), merged
        if pick is None:
            break
        chosen.append(pick); cur[pick[0]] = newrow; best += gain
    return chosen, best


def build_comps_from_selection(P, valid, comps, sel):
    """Exact (unfrozen) component matrices implied by a per-query paraphrase selection."""
    out = {p: comps[p].clone() for p in ["query_vs_video", "query_vs_captions"]}
    T, V = comps["query_vs_captions"].shape
    for e in EVENTS:
        M = torch.zeros(T, V)
        for q in range(T):
            ps = [p for (ee, p) in sel[q] if ee == e]
            if ps:
                M[q] = P[e][q, ps].max(dim=0).values
        out[e] = M
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="noASR", choices=list(SETTINGS))
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N queries")
    a = ap.parse_args()

    # CPU-only: gpu_ids=[] so codecarbon does not bill this run for the whisper server on GPU0.
    with track(f"tierC-oracle-{a.setting}", gpu_ids=[], tags=["tierC", "oracle"],
               config={"setting": a.setting, "limit": a.limit}) as tr:
        _run(a, tr)


def _run(a, tr):
    queries, video_ids, target, comps, P, valid, nd_published, recon_delta = load_cell(a.setting)
    T = len(queries)
    D = frozen_softmax_parts(comps)

    # --- baselines (exact fuse) ---
    # comps[e] is now the paracache max-pool, i.e. exactly the "keep every paraphrase" subset.
    ndA = per_query(fuse(comps, LADDER["A_visual"]), target)[0]
    ndB = per_query(fuse(comps, LADDER["B_noevents"]), target)[0]
    ndF = per_query(fuse(comps, LADDER["C_full"]), target)[0]
    print(f"\n### MultiVENT {a.setting}  (T={T})")
    print(f"  Fixed-A    {ndA.mean()*100:.2f}")
    print(f"  Fixed-B    {ndB.mean()*100:.2f}")
    print(f"  Fixed-Full {ndF.mean()*100:.2f}   <- all paraphrases, max-pooled (Q2E)")
    print(f"             (published {nd_published:.2f}; fp16 reconstruction delta {recon_delta:.4f},")
    print(f"              identical on both arms, so it cancels in every gain below)")

    # SANITY, on every query. This simultaneously validates two things the greedy search depends
    # on: (i) the frozen-softmax reconstruction reproduces the exact fuse, and (ii) our fast
    # ndcg10 agrees with the torchmetrics nDCG used everywhere else. The latter is not free --
    # ndcg10 and torchmetrics break exact SCORE TIES differently, and an empty event subset
    # writes a zeros row, which is maximally tie-heavy. Fused rows are continuous sums so ties
    # should be measure-zero; assert it rather than assume it.
    err = []
    for q in range(T):
        rows = {p: comps[p][q] for p in ALL5}
        err.append(abs(ndcg10(fused_row(rows, D).numpy(), target[q].numpy().astype(float)) - float(ndF[q])))
    mx = max(err)
    print(f"  [sanity] frozen-softmax + ndcg10 vs exact fuse + torchmetrics: max|diff| over all "
          f"{T} queries = {mx:.2e}")
    if mx > 1e-5:
        raise SystemExit(f"greedy objective does not match the reported metric (max|diff|={mx:.2e}). "
                         f"Refusing to optimise a different function than we score.")

    n_paras = {e: int(valid[e].sum()) for e in EVENTS}
    print(f"  paraphrases available: {n_paras} (total {sum(n_paras.values())})")

    # --- oracle over paraphrase subsets ---
    rng = range(T if not a.limit else min(T, a.limit))
    sel, approx = [], []
    for q in rng:
        ch, s = greedy_select(q, P, valid, comps, D, target)
        sel.append(ch); approx.append(s)
        if (q + 1) % 25 == 0:
            print(f"  ...{q+1}/{len(rng)} greedy nDCG so far {np.mean(approx)*100:.2f}", flush=True)

    if a.limit:
        print(f"[debug limit] approx oracle nDCG {np.mean(approx)*100:.2f} "
              f"vs Full {ndF[:len(rng)].mean()*100:.2f}")
        return

    kept = np.array([len(s) for s in sel])
    ocomps = build_comps_from_selection(P, valid, comps, sel)
    ndSel = per_query(fuse(ocomps, LADDER["C_full"]), target)[0]

    print(f"\n  ORACLE paraphrase selection (exact refuse): {ndSel.mean()*100:.2f} nDCG")
    print(f"    vs Fixed-Full {ndF.mean()*100:.2f}  -> +{(ndSel.mean()-ndF.mean())*100:.2f}")
    print(f"    vs Fixed-B    {ndB.mean()*100:.2f}  -> +{(ndSel.mean()-ndB.mean())*100:.2f}")
    print(f"    (frozen-softmax greedy estimate was {np.mean(approx)*100:.2f})")
    print(f"    paraphrases kept: mean {kept.mean():.1f} of {sum(n_paras.values())/T:.1f} "
          f"available; {100*(kept==0).mean():.0f}% of queries keep NONE")
    for e in EVENTS:
        c = np.array([sum(1 for (ee, _) in s if ee == e) for s in sel])
        print(f"      {e:22} kept mean {c.mean():.2f}  ({100*(c==0).mean():.0f}% queries drop it entirely)")

    hurt_full = float(((ndF - ndB) < -1e-9).float().mean())
    hurt_sel = float(((ndSel - ndB) < -1e-9).float().mean())
    print(f"    queries HURT vs Fixed-B: Q2E-Full {100*hurt_full:.0f}%  ->  oracle-select {100*hurt_sel:.0f}%")

    out = {
        "setting": a.setting, "T": T,
        "fixedA": float(ndA.mean() * 100), "fixedB": float(ndB.mean() * 100),
        "fixedFull": float(ndF.mean() * 100), "oracle_select": float(ndSel.mean() * 100),
        "greedy_frozen_estimate": float(np.mean(approx) * 100),
        "kept_mean": float(kept.mean()), "avail_mean": float(sum(n_paras.values()) / T),
        "pct_queries_keep_none": float((kept == 0).mean() * 100),
        "hurt_vs_B_full_pct": 100 * hurt_full, "hurt_vs_B_oracle_pct": 100 * hurt_sel,
        "per_query_ndcg": {"A": ndA.tolist(), "B": ndB.tolist(), "Full": ndF.tolist(),
                           "oracle_select": ndSel.tolist()},
        "selection": [[[e, int(p)] for (e, p) in s] for s in sel],
    }
    of = f"{REPO}/results/ablations/tierC_selection_oracle_{a.setting}.json"
    json.dump(out, open(of, "w"), indent=2)
    print(f"\nwrote {of}")
    tr.summary({k: out[k] for k in ["fixedA", "fixedB", "fixedFull", "oracle_select",
                                    "kept_mean", "avail_mean", "pct_queries_keep_none",
                                    "hurt_vs_B_full_pct", "hurt_vs_B_oracle_pct"]})


if __name__ == "__main__":
    main()
