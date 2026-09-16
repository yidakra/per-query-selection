"""Per-query system selection on MVEB, supervision's additional test bed.

The three axes so far decide what to search, how to phrase it, and which language to ask in. MVEB
supplies a fourth decision of the same shape and a different kind: two finished retrieval systems per
pool, a MultiVENT-specialised one and a generic base, with neither dominating. Supervision's own
summary has the base ahead on five of seven pools, so choosing per query is a live decision rather
than a formality. The runs already exist, so nothing is retrieved here.

Judgments are single-gold identity: query i's one relevant video is the i-th entry of the pool's id
list. That reconstruction is checked against supervision's own reported nDCG before anything is
computed on top of it, and the check has to pass exactly.

Two predictor families, the same ones the rest of the study uses. Score-only predictors read the
first-stage score distribution and need nothing else, so they run on every pool. Pre-retrieval
predictors need the query text and a lexical index, so they run only on pools where a query file is
supplied. Calibration is single-feature out-of-fold ridge under shuffled five-fold CV, as on MSR-VTT:
one phrasing per information need here, so there is no event-duplicate grouping analogue.

  python src/multivent2/mv2_mveb_selection.py --pool mrvmteb --pool didemo
"""
import os
import sys
import json
import math
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
ABL = os.path.join(_ROOT, "results", "ablations")
MVEB = "/mnt/data/q2e/mveb"
MARGIN = 5e-4      # how far above the best fixed system a router must land to count, fixed so that
                   # loosening the reconstruction tolerance cannot loosen the conclusion

# pool -> (caption directory, suffix on the run filename, supervision's reported nDCG per system)
POOLS = {
    "mrvmteb": ("MSR-VTT", "", {"base": 0.6969, "01mv": 0.6913}),
    "didemo": ("DiDeMo", "", {"base": 0.5615, "01mv": 0.5851}),
    "vatex": ("VATEX_test_1k", "", {"base": 0.7773, "01mv": 0.7928}),
    "anet": ("ActivityNet_Captions_val2", "", {"base": 0.6715, "01mv": 0.6054}),
    "acaps": ("AudioCaps_AV", "", {"base": 0.4446, "01mv": 0.3801}),
    "vggv": ("VGGSound_AV_RETRIEVAL", "_v", {"base": 0.9553, "01mv": 0.9686}),
    "vgga": ("VGGSound_AV_RETRIEVAL", "_a", {"base": 0.3567, "01mv": 0.3178}),
}


def sign_flip_p(diff, n_draws=2000, seed=0):
    """Two-sided sign-flip test over queries. One phrasing per information need here, so queries are
    exchangeable and there is no event-duplicate grouping to respect."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    obs = abs(float(diff.mean()))
    cnt = sum(1 for _ in range(n_draws)
              if abs(float((diff * rng.choice([-1.0, 1.0], size=len(diff))).mean())) >= obs)
    return float((1 + cnt) / (n_draws + 1))


def per_query_ndcg10(run, qrels):
    """nDCG@10 with one relevant document, so the ideal gain is 1 and this is a rank discount."""
    out = {}
    for q, gold in qrels.items():
        r = run.get(q, {})
        ranked = sorted(r, key=r.get, reverse=True)[:10]
        out[q] = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if d in gold)
    return out


def load_pool(pool, root):
    """Runs, reconstructed qrels and the query order. Raises if the reconstruction is unavailable."""
    d, sfx, _ = POOLS[pool]
    prefix = "vggsound" if pool in ("vggv", "vgga") else pool
    ids_file = os.path.join(root, "backfill_request", f"{prefix}_video_ids.txt")
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"no id list for {pool}; supervision has it as {prefix}_video_ids.txt")
    ids = [l.strip() for l in open(ids_file) if l.strip()]
    qids = [f"{prefix}_q{i:06d}" for i in range(len(ids))]
    qrels = {q: {v} for q, v in zip(qids, ids)}
    runs = {}
    for sysname in ("base", "01mv"):
        p = os.path.join(root, "first_stage", f"{prefix}_run_{sysname}_top100{sfx}.json")
        runs[sysname] = json.load(open(p))
    return runs, qrels, qids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="append", default=[], choices=sorted(POOLS),
                    help="repeat for more than one; default is every pool with an id list")
    ap.add_argument("--root", default=MVEB)
    ap.add_argument("--queries", action="append", default=[], metavar="POOL=FILE",
                    help="JSON of {query_id: text}; enables the pre-retrieval family for that pool")
    ap.add_argument("--index", action="append", default=[], metavar="POOL=FILE",
                    help="JSONL of {text: ...} for the lexical index; defaults to the pool's captions")
    ap.add_argument("--tol", type=float, default=5e-4,
                    help="how far the reconstructed nDCG may sit from supervision's own number; it "
                         "gates the reconstruction only and never the above-fixed comparison")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.tag and not a.tag.startswith("_"):
        a.tag = "_" + a.tag

    from sklearn.model_selection import KFold, cross_val_predict
    from mv2_qpp_predictors import Index, PRE_RETRIEVAL, SCORE_ONLY, \
        pre_retrieval_suite, score_only_suite
    from mv2_msrvtt_source_replication import route, model
    from retrieve import FEATURE_ORDER, conf_features

    qfiles = dict(s.split("=", 1) for s in a.queries)
    ifiles = dict(s.split("=", 1) for s in a.index)
    pools = a.pool or sorted(POOLS)
    results = {}
    for pool in pools:
        try:
            runs, qrels, qids = load_pool(pool, a.root)
        except FileNotFoundError as e:
            print(f"[{pool}] skipped: {e}", flush=True)
            continue

        # the reconstruction is only usable if it reproduces supervision's own numbers
        reported = POOLS[pool][2]
        means, bad = {}, []
        yq = {}
        for sysname, run in runs.items():
            yq[sysname] = per_query_ndcg10(run, qrels)
            means[sysname] = float(np.mean([yq[sysname][q] for q in qids]))
            if abs(means[sysname] - reported[sysname]) > a.tol:
                bad.append(f"{sysname} {means[sysname]:.4f} vs reported {reported[sysname]:.4f}")
        if bad:
            raise SystemExit(f"[{pool}] reconstruction does not match supervision: {'; '.join(bad)}")
        print(f"[{pool}] {len(qids)} queries; base {means['base']:.4f} 01mv {means['01mv']:.4f} "
              f"(both match the reported numbers within {a.tol})", flush=True)

        cheap = np.array([yq["base"][q] for q in qids])          # option A, the better fixed system
        expensive = np.array([yq["01mv"][q] for q in qids])      # option B
        gain = expensive - cheap
        cv = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(qids))))

        # WIG and sigma_x divide by sqrt(query length). Without the query text there is no length to
        # divide by, so nq is 1 for every query and those three features are not length-normalised.
        # No routing result moves, because the single-feature ridge standardises its one input and a
        # constant divisor cannot survive that, but the protocol only matches the sibling study once
        # the text arrives.
        qtext_pool = json.load(open(qfiles[pool])) if pool in qfiles else None
        if qtext_pool is None:
            print(f"[{pool}] no query text: WIG, WIG_norm and sigma_x0.5 run unnormalised", flush=True)
        # The released runs are top-100, a tenth of the 1000 the MultiVENT protocol scores over, so
        # the top-k window shrinks with them and keeps the same 1:10 ratio. At the default k=100 the
        # top-k mean would equal the list mean and WIG_norm would be identically zero, and RSD,
        # defined as unnormalised SMV at k=1000, would clamp onto SMV and duplicate it.
        depth_k = max(1, min(len(r) for r in runs["base"].values()) // 10)
        print(f"[{pool}] top-k window {depth_k} over runs of {min(len(r) for r in runs['base'].values())}",
              flush=True)
        score_raw = {n: [] for n in SCORE_ONLY}
        for q in qids:
            srt = sorted(runs["base"][q].values(), reverse=True)
            nq = len(qtext_pool[q].split()) if qtext_pool else 1
            feats = score_only_suite(np.asarray(srt, dtype=float), max(1, nq), k=depth_k)
            for n in SCORE_ONLY:
                score_raw[n].append(feats[n])
        dup = [n for n in SCORE_ONLY
               if n != "SMV" and np.allclose(score_raw[n], score_raw["SMV"])]
        const = [n for n in SCORE_ONLY if np.allclose(np.std(score_raw[n]), 0)]
        if dup or const:
            print(f"[{pool}] dropping {sorted(set(dup + const))}: duplicate of SMV or constant",
                  flush=True)
            for n in set(dup + const):
                score_raw.pop(n)
        score = {n: route(v, gain, cheap, expensive, cv) for n, v in score_raw.items()}
        results_extra = {"score_k": depth_k, "score_dropped": sorted(set(dup + const))}

        # the positive control: one learner over both systems' score distributions and their rank
        # agreement. Without it a weak score family is unreadable, because a predictor that misses
        # cannot be told apart from a decision that nothing can predict.
        feats = []
        for q in qids:
            rb, ro = runs["base"][q], runs["01mv"][q]
            cb = conf_features(np.asarray(sorted(rb.values(), reverse=True), dtype=float))
            co = conf_features(np.asarray(sorted(ro.values(), reverse=True), dtype=float))
            row = [cb[k] for k in FEATURE_ORDER] + [co[k] for k in FEATURE_ORDER]
            for depth in (10, 100):
                top_b = set(sorted(rb, key=rb.get, reverse=True)[:depth])
                top_o = set(sorted(ro, key=ro.get, reverse=True)[:depth])
                row.append(len(top_b & top_o) / max(1, len(top_b | top_o)))
            feats.append(row)
        ctrl_pred = cross_val_predict(model(), np.asarray(feats), gain, cv=cv)
        ctrl_routed = np.where(ctrl_pred > 0, expensive, cheap)
        best_fixed_vec = cheap if means["base"] >= means["01mv"] else expensive
        from scipy.stats import kendalltau
        control = {"tau": float(kendalltau(ctrl_pred, gain).statistic),
                   "frac_tied": float((np.abs(gain) < 1e-9).mean()),
                   "oracle_headroom": float(np.maximum(cheap, expensive).mean() - max(means.values())),
                   "routed_ndcg10": float(ctrl_routed.mean()),
                   "vs_best_fixed": float(ctrl_routed.mean() - best_fixed_vec.mean()),
                   "frac_switched": float((ctrl_pred > 0).mean()),
                   "p_two_sided": sign_flip_p(ctrl_routed - best_fixed_vec),
                   "n_features": len(feats[0])}

        pre = {}
        if qtext_pool is not None:
            qtext = qtext_pool
            src = ifiles.get(pool, os.path.join(a.root, POOLS[pool][0], "captions.jsonl"))
            texts = []
            for line in open(src):
                r = json.loads(line)
                t = (r.get("caption") or r.get("text") or "").strip()
                if t:
                    texts.append(t)
            index = Index(texts)
            print(f"[{pool}] lexical index over {len(texts)} documents", flush=True)
            pre_raw = {n: [] for n in PRE_RETRIEVAL}
            for q in qids:
                toks = qtext[q].lower().split()
                feats = pre_retrieval_suite(toks, index)
                for n in PRE_RETRIEVAL:
                    pre_raw[n].append(feats[n])
            pre = {n: route(v, gain, cheap, expensive, cv) for n, v in pre_raw.items()}

        fixed = max(means["base"], means["01mv"])

        def summary(rows):
            if not rows:
                return {"above_fixed": None, "n": 0, "degenerate": None, "max_abs_tau": None}
            return {"above_fixed": sum(v["routed_ndcg10"] > fixed + MARGIN for v in rows.values()),
                    "n": len(rows),
                    "degenerate": sum(v["degenerate"] for v in rows.values()),
                    "max_abs_tau": max(abs(v["tau"]) for v in rows.values())}

        results[pool] = {
            "n_queries": len(qids), "base": means["base"], "01mv": means["01mv"],
            "best_fixed": fixed, "gain_mean": float(gain.mean()), "gain_sd": float(gain.std()),
            "frac_01mv_better": float((gain > 1e-9).mean()),
            "frac_base_better": float((gain < -1e-9).mean()),
            "query_text": qtext_pool is not None,
            "oracle": float(np.maximum(cheap, expensive).mean()),
            "pre": pre, "score": score, "control": control, **results_extra,
            "pre_summary": summary(pre), "score_summary": summary(score),
        }
        ps, ss = results[pool]["pre_summary"], results[pool]["score_summary"]
        print(f"[{pool}] oracle {results[pool]['oracle']:.4f} vs best fixed {fixed:.4f}; "
              f"pre {ps['above_fixed']}/{ps['n']} above fixed, score {ss['above_fixed']}/{ss['n']}; "
              f"control {control['routed_ndcg10']:.4f} ({control['vs_best_fixed']:+.4f} of "
              f"{control['oracle_headroom']:+.4f} headroom, p={control['p_two_sided']:.4f}, "
              f"tau {control['tau']:+.3f}, tied {100*control['frac_tied']:.1f}%)",
              flush=True)

    out = os.path.join(ABL, f"mv2_mveb_selection{a.tag}.json")
    json.dump(results, open(out, "w"), indent=2)
    lines = ["# MVEB per-query system selection", "",
             "Choose between supervision's two first-stage systems per query. Single-gold identity "
             "judgments, five-fold out-of-fold calibration, no retrieval run here.", "",
             "| pool | n | base | 01mv | best fixed | oracle | pre above | score above | control |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pool, r in results.items():
        ps, ss = r["pre_summary"], r["score_summary"]
        pa = "--" if ps["n"] == 0 else f"{ps['above_fixed']}/{ps['n']}"
        lines.append(f"| {pool} | {r['n_queries']} | {r['base']:.4f} | {r['01mv']:.4f} | "
                     f"{r['best_fixed']:.4f} | {r['oracle']:.4f} | {pa} | "
                     f"{ss['above_fixed']}/{ss['n']} | {r['control']['vs_best_fixed']:+.4f} "
                     f"(p={r['control']['p_two_sided']:.3f}) |")
    open(os.path.join(ABL, f"mv2_mveb_selection{a.tag}.md"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
