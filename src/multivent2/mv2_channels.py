"""Multi-channel retrieval on MultiVENT 2.0: fuse the visual, ASR and OCR ranked lists, and set up
per-query routing *across modality channels* instead of across LLM-expansion tiers.

Our cascade so far is a cheap visual+caption stack topping out near nDCG@10 0.37, while the strong
systems on this benchmark -- MMMORRF ~0.586, CLaMR ~0.585 -- get their lift from ASR and OCR retrieval
channels we were not using.
Those channels are not something we have to build: the benchmark ships them as ranked lists
(`ranked_lists/whisperASR_clip.json`, `ranked_lists/10pyscene_paddleOCR_clip.json`), so a MMMORRF-style
weighted RRF over visual + ASR + OCR is reproducible locally at zero GPU cost.

Why this matters for the router: whether ASR helps is a *per-query, per-video* property -- a silent
protest video has no speech to transcribe, a news broadcast is mostly speech. That makes "which
channels are worth scoring for this query" a genuine routing decision on a competitive stack, rather
than a routing decision on a weak one. This script measures each channel alone, the fused ceiling, and
emits a router-ready JSON per escalation step (visual -> +ASR, visual -> +ASR+OCR).

CPU-only, no model loads.

  python src/multivent2/mv2_channels.py
"""
import os
import sys
import json
import argparse
import itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run, load_queries  # noqa: E402
from mv2_eval import ndcg10  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
RRF_K = 60

CHANNELS = {"visual": "10pyscene_clip.json",
            "asr": "whisperASR_clip.json",
            "ocr": "10pyscene_paddleOCR_clip.json"}


def rank_map(scores):
    """doc -> RRF contribution 1/(K+rank) for one query's score dict."""
    order = sorted(scores, key=lambda v: -scores[v])
    return {v: 1.0 / (RRF_K + i) for i, v in enumerate(order)}


def fuse(runs, weights, qids):
    """Weighted RRF over several runs. Union of each channel's candidates; a doc absent from a channel
    simply gets no contribution from it (the standard RRF treatment of a truncated list)."""
    out = {}
    for q in qids:
        acc = {}
        for name, w in weights.items():
            if w == 0 or q not in runs[name]:
                continue
            for v, c in rank_map(runs[name][q]).items():
                acc[v] = acc.get(v, 0.0) + w * c
        out[q] = acc
    return out


def routed_curve(path, steps=21):
    """Out-of-fold routed escalation curve for one cell: nDCG when the top-f queries by predicted gain
    take the expensive fusion and the rest keep the cheap channel. Reports the routed optimum against
    the two policies a practitioner would otherwise pick -- cheap-only (f=0) and uniform fusion (f=1)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, KFold
    d = json.load(open(path))
    qs = [q for q in d["features"] if q in d["per_query"]]
    X = np.array([d["features"][q] for q in qs])
    cheap = np.array([d["per_query"][q]["ndA"] for q in qs])
    exp_ = np.array([d["per_query"][q]["ndB"] for q in qs])
    g = exp_ - cheap
    n = len(g)
    ghat = cross_val_predict(Pipeline([("s", StandardScaler()),
                                       ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))]),
                             X, g, cv=KFold(5, shuffle=True, random_state=0))
    o, oo = np.argsort(-ghat), np.argsort(-g)
    fr = np.linspace(0, 1, steps)
    rows = []
    for f in fr:
        k = int(round(f * n))
        m = np.zeros(n, bool); m[o[:k]] = True
        mo = np.zeros(n, bool); mo[oo[:k]] = True
        rows.append({"f": float(f), "routed": float(np.where(m, exp_, cheap).mean()),
                     "oracle": float(np.where(mo, exp_, cheap).mean()),
                     "random": float(cheap.mean() + f * g.mean())})
    best = max(rows, key=lambda r: r["routed"])
    return {"curve": rows, "cheap_only": float(cheap.mean()), "uniform_fusion": float(exp_.mean()),
            "routed_best": best["routed"], "routed_best_f": best["f"],
            "oracle_best": max(r["oracle"] for r in rows),
            "gain_vs_cheap": 100 * (best["routed"] - cheap.mean()),
            "gain_vs_uniform": 100 * (best["routed"] - exp_.mean())}


def router_json(qrels, cheap_run, exp_run, qids, path, scorer):
    """Emit the standard router-ready cell: gain = nDCG(expensive) - nDCG(cheap), features from the
    cheap channel's own score distribution (legal to compute before paying for the extra channels)."""
    pq_c, pq_e = per_query_ndcg(qrels, cheap_run), per_query_ndcg(qrels, exp_run)
    common = [q for q in qids if q in pq_c and q in pq_e]
    feats = {}
    for q in common:
        s = np.array(sorted(cheap_run[q].values(), reverse=True))
        f = conf_features(s)
        feats[q] = [f[k] for k in FEATURE_ORDER]
    g = np.array([pq_e[q] - pq_c[q] for q in common])
    json.dump({"scorer": scorer, "ndcgA": float(np.mean([pq_c[q] for q in common])),
               "ndcgB": float(np.mean([pq_e[q] for q in common])),
               "per_query": {q: {"ndA": pq_c[q], "ndB": pq_e[q]} for q in common},
               "features": feats, "feature_order": FEATURE_ORDER}, open(path, "w"))
    print(f"  -> {os.path.basename(path)}: gain mean {100*g.mean():+.2f}  sd {100*g.std():.2f}  "
          f"help {100*(g>1e-9).mean():.0f}%  hurt {100*(g<-1e-9).mean():.0f}%")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_channels.json"))
    ap.add_argument("--sweep", action="store_true", help="sweep per-channel RRF weights")
    ap.add_argument("--channel", action="append", default=[], metavar="NAME=FILE",
                    help="add or override a channel, e.g. asr=asr_dense_multilingual-e5-base.json")
    ap.add_argument("--cell-tag", default="", help="suffix for the per-cell router JSONs, so a run "
                    "with substituted channels does not overwrite the shipped-channel cells")
    a = ap.parse_args()

    channels = dict(CHANNELS)
    for spec in a.channel:
        name, _, fn = spec.partition("=")
        channels[name] = fn

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    runs = {}
    for name, fn in channels.items():
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            print(f"MISSING {fn} -- download from hltcoe/MultiVENT2.0 ranked_lists/")
            continue
        runs[name] = load_run(p)
        print(f"{name:<7} {len(runs[name])} queries, "
              f"{np.mean([len(v) for v in runs[name].values()]):.0f} cands/query")
    qids = sorted(set.intersection(*[set(r) for r in runs.values()]))
    print(f"queries common to all channels: {len(qids)}\n")

    single = {}
    for name, r in runs.items():
        single[name] = ndcg10(qrels, {q: r[q] for q in qids})
        print(f"channel {name:<7} nDCG@10 = {single[name]:.5f}")

    combos = {}
    for k in (2, 3):
        for c in itertools.combinations(runs, k):
            w = {n: (1.0 if n in c else 0.0) for n in runs}
            combos["+".join(c)] = ndcg10(qrels, fuse(runs, w, qids))
    print()
    for k, v in sorted(combos.items(), key=lambda x: -x[1]):
        print(f"fusion {k:<22} nDCG@10 = {v:.5f}")

    best_w = {n: 1.0 for n in runs}
    if a.sweep and len(runs) == 3:
        print("\nweight sweep (visual fixed at 1.0):")
        best = -1
        for wa in (0.5, 1.0, 1.5, 2.0):
            for wo in (0.0, 0.25, 0.5, 1.0):
                w = {"visual": 1.0, "asr": wa, "ocr": wo}
                nd = ndcg10(qrels, fuse(runs, w, qids))
                if nd > best:
                    best, best_w = nd, dict(w)
                print(f"  asr={wa:<4} ocr={wo:<5} nDCG@10 = {nd:.5f}")
        print(f"best weights {best_w} -> {best:.5f}")

    # routing cells: cheap visual-only -> escalate to the extra channels
    print("\nrouter-ready cells:")
    vis = {q: runs["visual"][q] for q in qids}
    # one cell per escalation target. The weights are the sweep's best for that target, so each cell is
    # the *most favourable* uniform-fusion policy -- routing is compared against fusion at its best.
    targets = {"visual_to_asr": {"visual": 1.0, "asr": best_w.get("asr", 1.0), "ocr": 0.0},
               "visual_to_ocr": {"visual": 1.0, "asr": 0.0, "ocr": 0.5},
               "visual_to_all": {"visual": 1.0, "asr": best_w.get("asr", 1.0), "ocr": 0.5}}
    cells, routed = {}, {}
    for name, w in targets.items():
        if any(k not in runs for k, v in w.items() if v > 0):
            continue
        p = os.path.join(ABL, f"mv2_chan_{name}{a.cell_tag}.json")
        cells[name] = router_json(qrels, vis, fuse(runs, w, qids), qids, p, f"channel_{name}")
        routed[name] = routed_curve(p)
        r = routed[name]
        print(f"     cheap {r['cheap_only']:.5f} | uniform {r['uniform_fusion']:.5f} | "
              f"routed {r['routed_best']:.5f} @ f={r['routed_best_f']:.2f} "
              f"({r['gain_vs_cheap']:+.2f} vs cheap, {r['gain_vs_uniform']:+.2f} vs uniform) | "
              f"oracle {r['oracle_best']:.5f}")

    # per-query oracle over single channels: the ceiling if you could always pick the right one channel
    pq = {n: per_query_ndcg(qrels, {q: runs[n][q] for q in qids}) for n in runs}
    oracle_single = float(np.mean([max(pq[n].get(q, 0.0) for n in runs) for q in qids]))
    print(f"\noracle best-single-channel per query: {oracle_single:.5f} "
          f"(vs best fixed channel {max(single.values()):.5f})")

    json.dump({"single": single, "fusions": combos, "best_weights": best_w,
               "n_queries": len(qids), "oracle_best_single_channel": oracle_single,
               "routed": routed,
               "gain_stats": {k: {"mean": float(v.mean()), "sd": float(v.std()),
                                  "help": float((v > 1e-9).mean()), "hurt": float((v < -1e-9).mean())}
                              for k, v in cells.items()}},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
