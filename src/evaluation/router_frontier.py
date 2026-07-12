#!/usr/bin/env python
"""CPU router baseline + accuracy-compute frontier for the adaptive-Q2E extension.

Reads results/ablations/router_trainset.jsonl (per-query, per-tier NDCG + query text)
and produces, for each dataset slice:

  * Fixed-A / Fixed-B / Fixed-Full        -- static single-tier baselines (points)
  * Oracle frontier                       -- per-query cheapest-within-eps, TRUE ndcg (upper bound)
  * Trained-router frontier               -- TF-IDF -> logistic regression, honest out-of-fold

The frontier is traced by sweeping the cheapest-within-eps label rule over eps: a larger
eps trades a little NDCG for a cheaper tier, moving down-left along the curve. The x-axis is
mean similarity-components scored per query (A=1, B=2, Full=5), normalized to Full=1.0; this
is also the LLM event-decomposition rate story (only Full pays event decomposition).

CPU-only. Writes results/ablations/router_frontier.json and reports/figures/*.{pdf,png}.
"""
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINSET = os.path.join(ROOT, "results/ablations/router_trainset.jsonl")
OUT_JSON = os.path.join(ROOT, "results/ablations/router_frontier.json")
FIG_DIR = os.path.join(ROOT, "reports/figures")

# A/B/C ladder, ordered cheap -> expensive. Cost = measured J/query, normalized to Full=1.0
# (retired proxy was # similarity components scored). See tier_cost.py.
from tier_cost import COST_NORM, LADDER  # noqa: E402
EPS_SWEEP = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]
SEED = 0


def cheapest_within_eps(row, eps):
    vals = {t: row["_ndcg"][t] for t in LADDER}
    best = max(vals.values())
    for t in LADDER:  # cheap -> expensive; first within eps of best wins
        if vals[t] >= best - eps:
            return t
    return LADDER[-1]


def policy_stats(rows, chosen):
    """chosen: list of tier per row. Returns mean NDCG, mean normalized cost, Full-share."""
    ndcg = np.mean([r["_ndcg"][t] for r, t in zip(rows, chosen)]) * 100.0
    cost = float(np.mean([COST_NORM[t] for t in chosen]))
    full_share = float(np.mean([t == "Full" for t in chosen]))
    return {"ndcg": round(ndcg, 3), "cost": round(cost, 4), "full_share": round(full_share, 4)}


def make_pipeline():
    return Pipeline([
        ("feat", FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])


def oof_predict(texts, y, n_splits=5):
    """Honest out-of-fold tier predictions. Stratified when class counts allow, else plain KFold."""
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:                       # degenerate: only one tier ever labeled
        return y.copy()
    k = int(min(n_splits, counts.min()))
    if k < 2:                                 # a class with a single member -> can't stratify
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    else:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    return cross_val_predict(make_pipeline(), texts, y, cv=cv)


def slice_frontier(rows):
    texts = [r["query"] for r in rows]
    out = {"n": len(rows), "fixed": {}, "oracle": [], "router": [],
           "gold_per_query": round(float(np.mean([len(r["gold_video_ids"]) for r in rows])), 2)}

    for t in LADDER:
        out["fixed"][t] = policy_stats(rows, [t] * len(rows))

    for eps in EPS_SWEEP:
        y = [cheapest_within_eps(r, eps) for r in rows]
        out["oracle"].append({"eps": eps, **policy_stats(rows, y)})
        pred = oof_predict(texts, y)
        rp = policy_stats(rows, list(pred))
        # accuracy of tier prediction vs the oracle label (diagnostic only)
        rp["tier_acc"] = round(float(np.mean(np.asarray(pred) == np.asarray(y))), 4)
        out["router"].append({"eps": eps, **rp})
    return out


def dominates_or_matches(cost_a, ndcg_a, cost_b, ndcg_b):
    return cost_a <= cost_b + 1e-9 and ndcg_a >= ndcg_b - 1e-9


def headline(sl):
    """Cheapest router point that matches Fixed-Full NDCG (within 0.1 NDCG); report the saving."""
    full = sl["fixed"]["Full"]
    best = None
    for p in sl["router"]:
        if p["ndcg"] >= full["ndcg"] - 0.1:
            if best is None or p["cost"] < best["cost"]:
                best = p
    if best is None:
        return None
    return {"router_cost": best["cost"], "full_cost": full["cost"],
            "cost_saving_pct": round(100 * (full["cost"] - best["cost"]) / full["cost"], 1),
            "router_ndcg": best["ndcg"], "full_ndcg": full["ndcg"],
            "router_full_share": best["full_share"], "eps": best["eps"]}


def main():
    rows_all = [json.loads(l) for l in open(TRAINSET)]

    def sel(ds, enc, setting, pool):
        return [r for r in rows_all if r["dataset"] == ds and r["encoder"] == enc
                and r["setting"] == setting and r.get("pool") in (pool, None)]

    slices = {
        "MultiVENT_multiclip_noASR_fullpool": sel("MultiVENT", "multiclip", "noASR", "full_pool"),
        "MultiVENT_multiclip_ASR_fullpool":   sel("MultiVENT", "multiclip", "ASR", "full_pool"),
        "MultiVENT_multiclip_noASR_subset":   sel("MultiVENT", "multiclip", "noASR", "subset"),
        "MSRVTT_multiclip_noASR":             sel("MSR-VTT", "multiclip", "noASR", "full"),
        "MSRVTT_multiclip_ASR":               sel("MSR-VTT", "multiclip", "ASR", "full"),
        "MSRVTT_internvideo2_noASR":          sel("MSR-VTT", "internvideo2", "noASR", "full"),
    }

    results = {}
    for name, rows in slices.items():
        if not rows:
            continue
        sl = slice_frontier(rows)
        sl["headline"] = headline(sl)
        results[name] = sl
        h = sl["headline"]
        print(f"\n=== {name}  (n={sl['n']}, gold/q={sl['gold_per_query']}) ===")
        print(f"  Fixed-A   NDCG {sl['fixed']['A_visual']['ndcg']:.2f} @ cost {sl['fixed']['A_visual']['cost']:.2f}")
        print(f"  Fixed-B   NDCG {sl['fixed']['-Events']['ndcg']:.2f} @ cost {sl['fixed']['-Events']['cost']:.2f}")
        print(f"  Fixed-Full NDCG {sl['fixed']['Full']['ndcg']:.2f} @ cost {sl['fixed']['Full']['cost']:.2f}")
        if h:
            print(f"  ROUTER matches Full ({h['router_ndcg']:.2f} vs {h['full_ndcg']:.2f}) "
                  f"at cost {h['router_cost']:.2f} = {h['cost_saving_pct']:.0f}% cheaper "
                  f"(Full only {h['router_full_share']*100:.0f}% of queries, eps={h['eps']})")
        else:
            print("  ROUTER: no point matches Full within 0.1 NDCG")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    print(f"\nwrote {OUT_JSON}")

    make_figures(results)


def make_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG_DIR, exist_ok=True)

    def panel(ax, sl, title):
        orc = sl["oracle"]
        rtr = sl["router"]
        ax.plot([p["cost"] for p in orc], [p["ndcg"] for p in orc],
                "--", color="#888", marker="o", ms=3, lw=1.3, label="Oracle (upper bound)", zorder=2)
        ax.plot([p["cost"] for p in rtr], [p["ndcg"] for p in rtr],
                "-", color="#1f77b4", marker="o", ms=4, lw=1.8, label="Trained router (out-of-fold)", zorder=3)
        marks = {"A_visual": ("A: visual-only", "#2ca02c", "s"),
                 "-Events": ("B: +captions", "#ff7f0e", "^"),
                 "Full": ("Full Q2E (fixed)", "#d62728", "*")}
        for t, (lab, col, mk) in marks.items():
            f = sl["fixed"][t]
            ax.scatter([f["cost"]], [f["ndcg"]], c=col, marker=mk,
                       s=180 if t == "Full" else 90, zorder=5, edgecolors="k", linewidths=0.5, label=lab)
        ax.set_title(f"{title}\n(n={sl['n']}, {sl['gold_per_query']} gold/query)", fontsize=10)
        # measured joules span A=0.5% to Full=100% of the budget (200x): log x-axis to show all tiers.
        ax.set_xscale("log")
        ax.set_xlabel("mean cost / query  (measured J, normalized to Full = 1.0)")
        ax.set_ylabel("NDCG@10")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7.5, loc="lower right")

    # Headline: MultiVENT vs MSR-VTT side by side (the heterogeneity thesis)
    mv = results.get("MultiVENT_multiclip_noASR_fullpool")
    mr = results.get("MSRVTT_multiclip_noASR")
    if mv and mr:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        panel(axes[0], mv, "MultiVENT (complex / multilingual news)")
        panel(axes[1], mr, "MSR-VTT (simple / single-event)")
        fig.suptitle("Adaptive routing over Q2E fusion tiers: accuracy–compute frontier", fontsize=12, y=1.02)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG_DIR, f"router_frontier_heterogeneity.{ext}"),
                        bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"wrote {FIG_DIR}/router_frontier_heterogeneity.{{pdf,png}}")

    # Single MultiVENT panel (headline figure)
    if mv:
        fig, ax = plt.subplots(figsize=(6, 5))
        panel(ax, mv, "MultiVENT / MultiCLIP (noASR)")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG_DIR, f"router_frontier_multivent.{ext}"),
                        bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"wrote {FIG_DIR}/router_frontier_multivent.{{pdf,png}}")


if __name__ == "__main__":
    main()
