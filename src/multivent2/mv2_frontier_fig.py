"""Plot the B->Full accuracy-energy frontier on MultiVENT 2.0 (reads mv2_frontier.json).

The story of the figure: escalating queries to the Full (LLM event-decomposition) tier costs measured
GPU joules on the x-axis; nDCG@10 is on the y-axis. Three curves as the escalated fraction sweeps 0->1:
an oracle that escalates the right queries first (dashed, upper bound), the realizable out-of-fold router
(solid), and the random-escalation chord (dotted). The router hugs the random chord -- it cannot tell
which queries the event tier will help -- while the oracle opens a gap the router never reaches. That is
the "Full tier is dominated" result: expensive, tiny gain, and unroutable. Matches the house figure style
(dashed-grey oracle / solid-blue router, dpi=150). CPU-only.

  python src/multivent2/mv2_frontier_fig.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ABL = os.path.join(_ROOT, "results", "ablations")
FIG_DIR = os.path.join(_ROOT, "reports", "figures")


def main():
    d = json.load(open(os.path.join(ABL, "mv2_frontier.json")))
    cost = np.array(d["cost_j_per_query"])
    router = 100 * np.array(d["ndcg_router"])
    random = 100 * np.array(d["ndcg_random"])
    oracle = 100 * np.array(d["ndcg_oracle"])
    fracs = np.array(d["fracs"])
    jll = d["j_llm_per_query"]
    fi = int(np.argmax(oracle - random))          # peak oracle headroom over the random chord

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 5))

    ax.plot(cost, oracle, "--", color="#888", marker="o", ms=3, lw=1.3,
            label="Oracle escalation (upper bound)", zorder=2)
    ax.plot(cost, random, ":", color="#555", lw=1.5,
            label="Random escalation (chord)", zorder=2)
    ax.plot(cost, router, "-", color="#1f77b4", marker="o", ms=4, lw=1.9,
            label="Out-of-fold router", zorder=3)

    # endpoint tiers
    ax.scatter([cost[0]], [random[0]], c="#ff7f0e", marker="^", s=90, zorder=5,
               edgecolors="k", linewidths=0.5, label="Tier B (+captions), all")
    ax.scatter([cost[-1]], [random[-1]], c="#d62728", marker="*", s=200, zorder=5,
               edgecolors="k", linewidths=0.5, label="Full Q2E (fixed), all")

    # annotate the unreachable oracle headroom at its peak
    ax.annotate("", xy=(cost[fi], oracle[fi]), xytext=(cost[fi], router[fi]),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1.1))
    ax.text(cost[fi] + 3, (oracle[fi] + router[fi]) / 2,
            f"oracle headroom\n+{oracle[fi]-random[fi]:.2f} nDCG @ {100*fracs[fi]:.0f}% escalated\n"
            f"router captures {100*d['oracle_headroom_captured']:.0f}%",
            fontsize=7.5, va="center", color="#555")

    ax.set_title("The Full event tier is dominated on the energy frontier\n"
                 "(MultiVENT 2.0, 2,544 test queries; qwen2.5:7b on GPU1)", fontsize=10)
    ax.set_xlabel(f"mean cost / query  (measured GPU joules; Full = {jll:.0f} J/query)")
    ax.set_ylabel("nDCG@10")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")

    # secondary caption of the money number, parked in the open center-right whitespace
    ax.text(0.70, 0.80, f"Full tier: {jll:.0f} J/query\n{d['j_per_ndcg10_point']:.0f} J per nDCG@10 point gained",
            transform=ax.transAxes, fontsize=8, color="#333", va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc", lw=0.6))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"mv2_frontier.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {FIG_DIR}/mv2_frontier.{{pdf,png}}")


if __name__ == "__main__":
    main()
