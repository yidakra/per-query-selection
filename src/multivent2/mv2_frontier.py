"""Measured-joules accuracy-energy frontier for the B->Full step on MultiVENT 2.0.

Combines the per-query Full-tier gains (mv2_full.json) with the measured qwen2.5:7b generation energy
(mv2_energy.json) to plot nDCG@10 vs mean GPU joules/query as the escalated fraction f sweeps 0->1,
for the router ordering (out-of-fold predicted gain) against a random-escalation chord. Escalating f of
the queries to Full costs f * J_llm/query; nDCG(f) is the mean of (Full if escalated else tier B).

If the router curve tracks the random chord, no Full-tier spend is on an efficient frontier: the tier is
dominated -- expensive, tiny gain, and unroutable. CPU-only (reads two JSONs).

  python src/multivent2/mv2_frontier.py
"""
import os
import sys
import json
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ABL = os.path.join(_ROOT, "results", "ablations")
RNG = np.random.default_rng(0)


def mk():
    return Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])


def curve(order, base, full, jll, fracs):
    """nDCG and mean J/query along an escalation order (indices best-first)."""
    n = len(base)
    nd, cost = [], []
    for f in fracs:
        k = int(round(f * n))
        esc = np.zeros(n, bool); esc[order[:k]] = True
        nd.append(float(np.where(esc, full, base).mean()))
        cost.append(float(f * jll))
    return np.array(nd), np.array(cost)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=os.path.join(ABL, "mv2_full.json"))
    ap.add_argument("--energy", default=os.path.join(ABL, "mv2_energy.json"))
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_frontier.json"))
    a = ap.parse_args()
    d = json.load(open(a.full))
    e = json.load(open(a.energy))
    jll = e["gen_j_per_query_net"]

    qids = [q for q in d["features"] if q in d["per_query"]]
    X = np.array([d["features"][q] for q in qids])
    base = np.array([d["per_query"][q]["ndA"] for q in qids])        # tier B nDCG per query
    full = np.array([d["per_query"][q]["ndB"] for q in qids])        # Full nDCG per query
    g = full - base
    n = len(g)

    ghat = cross_val_predict(mk(), X, g, cv=KFold(5, shuffle=True, random_state=0))
    fracs = np.linspace(0, 1, 51)
    nd_r, cost = curve(np.argsort(-ghat), base, full, jll, fracs)     # router order
    # random chord: average many random escalation orders
    nd_rand = np.zeros_like(fracs)
    for _ in range(200):
        nd_rand += curve(RNG.permutation(n), base, full, jll, fracs)[0]
    nd_rand /= 200
    nd_oracle = curve(np.argsort(-g), base, full, jll, fracs)[0]      # cheating upper bound

    ndB, ndF = base.mean(), full.mean()
    area_gap = float(np.trapz(nd_r - nd_rand, cost))                  # router advantage over random (nDCG*J)
    j_per_point = jll / (100 * (ndF - ndB)) if ndF > ndB else float("inf")
    # fraction of the oracle's selective-escalation headroom the real router captures (at its peak f)
    fi = int(np.argmax(nd_oracle - nd_rand))
    head_oracle = float(nd_oracle[fi] - nd_rand[fi])
    head_router = float(nd_r[fi] - nd_rand[fi])
    frac_captured = head_router / head_oracle if head_oracle > 1e-9 else float("nan")

    out = {"j_llm_per_query": jll, "ndcgB": float(ndB), "ndcgFull": float(ndF),
           "j_per_ndcg10_point": j_per_point,
           "oracle_headroom_ndcg": head_oracle, "router_headroom_ndcg": head_router,
           "oracle_headroom_captured": frac_captured, "peak_frac": float(fracs[fi]),
           "fracs": fracs.tolist(), "cost_j_per_query": cost.tolist(),
           "ndcg_router": nd_r.tolist(), "ndcg_random": nd_rand.tolist(), "ndcg_oracle": nd_oracle.tolist(),
           "router_vs_random_area": area_gap}
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"measured LLM cost: {jll:.1f} J/query ({e.get('model','?')}, GPU1)")
    print(f"tier B nDCG@10 {ndB:.5f}  ->  Full (all escalated) {ndF:.5f}   (+{100*(ndF-ndB):.2f})")
    print(f"energy per nDCG@10 point gained: {j_per_point:.0f} J/query  (= {j_per_point*n/1000:.0f} kJ over {n} q)")
    print(f"\nfrontier (mean J/query -> nDCG@10):")
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = int(round(f * (len(fracs) - 1)))
        print(f"  f={f:<4} cost={cost[i]:6.1f} J   router {nd_r[i]:.5f}   random {nd_rand[i]:.5f}   "
              f"oracle {nd_oracle[i]:.5f}")
    print(f"\noracle can lift nDCG to {nd_oracle[fi]:.5f} (+{100*head_oracle:.2f}) by escalating the best "
          f"{100*fracs[fi]:.0f}% only;")
    print(f"the realizable router captures {100*frac_captured:.0f}% of that headroom "
          f"(+{100*head_router:.2f}).")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
