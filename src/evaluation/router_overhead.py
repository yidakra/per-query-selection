#!/usr/bin/env python
"""The router's own online cost, so the frontier x-axis is router-INCLUSIVE.

Every frontier figure charges tier-A/B similarity energy but not the cost of the routing DECISION
itself. A reviewer will ask whether that overhead eats the saving at the 15.5 J A->B escalation. This
measures it.

Per query, the decision is:
  1. conf_feats(fA[i]) -- softmax / sort / std over the V-dim gallery score vector tier A ALREADY
     produced for ranking (that vector is the tier-A cost we already count; it is not re-charged here).
  2. StandardScaler + ridge dot product over ~10 features -- a handful of FLOPs.

Pure CPU tensor ops. Timed on real MultiVENT noASR score vectors (T=259, V=2393), averaged over many
repeats. Reported as wall-clock per decision and as a CPU-TDP energy estimate (no RAPL on this host --
same caveat as cost_model_findings.md's CPU figures; we bracket a range of single-core power). Compared
to the measured A->B marginal (15.52 J) and Full (1418.36 J).
"""
import os, sys, time, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_hetero import comps_multivent, mk, A_COLS  # noqa: E402
from oracle_router_headroom import fuse, per_query, LADDER  # noqa: E402
from router_cascade_exp import conf_feats  # noqa: E402
from tier_cost import JOULES  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{_ROOT}/results/ablations/router_overhead.json"

REPEATS = 200                         # over all T queries -> 200*T timed decisions
CPU_W_RANGE = (5.0, 25.0)             # single active core, TDP-estimate bracket (W); cost_model caveat
J_ATOB = JOULES["-Events"] - JOULES["A_visual"]   # 15.52 J measured A->B marginal
J_A = JOULES["A_visual"]                            # 6.83 J
J_FULL = JOULES["Full"]                            # 1418.36 J


def build(setting="noASR"):
    queries, target, comps = comps_multivent(setting)
    fA = fuse(comps, LADDER["A_visual"]); fB = fuse(comps, LADDER["B_noevents"])
    ndA = per_query(fA, target)[0].numpy(); ndB = per_query(fB, target)[0].numpy()
    g = ndB - ndA
    rows = []
    for i in range(len(queries)):
        r = {"charlen": len(queries[i]), "wordlen": len(queries[i].split())}
        for k, v in conf_feats(fA[i]).items():
            r[f"A_{k}"] = v
        rows.append(r)
    X = np.array([[r[c] for c in A_COLS] for r in rows], dtype=float)
    return fA, queries, X, g


def main():
    fA, queries, X, g = build("noASR")
    T, V = fA.shape
    pipe = mk().fit(X, g)                                   # the deployed router
    sc, m = pipe["sc"], pipe["m"]
    mean_, scale_, coef_, b_ = sc.mean_, sc.scale_, m.coef_, m.intercept_

    # ---- 1. feature extraction: conf_feats over the gallery vector, per query ----
    for i in range(T):                                     # warm up
        conf_feats(fA[i])
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        for i in range(T):
            conf_feats(fA[i])
    t_feat = (time.perf_counter() - t0) / (REPEATS * T)

    # ---- 2. ridge inference: standardize + dot, as raw numpy (intrinsic FLOPs, no sklearn overhead) ----
    xs = X.copy()
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        for i in range(T):
            _ = float(((xs[i] - mean_) / scale_) @ coef_ + b_)
    t_pred = (time.perf_counter() - t0) / (REPEATS * T)

    # sklearn .predict(1 row) as an "as-implemented" upper bound (dominated by Python dispatch)
    row = X[:1]
    for _ in range(50):
        pipe.predict(row)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        for i in range(T):
            pipe.predict(X[i:i + 1])
    t_pred_sklearn = (time.perf_counter() - t0) / (REPEATS * T)

    t_decision = t_feat + t_pred                           # intrinsic per-query router overhead
    e_lo = t_decision * CPU_W_RANGE[0]                     # J, low CPU-power estimate
    e_hi = t_decision * CPU_W_RANGE[1]                     # J, high
    e_mid = t_decision * float(np.mean(CPU_W_RANGE))

    out = {
        "cell": "MultiVENT/noASR", "T": T, "V": V, "repeats": REPEATS,
        "t_feat_us": t_feat * 1e6, "t_pred_us": t_pred * 1e6,
        "t_pred_sklearn_us": t_pred_sklearn * 1e6, "t_decision_us": t_decision * 1e6,
        "cpu_w_range": CPU_W_RANGE,
        "e_router_mJ": {"lo": e_lo * 1e3, "mid": e_mid * 1e3, "hi": e_hi * 1e3},
        "J_A": J_A, "J_AtoB_marginal": J_ATOB, "J_Full": J_FULL,
        # the router pays its overhead on EVERY query; escalation only on the top-f
        "overhead_vs_AtoB_pct": 100 * e_mid / J_ATOB,
        "overhead_vs_A_pct": 100 * e_mid / J_A,
        "overhead_vs_Full_pct": 100 * e_mid / J_FULL,
        "AtoB_over_router_factor": J_ATOB / e_mid,
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print(f"router decision on real score vectors (MultiVENT noASR, T={T}, V={V}):")
    print(f"  feature extraction  conf_feats : {t_feat*1e6:8.2f} us/query")
    print(f"  ridge inference (raw dot)      : {t_pred*1e6:8.4f} us/query")
    print(f"  ridge inference (sklearn .predict, upper bound): {t_pred_sklearn*1e6:8.2f} us/query")
    print(f"  -> per-query decision          : {t_decision*1e6:8.2f} us")
    print(f"\nenergy (CPU-TDP estimate, {CPU_W_RANGE[0]:.0f}-{CPU_W_RANGE[1]:.0f} W single core):")
    print(f"  E_router  ~ {e_mid*1e3:.4f} mJ/query  ({e_lo*1e3:.4f}-{e_hi*1e3:.4f} mJ)")
    print(f"\nrouter-inclusive frontier:")
    print(f"  A->B marginal (measured)       : {J_ATOB:.2f} J")
    print(f"  router overhead / A->B marginal: {100*e_mid/J_ATOB:.5f} %   "
          f"(A->B is {J_ATOB/e_mid:,.0f}x the router)")
    print(f"  router overhead / tier A (6.83): {100*e_mid/J_A:.5f} %")
    print(f"  router overhead / Full (1418)  : {100*e_mid/J_FULL:.7f} %")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
