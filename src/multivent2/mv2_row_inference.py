"""Per-row inference for Table 1: turns the margin counts into tests.

Two complaints from review, answered together. The underline rule (routed > Original + 5e-4) is a
margin over point estimates with no uncertainty, and the 0-of-33 null carries no equivalence
argument. This script reads the symmetric nested-calibration table's stored decisions, rebuilds each
row's per-query routed value from the cell's own per-query nDCG, and reports per row:

- a one-sided sign-flip test of routed > Original at the event-group level (the exchangeable unit),
- Holm-corrected significance across each family's positive claims,
- a cluster-bootstrap 95% CI on the mean difference, and whether it sits inside +/- delta
  (default 0.005 nDCG, the smallest effect of interest), which is the TOST-style equivalence
  statement for the null family.

  python src/multivent2/mv2_row_inference.py
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from mv2_qpp_table import CELLS  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
RNG = np.random.default_rng(0)


def group_stats(diff, grp, n_draws=2000):
    """(one-sided sign-flip p for mean > 0, cluster-bootstrap 95% CI on the mean)."""
    groups = np.unique(grp)
    dg = np.array([diff[grp == g].mean() for g in groups])
    wg = np.array([(grp == g).sum() for g in groups], dtype=float)
    wg /= wg.sum()
    obs = float((wg * dg).sum())
    cnt = 0
    for _ in range(n_draws):
        s = RNG.choice([-1.0, 1.0], size=len(dg))
        if float((wg * dg * s).sum()) >= obs:
            cnt += 1
    p = float((1 + cnt) / (n_draws + 1))
    boots = []
    for _ in range(n_draws):
        pick = RNG.integers(0, len(groups), len(groups))
        boots.append(float((wg[pick] * dg[pick]).sum() / wg[pick].sum()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return p, obs, float(lo), float(hi)


def holm(pvals):
    """Holm step-down: returns adjusted p-values in the original order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="mv2_qpp_table_sym_grouped.json")
    ap.add_argument("--delta", type=float, default=0.005,
                    help="smallest effect of practical interest for the equivalence statement")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_row_inference.json"))
    a = ap.parse_args()

    table = json.load(open(os.path.join(ABL, a.table)))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))

    out = {"table": a.table, "delta": a.delta, "rows": []}
    fam_pvals = {"pre": [], "post": []}
    fam_keys = {"pre": [], "post": []}
    for label, fn in CELLS:
        if label not in table:
            continue
        cell = json.load(open(os.path.join(ABL, fn)))
        qids = table[label]["qids"]
        ndA = np.array([cell["per_query"][q]["ndA"] for q in qids])
        ndB = np.array([cell["per_query"][q]["ndB"] for q in qids])
        grp = event_groups(qids, qrels)
        # Original: the better fixed endpoint, per query
        fixed = ndB if table[label]["uniform"] >= table[label]["cheap"] else ndA

        for fam in ("pre", "post"):
            for name, dec in table[label]["decisions"].get(fam, {}).items():
                esc = np.frombuffer(dec.encode(), dtype=np.uint8) == ord("1")
                routed = np.where(esc, ndB, ndA)
                p, obs, lo, hi = group_stats(routed - fixed, grp)
                equiv = bool(-a.delta < lo and hi < a.delta)
                out["rows"].append({"cell": label, "family": fam, "predictor": name,
                                    "mean_diff": obs, "p_signflip_group": p,
                                    "ci95": [lo, hi], "within_delta": equiv})
                fam_pvals[fam].append(p)
                fam_keys[fam].append((label, name))

    for fam in ("pre", "post"):
        adj = holm(np.array(fam_pvals[fam]))
        lookup = {k: float(v) for k, v in zip(fam_keys[fam], adj)}
        for r in out["rows"]:
            if r["family"] == fam:
                r["p_holm"] = lookup[(r["cell"], r["predictor"])]

    n_sig = {"pre": 0, "post": 0}
    n_equiv = {"pre": 0, "post": 0}
    n_tot = {"pre": 0, "post": 0}
    for r in out["rows"]:
        f = r["family"]
        n_tot[f] += 1
        n_sig[f] += r["p_holm"] < 0.05
        n_equiv[f] += r["within_delta"]
        flag = "SIG " if r["p_holm"] < 0.05 else ("equiv" if r["within_delta"] else "     ")
        print(f"{r['cell']:12s} {r['family']:4s} {r['predictor']:10s} "
              f"diff={r['mean_diff']:+.4f} CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"p={r['p_signflip_group']:.4f} holm={r['p_holm']:.4f} {flag}")
    for f in ("pre", "post"):
        print(f"{f}: {n_sig[f]}/{n_tot[f]} significant above Original after Holm; "
              f"{n_equiv[f]}/{n_tot[f]} equivalent to Original within +/-{a.delta}")
    out["summary"] = {f: {"significant_holm": n_sig[f], "equivalent": n_equiv[f], "of": n_tot[f]}
                      for f in ("pre", "post")}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
