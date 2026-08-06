"""Does rank correlation predict selection utility? Measured over every row of the RQ2 table.

QPP has been evaluated by rank correlation for two decades: a predictor is good if its ordering of
queries matches the ordering by realised effectiveness. Arabzadeh et al. re-purpose predictors as
selectors, and that changes the quantity that matters. A selector is used once per query, to make one
binary decision, and only the sign of its prediction at the decision boundary is consulted. Kendall tau
scores the whole ordering; the decision reads one point of it.

Those two need not agree.  The original zero-crossing analysis found a negative relationship, but that
comparison confounded ranking quality with an uncalibrated operating point for the learned methods.
The optional nested mode replaces QSD and BERT-QPP with their group-disjoint, nested-calibration runs so
that we can measure which part of that conclusion survives honest operating-point selection.

Utility is measured against the best FIXED policy in the cell, not against the cheap channel. A selector
that beats cheap-only but loses to always-fusing has bought nothing: the practitioner would have run the
fixed policy and skipped the predictor. Reporting against cheap-only instead flatters every row in the
two cells where fusing helps on average.

Three populations, because the artifact objection has to be answered rather than asserted:
  all rows                      includes predictors that make one decision for every query
  non-degenerate                excludes them
  escalating between 5 and 95%  excludes the near-degenerate ones too

CPU only, reads three JSONs.

  python src/multivent2/mv2_qpp_utility.py [--tag _grouped] [--nested]
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy.stats import kendalltau, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
ABL = os.path.join(_ROOT, "results", "ablations")

SNAKE = {"ASR-shipped": "asr_shipped", "ASR-dense": "asr_dense", "OCR": "ocr"}


def collect(tag, nested=False):
    """-> list of rows, one per (predictor, cell). Each carries its cell's two fixed policies too, so
    utility can be computed against whichever of them is stronger there."""
    tab = json.load(open(os.path.join(ABL, f"mv2_qpp_table{tag}.json")))
    qsd = json.load(open(os.path.join(
        ABL, "mv2_qsd_pre_nested_grouped.json" if nested else f"mv2_qsd{tag}.json")))
    bert = json.load(open(os.path.join(
        ABL, "mv2_bertqpp_cross_3ep_nested_grouped.json" if nested else f"mv2_bertqpp{tag}.json")))
    p_bi = os.path.join(ABL, "mv2_bertqpp_bi_3ep_nested_grouped.json"
                        if nested else f"mv2_bertqpp_bi{tag}.json")
    bert_bi = json.load(open(p_bi)) if os.path.exists(p_bi) else {}
    p_qp = os.path.join(ABL, "mv2_qsd_post_5ep_nested_grouped.json"
                        if nested else f"mv2_qsd_post{tag}.json")
    qsd_post = json.load(open(p_qp)) if os.path.exists(p_qp) else {}
    # QSD's neighbourhood size is chosen per split scheme, matching how the table reports it
    qsd_k = "100_inv_dist" if tag.endswith("_grouped") or nested else "5_inv_dist"

    rows = []
    for cell, c in tab.items():
        if not isinstance(c, dict) or "post" not in c:
            continue
        cheap, uni = c["cheap"], c["uniform"]

        def add(block, name, tau, routed, frac):
            rows.append({"cell": cell, "block": block, "name": name, "tau": tau,
                         "routed": routed, "frac": frac, "cheap": cheap, "uniform": uni})

        for block in ("pre", "post"):
            for name, v in c.get(block, {}).items():
                add(block, name, v[1], v[0], v[3])
        add("ours", "cheap-feature ridge", c["ours"][1], c["ours"][0], c["ours"][3])
        sn = SNAKE[cell]
        q = qsd[sn]["k"][qsd_k]
        add("pre", "QSD_pre", q["tau"], q["routed_ndcg10"], q["frac_escalated"])
        b = bert[sn]
        add("post", "BERT-QPP", b["tau"], b["routed_ndcg10"], b.get("frac_escalated"))
        if sn in bert_bi:
            bb = bert_bi[sn]
            add("post", "BERT-QPP_bi", bb["tau"], bb["routed_ndcg10"], bb.get("frac_escalated"))
        if sn in qsd_post:
            qp = qsd_post[sn]
            add("post", "QSD_post", qp["tau"], qp["routed_ndcg10"], qp.get("frac_escalated"))
    return rows


def util(r):
    """Routed nDCG minus the better of the two fixed policies available in that cell."""
    return r["routed"] - max(r["cheap"], r["uniform"])


def report(rows, label):
    if len(rows) < 4:
        return None
    t = np.array([r["tau"] for r in rows])
    u = np.array([util(r) for r in rows])
    kt, pr = kendalltau(t, u), pearsonr(t, u)
    print(f"  {label:<34} n={len(rows):3d}  kendall={kt.statistic:+.3f} (p={kt.pvalue:.3g})"
          f"   pearson={pr.statistic:+.3f} (p={pr.pvalue:.3g})")
    return {"n": len(rows), "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue),
            "pearson": float(pr.statistic), "pearson_p": float(pr.pvalue)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_grouped")
    ap.add_argument("--nested", action="store_true",
                    help="use the matched nested-calibration QSD and BERT-QPP artifacts")
    a = ap.parse_args()

    rows = collect(a.tag, a.nested)
    print(f"{len(rows)} predictor-cell rows\n")

    print("Kendall tau against utility over the best fixed policy:")
    out = {"tag": a.tag, "nested_calibration": a.nested,
           "n_rows": len(rows), "populations": {}}
    pops = [("all rows", rows),
            ("non-degenerate only", [r for r in rows if r["frac"] not in (0.0, 1.0, None)]),
            ("escalating 5-95%", [r for r in rows
                                  if r["frac"] is not None and 0.05 <= r["frac"] <= 0.95])]
    for label, sel in pops:
        out["populations"][label] = report(sel, label)

    # the same thing computed against cheap-only, which is the flattering denominator, so the choice
    # of denominator is on the record rather than buried
    t = np.array([r["tau"] for r in rows])
    uc = np.array([r["routed"] - r["cheap"] for r in rows])
    kt = kendalltau(t, uc)
    print(f"\n  (against cheap-only instead:        n={len(rows):3d}  "
          f"kendall={kt.statistic:+.3f} (p={kt.pvalue:.3g}))")
    out["vs_cheap_only"] = {"kendall": float(kt.statistic), "kendall_p": float(kt.pvalue)}

    print("\nPer cell, which removes any cell-level offset:")
    for cell in dict.fromkeys(r["cell"] for r in rows):
        sel = [r for r in rows if r["cell"] == cell]
        nd = [r for r in sel if r["frac"] not in (0.0, 1.0, None)]
        st = report(sel, cell)
        if st:
            st["non_degenerate"] = len(nd)
            out["populations"][f"cell:{cell}"] = st

    deg = [r for r in rows if r["frac"] in (0.0, 1.0)]
    beat = [r for r in rows if util(r) > 1e-6]
    print(f"\n{len(deg)}/{len(rows)} rows make one decision for every query "
          f"({100*len(deg)/len(rows):.0f}%)")
    print(f"{len(beat)}/{len(rows)} rows beat the best fixed policy in their cell")
    out["degenerate"] = len(deg)
    out["beat_best_fixed"] = len(beat)

    print(f"\nHighest tau in the table, with what each one actually bought:")
    print(f"  {'cell':<12}{'block':<7}{'method':<22}{'tau':>8}{'routed':>9}{'bestfix':>9}"
          f"{'utility':>9}{'esc':>7}")
    for r in sorted(rows, key=lambda r: -r["tau"])[:10]:
        bf = max(r["cheap"], r["uniform"])
        fr = "n/a" if r["frac"] is None else f"{r['frac']:.2f}"
        print(f"  {r['cell']:<12}{r['block']:<7}{r['name']:<22}{r['tau']:>+8.3f}{r['routed']:>9.4f}"
              f"{bf:>9.4f}{util(r):>+9.4f}{fr:>7}")

    out["rows"] = rows
    stem = "mv2_qpp_utility_nested" if a.nested else "mv2_qpp_utility"
    dest = os.path.join(ABL, f"{stem}{a.tag}.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
