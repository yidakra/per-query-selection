"""Risk-coverage curves and AURC for escalation routing (the selective-prediction view).

The mapping from selective prediction to cascading is exact: a cascade that answers a query from the
cheap tier is *abstaining* from the expensive one, so

  coverage c  = fraction of queries answered by the cheap tier (i.e. 1 - escalation fraction f)
  risk R(c)   = mean regret on those covered queries = mean(nDCG_expensive - nDCG_cheap) over them
  AURC        = area under R(c) as c sweeps 1 -> 0

Lower AURC is better. A router that abstains well answers cheaply exactly where the cheap tier loses
nothing, so its risk falls fast as it starts escalating. Reported against two references: the oracle
(escalate by true gain -- the attainable floor) and random escalation (the no-skill line, whose risk is
flat at the mean regret). E-AURC = AURC(router) - AURC(oracle) is the excess a perfect predictor would
remove, and the normalized version reports the share of that excess the router already closes.

Predictions are out-of-fold (KFold-5 ridge on cheap-tier confidence features), so nothing here is fit
on the queries it is scored on. CPU-only, reads the router-ready JSONs other scripts emit.

  python src/multivent2/mv2_riskcov.py --in results/ablations/mv2_ab_dense.json --cell A_to_B
  python src/multivent2/mv2_riskcov.py --in results/ablations/mv2_full_qwen14b.json --cell B_to_Full
"""
import os
import json
import argparse
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ABL = os.path.join(_ROOT, "results", "ablations")


def risk_curve(order, gain, cvg):
    """Risk at each coverage: escalate along `order` (best-first), risk = mean regret on the rest."""
    n = len(gain)
    out = []
    for c in cvg:
        k = int(round((1.0 - c) * n))          # escalated count
        covered = np.ones(n, bool); covered[order[:k]] = False
        out.append(float(gain[covered].mean()) if covered.any() else 0.0)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(ABL, "mv2_ab_dense.json"))
    ap.add_argument("--cell", default="A_to_B", help="label for the JSON")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_riskcov.json"))
    ap.add_argument("--append", action="store_true", help="merge into an existing --out instead of overwriting")
    a = ap.parse_args()

    d = json.load(open(a.inp))
    qids = [q for q in d["features"] if q in d["per_query"]]
    X = np.array([d["features"][q] for q in qids])
    cheap = np.array([d["per_query"][q]["ndA"] for q in qids])
    exp_ = np.array([d["per_query"][q]["ndB"] for q in qids])
    g = exp_ - cheap                                   # regret of answering cheaply
    n = len(g)

    ghat = cross_val_predict(Pipeline([("sc", StandardScaler()),
                                       ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))]),
                             X, g, cv=KFold(5, shuffle=True, random_state=0))
    cvg = np.linspace(1.0, 0.0, 101)
    r_router = risk_curve(np.argsort(-ghat), g, cvg)
    r_oracle = risk_curve(np.argsort(-g), g, cvg)
    rng = np.random.default_rng(0)
    r_rand = np.mean([risk_curve(rng.permutation(n), g, cvg) for _ in range(200)], axis=0)

    # AURC over coverage (integrate in increasing coverage; curves are stored decreasing)
    def aurc(r):
        return float(np.trapz(r[::-1], cvg[::-1]))
    A_router, A_oracle, A_rand = aurc(r_router), aurc(r_oracle), aurc(r_rand)
    e_router, e_rand = A_router - A_oracle, A_rand - A_oracle
    closed = (e_rand - e_router) / e_rand if abs(e_rand) > 1e-12 else float("nan")

    rec = {"cell": a.cell, "source": os.path.basename(a.inp), "n": n,
           "mean_regret": float(g.mean()), "sd_regret": float(g.std()),
           "aurc_router": A_router, "aurc_oracle": A_oracle, "aurc_random": A_rand,
           "e_aurc_router": e_router, "e_aurc_random": e_rand,
           "excess_risk_closed": float(closed),
           "coverage": cvg.tolist(), "risk_router": r_router.tolist(),
           "risk_oracle": r_oracle.tolist(), "risk_random": r_rand.tolist()}

    all_ = {}
    if a.append and os.path.exists(a.out):
        all_ = json.load(open(a.out))
    all_[a.cell] = rec
    json.dump(all_, open(a.out, "w"), indent=2)

    print(f"{a.cell}  (n={n}, mean regret {100*g.mean():+.2f} nDCG@10 points)")
    print(f"  AURC  router {100*A_router:.3f}   oracle {100*A_oracle:.3f}   random {100*A_rand:.3f}")
    print(f"  E-AURC router {100*e_router:.3f} vs random {100*e_rand:.3f}  "
          f"-> router closes {100*closed:.1f}% of the excess risk a perfect router would remove")
    print("  risk at selected coverages (nDCG@10 points of regret on the covered set):")
    for c in (1.0, 0.9, 0.8, 0.75, 0.5):
        i = int(round((1.0 - c) * (len(cvg) - 1)))
        print(f"    c={c:<5} router {100*r_router[i]:+.2f}   random {100*r_rand[i]:+.2f}   "
              f"oracle {100*r_oracle[i]:+.2f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
