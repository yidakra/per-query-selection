#!/usr/bin/env python
"""Tier C, the honest version: can a LEARNED selector capture the oracle's +2.00?

The subset oracle keeps 0.9 of 24 paraphrases and gains +2.00 nDCG over Q2E's Full tier, of which
+1.31 is selection (which paraphrases) rather than routing (whether any). Oracles are label-fitted.
The routing study watched a +5.04 oracle ceiling collapse to +0.73 once a real router had to
predict the choice under nested CV. This holds the selector to that same bar.

PROTOCOL
  Outer 5-fold over queries. Within each training split, an inner 3-fold picks the decision
  threshold tau. The model never sees a test query's label, and tau is never chosen on the data it
  is scored on. Selections from the five test folds are concatenated and scored ONCE with the exact
  unfrozen fuse -- the same metric Q2E reports.

  Training target: membership in the greedy oracle subset, computed on TRAIN queries only.
  Features are label-free (they depend only on scores), so they may be computed for all queries.

TWO CONTROLS, either of which would sink the result:

  random    Per query, drop paraphrases uniformly at random, matched to the number the learned
            selector dropped. If max-pool noise is the whole story, ANY pruning helps and the
            learned model has contributed nothing. Averaged over N_SEEDS draws.
  heuristic Keep the top-k paraphrases per event type by their own max score, k tuned by the same
            nested CV. If this matches the model, the learned features are decoration.

COST. Every arm here scores all five components, so all sit at the Full tier's cost. Tier C does
not move a query left on the cost axis; it raises the frontier's right-hand endpoint. It spends no
LLM call beyond what Q2E already paid (~30 generations/query, see llm_cost_accounting.py): the
paraphrases are already generated, we are only choosing among them.

CPU-only.
"""
import os, sys, json, argparse, numpy as np, torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tracking import track                                            # noqa: E402
import tierC_selection_oracle as O                                    # noqa: E402

REPO = "/home/ubuntu/q2e_repro"
N_SEEDS = 20
TAUS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
KS = [0, 1, 2, 3, 5]
FEATS = ["mx", "mean", "std", "margin", "z_mx", "rank_mx", "n_para",
         "spearman_B", "overlap10_B", "ent", "is_argmax_any"]


def rank(x):
    r = np.empty_like(x, dtype=np.float64)
    r[np.argsort(x)] = np.arange(len(x), dtype=np.float64)
    return r


def build_features(comps, P, valid, D, T):
    """Label-free features for every valid (query, event, paraphrase)."""
    base = ["query_vs_video", "query_vs_captions"]
    rows, meta = [], []
    for q in range(T):
        b = O.fused_row({p: comps[p][q] for p in base}, D).numpy()
        rb, top10b = rank(b), set(np.argsort(-b)[:10].tolist())
        for e in O.EVENTS:
            ps = torch.nonzero(valid[e][q]).flatten().tolist()
            mxs = np.array([float(P[e][q, p].max()) for p in ps])
            # which paraphrase wins the max-pool for at least one video?
            stack = torch.stack([P[e][q, p] for p in ps]).numpy()      # (n_p, V)
            argmax_any = np.zeros(len(ps), bool)
            argmax_any[np.unique(stack.argmax(axis=0))] = True
            mu, sd = mxs.mean(), mxs.std() + 1e-9
            order = (-mxs).argsort().argsort()                          # 0 = best
            for i, p in enumerate(ps):
                r = stack[i]
                s = np.sort(r)[::-1]
                sm = np.exp(r - r.max()); sm /= sm.sum()
                rr = rank(r)
                sp = np.corrcoef(rr, rb)[0, 1]
                ov = len(top10b & set(np.argsort(-r)[:10].tolist())) / 10.0
                rows.append([s[0], r.mean(), r.std(), s[0] - s[1], (mxs[i] - mu) / sd,
                             float(order[i]), float(len(ps)), sp, ov,
                             float(-(sm * np.log(sm + 1e-12)).sum()), float(argmax_any[i])])
                meta.append((q, e, p))
    return np.asarray(rows, dtype=np.float64), meta


def sel_from_mask(meta, mask, T):
    sel = [[] for _ in range(T)]
    for (q, e, p), m in zip(meta, mask):
        if m:
            sel[q].append((e, p))
    return sel


def exact_ndcg(P, valid, comps, sel, target):
    c = O.build_comps_from_selection(P, valid, comps, sel)
    return O.per_query(O.fuse(c, O.LADDER["C_full"]), target)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="noASR")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    with track(f"tierC-learned-{a.setting}", gpu_ids=[], tags=["tierC"],
               config={"setting": a.setting, "folds": a.folds, "n_seeds": N_SEEDS,
                       "taus": TAUS, "ks": KS}) as tr:
        queries, video_ids, target, comps, P, valid, nd_pub, delta = O.load_cell(a.setting)
        T = len(queries)
        D = O.frozen_softmax_parts(comps)

        ndB = exact_ndcg(P, valid, comps, [[] for _ in range(T)], target)
        allsel = [[(e, p) for e in O.EVENTS for p in torch.nonzero(valid[e][q]).flatten().tolist()]
                  for q in range(T)]
        ndF = exact_ndcg(P, valid, comps, allsel, target)

        print("  building features...", flush=True)
        X, meta = build_features(comps, P, valid, D, T)
        qidx = np.array([m[0] for m in meta])
        print(f"  {X.shape[0]} paraphrase instances x {X.shape[1]} features", flush=True)

        orc = json.load(open(f"{REPO}/results/ablations/tierC_selection_oracle_{a.setting}.json"))
        okeep = {(q, e, p) for q, s in enumerate(orc["selection"]) for (e, p) in
                 [tuple(x) for x in s]}
        y = np.array([1 if m in okeep else 0 for m in meta], dtype=int)
        print(f"  oracle keeps {y.sum()}/{len(y)} paraphrases ({100*y.mean():.1f}%)", flush=True)

        def fit_predict(tr_q, te_q):
            trm, tem = np.isin(qidx, tr_q), np.isin(qidx, te_q)
            clf = GradientBoostingClassifier(random_state=0)
            clf.fit(X[trm], y[trm])
            return clf, trm, tem

        def score_mask(mask_global, sel_q=None):
            sel = sel_from_mask(meta, mask_global, T)
            return exact_ndcg(P, valid, comps, sel, target)

        outer = KFold(n_splits=a.folds, shuffle=True, random_state=0)
        m_learn = np.zeros(len(y), bool)
        m_heur = np.zeros(len(y), bool)
        chosen_tau, chosen_k = [], []

        for tr_i, te_i in outer.split(np.arange(T)):
            # ---- inner CV picks tau (model) and k (heuristic) on TRAIN queries only ----
            inner = KFold(n_splits=3, shuffle=True, random_state=1)
            tau_sc = {t: [] for t in TAUS}
            k_sc = {k: [] for k in KS}
            for itr, ite in inner.split(tr_i):
                a_q, b_q = tr_i[itr], tr_i[ite]
                clf, _, _ = fit_predict(a_q, b_q)
                bm = np.isin(qidx, b_q)
                b_set = set(b_q.tolist())
                pr = clf.predict_proba(X[bm])[:, 1]

                def score_val(mask_val):
                    """Apply the candidate selection to inner-val queries; every other query keeps
                    ALL paraphrases. The softmax denominators in fuse() run across queries, so the
                    non-val rows are not inert -- they must be held to a FIXED reference (Full)
                    rather than left empty, or tau would be tuned against a shifting context."""
                    mg = np.zeros(len(y), bool); mg[bm] = mask_val
                    sel = sel_from_mask(meta, mg, T)
                    for q in range(T):
                        if q not in b_set:
                            sel[q] = allsel[q]
                    return exact_ndcg(P, valid, comps, sel, target)[b_q].mean()

                for t in TAUS:
                    tau_sc[t].append(float(score_val(pr >= t)))
                rk = X[bm][:, FEATS.index("rank_mx")]
                for k in KS:
                    k_sc[k].append(float(score_val(rk < k)))

            t_star = max(TAUS, key=lambda t: np.mean(tau_sc[t]))
            k_star = max(KS, key=lambda k: np.mean(k_sc[k]))
            chosen_tau.append(t_star); chosen_k.append(k_star)

            clf, _, tem = fit_predict(tr_i, te_i)
            pr = clf.predict_proba(X[tem])[:, 1]
            m_learn[tem] = pr >= t_star
            m_heur[tem] = X[tem][:, FEATS.index("rank_mx")] < k_star

        ndL = score_mask(m_learn)
        ndH = score_mask(m_heur)

        # ---- cost-matched random control ----
        rnd = []
        for s in range(N_SEEDS):
            rs = np.random.RandomState(1000 + s)
            mr = np.zeros(len(y), bool)
            for q in range(T):
                for e in O.EVENTS:
                    idx = np.array([i for i, m in enumerate(meta) if m[0] == q and m[1] == e])
                    if not len(idx):
                        continue
                    nkeep = int(m_learn[idx].sum())
                    if nkeep:
                        mr[rs.choice(idx, nkeep, replace=False)] = True
            rnd.append(float(score_mask(mr).mean()))
        rnd = np.array(rnd) * 100

        ndSUB = exact_ndcg(P, valid, comps,
                           [[tuple(x) for x in s] for s in orc["selection"]], target)

        f = lambda x: float(x.mean()) * 100
        gB, gF, gL, gH, gO = f(ndB), f(ndF), f(ndL), f(ndH), f(ndSUB)
        print(f"\n### tier-C learned selector, nested CV   MultiVENT {a.setting}  (T={T})\n")
        print(f"  Fixed-B                              {gB:.2f}")
        print(f"  Fixed-Full  (Q2E, all paraphrases)   {gF:.2f}")
        print(f"  random prune (cost-matched)          {rnd.mean():.2f} +- {rnd.std():.2f}"
              f"   [vs Full {rnd.mean()-gF:+.2f}]")
        print(f"  heuristic top-k by own max score      {gH:.2f}   [vs Full {gH-gF:+.2f}]")
        print(f"  LEARNED selector (nested CV)          {gL:.2f}   [vs Full {gL-gF:+.2f}]")
        print(f"  oracle subset (upper bound)          {gO:.2f}   [vs Full {gO-gF:+.2f}]")
        realised = (gL - gF) / (gO - gF) * 100 if gO != gF else float("nan")
        print(f"\n  learned selector realises {realised:.0f}% of the oracle gain")
        print(f"  learned vs random control: {gL - rnd.mean():+.2f} "
              f"({(gL-rnd.mean())/(rnd.std()+1e-9):.1f} sd of the random draw)")
        print(f"  kept: learned {m_learn.sum()}, heuristic {m_heur.sum()}, "
              f"oracle {y.sum()}, all {len(y)}")
        print(f"  tau* per fold {chosen_tau} | k* per fold {chosen_k}")

        # paired bootstrap on per-query nDCG, learned vs Fixed-Full
        d = (ndL - ndF).numpy()
        rs = np.random.RandomState(0)
        bs = np.array([d[rs.randint(0, T, T)].mean() for _ in range(5000)]) * 100
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  learned - Full: {d.mean()*100:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

        out = {"setting": a.setting, "T": T, "fixedB": gB, "fixedFull": gF,
               "random_prune_mean": float(rnd.mean()), "random_prune_sd": float(rnd.std()),
               "heuristic_topk": gH, "learned": gL, "oracle_subset": gO,
               "realised_pct_of_oracle": float(realised),
               "learned_minus_full": float(d.mean() * 100), "ci95": [float(lo), float(hi)],
               "tau_star_per_fold": chosen_tau, "k_star_per_fold": chosen_k,
               "kept": {"learned": int(m_learn.sum()), "heuristic": int(m_heur.sum()),
                        "oracle": int(y.sum()), "all": int(len(y))}}
        of = f"{REPO}/results/ablations/tierC_learned_{a.setting}.json"
        json.dump(out, open(of, "w"), indent=2)
        print(f"\nwrote {of}")
        tr.summary({k: v for k, v in out.items() if not isinstance(v, (list, dict))})


if __name__ == "__main__":
    main()
