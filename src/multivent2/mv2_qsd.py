"""QSD-QPP as a routing baseline: Bigdeli et al., "Query Performance Prediction Using Neural Query
Space Proximity", ACM TIST, doi:10.1145/3762197. This is the QSD_pre / QSD_post entry in the suite, whose
repository ships only consumers of precomputed QSD outputs, so the method is implemented here from the
paper's equations.

QSD-QPP_Pre (their Eqs. 5-7): embed queries into a Query Space, collect the historical queries closest
to the new query, and interpolate their *known* effectiveness, weighted by inverse distance

    M_hat(q) = sum_{(e(qi), M_qi) in S_q}  w(e(q), e(qi)) * M_qi        (Eq. 5)
    w(e(q), e(qi)) = 1 / (1 + psi(e(q), e(qi)))                        (Eq. 6)
    uniform variant: M_hat(q) = mean of M_qi over S_q                  (Eq. 7)

Two deviations, both stated in the table notes. Their subspace is everything inside a distance
threshold gamma; we take the k nearest instead, because a fixed gamma transfers badly across embedding
spaces and k is the form their post-retrieval variant already uses. And Eq. 5's weights are unnormalized,
which makes the prediction scale with neighbour count, so we normalize them (Shepard interpolation);
the uniform variant of Eq. 7 is reported alongside.

Why this predictor matters for RQ4. Every other pre-retrieval predictor in our table (IDF, ICTF, SCQ,
SCS) needs corpus term statistics, which do not exist for a visual channel, and all of them score
tau ~ 0 here. QSD_pre needs no index at all, only historical queries with known effectiveness, so it is
a pre-retrieval predictor that *is* available in the multimodal setting. If it works where the
term-statistic family fails, the boundary is not pre-retrieval versus post-retrieval but whether a
predictor needs document-side language statistics.

The "historical queries with known effectiveness" are the training folds: neighbours are drawn only from
them and predictions are made on the held-out fold, so nothing leaks. The target is the escalation gain,
matching every other predictor in our table rather than their absolute-metric target.

QSD_post (their Eq. 8) needs a trained transformer over the query, its neighbours with their scores, and
the retrieved documents. It lives in `mv2_qsd_post.py`, because it is a training job rather than a
closed-form interpolation. Worth reading the two rows together: QSD_post is this predictor plus document
evidence, and it scores *worse* on both metrics in all three cells, which is the RQ4 boundary reproduced
inside one family instead of across families.

  python src/multivent2/mv2_qsd.py
"""
import os, sys, json, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # before torch: both GPUs are in use elsewhere
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np                            # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_queries, load_run       # noqa: E402
from mv2_recall_sidecar import load_cell_recall  # noqa: E402
from mv2_qpp_table import bits                   # noqa: E402
from scipy.stats import kendalltau             # noqa: E402
from sklearn.model_selection import KFold      # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

CELLS = {"asr_shipped": "mv2_chan_visual_to_asr.json",
         "asr_dense":   "mv2_chan_visual_to_asr_dense_m3.json",
         "ocr":         "mv2_chan_visual_to_ocr_dense_m3.json"}


def qsd_predict(E, g, tr, te, k, weighted=True):
    """Interpolate held-out gain from the k nearest training queries in embedding space."""
    Etr, Ete = E[tr], E[te]
    # cosine distance psi in [0, 2]; embeddings are L2-normalised
    sims = Ete @ Etr.T
    psi = 1.0 - sims
    idx = np.argsort(psi, axis=1)[:, :k]
    out = np.zeros(len(te))
    for i in range(len(te)):
        nb = idx[i]
        m = g[tr][nb]
        if weighted:
            w = 1.0 / (1.0 + psi[i, nb])       # Eq. 6
            out[i] = float((w * m).sum() / w.sum())
        else:
            out[i] = float(m.mean())           # Eq. 7
    return out


def event_groups(qids, qrels):
    """Group queries that describe the same event, so a duplicate cannot sit across a fold boundary.

    MultiVENT 2.0 carries several phrasings of one event ("New York Times coverage Hurricane Irma" and
    "2017 Hurricane Irma" have identical relevant sets). Queries are linked when they share any relevant
    document and grouped by connected component. Without this, QSD's historical-query interpolation
    reads its own answer off a near-duplicate in another fold: the nearest neighbour shares 60% of its
    relevant documents on average against 0.2% for a random query.
    """
    rel = {q: {d for d, r in qrels.get(q, {}).items() if r > 0} for q in qids}
    doc2q = {}
    for q in qids:
        for d in rel[q]:
            doc2q.setdefault(d, []).append(q)
    parent = {q: q for q in qids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for d, qs in doc2q.items():
        for other in qs[1:]:
            union(qs[0], other)
    roots = {}
    return np.array([roots.setdefault(find(q), len(roots)) for q in qids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--ks", default="5,10,25,50,100")
    ap.add_argument("--group-cv", action="store_true",
                    help="split by event group instead of by query, removing duplicate-topic leakage")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_qsd.json"))
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.model_selection import GroupKFold
    from mv2_io import load_qrels
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    m = SentenceTransformer(a.model, device="cpu")
    visual = load_run(os.path.join(DATA, "10pyscene_clip.json"))

    res = {"group_cv": bool(a.group_cv)}
    for cell, fn in CELLS.items():
        d = json.load(open(os.path.join(ABL, fn)))
        qids = [q for q in d["per_query"] if q in queries]
        ndA = np.array([d["per_query"][q]["ndA"] for q in qids])
        ndB = np.array([d["per_query"][q]["ndB"] for q in qids])
        g = ndB - ndA
        E = m.encode([queries[q] for q in qids], batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
        if a.group_cv:
            grp = event_groups(qids, qrels)
            splits = list(GroupKFold(5).split(np.arange(len(qids)), groups=grp))
            print(f"{cell}: {len(qids)} queries in {len(set(grp))} event groups, dim {E.shape[1]}",
                  flush=True)
        else:
            splits = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(qids))))
            print(f"{cell}: {len(qids)} queries, dim {E.shape[1]}", flush=True)

        # verified per-query Recall@100 for this cell's A and B runs. The SAME escalation decision
        # scored under a second metric, not a second decision: Table 1 reports one selector per row.
        recA, recB = load_cell_recall(cell, qids)

        def routed_recall(pred):
            if recA is None:
                return None
            return float(np.where(pred > 0, recB, recA).mean())

        cell_out = {"cheap": float(ndA.mean()), "uniform": float(ndB.mean()), "n": len(qids), "k": {},
                    "qids": qids,
                    "recall_cheap": float(recA.mean()) if recA is not None else None,
                    "recall_uniform": float(recB.mean()) if recB is not None else None}
        for k in [int(x) for x in a.ks.split(",")]:
            for weighted, tag in ((True, "inv_dist"), (False, "uniform_mean")):
                pred = np.zeros(len(qids))
                for tr, te in splits:
                    pred[te] = qsd_predict(E, g, tr, te, k, weighted)
                tau = float(kendalltau(pred, g).statistic)
                routed = float(np.where(pred > 0, ndB, ndA).mean())
                rec = routed_recall(pred)
                cell_out["k"][f"{k}_{tag}"] = {"tau": tau, "routed_ndcg10": routed,
                                               "routed_recall100": rec,
                                               "frac_escalated": float((pred > 0).mean()),
                                               "decisions": bits(pred > 0)}
                print(f"  k={k:<4} {tag:<13} tau={tau:+.3f}  routed nDCG@10={routed:.4f}"
                      + (f"  R@100={rec:.4f}" if rec is not None else ""), flush=True)
        # same splits, our cheap-feature router and the best analytic predictor, so the comparison is
        # not confounded by a different CV scheme
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from mv2_qpp_predictors import score_only_suite
        mk = lambda: Pipeline([("sc", StandardScaler()),
                               ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])
        X = np.array([d["features"][q] for q in qids])
        nqc = np.array([score_only_suite(list(visual[q].values()),
                                         len(queries[q].split()))["NQC"] for q in qids]).reshape(-1, 1)
        for tag, feats in (("ours_ridge", X), ("NQC_1feat", nqc)):
            pred = np.zeros(len(qids))
            for tr, te in splits:
                mm = mk().fit(feats[tr], g[tr])
                pred[te] = mm.predict(feats[te])
            cell_out[tag] = {"tau": float(kendalltau(pred, g).statistic),
                             "routed_ndcg10": float(np.where(pred > 0, ndB, ndA).mean()),
                             "routed_recall100": routed_recall(pred),
                             "frac_escalated": float((pred > 0).mean())}
            print(f"  {tag:<12} tau={cell_out[tag]['tau']:+.3f}  "
                  f"routed nDCG@10={cell_out[tag]['routed_ndcg10']:.4f}", flush=True)
        res[cell] = cell_out

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    print(f"\nbest per cell (grouped_cv={a.group_cv}):")
    for cell, c in res.items():
        if not isinstance(c, dict) or "k" not in c:
            continue
        b = max(c["k"].items(), key=lambda kv: kv[1]["tau"])
        print(f"  {cell:<12} QSD {b[0]:<16} tau={b[1]['tau']:+.3f} routed={b[1]['routed_ndcg10']:.4f}"
              f" | ours tau={c['ours_ridge']['tau']:+.3f} routed={c['ours_ridge']['routed_ndcg10']:.4f}"
              f" | NQC tau={c['NQC_1feat']['tau']:+.3f} routed={c['NQC_1feat']['routed_ndcg10']:.4f}")


if __name__ == "__main__":
    main()
