"""RQ2 baseline table: QPP predictors as routers over the modality-channel cells, laid out like
Arabzadeh et al. (2026) Table 1. Predictor formulas follow github.com/Narabzad/QPP-4-RAG
(see mv2_qpp_predictors.py).

Each predictor is computed for a query, run through a single-feature out-of-fold ridge so its sign and
scale are calibrated against the escalation gain, and then used to route: escalate the queries whose
predicted gain is positive to the fused visual+channel run, keep the rest on visual alone. The decision
metric is the resulting nDCG@10, the analogue of "execute the selected variant". The ordering metric is
Kendall tau between the raw predictor and the true gain.

The table has three predictor blocks, and the split is the multimodal finding:

  Score-only post-retrieval  - computable for every channel, including visual.
  Document-text post-retrieval (Clarity) - needs an RM1 language model over the retrieved documents.
  Pre-retrieval (IDF/ICTF/SCQ/SCS)  - needs a lexical index over the corpus.

The last two exist for the speech channel, whose documents are transcripts, and do not exist for the
visual channel, whose documents are frames. Pre-retrieval predictors are therefore reported over the ASR
text corpus and marked as unavailable for visual.

  python src/multivent2/mv2_qpp_table.py
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run, load_queries  # noqa: E402
from mv2_qpp_predictors import (score_only_suite, pre_retrieval_suite, Index,  # noqa: E402
                                SCORE_ONLY, PRE_RETRIEVAL)
from mv2_nested_calibration import (apply_fraction, calibration_split,  # noqa: E402
                                    choose_fraction)
from scipy.stats import kendalltau  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

CELLS = [
    ("ASR-shipped", "mv2_chan_visual_to_asr.json"),
    ("ASR-dense",   "mv2_chan_visual_to_asr_dense_m3.json"),
    ("OCR",         "mv2_chan_visual_to_ocr_dense_m3.json"),
]


def mk():
    return Pipeline([("sc", StandardScaler()),
                     ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])


def bits(esc):
    return "".join("1" if x else "0" for x in np.asarray(esc).ravel())


def route(raw, g, ndA, ndB, cv=None, recA=None, recB=None):
    """Single-feature OOF ridge -> escalate where predicted gain > 0.

    Returns (routed nDCG, raw tau, routed Recall@100, escalated fraction, decision bits).

    The recall is the SAME decision scored under a second metric, not a second decision: their Table 1
    reports each selector under four metrics, and a selector re-optimised per metric would not be the
    same selector. None where the sidecar has no verified recall for the cell.

    The fraction is what makes a degenerate row identifiable. At 0.0 or 1.0 the predictor made one
    decision for every query, so its nDCG is a fixed policy's nDCG and reports nothing about the
    predictor. Inferring that from the nDCG instead would be a guess: a predictor that escalates six
    queries out of 2,546 lands within float noise of never escalating, and is not the same thing.

    The last element is the decision itself, one bit per query in `qids` order, which is what the
    nugget columns are mixed from (`mv2_table1_nuggets.py`). Callers store it apart from the four
    reported numbers so the JSON's result tuples keep their shape.
    """
    raw = np.asarray(raw, dtype=np.float64).reshape(-1, 1)
    if np.allclose(raw.std(), 0):
        return (float(ndA.mean()), 0.0, (float(recA.mean()) if recA is not None else None),
                0.0, "0" * len(ndA))
    if cv is None:
        cv = KFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(mk(), raw, g, cv=cv)
    esc = pred > 0
    rec = float(np.where(esc, recB, recA).mean()) if recA is not None else None
    return (float(np.where(esc, ndB, ndA).mean()), float(kendalltau(raw.ravel(), g).statistic),
            rec, float(esc.mean()), bits(esc))


def route_nested(X, g, ndA, ndB, splits, grp, recA=None, recB=None, seed=0, cal_size=0.2):
    """The identical nested rescue the learned rows get, applied to an analytic predictor.

    Per outer fold: split the training fold into group-disjoint fit and calibration subsets, fit the
    ridge on fit, choose the escalation fraction on calibration (0.02 grid, work-minimising ties),
    refit on the full training fold, apply the frozen fraction to the test fold. No outer-test label
    enters the choice. Same tuple shape as route(), plus per-fold calibration records.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    tau = float(kendalltau(X[:, 0], g).statistic) if X.shape[1] == 1 else None
    if np.allclose(X.std(axis=0), 0):
        return (float(ndA.mean()), tau or 0.0,
                (float(recA.mean()) if recA is not None else None),
                0.0, "0" * len(ndA)), []
    esc = np.zeros(len(g), dtype=bool)
    pred = np.zeros(len(g))
    cal_log = []
    for fi, (tr, te) in enumerate(splits):
        fit, cal = calibration_split(tr, grp, cal_size, seed + fi)
        cal_pred = mk().fit(X[fit], g[fit]).predict(X[cal])
        fraction, cal_util = choose_fraction(cal_pred, ndA[cal], ndB[cal])
        pred[te] = mk().fit(X[tr], g[tr]).predict(X[te])
        esc[te] = apply_fraction(pred[te], fraction)
        cal_log.append({"fold": fi, "fraction": fraction, "calibration_utility": cal_util,
                        "n_fit": len(fit), "n_calibration": len(cal)})
    if tau is None:
        tau = float(kendalltau(pred, g).statistic)
    rec = float(np.where(esc, recB, recA).mean()) if recA is not None else None
    return (float(np.where(esc, ndB, ndA).mean()), tau, rec,
            float(esc.mean()), bits(esc)), cal_log


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-cv", action="store_true",
                    help="orient every predictor on event-grouped folds, so a near-duplicate phrasing "
                         "of the same event cannot sit on both sides of a fold boundary")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--cell", action="append", default=[], metavar="LABEL=JSON",
                    help="override the standard three cells; repeat for more than one")
    ap.add_argument("--index-text", default="asr_text.jsonl",
                    help="JSONL text corpus used for the lexical pre-retrieval index")
    ap.add_argument("--query-list", default="",
                    help="file with one query id per line; restrict every cell to these queries")
    ap.add_argument("--queries", default=None,
                    help="override the query CSV (same Query_id,query format); the pre-retrieval "
                         "features read this text, so variant experiments pass their variant CSV")
    ap.add_argument("--score-run", default=None,
                    help="run JSON whose scores feed the score-only features, replacing the shipped "
                         "visual run; variant cells pass the cheap option's own run")
    ap.add_argument("--nested-calibration", action="store_true",
                    help="give every analytic predictor (and the Ours row) the identical nested "
                         "escalation-fraction choice the learned rows get, so the family comparison "
                         "carries no protocol asymmetry")
    ap.add_argument("--calibration-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.nested_calibration and not a.group_cv:
        ap.error("--nested-calibration requires --group-cv")
    cells = CELLS
    if a.cell:
        cells = []
        for spec in a.cell:
            label, sep, filename = spec.partition("=")
            if not sep or not label or not filename:
                ap.error(f"invalid --cell {spec!r}; expected LABEL=JSON")
            cells.append((label, filename))
    visual = load_run(a.score_run if a.score_run
                      else os.path.join(DATA, "10pyscene_clip.json"))
    queries = load_queries(a.queries or os.path.join(DATA, "multivent_2_test_queries.csv"))

    # lexical index over the ASR transcripts: the speech channel's documents are text, so
    # pre-retrieval QPP exists for it. Nothing equivalent exists for the visual channel.
    asr_texts = []
    p_asr = os.path.join(DATA, a.index_text)
    if os.path.exists(p_asr):
        with open(p_asr) as f:
            for line in f:
                t = json.loads(line).get("text", "")
                if t.strip():
                    asr_texts.append(t)
    print(f"building lexical index over {len(asr_texts)} ASR transcripts...", flush=True)
    idx = Index(asr_texts) if asr_texts else None
    if idx:
        print(f"  index: {idx.n_docs} docs, {idx.total_terms} tokens, "
              f"{len(idx.cf)} vocab", flush=True)

    keep = None
    if a.query_list:
        keep = {line.strip() for line in open(a.query_list) if line.strip()}
        print(f"query list: restricting every cell to {len(keep)} queries", flush=True)

    results = {}
    for label, fn in cells:
        d = json.load(open(os.path.join(ABL, fn)))
        qids = [q for q in d["per_query"] if q in visual and q in queries]
        if keep is not None:
            qids = [q for q in qids if q in keep]
        ndA = np.array([d["per_query"][q]["ndA"] for q in qids])
        ndB = np.array([d["per_query"][q]["ndB"] for q in qids])
        g = ndB - ndA
        feats = np.array([d["features"][q] for q in qids])

        # verified per-query Recall@100 for this cell's A and B runs, if the sidecar reproduced the
        # cell's stored nDCG exactly. Absent -> the recall column stays empty for the cell.
        recA = recB = None
        sc_path = os.path.join(ABL, "mv2_recall_sidecar.json")
        if os.path.exists(sc_path):
            sc = json.load(open(sc_path)).get(label)
            if sc and all(q in sc["per_query"] for q in qids):
                recA = np.array([sc["per_query"][q][0] for q in qids])
                recB = np.array([sc["per_query"][q][1] for q in qids])

        if a.group_cv:
            from sklearn.model_selection import GroupKFold
            from mv2_io import load_qrels
            from mv2_qsd import event_groups
            qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
            grp = event_groups(qids, qrels)
            cv = list(GroupKFold(5).split(np.arange(len(qids)), groups=grp))
            print(f"{label}: {len(set(grp))} event groups", flush=True)
        else:
            cv = None

        # score-only predictors from the cheap visual channel
        post = {n: [] for n in SCORE_ONLY}
        for q in qids:
            nq = len(queries[q].split())
            for n, v in score_only_suite(list(visual[q].values()), nq).items():
                post[n].append(v)

        # decisions are kept out of the reported tuples: one bitstring per predictor, in `qids` order
        dec = {}
        cal_records = {}

        def run_family(values, names, key):
            if a.nested_calibration:
                out = {}
                for n in names:
                    tup, cal_log = route_nested(values[n], g, ndA, ndB, cv, grp, recA, recB,
                                                seed=a.seed, cal_size=a.calibration_size)
                    out[n] = tup
                    cal_records[f"{key}:{n}"] = cal_log
                return out
            return {n: route(values[n], g, ndA, ndB, cv, recA, recB) for n in names}

        full = run_family(post, SCORE_ONLY, "post")
        rows = {n: v[:4] for n, v in full.items()}
        dec["post"] = {n: v[4] for n, v in full.items()}

        # pre-retrieval predictors over the ASR lexical index (query-side only, no retrieval at all)
        pre_rows = {}
        if idx:
            pre = {n: [] for n in PRE_RETRIEVAL}
            for q in qids:
                s = pre_retrieval_suite(queries[q].lower().split(), idx)
                for n in PRE_RETRIEVAL:
                    pre[n].append(s[n])
            pfull = run_family(pre, PRE_RETRIEVAL, "pre")
            pre_rows = {n: v[:4] for n, v in pfull.items()}
            dec["pre"] = {n: v[4] for n, v in pfull.items()}

        if a.nested_calibration:
            ours_full, ours_cal = route_nested(feats, g, ndA, ndB, cv, grp, recA, recB,
                                               seed=a.seed, cal_size=a.calibration_size)
            ours = ours_full[:4]
            dec["ours"] = ours_full[4]
            cal_records["ours"] = ours_cal
        else:
            ours_pred = cross_val_predict(mk(), feats, g,
                                          cv=cv or KFold(5, shuffle=True, random_state=0))
            ours = (float(np.where(ours_pred > 0, ndB, ndA).mean()),
                    float(kendalltau(ours_pred, g).statistic),
                    float(np.where(ours_pred > 0, recB, recA).mean()) if recA is not None else None,
                    float((ours_pred > 0).mean()))
            dec["ours"] = bits(ours_pred > 0)
        oracle = (float(np.where(g > 0, ndB, ndA).mean()), 1.0,
                  float(np.where(g > 0, recB, recA).mean()) if recA is not None else None,
                  float((g > 0).mean()))

        dec["oracle"] = bits(g > 0)
        results[label] = {"n": len(qids), "cheap": float(ndA.mean()), "uniform": float(ndB.mean()),
                          "recall_cheap": float(recA.mean()) if recA is not None else None,
                          "recall_uniform": float(recB.mean()) if recB is not None else None,
                          "post": rows, "pre": pre_rows, "ours": ours, "oracle": oracle,
                          "decision_rule": ("nested_fraction" if a.nested_calibration else "zero"),
                          "nested_calibration": cal_records if a.nested_calibration else None,
                          "qids": qids, "decisions": dec}
        print(f"{label}: n={len(qids)} cheap={ndA.mean():.4f} uniform={ndB.mean():.4f} "
              f"ours={ours[0]:.4f} (tau {ours[1]:+.3f})", flush=True)

    json.dump(results, open(os.path.join(ABL, f"mv2_qpp_table{a.tag}.json"), "w"), indent=2)

    labels = [c[0] for c in cells]
    def fmt(v):
        return f"{v[0]:.4f} | {v[1]:+.3f}"    # markdown here stays nDCG|tau; recall lives in Table 1
    L = []
    L.append("| Category | Method | " + " | ".join(f"{l} nDCG@10 | τ" for l in labels) + " |")
    L.append("|" + "---|" * (2 + 2 * len(labels)))
    L.append("| Original | visual only (cheap) | " +
             " | ".join(f"{results[l]['cheap']:.4f} | --" for l in labels) + " |")
    L.append("| | uniform fusion (best w) | " +
             " | ".join(f"{results[l]['uniform']:.4f} | --" for l in labels) + " |")
    for i, n in enumerate(PRE_RETRIEVAL):
        if not results[labels[0]]["pre"]:
            break
        cat = "Pre-retrieval<br>(ASR text index)" if i == 0 else ""
        L.append(f"| {cat} | {n} | " + " | ".join(fmt(results[l]["pre"][n]) for l in labels) + " |")
    for i, n in enumerate(SCORE_ONLY):
        cat = "Post-retrieval<br>(score-only)" if i == 0 else ""
        L.append(f"| {cat} | {n} | " + " | ".join(fmt(results[l]["post"][n]) for l in labels) + " |")
    L.append("| Post-retrieval<br>(needs doc text) | clarity | " +
             " | ".join("n/a for visual" + " | --" for _ in labels) + " |")
    L.append("| **Ours** | **cheap-feature gain ridge** | " +
             " | ".join("**" + fmt(results[l]["ours"]) + "**" for l in labels) + " |")
    L.append("| Oracle | route by true gain | " +
             " | ".join(fmt(results[l]["oracle"]) for l in labels) + " |")
    md = "\n".join(L)
    open(os.path.join(ABL, f"mv2_qpp_table{a.tag}.md"), "w").write(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
