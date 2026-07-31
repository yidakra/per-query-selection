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


def route(raw, g, ndA, ndB, cv=None, recA=None, recB=None):
    """Single-feature OOF ridge -> escalate where predicted gain > 0.

    Returns (routed nDCG, raw tau, routed Recall@100). The recall is the SAME decision scored under a
    second metric, not a second decision: their Table 1 reports each selector under four metrics, and a
    selector re-optimised per metric would not be the same selector. None where the sidecar has no
    verified recall for the cell."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1, 1)
    if np.allclose(raw.std(), 0):
        return float(ndA.mean()), 0.0, (float(recA.mean()) if recA is not None else None)
    if cv is None:
        cv = KFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(mk(), raw, g, cv=cv)
    esc = pred > 0
    rec = float(np.where(esc, recB, recA).mean()) if recA is not None else None
    return float(np.where(esc, ndB, ndA).mean()), float(kendalltau(raw.ravel(), g).statistic), rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-cv", action="store_true",
                    help="orient every predictor on event-grouped folds, so a near-duplicate phrasing "
                         "of the same event cannot sit on both sides of a fold boundary")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    a = ap.parse_args()
    visual = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))

    # lexical index over the ASR transcripts: the speech channel's documents are text, so
    # pre-retrieval QPP exists for it. Nothing equivalent exists for the visual channel.
    asr_texts = []
    p_asr = os.path.join(DATA, "asr_text.jsonl")
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

    results = {}
    for label, fn in CELLS:
        d = json.load(open(os.path.join(ABL, fn)))
        qids = [q for q in d["per_query"] if q in visual and q in queries]
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

        rows = {n: route(post[n], g, ndA, ndB, cv, recA, recB) for n in SCORE_ONLY}

        # pre-retrieval predictors over the ASR lexical index (query-side only, no retrieval at all)
        pre_rows = {}
        if idx:
            pre = {n: [] for n in PRE_RETRIEVAL}
            for q in qids:
                s = pre_retrieval_suite(queries[q].lower().split(), idx)
                for n in PRE_RETRIEVAL:
                    pre[n].append(s[n])
            pre_rows = {n: route(pre[n], g, ndA, ndB, cv, recA, recB) for n in PRE_RETRIEVAL}

        ours_pred = cross_val_predict(mk(), feats, g,
                                     cv=cv or KFold(5, shuffle=True, random_state=0))
        ours = (float(np.where(ours_pred > 0, ndB, ndA).mean()),
                float(kendalltau(ours_pred, g).statistic),
                float(np.where(ours_pred > 0, recB, recA).mean()) if recA is not None else None)
        oracle = (float(np.where(g > 0, ndB, ndA).mean()), 1.0,
                  float(np.where(g > 0, recB, recA).mean()) if recA is not None else None)

        results[label] = {"n": len(qids), "cheap": float(ndA.mean()), "uniform": float(ndB.mean()),
                          "recall_cheap": float(recA.mean()) if recA is not None else None,
                          "recall_uniform": float(recB.mean()) if recB is not None else None,
                          "post": rows, "pre": pre_rows, "ours": ours, "oracle": oracle}
        print(f"{label}: n={len(qids)} cheap={ndA.mean():.4f} uniform={ndB.mean():.4f} "
              f"ours={ours[0]:.4f} (tau {ours[1]:+.3f})", flush=True)

    json.dump(results, open(os.path.join(ABL, f"mv2_qpp_table{a.tag}.json"), "w"), indent=2)

    labels = [c[0] for c in CELLS]
    def fmt(v):
        return f"{v[0]:.4f} | {v[1]:+.3f}"    # markdown here stays nDCG|tau; recall lives in Table 1
    L = []
    L.append("| Category | Method | " + " | ".join(f"{l} nDCG@10 | τ" for l in labels) + " |")
    L.append("|" + "---|" * (2 + 2 * len(labels)))
    L.append("| Original | visual only (cheap) | " +
             " | ".join(f"{results[l]['cheap']:.4f} | —" for l in labels) + " |")
    L.append("| | uniform fusion (best w) | " +
             " | ".join(f"{results[l]['uniform']:.4f} | —" for l in labels) + " |")
    for i, n in enumerate(PRE_RETRIEVAL):
        if not results[labels[0]]["pre"]:
            break
        cat = "Pre-retrieval<br>(ASR text index)" if i == 0 else ""
        L.append(f"| {cat} | {n} | " + " | ".join(fmt(results[l]["pre"][n]) for l in labels) + " |")
    for i, n in enumerate(SCORE_ONLY):
        cat = "Post-retrieval<br>(score-only)" if i == 0 else ""
        L.append(f"| {cat} | {n} | " + " | ".join(fmt(results[l]["post"][n]) for l in labels) + " |")
    L.append("| Post-retrieval<br>(needs doc text) | clarity | " +
             " | ".join("n/a for visual" + " | —" for _ in labels) + " |")
    L.append("| **Ours** | **cheap-feature gain ridge** | " +
             " | ".join("**" + fmt(results[l]["ours"]) + "**" for l in labels) + " |")
    L.append("| Oracle | route by true gain | " +
             " | ".join(fmt(results[l]["oracle"]) for l in labels) + " |")
    md = "\n".join(L)
    open(os.path.join(ABL, f"mv2_qpp_table{a.tag}.md"), "w").write(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
