"""RQ2 baseline table: QPP predictors as routers over the modality-channel cells, laid out like
Arabzadeh et al. (2026) Table 1 (Original / post-retrieval predictors / Ours / Oracle).

Each predictor is computed from the CHEAP visual channel's similarity-score vector (post-cheap-retrieval,
pre-expensive: the only pre-computable signal in the multimodal setting -- see related_work_qpp.md), run
through a single-feature out-of-fold ridge to orient it, and used to route: escalate the queries whose
predicted gain > 0 to the fused visual+channel run, keep the rest on visual. Decision metric = the
resulting nDCG@10 (analogue of their "execute the selected variant"); ordering metric = Kendall tau of
the raw predictor vs the true escalation gain.

Pre-retrieval QPP (IDF/ICTF/SCS/...) is intentionally absent: it needs corpus term statistics, which do
not exist for the visual channel. That gap is the multimodal point, reported as a table note rather than
a row of NAs.

  python src/multivent2/mv2_qpp_table.py
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run  # noqa: E402
from scipy.stats import kendalltau
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

# escalation cells: (column label, router-ready cell JSON with per_query ndA/ndB + features)
CELLS = [
    ("visual→+ASR (shipped)", "mv2_chan_visual_to_asr.json"),
    ("visual→+ASR (dense)",   "mv2_chan_visual_to_asr_dense_m3.json"),
    ("visual→+OCR",           "mv2_chan_visual_to_ocr_dense_m3.json"),
]

# post-retrieval QPP predictors over a query's cheap-channel score vector s (sorted desc), top-k depth.
# Standard families (Shtok/Zhou-Croft/Cronen-Townsend/Tao-Wu); to be reconciled with the exact
# definitions in github.com/Narabzad/QPP-4-RAG before the camera-ready.
def qpp_suite(s, k=10):
    s = np.sort(np.asarray(s, dtype=np.float64))[::-1]
    topk = s[:k]
    mu, sd = s.mean(), s.std() + 1e-9
    tmu = topk.mean()
    p = np.exp(s - s.max()); p /= p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum())
    half = s[: max(1, len(s) // 2)]
    return {
        "max (σ_max magnitude)":  float(s[0]),
        "clarity":                float(-ent),
        "NQC":                    float(topk.std() / (abs(mu) + 1e-9)),
        "NQC_norm":               float(topk.std() / (s[0] + 1e-9)),
        "σ_max":                  float(topk.std()),
        "σ_50%":                  float(half.std()),
        "SMV":                    float(np.mean(topk * np.abs(topk - tmu))),
        "SMV_norm":               float(np.mean(topk * np.abs(topk - tmu)) / (abs(mu) + 1e-9)),
        "WIG":                    float(tmu - mu),
        "WIG_norm":               float((tmu - mu) / sd),
        "RSD":                    float(topk.std() / (tmu + 1e-9)),
    }

PENDING = ["QSD_post (needs doc embeddings)", "BERT-QPP bi/cross (needs trained model)"]


def mk():
    return Pipeline([("sc", StandardScaler()),
                     ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))])


def oof(X, g):
    return cross_val_predict(mk(), X, g, cv=KFold(5, shuffle=True, random_state=0))


def routed_ndcg(predg, ndA, ndB):
    esc = predg > 0
    return float(np.where(esc, ndB, ndA).mean())


def main():
    visual = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    results = {}   # cell -> {"cheap":x,"uniform":x,"rows":{method:(ndcg,tau)},"ours":..,"oracle":..}
    for label, fn in CELLS:
        d = json.load(open(os.path.join(ABL, fn)))
        qids = [q for q in d["per_query"] if q in visual]
        ndA = np.array([d["per_query"][q]["ndA"] for q in qids])
        ndB = np.array([d["per_query"][q]["ndB"] for q in qids])
        g = ndB - ndA
        feats = np.array([d["features"][q] for q in qids])          # our 8 conf features

        # predictor matrix
        names = list(qpp_suite(sorted(visual[qids[0]].values())).keys())
        P = {n: [] for n in names}
        for q in qids:
            sv = list(visual[q].values())
            for n, v in qpp_suite(sv).items():
                P[n].append(v)

        rows = {}
        for n in names:
            raw = np.array(P[n])
            tau = float(kendalltau(raw, g).statistic)
            predg = oof(raw.reshape(-1, 1), g)
            rows[n] = (routed_ndcg(predg, ndA, ndB), tau)

        ours_pred = oof(feats, g)
        ours = (routed_ndcg(ours_pred, ndA, ndB), float(kendalltau(ours_pred, g).statistic))
        oracle = (routed_ndcg(g, ndA, ndB), 1.0)      # escalate exactly where g>0

        results[label] = {"n": len(qids), "cheap": float(ndA.mean()), "uniform": float(ndB.mean()),
                          "rows": rows, "ours": ours, "oracle": oracle, "names": names}

    json.dump(results, open(os.path.join(ABL, "mv2_qpp_table.json"), "w"), indent=2)

    # ---- render markdown, paper-1 layout ----
    labels = [c[0] for c in CELLS]
    def cell_cols(v):  # (ndcg, tau) -> "0.354 | +0.17"
        return f"{v[0]:.4f} | {v[1]:+.3f}"
    hdr = "| Category | Method | " + " | ".join(f"{l}<br>nDCG@10 \\| τ" for l in labels) + " |"
    sep = "|" + "---|" * (2 + len(labels))
    lines = [hdr, sep]
    r0 = results[labels[0]]
    lines.append("| Original | visual only (cheap) | " +
                 " | ".join(f"{results[l]['cheap']:.4f} | —" for l in labels) + " |")
    lines.append("| | uniform fusion (best w) | " +
                 " | ".join(f"{results[l]['uniform']:.4f} | —" for l in labels) + " |")
    for i, n in enumerate(r0["names"]):
        cat = "Post-retrieval" if i == 0 else ""
        lines.append(f"| {cat} | {n} | " +
                     " | ".join(cell_cols(results[l]["rows"][n]) for l in labels) + " |")
    lines.append("| **Ours** | **cheap-feature gain ridge** | " +
                 " | ".join("**" + cell_cols(results[l]["ours"]) + "**" for l in labels) + " |")
    lines.append("| Oracle | route by true gain | " +
                 " | ".join(cell_cols(results[l]["oracle"]) for l in labels) + " |")
    md = "\n".join(lines)
    open(os.path.join(ABL, "mv2_qpp_table.md"), "w").write(md + "\n")
    print(md)
    print("\nPending (need a model/embeddings): " + "; ".join(PENDING))
    print("Pre-retrieval QPP omitted: no corpus term statistics over video (multimodal note).")


if __name__ == "__main__":
    main()
