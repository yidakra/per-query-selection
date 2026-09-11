"""Task B of the pre-registered variant experiments: can QPP pick the right query variant here?

The source study's task, replicated on MultiVENT 2.0: each query brings a pool of QueryGym
reformulations plus the original; every candidate is executed against the dense speech channel (and
the speech+OCR RRF fusion as the secondary pipeline); each analytic predictor ranks the pool; the
predictor-picked candidate's realized quality is compared against the original query, the best
single reformulation method, and the per-need oracle. Correlations are computed within need and
aggregated across needs, the source study's own aggregation. Predictors are analytic and untrained,
so there is no fold structure to leak through; the oracle is labels and is reported as a bound only.

Runs after mv2_generate_variants.py, using the corpus embedding caches:

  python src/multivent2/mv2_variant_selection.py --samples 1   # complete 6x1 pool
  python src/multivent2/mv2_variant_selection.py --samples 5   # full pre-registered pool
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

METHODS = ["genqr", "genqr_ensemble", "mugi", "qa_expand", "query2doc", "query2e"]


def load_pool(path, samples):
    """candidates[qid] = list of (label, text); label 'original' or 'method#sample'."""
    by_key = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r or r["sample"] >= samples:
                continue
            by_key[(r["qid"], r["method"], r["sample"])] = r["variant"]
    return by_key


def encode_and_search(texts, cache_path, model_name, topk=1000, batch=64, device="cuda"):
    """Retrieve for every text against the cached document embeddings. `device` exists because the
    box's single GPU is shared: on "cpu" the maths is identical in float32, only slower."""
    import torch
    from sentence_transformers import SentenceTransformer
    cache = np.load(cache_path, allow_pickle=False)
    half = device != "cpu"                       # fp16 matmul has no CPU kernel here
    D = torch.from_numpy(cache["D"]).to(device)
    owner = torch.from_numpy(cache["owner"].astype(np.int64)).to(device)
    doc_ids = [str(x) for x in cache["doc_ids"]]
    m = SentenceTransformer(model_name, device=device)
    if half:
        m.half()
    else:
        D = D.float()
    Q = m.encode(texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=True).astype(np.float16 if half else np.float32)
    ndoc = len(doc_ids)
    runs = []
    for s in range(0, len(texts), 64):
        qb = torch.from_numpy(Q[s:s + 64]).to(device)
        sims = qb @ D.T
        pooled = torch.full((sims.shape[0], ndoc), -1e4, device=device, dtype=sims.dtype)
        pooled.scatter_reduce_(1, owner.expand(sims.shape[0], -1), sims, reduce="amax")
        vals, idx = torch.topk(pooled.float(), min(topk, ndoc), dim=1)
        for j in range(sims.shape[0]):
            runs.append({doc_ids[int(d)]: float(v)
                         for d, v in zip(idx[j].tolist(), vals[j].tolist())})
    del D, owner
    if half:
        torch.cuda.empty_cache()
    return runs


def rrf(a, b, k=60):
    ra = {d: i + 1 for i, d in enumerate(sorted(a, key=a.get, reverse=True))}
    rb = {d: i + 1 for i, d in enumerate(sorted(b, key=b.get, reverse=True))}
    return {d: 1.0 / (k + ra.get(d, 10**9)) + 1.0 / (k + rb.get(d, 10**9))
            for d in set(ra) | set(rb)}


def rrf_multi(runs, k=60):
    """Reciprocal-rank fusion over any number of ranked lists."""
    out = {}
    for r in runs:
        for i, d in enumerate(sorted(r, key=r.get, reverse=True)):
            out[d] = out.get(d, 0.0) + 1.0 / (k + i + 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=os.path.join(DATA, "query_variants.jsonl"))
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-features", default="",
                    help="write per-candidate confidence features and per-query nDCG to this JSON, "
                         "the input a learned variant selector needs")
    ap.add_argument("--save-picks", default="",
                    help="write per-query picks and top-100 doc lists for the key policies to this "
                         "JSONL, the input the nugget evaluation needs")
    ap.add_argument("--gpu", type=int, default=1,
                    help="physical GPU index; device 0 belongs to another service and is never used")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    from scipy.stats import kendalltau
    import ir_measures
    from ir_measures import nDCG, R
    from mv2_io import load_qrels, load_queries
    from mv2_qpp_predictors import Index, PRE_RETRIEVAL, SCORE_ONLY, \
        pre_retrieval_suite, score_only_suite

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    pool = load_pool(a.variants, a.samples)

    qids = sorted(q for q in queries if q in qrels)
    complete = [q for q in qids
                if all((q, m, s) in pool for m in METHODS for s in range(a.samples))]
    print(f"{len(complete)}/{len(qids)} queries have the complete "
          f"{len(METHODS)}x{a.samples} pool", flush=True)

    labels = ["original"] + [f"{m}#{s}" for m in METHODS for s in range(a.samples)]
    texts, owners = [], []
    for q in complete:
        for lab in labels:
            if lab == "original":
                texts.append(queries[q])
            else:
                m, s = lab.rsplit("#", 1)
                texts.append(pool[(q, m, int(s))])
            owners.append((q, lab))
    # the concatenate-everything baseline: all expansions in one query, retrieved once. It is the
    # alternative to selecting, so it stays out of the predictors' candidate pool. The encoder
    # truncates at its max sequence length, which is part of what concatenation costs.
    for q in complete:
        joined = " ".join([queries[q]] + [pool[(q, m, 0)] for m in METHODS])
        texts.append(joined)
        owners.append((q, "concat_all"))

    print(f"scoring {len(texts)} candidates on the dense speech channel", flush=True)
    asr_runs = encode_and_search(texts, os.path.join(_ROOT, "runs", "embcache", "asr_bge-m3.npz"),
                                 "BAAI/bge-m3")
    print("scoring on the OCR channel for the fusion pipeline", flush=True)
    ocr_runs = encode_and_search(texts, os.path.join(_ROOT, "runs", "embcache", "ocr_bge-m3.npz"),
                                 "BAAI/bge-m3")

    # per-candidate quality under both pipelines
    def score_run(run_by_cand):
        fake_run = {f"{q}##{lab}": run_by_cand[i] for i, (q, lab) in enumerate(owners)}
        fake_qrels = {f"{q}##{lab}": qrels[q] for (q, lab) in owners}
        vals = {}
        for met in ir_measures.iter_calc([nDCG @ 10, R @ 100], fake_qrels, fake_run):
            vals.setdefault(met.query_id, {})[str(met.measure)] = met.value
        return vals

    print("evaluating", flush=True)
    q_asr = score_run(asr_runs)
    fused = [rrf(asr_runs[i], ocr_runs[i]) for i in range(len(texts))]
    q_fus = score_run(fused)

    # predictor features per candidate: pre-retrieval over the standard speech index, score
    # features over the candidate's own dense speech scores
    print("building the lexical index for the pre-retrieval features", flush=True)
    idx_texts = []
    with open(os.path.join(DATA, "asr_text.jsonl")) as f:
        for line in f:
            t = json.loads(line).get("text", "")
            if t.strip():
                idx_texts.append(t)
    index = Index(idx_texts)

    feats = collections.defaultdict(dict)
    for i, (q, lab) in enumerate(owners):
        toks = texts[i].lower().split()
        pre = pre_retrieval_suite(toks, index)
        post = score_only_suite(list(asr_runs[i].values()), len(toks))
        feats[(q, lab)] = {**{f"pre:{k}": v for k, v in pre.items()},
                           **{f"post:{k}": v for k, v in post.items()}}

    predictors = [f"pre:{k}" for k in PRE_RETRIEVAL] + [f"post:{k}" for k in SCORE_ONLY]
    out = {"n_queries": len(complete), "samples": a.samples, "pipelines": {}}
    for pipe, qual in (("asr_dense", q_asr), ("asr_ocr_rrf", q_fus)):
        res = {"baselines": {}, "predictors": {}}
        nd = {(q, lab): qual[f"{q}##{lab}"]["nDCG@10"] for (q, lab) in owners}
        rc = {(q, lab): qual[f"{q}##{lab}"]["R@100"] for (q, lab) in owners}

        res["baselines"]["original"] = float(np.mean([nd[(q, "original")] for q in complete]))
        res["baselines"]["concat_all"] = float(np.mean([nd[(q, "concat_all")] for q in complete]))

        # fuse-everything baseline: RRF over all candidates' result lists, immune to any encoder
        # length limit; the rank-fusion version of "use every subquery at once"
        idx = {ow: i for i, ow in enumerate(owners)}
        run_src = asr_runs if pipe == "asr_dense" else fused
        fused_all = {q: rrf_multi([run_src[idx[(q, lab)]] for lab in labels]) for q in complete}
        fq = {f"{q}##x": fused_all[q] for q in complete}
        fqr = {f"{q}##x": qrels[q] for q in complete}
        vals = {m.query_id: m.value for m in ir_measures.iter_calc([nDCG @ 10], fqr, fq)}
        res["baselines"]["fuse_all"] = float(np.mean([vals[f"{q}##x"] for q in complete]))
        per_method = {m: float(np.mean([nd[(q, f"{m}#{s}")] for q in complete
                                        for s in range(a.samples)])) for m in METHODS}
        best_m = max(per_method, key=per_method.get)
        res["baselines"]["per_method"] = per_method
        res["baselines"]["best_single_method"] = {"method": best_m, "ndcg10": per_method[best_m]}
        res["baselines"]["oracle"] = float(np.mean([max(nd[(q, lab)] for lab in labels)
                                                    for q in complete]))

        for p in predictors:
            taus, picked_nd, picked_rc = [], [], []
            for q in complete:
                vals = np.array([feats[(q, lab)][p] for lab in labels])
                gold = np.array([nd[(q, lab)] for lab in labels])
                if vals.std() > 0 and gold.std() > 0:
                    t = kendalltau(vals, gold).statistic
                    if not np.isnan(t):
                        taus.append(t)
                pick = labels[int(np.argmax(vals))]
                picked_nd.append(nd[(q, pick)]); picked_rc.append(rc[(q, pick)])
            res["predictors"][p] = {
                "mean_within_need_tau": float(np.mean(taus)) if taus else 0.0,
                "selected_ndcg10": float(np.mean(picked_nd)),
                "selected_r100": float(np.mean(picked_rc)),
                "vs_original": float(np.mean(picked_nd) - res["baselines"]["original"])}
        out["pipelines"][pipe] = res
        best = max(res["predictors"].items(), key=lambda kv: kv[1]["selected_ndcg10"])
        print(f"[{pipe}] original {res['baselines']['original']:.4f}  "
              f"concat-all {res['baselines']['concat_all']:.4f}  "
              f"best-method {per_method[best_m]:.4f} ({best_m})  "
              f"oracle {res['baselines']['oracle']:.4f}  "
              f"best predictor {best[0]} {best[1]['selected_ndcg10']:.4f}", flush=True)

    if a.save_features:
        from retrieve import conf_features, FEATURE_ORDER
        idxf = {ow: i for i, ow in enumerate(owners)}
        feat_rows = []
        for q in complete:
            for lab in labels:
                cf = conf_features(list(asr_runs[idxf[(q, lab)]].values()))
                feat_rows.append({
                    "qid": q, "label": lab,
                    "method": "original" if lab == "original" else lab.rsplit("#", 1)[0],
                    "x": [cf[k] for k in FEATURE_ORDER],
                    "ndcg10": q_asr[f"{q}##{lab}"]["nDCG@10"]})
        json.dump({"feature_order": FEATURE_ORDER, "labels": labels, "rows": feat_rows},
                  open(a.save_features, "w"))
        print(f"wrote per-candidate features to {a.save_features} ({len(feat_rows)} rows)")

    if a.save_picks:
        # per-query document lists for the nugget phases: the original, the best pre and post
        # selectors' picks, the oracle's pick, and the two use-everything baselines, all on the
        # primary pipeline
        idx = {ow: i for i, ow in enumerate(owners)}
        nd1 = {(q, lab): q_asr[f"{q}##{lab}"]["nDCG@10"] for (q, lab) in owners}
        key_preds = {"sel_pre_QL": "pre:QL", "sel_post_NQC_norm": "post:NQC_norm"}
        with open(a.save_picks, "w") as fh:
            for q in complete:
                row = {"qid": q}
                row["original"] = {"pick": "original",
                                   "docs": list(asr_runs[idx[(q, "original")]])[:100]}
                for name, pred in key_preds.items():
                    vals = np.array([feats[(q, lab)][pred] for lab in labels])
                    pick = labels[int(np.argmax(vals))]
                    row[name] = {"pick": pick, "docs": list(asr_runs[idx[(q, pick)]])[:100]}
                opick = max(labels, key=lambda lab: nd1[(q, lab)])
                row["oracle"] = {"pick": opick, "docs": list(asr_runs[idx[(q, opick)]])[:100]}
                row["concat_all"] = {"pick": "concat_all",
                                     "docs": list(asr_runs[idx[(q, "concat_all")]])[:100]}
                fmerged = rrf_multi([asr_runs[idx[(q, lab)]] for lab in labels])
                row["fuse_all"] = {"pick": "fuse_all",
                                   "docs": sorted(fmerged, key=fmerged.get, reverse=True)[:100]}
                fh.write(json.dumps(row) + "\n")
        print(f"wrote picks to {a.save_picks}")

    path = os.path.join(ABL, f"mv2_variant_selection{a.tag}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
