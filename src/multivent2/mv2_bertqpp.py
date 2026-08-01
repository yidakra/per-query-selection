"""BERT-QPP (Arabzadeh et al. 2021) as a routing baseline, the one supervised predictor in the
QPP-4-RAG suite.

Their setup pairs a query with the text of its first retrieved document and fine-tunes a cross-encoder
to regress query performance. Two adaptations are needed here and both are stated in the table notes:

  Target. Their model predicts an absolute retrieval metric. Every other predictor in our table is
  oriented against the escalation gain by an out-of-fold ridge, so for an apples-to-apples comparison
  this one regresses the gain directly.

  Document text. The cheap channel is visual and its documents are video, so there is no first-document
  text to pair with the query. The benchmark's shipped captions stand in, the same substitution the RAG
  arm makes. Without them this predictor could not be run at all, which is itself the RQ4 point.

Trained out of fold (5 folds, train on four, predict the held-out one) so the reported numbers carry no
leakage, matching the protocol used for the analytic predictors. CPU only: GPU0 runs an unrelated
service and GPU1 is occupied, so CUDA is hidden before torch is imported.

  python src/multivent2/mv2_bertqpp.py --cells asr_shipped,asr_dense,ocr --epochs 1
"""
import os, sys, json, time, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # must precede torch: keep both GPUs untouched
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np                                # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run, load_queries         # noqa: E402
from scipy.stats import kendalltau                # noqa: E402
from sklearn.model_selection import KFold, GroupKFold  # noqa: E402
from mv2_qsd import event_groups                  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

CELLS = {"asr_shipped": "mv2_chan_visual_to_asr.json",
         "asr_dense":   "mv2_chan_visual_to_asr_dense_m3.json",
         "ocr":         "mv2_chan_visual_to_ocr_dense_m3.json"}


def load_captions():
    """Verbalization of the visual channel's documents: without this BERT-QPP has no input here."""
    cap = {}
    p = os.path.join(DATA, "qwen_captions_test.jsonl")
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("text", "").strip():
                cap[r["doc_id"]] = r["text"]
    return cap


def recall_only(path):
    """Fill routed_recall100 for each cell already in `path`, from its stored out-of-fold predictions.

    The nDCG is re-derived here too and checked against the stored value. It is a cheap way of proving
    that the predictions in the file are the ones that produced the number beside them, which is worth
    having before a recall computed from them goes into the paper's main table.
    """
    from mv2_recall_sidecar import load_cell_recall
    out = json.load(open(path))
    for cell, c in out.items():
        if "pred" not in c:
            continue
        d = json.load(open(os.path.join(ABL, CELLS[cell])))
        qids = list(c["pred"])
        pred = np.array([c["pred"][q] for q in qids])
        ndA = np.array([d["per_query"][q]["ndA"] for q in qids])
        ndB = np.array([d["per_query"][q]["ndB"] for q in qids])
        chk = float(np.where(pred > 0, ndB, ndA).mean())
        ok = abs(chk - c["routed_ndcg10"]) < 1e-9
        recA, recB = load_cell_recall(cell, qids)
        rec = None if recA is None else float(np.where(pred > 0, recB, recA).mean())
        if not ok:
            print(f"{cell}: stored predictions give {chk:.6f}, file says "
                  f"{c['routed_ndcg10']:.6f} -- recall not written")
            continue
        c["routed_recall100"] = rec
        c["recall_cheap"] = float(recA.mean()) if recA is not None else None
        c["recall_uniform"] = float(recB.mean()) if recB is not None else None
        c["frac_escalated"] = float((pred > 0).mean())
        print(f"{cell}: nDCG@10 {chk:.4f} reproduced; "
              + (f"R@100 {rec:.4f} (A {recA.mean():.4f}, B {recB.mean():.4f})"
                 if rec is not None else "no verified recall for this cell"))
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="asr_shipped,asr_dense,ocr")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many queries")
    ap.add_argument("--group-cv", action="store_true",
                    help="split by event group, not by query. Needed here for the same reason as QSD: "
                         "near-duplicate event queries often retrieve the SAME top document, so a plain "
                         "split lets the model memorise event -> gain across folds")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_bertqpp.json"))
    ap.add_argument("--recall-only", action="store_true",
                    help="score the ALREADY TRAINED predictions under Recall@100 and exit. The "
                         "out-of-fold predictions are stored per query, so the routing decision is "
                         "recoverable without a retrain, and re-scoring it is the point: one selector, "
                         "two metrics. Touches no GPU and no model.")
    a = ap.parse_args()

    if a.recall_only:
        recall_only(a.out)
        return

    import torch
    torch.set_num_threads(a.threads)
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    visual = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    caps = load_captions()
    print(f"captions for {len(caps)} docs", flush=True)

    out = {}
    if os.path.exists(a.out):
        out = json.load(open(a.out))

    for cell in a.cells.split(","):
        if cell in out:
            print(f"{cell}: already done, skipping", flush=True)
            continue
        d = json.load(open(os.path.join(ABL, CELLS[cell])))
        qids = [q for q in d["per_query"] if q in visual and q in queries]
        if a.limit:
            qids = qids[:a.limit]
        ndA = np.array([d["per_query"][q]["ndA"] for q in qids])
        ndB = np.array([d["per_query"][q]["ndB"] for q in qids])
        g = ndB - ndA

        # pair each query with its top-1 retrieved document, verbalized through the caption
        texts = []
        for q in qids:
            top = max(visual[q], key=lambda dd: visual[q][dd])
            texts.append(caps.get(top, ""))
        n_missing = sum(1 for t in texts if not t)
        print(f"{cell}: {len(qids)} queries, {n_missing} without a caption for their top doc",
              flush=True)

        if a.group_cv:
            from mv2_io import load_qrels
            qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
            grp = event_groups(qids, qrels)
            splits = list(GroupKFold(5).split(np.arange(len(qids)), groups=grp))
            print(f"  {len(set(grp))} event groups (grouped CV)", flush=True)
        else:
            splits = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(qids))))

        pred = np.zeros(len(qids))
        t0 = time.time()
        for fi, (tr, te) in enumerate(splits):
            ex = [InputExample(texts=[queries[qids[i]], texts[i]], label=float(g[i])) for i in tr]
            dl = DataLoader(ex, shuffle=True, batch_size=a.batch)
            m = CrossEncoder(a.model, num_labels=1, max_length=a.max_len, device="cpu")
            # CrossEncoder defaults to BCEWithLogitsLoss when num_labels == 1, which expects targets in
            # [0, 1]. Our target is an escalation gain in roughly [-1, 1], so the default objective is
            # invalid: an earlier run reached tau +0.240 yet escalated every query, because the ranking
            # carried signal while the zero-crossing was meaningless. Their setup avoids this because a
            # ranking metric is already in [0, 1]. Regress the gain with MSE instead.
            m.fit(train_dataloader=dl, epochs=a.epochs, warmup_steps=max(10, len(dl) // 10),
                  loss_fct=torch.nn.MSELoss(), show_progress_bar=False)
            pred[te] = m.predict([[queries[qids[i]], texts[i]] for i in te],
                                 batch_size=32, show_progress_bar=False)
            print(f"  fold {fi+1}/5 done ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
            del m

        tau = float(kendalltau(pred, g).statistic)
        routed = float(np.where(pred > 0, ndB, ndA).mean())
        # a regression head's zero-crossing may still be off even under MSE, so report the best
        # escalation fraction too: it separates ranking quality from calibration
        order = np.argsort(-pred)
        best_v, best_f = routed, 0.0
        for f in np.arange(0.02, 1.0, 0.02):
            k = int(round(f * len(qids)))
            mask = np.zeros(len(qids), bool); mask[order[:k]] = True
            v = float(np.where(mask, ndB, ndA).mean())
            if v > best_v:
                best_v, best_f = v, float(f)
        out[cell] = {"tau": tau, "routed_ndcg10": routed,
                     "routed_best_f": best_v, "best_f": best_f, "n": len(qids),
                     "cheap": float(ndA.mean()), "uniform": float(ndB.mean()),
                     "model": a.model, "epochs": a.epochs, "group_cv": bool(a.group_cv),
                     "pred": {q: float(p) for q, p in zip(qids, pred)}}
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"{cell}: tau={tau:+.3f}  routed nDCG@10={routed:.4f} (thresh>0)  "
              f"{best_v:.4f} @f={best_f:.2f} (best)  "
              f"(cheap {ndA.mean():.4f}, uniform {ndB.mean():.4f})", flush=True)

    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
