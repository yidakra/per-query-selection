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
from sklearn.model_selection import KFold         # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="asr_shipped,asr_dense,ocr")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many queries")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_bertqpp.json"))
    a = ap.parse_args()

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

        pred = np.zeros(len(qids))
        t0 = time.time()
        for fi, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(qids)):
            ex = [InputExample(texts=[queries[qids[i]], texts[i]], label=float(g[i])) for i in tr]
            dl = DataLoader(ex, shuffle=True, batch_size=a.batch)
            m = CrossEncoder(a.model, num_labels=1, max_length=a.max_len, device="cpu")
            m.fit(train_dataloader=dl, epochs=a.epochs, warmup_steps=max(10, len(dl) // 10),
                  show_progress_bar=False)
            pred[te] = m.predict([[queries[qids[i]], texts[i]] for i in te],
                                 batch_size=32, show_progress_bar=False)
            print(f"  fold {fi+1}/5 done ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
            del m

        tau = float(kendalltau(pred, g).statistic)
        routed = float(np.where(pred > 0, ndB, ndA).mean())
        out[cell] = {"tau": tau, "routed_ndcg10": routed, "n": len(qids),
                     "cheap": float(ndA.mean()), "uniform": float(ndB.mean()),
                     "model": a.model, "epochs": a.epochs,
                     "pred": {q: float(p) for q, p in zip(qids, pred)}}
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"{cell}: tau={tau:+.3f}  routed nDCG@10={routed:.4f}  "
              f"(cheap {ndA.mean():.4f}, uniform {ndB.mean():.4f})", flush=True)

    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
