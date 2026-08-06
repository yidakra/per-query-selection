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
leakage, matching the protocol used for the analytic predictors. CPU is the default. A GPU run must be
launched with an explicit nonzero CUDA visibility mask; the script refuses an unmasked CUDA request so
GPU0 cannot be selected accidentally.

  python src/multivent2/mv2_bertqpp.py --cells asr_shipped,asr_dense,ocr --epochs 1
"""
import os, sys, json, time, argparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # hide every GPU unless caller names a safe one
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np                                # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run, load_queries         # noqa: E402
from scipy.stats import kendalltau                # noqa: E402
from sklearn.model_selection import KFold, GroupKFold  # noqa: E402
from mv2_qsd import event_groups                  # noqa: E402
from mv2_nested_calibration import (apply_fraction, calibration_split,
                                    choose_fraction)  # noqa: E402

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
        if c.get("decision_rule") == "nested_fraction" and "decisions" in c:
            decisions = np.array(list(c["decisions"])) == "1"
        else:
            decisions = pred > 0
        chk = float(np.where(decisions, ndB, ndA).mean())
        ok = abs(chk - c["routed_ndcg10"]) < 1e-9
        recA, recB = load_cell_recall(cell, qids)
        rec = None if recA is None else float(np.where(decisions, recB, recA).mean())
        if not ok:
            print(f"{cell}: stored predictions give {chk:.6f}, file says "
                  f"{c['routed_ndcg10']:.6f} -- recall not written")
            continue
        c["routed_recall100"] = rec
        c["recall_cheap"] = float(recA.mean()) if recA is not None else None
        c["recall_uniform"] = float(recB.mean()) if recB is not None else None
        c["frac_escalated"] = float(decisions.mean())
        c["qids"] = qids
        c["decisions"] = "".join("1" if x else "0" for x in decisions)
        print(f"{cell}: nDCG@10 {chk:.4f} reproduced; "
              + (f"R@100 {rec:.4f} (A {recA.mean():.4f}, B {recB.mean():.4f})"
                 if rec is not None else "no verified recall for this cell"))
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


def fit_bi(queries, qids, texts, g, tr, te, a, torch, seed):
    """The bi-encoder variant: encode query and document separately, score their dot product.

    One shared encoder rather than two, which is the usual Siamese arrangement and halves the parameters;
    stated here because their paper is not explicit about weight sharing and the choice is ours. Mean
    pooling over unmasked tokens, then a dot product, then MSE against the escalation gain -- the same
    target and the same objective as the cross-encoder path, so the two differ only in whether query and
    document attend to each other.

    That difference is the point of running both. The cross-encoder must see the pair, so at query time
    it costs a forward pass per candidate document. The bi-encoder's document side can be encoded once,
    offline, leaving one short query pass at serving time. A router that costs more than the retrieval it
    is routing has no reason to exist, so the cheap variant is the one that could actually be deployed --
    and if it degenerates the same way the expensive one does, that is worth knowing.
    """
    from transformers import AutoTokenizer, AutoModel
    np.random.seed(seed)
    torch.manual_seed(seed)
    if a.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = AutoModel.from_pretrained(a.model).to(a.device)
    opt = torch.optim.AdamW(enc.parameters(), lr=a.lr)
    lossf = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)

    def embed(strings, max_len):
        b = tok(strings, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        b = {k: v.to(a.device) for k, v in b.items()}
        out = enc(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    enc.train()
    idx = np.array(tr)
    rng = np.random.default_rng(seed)
    for _ in range(a.epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), a.batch):
            chunk = idx[s:s + a.batch]
            if len(chunk) < 2:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=a.amp):
                qe = embed([queries[qids[i]] for i in chunk], 64)
                de = embed([texts[i] for i in chunk], a.max_len)
                score = (qe * de).sum(-1)
                target = torch.as_tensor([float(g[i]) for i in chunk],
                                         dtype=torch.float32, device=a.device)
                loss = lossf(score, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

    enc.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(te), 32):
            chunk = te[s:s + 32]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=a.amp):
                qe = embed([queries[qids[i]] for i in chunk], 64)
                de = embed([texts[i] for i in chunk], a.max_len)
                score = (qe * de).sum(-1)
            out.extend(score.float().cpu().tolist())
    del enc
    if a.device == "cuda":
        torch.cuda.empty_cache()
    return np.array(out)


def fit_cross(queries, qids, texts, g, tr, te, a, torch, seed):
    """Fit one cross-encoder and return predictions for `te`."""
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    np.random.seed(seed)
    torch.manual_seed(seed)
    if a.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    ex = [InputExample(texts=[queries[qids[i]], texts[i]], label=float(g[i])) for i in tr]
    generator = torch.Generator().manual_seed(seed)
    dl = DataLoader(ex, shuffle=True, batch_size=a.batch, generator=generator)
    model = CrossEncoder(a.model, num_labels=1, max_length=a.max_len, device=a.device)
    # CrossEncoder defaults to BCEWithLogitsLoss for a scalar head, but escalation gain is signed.
    model.fit(train_dataloader=dl, epochs=a.epochs, warmup_steps=max(10, len(dl) // 10),
              loss_fct=torch.nn.MSELoss(), optimizer_params={"lr": a.lr},
              show_progress_bar=False, use_amp=a.amp)
    pred = np.asarray(model.predict([[queries[qids[i]], texts[i]] for i in te],
                                    batch_size=32, show_progress_bar=False))
    del model
    if a.device == "cuda":
        torch.cuda.empty_cache()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="asr_shipped,asr_dense,ocr")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="CUDA is accepted only when CUDA_VISIBLE_DEVICES names nonzero physical GPUs")
    ap.add_argument("--amp", action="store_true",
                    help="use CUDA mixed precision; requires --device cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nested-calibration", action="store_true",
                    help="choose an escalation fraction on a group-disjoint subset of each outer "
                         "training fold, never on outer-test labels")
    ap.add_argument("--calibration-size", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many queries")
    ap.add_argument("--group-cv", action="store_true",
                    help="split by event group, not by query. Needed here for the same reason as QSD: "
                         "near-duplicate event queries often retrieve the SAME top document, so a plain "
                         "split lets the model memorise event -> gain across folds")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_bertqpp.json"))
    ap.add_argument("--variant", default="cross", choices=["cross", "bi"],
                    help="Arabzadeh et al. give both. `cross` concatenates query and document into one "
                         "encoder pass and reads the gain off the joint representation. `bi` encodes "
                         "each separately and scores their dot product, which is the cheaper and more "
                         "deployable of the two -- the document side can be encoded offline, so at "
                         "query time it costs one short forward pass rather than one per candidate. "
                         "That matters for a router, whose whole justification is being cheaper than "
                         "the thing it is deciding about.")
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
    if a.device == "cuda":
        visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
        if not visible or "0" in visible:
            sys.exit("refusing CUDA: set CUDA_VISIBLE_DEVICES to nonzero physical GPU indices only")
        if not torch.cuda.is_available():
            sys.exit("CUDA requested but unavailable under the current visibility mask")
    if a.amp and a.device != "cuda":
        sys.exit("--amp requires --device cuda")
    if a.nested_calibration and not a.group_cv:
        sys.exit("--nested-calibration requires --group-cv")
    torch.set_num_threads(a.threads)

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

        grp = None
        if a.group_cv:
            from mv2_io import load_qrels
            qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
            grp = event_groups(qids, qrels)
            splits = list(GroupKFold(5).split(np.arange(len(qids)), groups=grp))
            print(f"  {len(set(grp))} event groups (grouped CV)", flush=True)
        else:
            splits = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(qids))))

        pred = np.zeros(len(qids))
        nested_decisions = np.zeros(len(qids), dtype=bool)
        calibration = []
        t0 = time.time()
        for fi, (tr, te) in enumerate(splits):
            fit_predict = fit_bi if a.variant == "bi" else fit_cross
            if a.nested_calibration:
                fit, cal = calibration_split(tr, grp, a.calibration_size, a.seed + fi)
                cal_pred = fit_predict(queries, qids, texts, g, fit, cal, a, torch,
                                       a.seed + 100 * fi)
                fraction, cal_utility = choose_fraction(cal_pred, ndA[cal], ndB[cal])
                calibration.append({"fold": fi, "fraction": fraction,
                                    "calibration_utility": cal_utility,
                                    "n_fit": len(fit), "n_calibration": len(cal)})
                pred[te] = fit_predict(queries, qids, texts, g, tr, te, a, torch,
                                       a.seed + 100 * fi + 1)
                nested_decisions[te] = apply_fraction(pred[te], fraction)
            else:
                pred[te] = fit_predict(queries, qids, texts, g, tr, te, a, torch,
                                       a.seed + fi)
            print(f"  fold {fi+1}/5 done ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

        tau = float(kendalltau(pred, g).statistic)
        zero_decisions = pred > 0
        decisions = nested_decisions if a.nested_calibration else zero_decisions
        routed = float(np.where(decisions, ndB, ndA).mean())
        routed_zero = float(np.where(zero_decisions, ndB, ndA).mean())
        from mv2_recall_sidecar import load_cell_recall
        recA, recB = load_cell_recall(cell, qids)
        recall = None if recA is None else float(np.where(decisions, recB, recA).mean())
        out[cell] = {"tau": tau, "routed_ndcg10": routed,
                     "routed_recall100": recall, "frac_escalated": float(decisions.mean()),
                     "routed_ndcg10_zero": routed_zero,
                     "frac_escalated_zero": float(zero_decisions.mean()), "n": len(qids),
                     "cheap": float(ndA.mean()), "uniform": float(ndB.mean()),
                     "model": a.model, "epochs": a.epochs, "group_cv": bool(a.group_cv),
                     "variant": a.variant, "device": a.device,
                     "precision": "amp-fp16" if a.amp else "fp32",
                     "batch_size": a.batch, "max_length": a.max_len, "seed": a.seed,
                     "decision_rule": "nested_fraction" if a.nested_calibration else "zero",
                     "nested_calibration": calibration,
                     "qids": qids,
                     "decisions": "".join("1" if x else "0" for x in decisions),
                     "zero_decisions": "".join("1" if x else "0" for x in zero_decisions),
                     "pred": {q: float(p) for q, p in zip(qids, pred)}}
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"{cell}: tau={tau:+.3f}  routed nDCG@10={routed:.4f}  "
              f"zero={routed_zero:.4f}  esc={100*decisions.mean():.1f}%  "
              f"(cheap {ndA.mean():.4f}, uniform {ndB.mean():.4f})", flush=True)

    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
