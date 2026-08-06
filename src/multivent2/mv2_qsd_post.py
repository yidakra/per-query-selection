"""QSD-QPP_Post (Bigdeli et al., "Query Performance Prediction Using Neural Query Space Proximity",
ACM TIST, doi:10.1145/3762197, their Eq. 8) as a routing baseline.

The post-retrieval half of QSD. Where QSD_pre interpolates a prediction straight from the known
effectiveness of nearby historical queries (`mv2_qsd.py`, Eqs. 5-7), QSD_post learns the mapping instead:
a transformer reads the query together with its neighbours in Query Space, those neighbours' known
performance, and the documents retrieved for the query, and regresses performance from all three.

That makes it the only predictor in the suite that sees every kind of evidence at once -- query text,
other queries' labels, and document text -- which is why it is worth the build. If the boundary this
paper argues for is real, QSD_post should behave like the *document-side* family rather than like
QSD_pre, because it is the neighbour signal plus exactly the thing QSD_pre does without.

Implementation, with the deviations stated because their repository ships only consumers of precomputed
QSD outputs:

  Input.  [CLS] query [SEP] the k nearest training queries, each followed by its known escalation gain
          [SEP] the caption of the query's top-1 visually retrieved document. Their formulation pairs
          the query with retrieved document text; the visual channel's documents are video, so the
          shipped captions stand in, the same substitution BERT-QPP and the RAG arm make.

  Head.   The pooled representation is concatenated with four numeric features computed from the same
          neighbourhood -- the inverse-distance interpolation of Eq. 5, the uniform mean of Eq. 7, the
          spread of the neighbour gains, and the mean cosine distance -- and a linear layer regresses
          the gain. Serialising floats into the text alone would leave the model to parse numbers out of
          wordpieces, which is a worse test of the method than giving it the numbers directly.

  Target. The escalation gain, matching every other predictor in our table rather than their absolute
          metric, so the row is comparable with the rest of Table 1.

Leakage discipline is the same as QSD_pre's and matters more here, because the model can memorise:
neighbours are drawn ONLY from the training fold, a training query never retrieves itself, and folds are
grouped by event so a near-duplicate phrasing cannot sit on both sides of the split. Without event
grouping this predictor reads its own answer off a duplicate -- QSD_pre loses 52% of its tau to that
correction.

CPU is the default. A GPU run must be launched with an explicit nonzero CUDA visibility mask, for
example `CUDA_VISIBLE_DEVICES=1 ... --device cuda`; the script refuses an unmasked CUDA request so GPU0
cannot be selected accidentally.

  python src/multivent2/mv2_qsd_post.py --group-cv
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # hide every GPU unless the caller names a safe one
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np                                # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run, load_queries, load_qrels   # noqa: E402
from mv2_qsd import event_groups                        # noqa: E402
from mv2_bertqpp import load_captions, CELLS            # noqa: E402
from mv2_recall_sidecar import load_cell_recall         # noqa: E402
from mv2_nested_calibration import (apply_fraction, calibration_split,
                                    choose_fraction)     # noqa: E402
from scipy.stats import kendalltau                      # noqa: E402
from sklearn.model_selection import KFold, GroupKFold   # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")


def neighbourhood(E, g, tr, targets, k, exclude_self=True):
    """For each index in `targets`, the k nearest TRAINING queries, their gains and distances.

    Returns (idx, psi, feats). `feats` is the four-number summary the head consumes: Eq. 5's
    inverse-distance interpolation, Eq. 7's uniform mean, the spread of the neighbour gains, and the
    mean distance. Everything is computed against `tr` only, which is what keeps the held-out fold out
    of its own prediction.
    """
    sims = E[targets] @ E[tr].T
    psi = 1.0 - sims                                   # cosine distance in [0, 2]
    if exclude_self:
        pos = {t: i for i, t in enumerate(tr)}
        for r, t in enumerate(targets):
            if t in pos:
                psi[r, pos[t]] = np.inf
    order = np.argsort(psi, axis=1)[:, :k]
    gains = g[tr][order]                               # (n_targets, k)
    d = np.take_along_axis(psi, order, axis=1)
    d = np.where(np.isfinite(d), d, 0.0)
    w = 1.0 / (1.0 + d)
    feats = np.stack([(w * gains).sum(1) / w.sum(1),   # Eq. 5, normalised (Shepard)
                      gains.mean(1),                   # Eq. 7
                      gains.std(1),
                      d.mean(1)], axis=1)
    return order, d, gains, feats


def build_text(qtext, nb_texts, nb_gains, doc_text):
    """query [SEP] neighbours with their gains [SEP] top document."""
    nb = " ; ".join(f"{t} ({v:+.2f})" for t, v in zip(nb_texts, nb_gains))
    return qtext, f"similar queries: {nb} | document: {doc_text}"


def run_fold(a, torch, queries, qids, E, g, caps_for, tr, te, seed):
    from transformers import AutoTokenizer, AutoModel
    np.random.seed(seed)
    torch.manual_seed(seed)
    if a.device == "cuda":
        torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = AutoModel.from_pretrained(a.model).to(a.device)
    head = torch.nn.Linear(enc.config.hidden_size + 4, 1).to(a.device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=a.lr)
    lossf = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)

    def prep(targets, exclude_self):
        order, d, gains, feats = neighbourhood(E, g, tr, targets, a.k, exclude_self)
        pairs = []
        for r, t in enumerate(targets):
            nb_texts = [queries[qids[tr[j]]] for j in order[r]]
            pairs.append(build_text(queries[qids[t]], nb_texts, gains[r], caps_for[t]))
        return pairs, feats.astype(np.float32)

    def forward(pairs, feats):
        b = tok([p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True,
                max_length=a.max_len, return_tensors="pt")
        b = {k: v.to(a.device) for k, v in b.items()}
        out = enc(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        numeric = torch.as_tensor(feats, dtype=torch.float32, device=a.device)
        return head(torch.cat([pooled, numeric], dim=1)).squeeze(-1)

    tr_pairs, tr_feats = prep(list(tr), exclude_self=True)
    enc.train(); head.train()
    idx = np.arange(len(tr))
    rng = np.random.default_rng(seed)
    for _ in range(a.epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), a.batch):
            ch = idx[s:s + a.batch]
            if len(ch) < 2:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=a.amp):
                pred = forward([tr_pairs[i] for i in ch], tr_feats[ch])
                target = torch.as_tensor([float(g[tr[i]]) for i in ch],
                                         dtype=torch.float32, device=a.device)
                loss = lossf(pred, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

    te_pairs, te_feats = prep(list(te), exclude_self=False)
    enc.eval(); head.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(te), 32):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=a.amp):
                batch_out = forward(te_pairs[s:s + 32], te_feats[s:s + 32])
            out.extend(batch_out.float().cpu().tolist())
    del enc, head
    if a.device == "cuda":
        torch.cuda.empty_cache()
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="asr_shipped,asr_dense,ocr")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--embed", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="query-space encoder, the same one QSD_pre builds its neighbourhoods in")
    ap.add_argument("--k", type=int, default=5, help="neighbours shown to the model")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=320)
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
                    help="split by event group. Strongly recommended: this predictor is handed other "
                         "queries' labels, so a plain split lets it read its answer off a duplicate")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_qsd_post.json"))
    a = ap.parse_args()

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
    from sentence_transformers import SentenceTransformer

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    visual = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    caps = load_captions()
    sbert = SentenceTransformer(a.embed, device="cpu")
    print(f"captions for {len(caps)} docs", flush=True)

    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
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
        E = sbert.encode([queries[q] for q in qids], batch_size=64, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        caps_for = [caps.get(max(visual[q], key=lambda dd: visual[q][dd]), "") for q in qids]
        n_missing = sum(1 for t in caps_for if not t)
        print(f"{cell}: {len(qids)} queries, {n_missing} without a caption for their top doc",
              flush=True)

        grp = event_groups(qids, qrels) if a.group_cv else None
        if a.group_cv:
            splits = list(GroupKFold(5).split(np.arange(len(qids)), groups=grp))
            print(f"  {len(set(grp))} event groups (grouped CV)", flush=True)
        else:
            splits = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(qids))))

        pred = np.zeros(len(qids))
        nested_decisions = np.zeros(len(qids), dtype=bool)
        calibration = []
        t0 = time.time()
        for fi, (tr, te) in enumerate(splits):
            if a.nested_calibration:
                fit, cal = calibration_split(tr, grp, a.calibration_size, a.seed + fi)
                cal_pred = run_fold(a, torch, queries, qids, E, g, caps_for, fit, cal,
                                    a.seed + 100 * fi)
                fraction, cal_utility = choose_fraction(cal_pred, ndA[cal], ndB[cal])
                calibration.append({"fold": fi, "fraction": fraction,
                                    "calibration_utility": cal_utility,
                                    "n_fit": len(fit), "n_calibration": len(cal)})
                pred[te] = run_fold(a, torch, queries, qids, E, g, caps_for, tr, te,
                                    a.seed + 100 * fi + 1)
                nested_decisions[te] = apply_fraction(pred[te], fraction)
            else:
                pred[te] = run_fold(a, torch, queries, qids, E, g, caps_for, tr, te,
                                    a.seed + fi)
            print(f"  fold {fi+1}/5 done ({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

        tau = float(kendalltau(pred, g).statistic)
        zero_decisions = pred > 0
        decisions = nested_decisions if a.nested_calibration else zero_decisions
        routed = float(np.where(decisions, ndB, ndA).mean())
        routed_zero = float(np.where(zero_decisions, ndB, ndA).mean())
        recA, recB = load_cell_recall(cell, qids)
        rec = None if recA is None else float(np.where(decisions, recB, recA).mean())

        out[cell] = {"tau": tau, "routed_ndcg10": routed, "routed_recall100": rec,
                     "frac_escalated": float(decisions.mean()),
                     "routed_ndcg10_zero": routed_zero,
                     "frac_escalated_zero": float(zero_decisions.mean()), "n": len(qids),
                     "cheap": float(ndA.mean()), "uniform": float(ndB.mean()),
                     "recall_cheap": float(recA.mean()) if recA is not None else None,
                     "recall_uniform": float(recB.mean()) if recB is not None else None,
                     "model": a.model, "k": a.k, "epochs": a.epochs, "group_cv": bool(a.group_cv),
                     "device": a.device, "precision": "amp-fp16" if a.amp else "fp32",
                     "batch_size": a.batch, "max_length": a.max_len, "seed": a.seed,
                     "decision_rule": "nested_fraction" if a.nested_calibration else "zero",
                     "nested_calibration": calibration,
                     "qids": qids,
                     "decisions": "".join("1" if x else "0" for x in decisions),
                     "zero_decisions": "".join("1" if x else "0" for x in zero_decisions),
                     "pred": {q: float(p) for q, p in zip(qids, pred)}}
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"{cell}: tau={tau:+.3f}  routed nDCG@10={routed:.4f}  esc={100*decisions.mean():.1f}%"
              + (f"  R@100={rec:.4f}" if rec is not None else "")
              + f"  (cheap {ndA.mean():.4f}, uniform {ndB.mean():.4f})", flush=True)

    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
