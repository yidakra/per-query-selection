"""Build a stronger speech channel by cross-encoder reranking, then let selection face it.

The review's standing objection: the routing gap is measured on channels roughly 2x below the
benchmark's best systems, and only a stronger channel can say whether the gap survives. External run
files are not in hand, so this builds the strongest channel available locally: rerank the translated
dense channel's top candidates with a multilingual cross-encoder over the translated transcripts.
Reranking is the standard second stage of strong systems, so the result is a fair step toward their
regime, not a toy.

Per query: take the top K documents from the dense run, score (query, transcript) pairs with the
cross-encoder on GPU1, and write a run whose top K carries the reranker's scores. Documents below K
are dropped; RRF fusion and nDCG@10 read ranks, and the selection experiment trims to the top 100
everywhere else. Checkpoints every 200 queries and resumes from its own output.

  CUDA_VISIBLE_DEVICES=1 python src/multivent2/mv2_rerank_channel.py
"""
import os
import sys
import json
import time
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_run, load_queries, load_qrels  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="asr_dense_bge-m3-mt.json",
                    help="first-stage run to rerank, under data/multivent2/")
    ap.add_argument("--text", default="asr_text_en.jsonl",
                    help="doc_id -> text corpus the cross-encoder reads")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--out", default=os.path.join(DATA, "asr_rerank_v2m3_mt.json"))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu keeps the shared GPU free; the cross-encoder scores identically, "
                         "in float32 and slower")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new queries, for timing")
    a = ap.parse_args()
    ckpt = a.out + ".partial.jsonl"

    # The box once had a GPU 0 belonging to another project and this refused to touch it. It now has
    # a single shared device, which is GPU 0, so the refusal only applies to the CUDA path and only
    # when no device has been named at all.
    if a.device == "cuda" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        sys.exit("refusing to run: name a GPU in CUDA_VISIBLE_DEVICES, or pass --device cpu")

    from sentence_transformers import CrossEncoder

    run = load_run(os.path.join(DATA, a.run))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    texts = {}
    with open(os.path.join(DATA, a.text)) as f:
        for line in f:
            r = json.loads(line)
            if r.get("text", "").strip():
                texts[r["doc_id"]] = r["text"]
    qids = sorted(q for q in run if q in queries)
    print(f"{len(qids)} queries, {len(texts)} docs with text, top {a.topk} per query", flush=True)

    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["qid"]] = r["scores"]
                except Exception:
                    pass
        print(f"resuming: {len(done)} queries already reranked", flush=True)

    kwargs = {"automodel_args": {"torch_dtype": "float16"}} if a.device == "cuda" else {}
    model = CrossEncoder(a.model, max_length=a.max_len, device=a.device, **kwargs)
    t0 = time.time()
    n_new = 0
    with open(ckpt, "a", buffering=1) as fh:
        for q in qids:
            if q in done:
                continue
            top = sorted(run[q].items(), key=lambda kv: -kv[1])[:a.topk]
            pairs = [(queries[q], texts.get(d, "")[:4000]) for d, _ in top]
            scores = model.predict(pairs, batch_size=a.batch, show_progress_bar=False)
            rec = {d: float(s) for (d, _), s in zip(top, scores)}
            fh.write(json.dumps({"qid": q, "scores": rec}) + "\n")
            done[q] = rec
            n_new += 1
            if a.limit and n_new >= a.limit:
                rate = n_new / (time.time() - t0)
                todo = len(qids) - len(done)
                print(f"timing: {rate:.2f} queries/s, {todo} left, "
                      f"eta {todo / max(rate, 1e-9) / 3600:.1f} h", flush=True)
                return
            if n_new % 200 == 0:
                rate = n_new / (time.time() - t0)
                eta = (len(qids) - len(done)) / max(rate, 1e-9) / 3600
                print(f"  {len(done)}/{len(qids)} ({rate:.1f} q/s, eta {eta:.1f} h)", flush=True)

    out_run = {q: done[q] for q in qids}
    json.dump(out_run, open(a.out, "w"))
    print(f"wrote {a.out}", flush=True)

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    nd_new = per_query_ndcg(qrels, out_run)
    nd_old = per_query_ndcg(qrels, {q: run[q] for q in qids})
    common = sorted(set(nd_new) & set(nd_old))
    print(f"reranked channel nDCG@10: {np.mean([nd_new[q] for q in common]):.4f}  "
          f"(first stage: {np.mean([nd_old[q] for q in common]):.4f})", flush=True)


if __name__ == "__main__":
    main()
