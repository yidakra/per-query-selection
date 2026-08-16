"""Build a dense retrieval channel over the raw ASR transcripts / OCR text and emit a ranked list in
the same format as the ranked lists the benchmark ships.

The provided `whisperASR_clip.json` scores transcripts with CLIP's text tower, which was trained to
match images and truncates at 77 tokens -- it is a weak use of the transcript. MMMORRF's lift comes
from running a *real* text retriever over the same transcripts. This does that with a multilingual
bi-encoder: transcripts stay in the language they were spoken in, queries are English, so the encoder
has to align across languages.

Long transcripts are split into overlapping windows and a document scores as its best window
(max-pool), the standard treatment for passage-level evidence inside a long document.

GPU note: pinned to CUDA device 1 by default. Device 0 is in use by another service and is never
touched.

  python src/multivent2/mv2_dense_channel.py --which asr --model intfloat/multilingual-e5-base
"""
import os
import sys
import json
import time
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")


def windows(text, size, overlap):
    """Character windows with overlap; short texts stay whole."""
    text = " ".join(text.split())
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["asr", "ocr"], default="asr")
    ap.add_argument("--text", default=None, help="override the text JSONL (e.g. asr_text_en.jsonl)")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--gpu", type=int, default=1, help="physical GPU index")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-seq", type=int, default=512)
    ap.add_argument("--win", type=int, default=1200, help="window size in characters")
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many docs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--queries", default=None,
                    help="override the query CSV (same Query_id,query format); use for variant runs")
    ap.add_argument("--save-embeddings", default=None,
                    help="write the corpus window embeddings to this .npz after encoding")
    ap.add_argument("--load-embeddings", default=None,
                    help="load corpus window embeddings from this .npz instead of encoding")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    from sentence_transformers import SentenceTransformer
    from mv2_io import load_queries  # noqa: E402

    tag = a.model.split("/")[-1]
    out_path = a.out or os.path.join(DATA, f"{a.which}_dense_{tag}.json")
    e5 = "e5" in a.model.lower()          # e5 needs its asymmetric prefixes to work at all

    docs = []
    text_path = a.text or os.path.join(DATA, f"{a.which}_text.jsonl")
    with open(text_path) as f:
        for line in f:
            d = json.loads(line)
            if d["text"].strip():
                docs.append((d["doc_id"], d["text"]))
    if a.limit:
        docs = docs[:a.limit]

    doc_ids, chunks, owner = [], [], []
    for i, (did, txt) in enumerate(docs):
        doc_ids.append(did)
        for w in windows(txt, a.win, a.overlap):
            chunks.append(("passage: " + w) if e5 else w)
            owner.append(i)
    owner = np.asarray(owner, dtype=np.int32)
    print(f"{a.which}: {len(docs)} docs -> {len(chunks)} windows "
          f"({len(chunks)/max(1,len(docs)):.2f} per doc)")

    m = SentenceTransformer(a.model, device="cuda")
    m.max_seq_length = a.max_seq
    m.half()

    if a.load_embeddings:
        cache = np.load(a.load_embeddings, allow_pickle=False)
        if str(cache["model"]) != a.model or int(cache["win"]) != a.win \
                or int(cache["overlap"]) != a.overlap:
            raise RuntimeError(f"embedding cache {a.load_embeddings} was built with "
                               f"model={cache['model']} win={cache['win']} overlap={cache['overlap']}, "
                               f"which does not match the requested settings")
        D = cache["D"]
        owner = cache["owner"]
        doc_ids = [str(x) for x in cache["doc_ids"]]
        enc_s = 0.0
        print(f"loaded {D.shape[0]} cached window embeddings from {a.load_embeddings}")
    else:
        t0 = time.time()
        D = m.encode(chunks, batch_size=a.batch, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=True).astype(np.float16)
        enc_s = time.time() - t0
        print(f"encoded {len(chunks)} windows in {enc_s/60:.1f} min "
              f"({len(chunks)/enc_s:.0f} win/s), dim {D.shape[1]}")
        if a.save_embeddings:
            np.savez(a.save_embeddings, D=D, owner=owner,
                     doc_ids=np.array(doc_ids), model=a.model, win=a.win, overlap=a.overlap)
            print(f"saved embeddings to {a.save_embeddings}")

    queries = load_queries(a.queries or os.path.join(DATA, "multivent_2_test_queries.csv"))
    qids = sorted(queries)
    qtexts = [("query: " + queries[q]) if e5 else queries[q] for q in qids]
    t1 = time.time()
    Q = m.encode(qtexts, batch_size=a.batch, convert_to_numpy=True,
                 normalize_embeddings=True).astype(np.float16)
    q_s = (time.time() - t1) / len(qids)
    print(f"encoded {len(qids)} queries, {1000*q_s:.1f} ms/query")

    Dt = torch.from_numpy(D).cuda()
    run, t2 = {}, time.time()
    own = torch.from_numpy(owner.astype(np.int64)).cuda()
    ndoc = len(doc_ids)
    for s in range(0, len(qids), 64):
        qb = torch.from_numpy(Q[s:s + 64]).cuda()
        sims = qb @ Dt.T                                       # (b, n_windows)
        # max-pool windows into their document
        pooled = torch.full((sims.shape[0], ndoc), -1e4, device="cuda", dtype=sims.dtype)
        pooled.scatter_reduce_(1, own.expand(sims.shape[0], -1), sims, reduce="amax")
        k = min(a.topk, ndoc)
        vals, idx = torch.topk(pooled.float(), k, dim=1)
        for j, qi in enumerate(qids[s:s + 64]):
            run[qi] = {doc_ids[int(d)]: float(v) for d, v in zip(idx[j].tolist(), vals[j].tolist())}
    search_s = (time.time() - t2) / len(qids)
    print(f"search {1000*search_s:.1f} ms/query over {ndoc} docs")

    json.dump(run, open(out_path, "w"))
    meta = {"model": a.model, "which": a.which, "n_docs": ndoc, "n_windows": len(chunks),
            "dim": int(D.shape[1]), "encode_s_total": enc_s, "query_encode_ms": 1000 * q_s,
            "search_ms": 1000 * search_s, "win": a.win, "overlap": a.overlap,
            "max_seq": a.max_seq, "topk": a.topk}
    json.dump(meta, open(out_path.replace(".json", "_meta.json"), "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
