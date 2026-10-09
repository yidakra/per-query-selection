"""The video-embedding channel: a reviewer's objection, made runnable.

A caption is a lossy compression of the video, so the retrieval index should carry the video
representation and not only caption text. The text half of that objection is closed (a union index
holding caption, transcript and on-screen text is 0 of 33). This is the other half. The reviewer's
Qwen3.5-9B release ships a pooled vector per video for the 55,388-video test split, in three
poolings, and the ids are our judgment doc ids. We put our queries into the same space by running
the same model over the query text and pooling the same way, then score every video by cosine.

`mean_caption` pools the generated caption's token states, so query-to-document is text-to-text and
should behave like a normal dense channel. `mean_video` pools the video's token states, so
query-to-document is text-to-video with no contrastive training between them. Whether that is a
retrieval space at all is the question; a near-zero nDCG is an answer, not a failure.

The vectors are unpublished and stay outside the repo. CPU by default: a 9B model in bf16 needs
about 18 GB, which the box has in RAM and does not have free on the shared GPU.

  python src/multivent2/mv2_video_channel.py --pooling mean_video --limit 32   # smoke test
  python src/multivent2/mv2_video_channel.py --pooling mean_video
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
# Caption release (not redistributed); set MV2_CAPTIONS_DIR to where it is unpacked.
CAP_DIR = os.path.expanduser(os.environ.get("MV2_CAPTIONS_DIR", "~/captions"))
VEC = os.path.join(CAP_DIR, "vectors9b", "eval", "mv2_test")
IDS = os.path.join(CAP_DIR, "eval", "mv2_test", "ids.json")


def encode_queries(texts, model_id, device, batch, max_len):
    """Mean-pool the last hidden state over the real tokens, matching the release's pooling."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            chunk = texts[s:s + batch]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
            print(f"  {min(s + batch, len(texts))}/{len(texts)} queries encoded", flush=True)
    return np.concatenate(out, axis=0)


def self_test(a):
    """Encode a video's own caption and look for that video's row. If the pooling matches the one
    the release used, a caption must retrieve its own vector; if it does not, any retrieval number
    from this script is measuring our reconstruction rather than the release."""
    caps = os.path.join(CAP_DIR, "eval", "mv2_test", "captions.jsonl")
    ids = json.load(open(IDS))["ids"]
    row = {v: i for i, v in enumerate(ids)}
    picked, texts, rows = [], [], []
    rng = np.random.default_rng(0)
    keep = set(rng.choice(len(ids), size=min(a.self_test * 4, len(ids)), replace=False).tolist())
    with open(caps) as fh:
        for i, line in enumerate(fh):
            if len(picked) >= a.self_test:
                break
            r = json.loads(line)
            if r["video_id"] in row and row[r["video_id"]] in keep and r.get("caption", "").strip():
                picked.append(r["video_id"]); texts.append(r["caption"]); rows.append(row[r["video_id"]])
    print(f"self-test on {len(picked)} captions, pooling {a.pooling}", flush=True)
    Q = encode_queries(texts, a.model, a.device, a.batch, a.max_len)

    D = np.asarray(np.load(os.path.join(VEC, f"{a.pooling}.f16.npy"), mmap_mode="r"), dtype=np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True).clip(min=1e-6)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(min=1e-6)
    ranks = []
    for i, r in enumerate(rows):
        sims = D @ Q[i]
        ranks.append(int((sims > sims[r]).sum()) + 1)
    ranks = np.array(ranks)
    print(f"own vector at rank 1 for {int((ranks == 1).sum())}/{len(ranks)}; "
          f"median rank {int(np.median(ranks))}; in top 10 for {int((ranks <= 10).sum())}", flush=True)
    out = os.path.join(ABL, f"mv2_video_channel_selftest_{a.pooling}.json")
    json.dump({"pooling": a.pooling, "model": a.model, "n": len(ranks),
               "rank1": int((ranks == 1).sum()), "top10": int((ranks <= 10).sum()),
               "median_rank": int(np.median(ranks)), "n_videos": int(D.shape[0]),
               "max_len": a.max_len}, open(out, "w"), indent=2)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", default="mean_video", choices=["mean_video", "mean_caption"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N queries")
    ap.add_argument("--out", default="")
    ap.add_argument("--save-query-vectors", default="")
    ap.add_argument("--load-query-vectors", default="",
                    help="skip the encoder and read vectors written by an earlier --encode-only run")
    ap.add_argument("--self-test", type=int, default=0,
                    help="encode N videos' own captions and check they retrieve their own row; this "
                         "is how we tell a mismatched pooling from an unusable space")
    ap.add_argument("--encode-only", action="store_true",
                    help="write the query vectors and stop; the model needs a transformers new "
                         "enough for this architecture, which the evaluation environment is not")
    a = ap.parse_args()

    if a.self_test:                         # needs only the release, not the query set or scorer
        self_test(a)
        return

    from mv2_io import load_qrels, load_queries

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    qids = sorted(q for q in queries if q in qrels)
    if a.limit:
        qids = qids[:a.limit]

    if a.encode_only:                       # nothing below this needs the release or the scorer
        print(f"encoding {len(qids)} queries with {a.model} on {a.device}", flush=True)
        Q = encode_queries([queries[q] for q in qids], a.model, a.device, a.batch, a.max_len)
        target = a.save_query_vectors or os.path.join(ABL, "mv2_video_query_vectors.npy")
        np.save(target, Q.astype(np.float16))
        json.dump({"qids": qids}, open(target + ".qids.json", "w"))
        print(f"wrote {target} with shape {Q.shape}")
        return

    from mv2_ab import per_query_ndcg
    ids = json.load(open(IDS))["ids"]
    D = np.load(os.path.join(VEC, f"{a.pooling}.f16.npy"), mmap_mode="r")
    if len(ids) != D.shape[0]:
        raise SystemExit(f"id list is {len(ids)} but the matrix has {D.shape[0]} rows")
    judged = {d for q in qids for d in qrels[q]}
    covered = len(judged & set(ids))
    print(f"{D.shape[0]} videos x {D.shape[1]} dims; {covered}/{len(judged)} judged documents "
          f"({100 * covered / len(judged):.1f}%) are in the release", flush=True)

    if a.load_query_vectors:
        Q = np.load(a.load_query_vectors).astype(np.float32)
        cached = json.load(open(a.load_query_vectors + ".qids.json"))["qids"]
        if cached != qids:                  # a vector file is only valid for the queries it encoded
            raise SystemExit(f"the vector file holds {len(cached)} queries that do not match the "
                             f"{len(qids)} being scored")
        print(f"loaded {Q.shape[0]} query vectors from {a.load_query_vectors}", flush=True)
    else:
        print(f"encoding {len(qids)} queries with {a.model} on {a.device}", flush=True)
        Q = encode_queries([queries[q] for q in qids], a.model, a.device, a.batch, a.max_len)
        if a.save_query_vectors:
            np.save(a.save_query_vectors, Q.astype(np.float16))

    Dn = np.asarray(D, dtype=np.float32)
    Dn /= np.linalg.norm(Dn, axis=1, keepdims=True).clip(min=1e-6)   # cosine: the release is
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(min=1e-6)     # stored unnormalized
    run = {}
    for i, q in enumerate(qids):
        sims = Dn @ Q[i]
        top = np.argpartition(-sims, min(a.topk, len(sims) - 1))[:a.topk]
        top = top[np.argsort(-sims[top])]
        run[q] = {ids[j]: float(sims[j]) for j in top}

    pq = per_query_ndcg(qrels, run)
    ndcg = float(np.mean([pq.get(q, 0.0) for q in qids]))
    print(f"[{a.pooling}] nDCG@10 {ndcg:.4f} over {len(qids)} queries", flush=True)

    out = a.out or os.path.join(DATA, f"video_dense_qwen35-9b_{a.pooling}.json")
    json.dump(run, open(out, "w"))
    summary = os.path.join(ABL, f"mv2_video_channel_{a.pooling}.json")
    json.dump({"pooling": a.pooling, "model": a.model, "n_queries": len(qids),
               "n_videos": int(D.shape[0]), "dims": int(D.shape[1]),
               "judged_covered": covered, "judged_total": len(judged),
               "ndcg10": ndcg, "run": os.path.basename(out)}, open(summary, "w"), indent=2)
    print(f"wrote {out} and {summary}")


if __name__ == "__main__":
    main()
