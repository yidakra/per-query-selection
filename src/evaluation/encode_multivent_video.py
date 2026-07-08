"""Compute the MultiVENT `query_vs_video` component (MultiCLIP) from the fetched frame caches.

Memory-safe rewrite: the repo's get_query_vs_video_score() stacks the raw frames of ALL videos
in CPU RAM before embedding (2,393 x 16 x 3 x 224 x 224 float32 ~= 30 GB) which OOM-kills the
process on a 31 GB box. We instead embed videos in CHUNKS, keeping only one chunk's frames in
RAM at a time; the resulting per-video embeddings are tiny (V x D). Numerically identical to the
repo path (each video's embedding is independent) — we just call the same lower-level functions
(process_query/get_text_embedding, process_video/get_video_embedding, get_score) in chunks.

Missing videos (not recoverable) have no frame cache; we feed them valid zero frames (so nothing
crashes) and then set their score COLUMN to a constant post-hoc. After the pipeline's
softmax-over-queries (dim=0) a constant column is uniform -> no spurious signal (option (b):
full 2,393 pool, missing = null video). The video signal is identical for noASR/ASR, so this is
computed once (on the noASR dataset) and reused for both when fusing with the cached text.

Outputs (runs/multivent_video_multiclip/):
  cache/query_vs_video.pt   (Q x V float, missing columns zeroed)
  video_order.json          {"video_ids":[...], "queries":[...], "missing":[...]}
Run from repo root with venv + GPU (CUDA_VISIBLE_DEVICES=1).
"""
import os, sys, json, time
from types import SimpleNamespace

REPO = "/home/ubuntu/q2e_repro/external/q2e_official"
DS = "/home/ubuntu/q2e_repro/data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"
RUN = "/home/ubuntu/q2e_repro/runs/multivent_video_multiclip"

sys.path.insert(0, REPO)
os.chdir(REPO)
import torch  # noqa: E402
import numpy as np  # noqa: E402
from datasets import load_from_disk  # noqa: E402


def canonical_order(ds):
    """Replicate run_eval.get_data's first-appearance dedup for queries and video_ids."""
    queries, video_ids, sq, sv = [], [], set(), set()
    for row in ds:
        q = row["query"]
        if q not in sq:
            sq.add(q); queries.append(q)
        v = row["video_id"]
        if v not in sv:
            sv.add(v); video_ids.append(v)
    return queries, video_ids


def main():
    os.makedirs(os.path.join(RUN, "cache"), exist_ok=True)
    ds = load_from_disk(DS)
    F = int(ds["num_of_frames"][0])
    queries, video_ids = canonical_order(ds)
    print(f"[encode] {len(queries)} queries x {len(video_ids)} videos, F={F}", flush=True)

    import src.eval.MultiCLIP.vision_embedder as mc
    cache_dir = f"{mc.CLIP_FRAME_DIR}/{F}"
    missing = [v for v in video_ids if not os.path.exists(f"{cache_dir}/{v}.npy")]
    miss_set = set(missing)
    print(f"[encode] {len(missing)} videos have no frame cache (null video) "
          f"= {100*len(missing)/len(video_ids):.1f}%", flush=True)

    mc.cfg.max_frames = F
    mc.cfg.batch_size = int(os.environ.get("Q2E_MC_VIDEO_BS", "8"))   # GPU micro-batch
    mc.video_path = os.path.join(REPO, "data", "MultiVENT", "_fetch_videos_tmp")  # unused (all cached)
    tokenizer, model = mc.get_model()

    def load_video(v):
        if v in miss_set:  # valid zero frames -> no crash; column zeroed post-hoc
            z = np.zeros((1, mc.cfg.max_frames, 1, 3, mc.cfg.image_resolution,
                          mc.cfg.image_resolution), dtype=np.float32)
            return torch.tensor(z).float()
        return mc.process_video(v)

    t0 = time.time()
    # Text embeddings (queries) once — small.
    with torch.no_grad():
        proc_q = torch.stack([mc.process_query(q, tokenizer) for q in queries])
        text_emb = mc.get_text_embedding(proc_q, model)
    del proc_q

    # Video embeddings in chunks -> bounded CPU RAM (one chunk of raw frames at a time).
    CHUNK = int(os.environ.get("Q2E_MC_VIDEO_CHUNK", "200"))
    vid_embs = []
    for i in range(0, len(video_ids), CHUNK):
        chunk = video_ids[i:i + CHUNK]
        with torch.no_grad():
            proc_v = torch.stack([load_video(v) for v in chunk])
            emb = mc.get_video_embedding(proc_v, model)   # [chunk x D] on CPU
        vid_embs.append(emb)
        del proc_v
        print(f"[encode] videos {min(i+CHUNK, len(video_ids))}/{len(video_ids)} "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)
    video_emb = torch.cat(vid_embs, dim=0)

    args = SimpleNamespace(num_of_frames=F)
    with torch.no_grad():
        scores = mc.get_score(args, text_emb, video_emb).float()   # [Q x V]
    print(f"[encode] query_vs_video {tuple(scores.shape)} in {time.time()-t0:.0f}s", flush=True)

    # Null out missing-video columns (constant column -> uniform after softmax dim=0).
    if missing:
        miss_idx = [i for i, v in enumerate(video_ids) if v in miss_set]
        scores[:, miss_idx] = 0.0

    torch.save(scores, os.path.join(RUN, "cache", "query_vs_video.pt"))
    json.dump({"video_ids": list(video_ids), "queries": list(queries), "missing": missing},
              open(os.path.join(RUN, "video_order.json"), "w"))
    print(f"[encode] saved -> {RUN}/cache/query_vs_video.pt", flush=True)


if __name__ == "__main__":
    main()
