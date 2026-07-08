"""Faithful, cached Q2E evaluation runner.

Reuses the OFFICIAL scoring / fusion / metric functions from
external/q2e_official/src/eval (imported verbatim), but:
  * computes each of the (up to) 5 component score matrices ONCE and caches them to
    disk (results/cache/<tag>/<component>.pt) so re-runs and offline fusion are instant;
  * lets us skip `query_vs_video` when raw videos are unavailable (--no_video) to still
    reproduce the text pipeline + fusion (= paper's "Q2E - Video" family);
  * evaluates the full 31-subset sweep AND the named paper rows, writing machine-readable
    metrics to runs/<tag>/metrics.json — no wandb.

This is faithful: identical encoders, identical ColBERT text scorer, identical
pre-softmax(dim=0) + inverse-entropy fusion + min_max_normalize + torchmetrics scoring
as src/eval/infer.py. It only adds caching and subset bookkeeping.

Usage (run from repo root of external/q2e_official with .venv-eval active, or set
PYTHONPATH to it):
  python run_eval.py --dataset_dir data/MSR-VTT-1kA/Q2E_..._ASR \
      --t2v_encoder multiclip --tag msrvtt_multiclip_asr --out /abs/runs
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk, load_dataset

# ---- locate official repo and import its modules verbatim ----
OFFICIAL = os.environ.get("Q2E_OFFICIAL",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "q2e_official")))
sys.path.insert(0, OFFICIAL)
os.chdir(OFFICIAL)  # so relative model paths (data/models/...) resolve like infer.py

from src.eval.evaluation import retrieval_score            # noqa: E402
from src.eval.fusion_score import (                        # noqa: E402
    fusion_inverse_entropy, fusion_exp_entropy, fusion_reciprocal_rank)
import src.eval.text_embedder as _te                       # noqa: E402
from src.eval.text_embedder import get_many_to_many_score  # noqa: E402
import numpy as _np                                        # noqa: E402

# --- Hardware fix (results-preserving): the official get_one_to_one_score hardcodes
# ColBERT encode bsize=1024 / search bsize=8192, which OOMs a 15GB A2 (authors used
# A100). Batch size only affects throughput, not scores. We patch in an identical
# function with smaller, env-tunable batch sizes. Everything else is byte-identical.
_ENC_BS = int(os.environ.get("Q2E_COLBERT_ENC_BS", "32"))
_SRCH_BS = int(os.environ.get("Q2E_COLBERT_SEARCH_BS", "256"))

def _patched_one_to_one(args, seqs1, seqs2):
    if args.text_emb_type != "colbert":
        return _te.get_one_to_one_score(args, seqs1, seqs2)  # fall back for sbert
    if _te.RAG is None:
        from ragatouille import RAGPretrainedModel
        _te.RAG = RAGPretrainedModel.from_pretrained(args.text_emb_model, verbose=0, n_gpu=1)
    _te.RAG.encode([x for x in seqs2], verbose=False, bsize=_ENC_BS)
    results = _te.RAG.search_encoded_docs(query=seqs1, k=len(seqs2), bsize=_SRCH_BS)
    scores = _np.zeros((len(seqs1), len(seqs2)))
    ri = _np.array([[res["result_index"] for res in row] for row in results])
    sv = _np.array([[res["score"] for res in row] for row in results])
    rows = _np.arange(len(results))[:, None]
    scores[rows, ri] = sv
    _te.RAG.clear_encoded_docs(force=True)
    return torch.tensor(scores)

_te.get_one_to_one_score = _patched_one_to_one  # used internally by get_many_to_many_score

# get_data / form_captions copied VERBATIM from src/eval/infer.py to avoid importing
# infer.py (whose module-level ARGS=get_args() would parse our argv).
def get_data(ds):
    queries, prequels, sequels, durings = [], [], [], []
    whole_video_captions, frame_captions, video_ids = [], [], []
    llm_translation_asrs, whisper_translation_asrs, refined_translation_asrs = [], [], []
    already_seen_query, already_seen_video = set(), set()
    for row in ds:
        query = row["query"]
        if query not in already_seen_query:
            already_seen_query.add(query)
            queries.append(query); prequels.append(row["prequel"])
            sequels.append(row["sequel"]); durings.append(row["during"])
        video_id = row["video_id"]
        if video_id not in already_seen_video:
            already_seen_video.add(video_id)
            video_ids.append(video_id)
            whole_video_captions.append(row["frame2video_caption"])
            frame_captions.append(row["frame_captions"])
            if "asr" in row:
                llm_translation_asrs.append(row["asr"]["translated_llm"])
                whisper_translation_asrs.append(row["asr"]["translated_whisper"])
                refined_translation_asrs.append(row["asr"]["refined"])
    query_to_videoind = defaultdict(list)
    for row in ds:
        query_to_videoind[row["query"]].append(video_ids.index(row["video_id"]))
    return (queries, prequels, sequels, durings, whole_video_captions, frame_captions,
            query_to_videoind, video_ids, llm_translation_asrs,
            whisper_translation_asrs, refined_translation_asrs)


def form_captions(params, **kwargs):
    assert len(params) > 0
    captions = [[] for _ in range(len(kwargs[params[0]]))]
    for param in params:
        if isinstance(kwargs[param][0], list):
            for i, row in enumerate(kwargs[param]):
                captions[i].extend(row)
        else:
            for i, row in enumerate(kwargs[param]):
                captions[i].append(row)
    assert set([len(c) for c in captions]) == {len(captions[0])}
    return captions


ALL5 = ["query_vs_video", "query_vs_captions", "prequel_vs_captions",
        "during_vs_captions", "sequel_vs_captions"]


def build_args(dataset_dir, encoder, num_frames, video_dir):
    class A: pass
    a = A()
    a.dataset_dir = dataset_dir; a.dataset_path = dataset_dir
    a.text_emb_type = "colbert"
    a.text_emb_model = "hltcoe/plaidx-large-eng-tdist-mt5xxl-engeng"
    a.t2v_encoder = encoder
    a.num_of_frames = num_frames
    a.video_dir = video_dir
    a.softmax = "pre"
    return a


def load_ds(path):
    try:
        ds = load_from_disk(path)
    except Exception:
        ds = load_dataset(path)
    if "train" in ds:
        ds = ds["train"]
    return ds


def component_matrices(args, no_video, cache_dir):
    ds = load_ds(args.dataset_dir)
    args.num_of_frames = int(ds["num_of_frames"][0])
    with_asr = "asr" in ds.column_names
    (queries, prequels, sequels, durings, vcaptions, ccaptions, q2vi, video_ids,
     llm_asr, whisper_asr, refined_asr) = get_data(ds)

    # DIVERGENCE (documented, env-gated): each event is a LIST of paraphrases; the many-to-many
    # score is a torch.max over them. LLaMA-1B emits degenerate events with up to 270 paraphrases
    # (vs 8B's max 35). get_many_to_many_score accumulates T*mx_queries ColBERT queries in one call
    # (T=259, mx_queries = max paraphrases over the batch), so 270 -> a ~70k-query encode that OOMs
    # /stalls. Cap paraphrases-per-event to Q2E_EVENT_MAXPARAS -> mx_queries <= K -> peak memory <=
    # the (successful) 8B run (max 35 paraphrases). Max-pool over the kept paraphrases => mild
    # approximation. Off by default -> all other runs bit-for-bit faithful.
    _kpar = os.environ.get("Q2E_EVENT_MAXPARAS")
    if _kpar:
        _k = int(_kpar)
        _cap = lambda e: e[:_k] if isinstance(e, list) and len(e) > _k else e
        _mx = max((len(e) for e in prequels + durings + sequels if isinstance(e, list)), default=1)
        prequels = [_cap(e) for e in prequels]
        durings = [_cap(e) for e in durings]
        sequels = [_cap(e) for e in sequels]
        print(f"[divergence] Q2E_EVENT_MAXPARAS={_k}: capped paraphrases/event "
              f"(pre-cap max {_mx}) -> <= {_k}", flush=True)

    if with_asr:
        cap_params = ["vcaptions", "ccaptions", "llm_translation_asrs",
                      "whisper_translation_asrs", "refined_translation_asrs"]
    else:
        cap_params = ["vcaptions", "ccaptions"]
    captions = form_captions(cap_params, vcaptions=vcaptions, ccaptions=ccaptions,
                             llm_translation_asrs=llm_asr,
                             whisper_translation_asrs=whisper_asr,
                             refined_translation_asrs=refined_asr)

    target = torch.zeros((len(queries), len(video_ids)))
    for i, q in enumerate(queries):
        target[i, q2vi[q]] = 1
    target = target.bool()

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    comps = {}
    todo = [p for p in ALL5 if not (no_video and p == "query_vs_video")]
    for p in todo:
        cf = os.path.join(cache_dir, f"{p}.pt")
        if os.path.exists(cf):
            comps[p] = torch.load(cf); print(f"[cache] {p} {tuple(comps[p].shape)}"); continue
        t0 = time.time()
        if p == "query_vs_video":
            from importlib import import_module
            if args.t2v_encoder == "multiclip":
                mc = import_module("src.eval.MultiCLIP.vision_embedder")
                # Hardware fix (results-preserving): default video batch=128 -> 128*16=2048
                # ViT-H frames per fwd OOMs a 15GB A2. Batch size only affects memory.
                mc.cfg.batch_size = int(os.environ.get("Q2E_MC_VIDEO_BS", "8"))
                s = mc.get_query_vs_video_score(args=args, queries=queries, video_ids=video_ids)
            else:
                iv = import_module("src.eval.InternVideo2.vision_embedder")
                iv_bs = int(os.environ.get("Q2E_IV2_VIDEO_BS", "8"))
                try:
                    iv.cfg.inputs.batch_size_test.video = iv_bs
                    iv.cfg.inputs.batch_size_test.image = iv_bs
                    iv.cfg.evaluation.eval_offload = True  # keep feats on CPU to save VRAM
                except Exception as e:
                    print("iv2 cfg override warn:", e)
                # Hardware fix (results-preserving): official get_text_embedding hardcodes
                # text_bs=256; one such batch builds a ~4 GiB attention tensor and OOMs a
                # 15 GB A2. Reduce the query text-encoding batch (batching is numerically
                # inert). Same encode_text math, smaller chunks.
                _iv_txt_bs = int(os.environ.get("Q2E_IV2_TEXT_BS", "32"))
                def _patched_get_text_embedding(queries, model, tokenizer, _iv=iv, _bs=_iv_txt_bs):
                    n = len(queries); feats = []; atts = []
                    mx = _iv.cfg.max_txt_l
                    for i in range(0, n, _bs):
                        chunk = queries[i:min(n, i + _bs)]
                        ti = tokenizer(chunk, padding="max_length", truncation=True,
                                       max_length=mx, return_tensors="pt").to(_iv.DEVICE)
                        feats.append(model.encode_text(ti)[0])
                        atts.append(ti.attention_mask)
                    return torch.cat(feats, dim=0), torch.cat(atts, dim=0)
                iv.get_text_embedding = _patched_get_text_embedding
                s = iv.get_query_vs_video_score(args=args, queries=queries, video_ids=video_ids)
        elif p == "query_vs_captions":
            s = get_many_to_many_score(args, queries, captions)
        elif p == "prequel_vs_captions":
            s = get_many_to_many_score(args, prequels, captions)
        elif p == "during_vs_captions":
            s = get_many_to_many_score(args, durings, captions)
        elif p == "sequel_vs_captions":
            s = get_many_to_many_score(args, sequels, captions)
        comps[p] = s.float()
        torch.save(comps[p], cf)
        print(f"[compute] {p} {tuple(s.shape)} in {time.time()-t0:.1f}s")
    return comps, target, queries, video_ids, q2vi, with_asr


def fuse(components, params, aggregation, softmax="pre"):
    results = []
    for p in params:
        m = components[p]
        if softmax == "pre":
            m = F.softmax(m, dim=0)
        results.append(m)
    if aggregation == "mean":
        sm = torch.stack(results, 0).mean(0)
    elif aggregation == "max":
        sm = torch.stack(results, 0).max(0).values
    elif aggregation == "inv_entropy":
        sm = fusion_inverse_entropy(None, results, params, range(results[0].shape[0]), range(results[0].shape[1]))
    elif aggregation == "exp_entropy":
        sm = fusion_exp_entropy(None, results, params, range(results[0].shape[0]), range(results[0].shape[1]))
    elif aggregation == "rrf":
        sm = fusion_reciprocal_rank(None, results, params, range(results[0].shape[0]), range(results[0].shape[1]))
    else:
        raise ValueError(aggregation)
    return sm


def subsets(available):
    import itertools
    out = []
    for r in range(1, len(available) + 1):
        for c in itertools.combinations(available, r):
            out.append(list(c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--t2v_encoder", choices=["multiclip", "internvideo2"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="abs path to runs/ dir")
    ap.add_argument("--aggregations", default="inv_entropy,mean,max,exp_entropy,rrf")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--video_dir", default="")
    args_cli = ap.parse_args()

    ddir = os.path.join(OFFICIAL, args_cli.dataset_dir) if not os.path.isabs(args_cli.dataset_dir) else args_cli.dataset_dir
    video_dir = args_cli.video_dir or os.path.join(str(Path(ddir).parent), "videos")
    a = build_args(ddir, args_cli.t2v_encoder, 16, video_dir)

    run_dir = os.path.join(args_cli.out, args_cli.tag)
    cache_dir = os.path.join(run_dir, "cache")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    comps, target, queries, video_ids, q2vi, with_asr = component_matrices(a, args_cli.no_video, cache_dir)
    available = [p for p in ALL5 if p in comps]

    records = []
    # raw (no-fusion) ranking for each single component == paper's encoder/text baselines
    for p in available:
        met = retrieval_score(comps[p], target)
        records.append({"aggregation": "raw", "params": [p], "n_params": 1, "metrics": met})
    for agg in args_cli.aggregations.split(","):
        for params in subsets(available):
            sm = fuse(comps, params, agg)
            met = retrieval_score(sm, target)
            records.append({"aggregation": agg, "params": params,
                            "n_params": len(params), "metrics": met})
    meta = {"tag": args_cli.tag, "dataset_dir": args_cli.dataset_dir,
            "encoder": args_cli.t2v_encoder, "with_asr": with_asr,
            "no_video": args_cli.no_video, "n_queries": len(queries),
            "n_videos": len(video_ids), "available_components": available}
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=2)
    print(f"[done] {args_cli.tag}: {len(records)} records -> {run_dir}/metrics.json")


if __name__ == "__main__":
    main()
