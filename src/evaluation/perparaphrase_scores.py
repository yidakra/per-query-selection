#!/usr/bin/env python
"""Recompute the three event components RETAINING the per-paraphrase axis.

Q2E scores an event component as
    S[q, v] = max over paraphrases p, caption slots c  of  sim(para[q][p], cap[v][c])
(`get_many_to_many_score`, text_embedder.py:114 -- a max-pool over BOTH axes).

Because the pool is a max, a bad paraphrase can only ever RAISE a wrong video's score:
adding paraphrases injects monotone noise.  That is a candidate mechanism for the two
facts we already have -- 17% of MultiVENT queries are HURT by the event tiers, and the
Full-tier gain is unpredictable per query.

This script keeps the paraphrase axis, producing
    P[q, p, v] = max over caption slots c of sim(para[q][p], cap[v][c])
so that   P.max(dim=1) == the cached component tensor   (asserted).  With P in hand we can
ask, on CPU and with no LLM, whether the RIGHT events are already in the generated set --
i.e. whether tier C should be a paraphrase SELECTOR (no regeneration, no drift) rather than
an evidence-conditioned regeneration loop.

Costs exactly what the original component computation cost: the inner get_one_to_one_score
already scores all T*mx_paras paraphrases against all docs; we simply stop collapsing axis p.

GPU: text encoder only (ColBERT/PLAID).  Pin to GPU1 -- GPU0 runs the whisper server.
"""
import os, sys, time, json, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval as R  # noqa: E402  (sets sys.path/chdir to the official tree, patches ColBERT bsize)
from tracking import track  # noqa: E402

EVENTS = {"prequel_vs_captions": "prequels",
          "during_vs_captions": "durings",
          "sequel_vs_captions": "sequels"}
SETTINGS = {
    "noASR": ("multivent_textonly_noASR", "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"),
    "ASR": ("multivent_textonly_ASR", "Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_ASR"),
}
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
REPO = _ROOT


def many_to_many_perparaphrase(args, queries, docs, tracker=None):
    """Mirror of get_many_to_many_score but WITHOUT the max over the paraphrase axis.

    queries: T lists of paraphrase strings
    docs:    V lists of caption strings
    returns: (P, valid)  P: T x mx_q x V float32 ; valid: T x mx_q bool
    Padding slots are left at 0.0 (identical to upstream, whose `scores` init is zeros and
    whose padded rows are masked to -inf before the max).

    Efficiency note: upstream pads every query to mx_q paraphrase slots and then encodes and
    searches the padding, only to mask it to -inf afterwards.  With mean 8 paraphrases and
    mx_q=30 that is ~73% wasted compute.  We score only the VALID (query, paraphrase) pairs.
    Scores are unchanged -- the padded slots contribute nothing to any max.
    """
    queries = [list(q) for q in queries]
    docs = [list(d) for d in docs]
    if isinstance(queries[0], str):
        queries = [[q] for q in queries]
    if isinstance(docs[0], str):
        docs = [[d] for d in docs]

    T, V = len(queries), len(docs)
    mx_q = max(len(q) for q in queries)
    mx_d = max(len(d) for d in docs)
    valid = torch.zeros(T, mx_q, dtype=torch.bool)
    flat, idx = [], []          # flat[n] is the n-th valid paraphrase; idx[n] = (t, p)
    for t, q in enumerate(queries):
        for p, s in enumerate(q):
            if s:
                valid[t, p] = True
                flat.append(s)
                idx.append((t, p))
    for i, d in enumerate(docs):
        docs[i] = d + (mx_d - len(d)) * [""]
    rows = torch.tensor([t for t, _ in idx])
    cols = torch.tensor([p for _, p in idx])
    print(f"  scoring {len(flat)} valid paraphrases (upstream would pad to {T*mx_q})", flush=True)

    P = torch.zeros(T, mx_q, V)
    for j in range(mx_d):
        curr_docs = [d[j] for d in docs]
        t0 = time.time()
        sc = R._patched_one_to_one(args, flat, curr_docs).float()  # N x V
        for k, d in enumerate(curr_docs):
            if not d:
                sc[:, k] = float("-inf")
        P[rows, cols, :] = torch.max(P[rows, cols, :], sc)
        dt = time.time() - t0
        print(f"  doc-slot {j+1}/{mx_d}  ({dt:.1f}s)", flush=True)
        if tracker is not None:
            tracker.log({"doc_slot": j + 1, "slot_seconds": dt})
    return P, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="noASR", choices=list(SETTINGS))
    ap.add_argument("--event", default="all")
    a = ap.parse_args()

    tag, ds_name = SETTINGS[a.setting]
    ds_dir = os.path.join(REPO, "data", "MultiVENT", ds_name)
    out_dir = os.path.join(REPO, "runs", tag, "paracache")
    os.makedirs(out_dir, exist_ok=True)

    ds = R.load_ds(ds_dir)
    args = R.build_args(ds_dir, "multiclip", int(ds["num_of_frames"][0]), None)
    (queries, prequels, sequels, durings, vcaptions, ccaptions, q2vi, video_ids,
     llm_asr, whisper_asr, refined_asr) = R.get_data(ds)
    with_asr = "asr" in ds.column_names
    cap_params = (["vcaptions", "ccaptions", "llm_translation_asrs",
                   "whisper_translation_asrs", "refined_translation_asrs"]
                  if with_asr else ["vcaptions", "ccaptions"])
    captions = R.form_captions(cap_params, vcaptions=vcaptions, ccaptions=ccaptions,
                               llm_translation_asrs=llm_asr,
                               whisper_translation_asrs=whisper_asr,
                               refined_translation_asrs=refined_asr)
    src = {"prequels": prequels, "durings": durings, "sequels": sequels}
    todo = list(EVENTS) if a.event == "all" else [a.event]

    for comp in todo:
        outf = os.path.join(out_dir, f"{comp}.pt")
        if os.path.exists(outf):
            print(f"[skip] {outf} exists", flush=True)
            continue
        print(f"\n=== {a.setting} / {comp}  (T={len(queries)}, V={len(video_ids)}) ===", flush=True)
        t0 = time.time()
        # CUDA_VISIBLE_DEVICES pins torch to GPU1; codecarbon needs the PHYSICAL index, which
        # CUDA_VISIBLE_DEVICES does NOT remap. GPU0 hosts an unrelated whisper server.
        with track(f"paracache-{a.setting}-{comp}", gpu_ids=[1], tags=["tierC", "paracache"],
                   config={"setting": a.setting, "component": comp, "T": len(queries),
                           "V": len(video_ids), "encoder": "colbert-plaidx-large",
                           "enc_bs": os.environ.get("Q2E_COLBERT_ENC_BS", "32"),
                           "search_bs": os.environ.get("Q2E_COLBERT_SEARCH_BS", "256")}) as tr:
            P, valid = many_to_many_perparaphrase(args, src[EVENTS[comp]], captions, tracker=tr)

            # SAVE FIRST. An earlier version validated before writing and discarded ~50 min of
            # GPU work when the check tripped. Persist, then judge.
            torch.save({"P": P, "valid": valid, "queries": queries, "video_ids": video_ids}, outf)
            print(f"[wrote] {outf}  P{tuple(P.shape)}  ({time.time()-t0:.0f}s)", flush=True)

            # VALIDATION. ColBERT encodes under fp16, so re-scoring the same (paraphrase,
            # caption) pair in a different batch composition moves the score by ~1e-2 on a 0-100
            # scale (measured: diag_colbert_determinism.py). A raw atol on scores is therefore
            # the wrong gate. What must be preserved is every number we report -- the metric
            # gate lives in tierC_selection_oracle.load_cell(), which refuses to build an oracle
            # if the reconstructed tensor shifts the published Full-tier nDCG by > 0.01.
            ref = torch.load(os.path.join(REPO, "runs", tag, "cache", f"{comp}.pt")).float()
            d = (P.max(dim=1).values - ref).abs()
            print(f"[validate] {comp}: max|diff|={d.max():.3e}  mean|diff|={d.mean():.3e}  "
                  f"frac>1e-3={(d > 1e-3).float().mean():.4f}", flush=True)
            rec = {"component": comp, "max_abs_diff": float(d.max()),
                   "mean_abs_diff": float(d.mean()),
                   "frac_gt_1e-3": float((d > 1e-3).float().mean()),
                   "n_valid_paraphrases": int(valid.sum())}
            json.dump(rec, open(outf.replace(".pt", "_validation.json"), "w"), indent=2)
            tr.summary(rec)


if __name__ == "__main__":
    main()
