#!/usr/bin/env python
"""Why does the per-paraphrase tensor not reproduce the cached component?

Two changes were made vs the upstream loop:
  (a) batch sizes raised (enc 32->128, search 256->1024)
  (b) padding paraphrases dropped (7770 padded rows -> 2078 valid rows)

Both alter ColBERT's batch composition. (b) should be EXACTLY score-preserving; (a) need not be,
since ColBERT encodes under fp16 and batch composition changes reduction order / padding length.

This isolates them on ONE doc slot:
   run1 = flat valid rows @ upstream bsize (32/256)
   run2 = flat valid rows @ raised  bsize (128/1024)
   run3 = padded 7770 rows @ upstream bsize (32/256)   <- the exact upstream arrangement
Compare run1 vs run3 (isolates (b)) and run1 vs run2 (isolates (a)).
"""
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_eval as R  # noqa: E402
import src.eval.text_embedder as _te  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
REPO = _ROOT
DS = f"{REPO}/data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"
SLOT = 0


def score(flat, docs, enc_bs, srch_bs):
    R._ENC_BS = enc_bs; R._SRCH_BS = srch_bs   # read at call time inside _patched_one_to_one
    t0 = time.time()
    s = R._patched_one_to_one(args, flat, docs).float()
    print(f"    ({len(flat)} rows, enc={enc_bs} srch={srch_bs}) {time.time()-t0:.0f}s", flush=True)
    return s


ds = R.load_ds(DS)
args = R.build_args(DS, "multiclip", int(ds["num_of_frames"][0]), None)
(queries, prequels, sequels, durings, vcaptions, ccaptions, q2vi, video_ids, la, wa, ra) = R.get_data(ds)
captions = R.form_captions(["vcaptions", "ccaptions"], vcaptions=vcaptions, ccaptions=ccaptions,
                           llm_translation_asrs=la, whisper_translation_asrs=wa, refined_translation_asrs=ra)

mx_q = max(len(q) for q in prequels)
mx_d = max(len(d) for d in captions)
docs = [list(d) + (mx_d - len(d)) * [""] for d in captions]
curr_docs = [d[SLOT] for d in docs]

flat, idx = [], []
for t, q in enumerate(prequels):
    for p, s in enumerate(q):
        if s:
            flat.append(s); idx.append((t, p))
print(f"doc slot {SLOT}: {len(flat)} valid paraphrases, {len(curr_docs)} docs, mx_q={mx_q}")

print("  run1: flat @ 32/256"); r1 = score(flat, curr_docs, 32, 256)
print("  run2: flat @ 128/1024"); r2 = score(flat, curr_docs, 128, 1024)

padded = [list(q) + (mx_q - len(q)) * [""] for q in prequels]
cum = []
for i in range(mx_q):
    cum.extend([q[i] for q in padded])
print("  run3: padded @ 32/256 (upstream arrangement)"); r3 = score(cum, curr_docs, 32, 256)

T = len(prequels)
# cum is laid out as mx_q blocks of T rows: row for (t, p) is p*T + t
r3v = r3[torch.tensor([p * T + t for (t, p) in idx])]

def cmp(a, b, name):
    d = (a - b).abs()
    print(f"  {name:26} max|d|={d.max():.3e}  mean|d|={d.mean():.3e}  "
          f"frac>1e-3: {(d>1e-3).float().mean():.4f}")

cmp(r1, r3v, "(b) flat vs padded  @32/256")
cmp(r1, r2, "(a) bsize 32/256 vs 128/1024")
cmp(r3v, r2, "both changes combined")
