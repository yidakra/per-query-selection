#!/usr/bin/env python
"""Marginal per-query energy of `query_vs_video` -- the tier-A component, and the denominator
of the whole cost axis (A = 0.2, B = 0.4, Full = 1.0).

`component_energy_bench.py` measured the four ColBERT caption components.  This measures the
fifth.  It is a different modality (MultiCLIP ViT-H) and shares no code with the others, so the
proxy's claim that it is "one unit, same as the rest" needs its own check.

WHAT IS AMORTISED.  get_query_vs_video_score() encodes the V=2393-video gallery and then scores.
Gallery encoding is offline, paid identically by every tier, and is not a per-query routing cost.
The marginal cost of tier A per query is:

    get_text_embedding([q])          # MultiCLIP text tower
  + get_score(1 x D, V x D)          # normalise + matmul + rescale

Video embeddings enter only as a (V, D) matrix in a matmul.  Energy depends on tensor SHAPES, not
on the values, so a random (V, D) matrix measures the same joules as the real one.  We therefore
do not re-encode 2393 videos.  (Stated explicitly because it looks like a shortcut and is not: no
score is reported from this script, only energy.)

THE DUMMY-VISION FORWARD.  vision_embedder.get_text_embedding (line ~148) does:

    video = torch.zeros((bsz * 1, 3, 224, 224))
    outputs = model(video, input_ids)
    sequence_output = outputs["text_features"]

i.e. it pushes a batch of BLACK IMAGES through the ViT-H vision tower on every text batch, then
throws the image features away and keeps only text_features.  That is pure waste in the query
path.  We measure the component both AS SHIPPED and with the vision tower bypassed
(`model.encode_text`), because the difference tells us whether tier A's measured cost is intrinsic
or an artefact of upstream's implementation.  The AS SHIPPED number is the honest one for the cost
axis, since it is what the reported nDCG figures actually paid.

Energy: NVML on PHYSICAL GPU1, 5 Hz, minus warm idle.  GPU0 hosts whisper; never touched.

Run:  CUDA_VISIBLE_DEVICES=1 python src/evaluation/component_energy_tierA.py
"""
import os, sys, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval as R  # noqa: E402  (chdirs into the official tree)
from tracking import track  # noqa: E402
from component_energy_bench import PowerMeter, measure, PHYS_GPU  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = _ROOT
DS = f"{REPO}/data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"
OUT = f"{REPO}/results/ablations/component_energy_tierA.json"
V = 2393
REPEATS = 3


def main():
    ds = R.load_ds(DS)
    args = R.build_args(DS, "multiclip", int(ds["num_of_frames"][0]), None)
    (queries, *_rest) = R.get_data(ds)
    T = len(queries)

    from importlib import import_module
    mc = import_module("src.eval.MultiCLIP.vision_embedder")
    mc.cfg.batch_size = int(os.environ.get("Q2E_MC_VIDEO_BS", "8"))
    tokenizer, model = mc.get_model()
    model = model.to(mc.DEVICE).eval()

    proc = torch.stack([mc.process_query(q, tokenizer) for q in queries])   # T x L
    D = int(model.text_projection.shape[-1]) if hasattr(model, "text_projection") else 1024
    meter = PowerMeter()
    rows = []

    with track("component-energy-tierA", gpu_ids=[PHYS_GPU], tags=["cost-model", "energy"],
               config={"T": T, "V": V, "encoder": "multiclip-vit-h", "phys_gpu": PHYS_GPU,
                       "batch_size": mc.cfg.batch_size, "repeats": REPEATS}) as tr:

        # warmup + discover D from a real forward
        with torch.no_grad():
            te = mc.get_text_embedding(proc[:8], model)
        D = te.shape[-1]
        vid_emb = torch.randn(V, D)      # shapes only; no scores reported from this script
        torch.cuda.synchronize()

        idle_w = meter.idle_watts()
        print(f"[idle] GPU{PHYS_GPU} = {idle_w:.1f} W ; D={D}", flush=True)

        def shipped(n):
            with torch.no_grad():
                te = mc.get_text_embedding(proc[:n], model)
                mc.get_score(args, te, vid_emb)

        def bypass(n):
            """Same text features, without pushing black images through the ViT-H vision tower."""
            with torch.no_grad():
                out = []
                for i in range(0, n, mc.cfg.batch_size):
                    ids = proc[i:i + mc.cfg.batch_size].to(mc.DEVICE)
                    out.append(model.encode_text(ids).cpu())
                te = torch.cat(out, 0)
                mc.get_score(args, te, vid_emb)

        Ns = [64, 128, 259]
        for label, fn in (("as_shipped", shipped), ("vision_bypassed", bypass)):
            for n in Ns:
                for r in range(REPEATS):
                    try:
                        m = measure(lambda n=n, fn=fn: fn(n), meter, idle_w)
                    except Exception as e:
                        print(f"  [{label}] N={n} FAILED: {e}", flush=True)
                        break
                    m.update({"label": label, "N": n, "rep": r})
                    rows.append(m)
                    print(f"  {label:16s} N={n:4d} rep{r}  {m['seconds']:6.2f}s  "
                          f"{m['joules_net']:8.1f} J net ({m['mean_watts']:.0f} W)", flush=True)
                    tr.log({f"{label}_N": n, "joules_net": m["joules_net"]})

        fits = {}
        for label in ("as_shipped", "vision_bypassed"):
            sub = [r_ for r_ in rows if r_["label"] == label]
            if len(sub) < 4:
                continue
            X = np.array([r_["N"] for r_ in sub], float)
            for key in ("joules_net", "joules_gross", "seconds"):
                y = np.array([r_[key] for r_ in sub], float)
                A = np.vstack([np.ones_like(X), X]).T
                (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
                r2 = 1 - ((y - (a + b * X)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
                fits.setdefault(label, {})[key] = {"a": float(a), "b": float(b), "r2": float(r2)}
            f = fits[label]["joules_net"]
            print(f"\n[fit:{label}] a={f['a']:.3g} J  b={f['b']:.5g} J/query  R2={f['r2']:.4f}",
                  flush=True)

        out = {"T": T, "V": V, "D": int(D), "idle_watts": idle_w, "batch_size": mc.cfg.batch_size,
               "measurements": rows, "fits": fits,
               "note": "video embeddings are random (V,D): energy depends on shape, not value. "
                       "No retrieval score is reported from this script."}
        json.dump(out, open(OUT, "w"), indent=2)
        print(f"\nwrote {OUT}", flush=True)
        if "as_shipped" in fits and "vision_bypassed" in fits:
            bs_, bb_ = fits["as_shipped"]["joules_net"]["b"], fits["vision_bypassed"]["joules_net"]["b"]
            print(f"\n  tier-A marginal, as shipped      : {bs_:.4g} J / query")
            print(f"  tier-A marginal, vision bypassed : {bb_:.4g} J / query  "
                  f"({bs_/max(bb_,1e-9):.1f}x cheaper)")
            tr.summary({"b_joules_per_query_shipped": bs_, "b_joules_per_query_bypassed": bb_})


if __name__ == "__main__":
    main()
