#!/usr/bin/env python
"""Is the component-count cost proxy right?  Measure joules per similarity component.

Every frontier figure in this repo plots cost = (number of similarity components scored),
normalised so Full = 1.0 (A = 0.2, B = 0.4).  That rates all five components at unit cost.
`llm_cost_accounting.py` already showed the proxy ignores the ~30 LLM generations the Full tier
issues per query.  This script attacks the other half of the assumption: are the SIMILARITY
components themselves equal cost?

Four of the five run through one code path (run_eval.py:221-228):

    query_vs_captions   -> get_many_to_many_score(args, queries,  captions)
    {prequel,during,sequel}_vs_captions -> get_many_to_many_score(args, <event paraphrases>, captions)

They differ only in how many query-side strings they feed it.  Upstream
(text_embedder.py:96-102) pads every query to `mx_queries` paraphrase slots and concatenates:

    cum_curr_queries = T * mx_queries  strings, scored per doc slot

    query_vs_captions : mx_q =  1  ->    259 strings / doc slot
    event component   : mx_q = 30  ->  7,770 strings / doc slot   (2,076 of them non-empty)

So the query-side work differs by 30x as published, 8x if the padding is skipped.  Pulling the
other way, `_patched_one_to_one` re-encodes all V=2,393 captions on EVERY call, a fixed cost
independent of N.  If that intercept dominates, the components really are near-equal and the
proxy survives.  Which effect wins is an empirical question.  Hence this script.

MODEL.  For a fixed doc slot the cost of one call is
        E(N) = a + b * N
    a = doc-side encode of V captions (amortised in any real deployment: index once, query many)
    b = marginal cost per query-side string (encode + MaxSim against V docs)
A component's total = 17 doc slots * E(N_component).

`b` is the number that matters for routing.  A router pays the marginal cost of the extra
components it buys per query; the caption index is built once, offline, by every tier alike.

DESIGN.  Sweep N on a single doc slot, fit (a, b) by least squares, then PREDICT the total
energy of `during_vs_captions` at N=2070 and compare against the 35.62 Wh independently recorded
by perparaphrase_scores.py (results/energy/runs.jsonl).  That prediction is held out: it is not
in the fit.  If the linear model is wrong, it will miss.

Also measures the real `query_vs_captions` at N=259 with the ACTUAL query strings (mean 172.7
chars) rather than paraphrases (mean 116.5), since ColBERT pads to query_maxlen and the two may
not cost the same per string.

Energy: NVML power sampled at 5 Hz on PHYSICAL GPU1 and integrated, minus a measured idle
baseline.  CodeCarbon's 15 s duty cycle is too coarse for calls this short.  GPU0 hosts an
unrelated whisper server and is never touched.

Run:  CUDA_VISIBLE_DEVICES=1 python src/evaluation/component_energy_bench.py
"""
import os, sys, time, json, threading, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval as R  # noqa: E402  (chdirs into the official tree, patches ColBERT bsize)
from tracking import track  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = _ROOT
DS = f"{REPO}/data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"
OUT = f"{REPO}/results/ablations/component_energy.json"

PHYS_GPU = 1          # NVML index. NOT remapped by CUDA_VISIBLE_DEVICES.
MX_D = 17             # doc slots; every MultiVENT video has exactly 17 caption slots
# Independently measured by perparaphrase_scores.py, from results/energy/runs.jsonl.
# Held out of the fit; used only to check the prediction.
RECORDED_WH = {"during_vs_captions": 35.62, "prequel_vs_captions": 35.55, "sequel_vs_captions": 35.72}
RECORDED_N = {"during_vs_captions": 2070, "prequel_vs_captions": 2078, "sequel_vs_captions": 2080}


class PowerMeter:
    """Integrate NVML power draw on one physical GPU. Joules = sum(P dt)."""

    def __init__(self, phys_gpu=PHYS_GPU, hz=5.0):
        import pynvml
        self.nvml = pynvml
        self.nvml.nvmlInit()
        self.h = self.nvml.nvmlDeviceGetHandleByIndex(phys_gpu)
        self.dt = 1.0 / hz
        self._stop = threading.Event()
        self._samples = []

    def _watts(self):
        return self.nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0

    def _loop(self):
        while not self._stop.is_set():
            self._samples.append((time.time(), self._watts()))
            time.sleep(self.dt)

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2.0)

    def joules(self):
        """Trapezoidal integration of the power trace."""
        s = self._samples
        if len(s) < 2:
            return float("nan"), float("nan"), 0
        t = np.array([x[0] for x in s])
        p = np.array([x[1] for x in s])
        return float(np.trapz(p, t)), float(p.mean()), len(s)

    def idle_watts(self, secs=6.0):
        p = []
        t0 = time.time()
        while time.time() - t0 < secs:
            p.append(self._watts())
            time.sleep(self.dt)
        return float(np.median(p))


def measure(fn, meter, idle_w):
    """Run fn once, return (seconds, gross_joules, net_joules_above_idle, mean_watts)."""
    torch.cuda.synchronize()
    with meter:
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        wall = time.time() - t0
    gross, mean_w, n = meter.joules()
    return {"seconds": wall, "joules_gross": gross, "joules_net": gross - idle_w * wall,
            "mean_watts": mean_w, "n_samples": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--slot", type=int, default=0, help="which doc slot to benchmark")
    ap.add_argument("--padded", action="store_true",
                    help="also time N=7770 (upstream's padded event component). ~10 min.")
    a = ap.parse_args()

    ds = R.load_ds(DS)
    args = R.build_args(DS, "multiclip", int(ds["num_of_frames"][0]), None)
    (queries, prequels, sequels, durings, vcaptions, ccaptions, q2vi, video_ids,
     la, wa, ra) = R.get_data(ds)
    captions = R.form_captions(["vcaptions", "ccaptions"], vcaptions=vcaptions, ccaptions=ccaptions,
                               llm_translation_asrs=la, whisper_translation_asrs=wa,
                               refined_translation_asrs=ra)
    T, V = len(queries), len(captions)
    curr_docs = [d[a.slot] for d in captions]
    assert all(curr_docs), "doc slot has empty captions; intercept would not be comparable"

    para = [s for q in durings for s in q if s]          # 2070 during-paraphrases
    Ns = [259, 519, 1038, 2076]

    meter = PowerMeter()
    rows = []

    with track("component-energy-bench", gpu_ids=[PHYS_GPU], tags=["cost-model", "energy"],
               config={"T": T, "V": V, "mx_d": MX_D, "doc_slot": a.slot, "repeats": a.repeats,
                       "encoder": "colbert-plaidx-large", "phys_gpu": PHYS_GPU,
                       "enc_bs": os.environ.get("Q2E_COLBERT_ENC_BS", "32"),
                       "search_bs": os.environ.get("Q2E_COLBERT_SEARCH_BS", "256")}) as tr:

        print("[warmup] loading ColBERT + one scoring call", flush=True)
        R._patched_one_to_one(args, para[:16], curr_docs[:64])
        torch.cuda.synchronize()

        idle_w = meter.idle_watts()
        print(f"[idle] GPU{PHYS_GPU} baseline = {idle_w:.1f} W", flush=True)
        tr.summary({"idle_watts": idle_w})

        # ---- the real query_vs_captions component (actual query strings, N=259) ----
        for r in range(a.repeats):
            m = measure(lambda: R._patched_one_to_one(args, list(queries), curr_docs), meter, idle_w)
            m.update({"label": "query_vs_captions(real queries)", "N": T, "rep": r,
                      "strings": "queries", "mean_chars": float(np.mean([len(s) for s in queries]))})
            rows.append(m)
            print(f"  qvc  N={T:5d} rep{r}  {m['seconds']:7.1f}s  {m['joules_net']:8.1f} J net "
                  f"({m['mean_watts']:.0f} W)", flush=True)
            tr.log({"N": T, "seconds": m["seconds"], "joules_net": m["joules_net"]})

        # ---- sweep N over event paraphrases (same code path, same docs) ----
        for N in Ns:
            for r in range(a.repeats):
                seqs = para[:N] if N <= len(para) else (para * (N // len(para) + 1))[:N]
                m = measure(lambda s=seqs: R._patched_one_to_one(args, s, curr_docs), meter, idle_w)
                m.update({"label": f"paraphrases N={N}", "N": N, "rep": r, "strings": "durings",
                          "mean_chars": float(np.mean([len(s) for s in seqs]))})
                rows.append(m)
                print(f"  para N={N:5d} rep{r}  {m['seconds']:7.1f}s  {m['joules_net']:8.1f} J net "
                      f"({m['mean_watts']:.0f} W)", flush=True)
                tr.log({"N": N, "seconds": m["seconds"], "joules_net": m["joules_net"]})

        if a.padded:
            N = T * 30      # 7770: what upstream actually scores, padding included
            seqs = (para * (N // len(para) + 1))[:N]
            m = measure(lambda: R._patched_one_to_one(args, seqs, curr_docs), meter, idle_w)
            m.update({"label": "paraphrases N=7770 (upstream padded)", "N": N, "rep": 0,
                      "strings": "durings"})
            rows.append(m)
            print(f"  PAD  N={N:5d}       {m['seconds']:7.1f}s  {m['joules_net']:8.1f} J net", flush=True)

        # ---- fit E(N) = a + b*N on the paraphrase sweep ONLY (qvc + N=2076 held out) ----
        fit_rows = [r_ for r_ in rows if r_["strings"] == "durings" and r_["N"] in (259, 519, 1038)]
        X = np.array([r_["N"] for r_ in fit_rows], float)
        fits = {}
        for key in ("joules_net", "joules_gross", "seconds"):
            y = np.array([r_[key] for r_ in fit_rows], float)
            A = np.vstack([np.ones_like(X), X]).T
            (a_hat, b_hat), *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - (a_hat + b_hat * X)
            r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
            fits[key] = {"a": float(a_hat), "b": float(b_hat), "r2": float(r2)}
            print(f"\n[fit:{key}]  a={a_hat:.4g}  b={b_hat:.6g}/string  R2={r2:.4f}", flush=True)
        aJ, bJ = fits["joules_net"]["a"], fits["joules_net"]["b"]

        # ---- HELD-OUT prediction: during_vs_captions total, vs the recorded 35.62 Wh ----
        # CodeCarbon reports GROSS whole-device energy (it integrates NVML power; it does not
        # subtract idle). So the prediction must be built from joules_GROSS, not joules_net.
        # Comparing a net prediction against a gross measurement would understate by idle*wall
        # -- ~21.1 W * ~2760 s = 16 Wh here, i.e. nearly half the recorded figure.
        aG, bG = fits["joules_gross"]["a"], fits["joules_gross"]["b"]
        Nd = RECORDED_N["during_vs_captions"]
        pred_J = MX_D * (aG + bG * Nd)
        pred_Wh = pred_J / 3600.0
        rec_Wh = RECORDED_WH["during_vs_captions"]
        err = (pred_Wh - rec_Wh) / rec_Wh * 100
        pred_wall = MX_D * (fits["seconds"]["a"] + fits["seconds"]["b"] * Nd)
        print(f"\n[held-out] during_vs_captions predicted {pred_Wh:.2f} Wh (gross) vs recorded "
              f"{rec_Wh:.2f} Wh ({err:+.1f}%)", flush=True)
        print(f"[held-out] predicted wall {pred_wall:.0f}s vs recorded 2760s "
              f"({(pred_wall-2760)/2760*100:+.1f}%)", flush=True)

        # ---- what the proxy gets wrong ----
        qvc_J = float(np.mean([r_["joules_net"] for r_ in rows if r_["strings"] == "queries"]))
        n_para_mean = 2076
        marg_qvc = bJ * T                    # marginal, per doc slot, whole query set
        marg_evt = bJ * n_para_mean
        marg_evt_pad = bJ * (T * 30)
        amort_qvc = aJ + bJ * T
        amort_evt = aJ + bJ * n_para_mean
        print(f"\n  amortised (a+bN, index rebuilt per call): event/qvc = {amort_evt/amort_qvc:.2f}x")
        print(f"  marginal  (bN, index built once)     : event/qvc = {marg_evt/marg_qvc:.2f}x")
        print(f"  marginal, upstream padded mx_q=30    : event/qvc = {marg_evt_pad/marg_qvc:.2f}x")
        print(f"  the cost proxy assumes                 event/qvc = 1.00x")

        out = {
            "T": T, "V": V, "mx_d": MX_D, "doc_slot": a.slot, "idle_watts": idle_w,
            "measurements": rows,
            "fit": {"a_joules": float(aJ), "b_joules_per_string": float(bJ),
                    "fitted_on_N": [259, 519, 1038], "note": "per doc slot",
                    "all": fits},
            "heldout_check": {"component": "during_vs_captions", "N": Nd,
                              "predicted_Wh": float(pred_Wh), "recorded_Wh": rec_Wh,
                              "pct_error": float(err), "basis": "gross (matches codecarbon)",
                              "predicted_wall_s": float(pred_wall), "recorded_wall_s": 2760.0,
                              "recorded_by": "perparaphrase_scores.py (results/energy/runs.jsonl)"},
            "query_vs_captions_measured_joules_per_slot": qvc_J,
            "ratios_event_over_qvc": {
                "amortised_index_per_call": float(amort_evt / amort_qvc),
                "marginal_index_amortised": float(marg_evt / marg_qvc),
                "marginal_upstream_padded_mxq30": float(marg_evt_pad / marg_qvc),
                "cost_proxy_assumes": 1.0,
            },
        }
        json.dump(out, open(OUT, "w"), indent=2)
        print(f"\nwrote {OUT}", flush=True)
        tr.summary({"a_joules": float(aJ), "b_joules_per_string": float(bJ),
                    "heldout_pct_error": float(err),
                    "marginal_event_over_qvc": float(marg_evt / marg_qvc)})


if __name__ == "__main__":
    main()
