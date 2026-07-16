"""Measured GPU energy of the Full tier's LLM step on MultiVENT 2.0 -- the cost tier_cost.py omitted.

tier_cost.py measured the similarity components in joules but explicitly EXCLUDED the LLM generations
the Full tier issues per query ("excludes the ~30 LLaMA-70B generations/query"). On 2.0 that generation
IS the dominant Full-tier cost. This measures it directly: NVML power on PHYSICAL GPU1 sampled at 5 Hz
and integrated, minus a model-loaded idle baseline, over a sample of real queries driven through the
same Ollama endpoint mv2_events.py uses. GPU0 hosts Whisper and is never touched.

  CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_energy.py --n 40
"""
import os
import sys
import time
import json
import argparse
import threading
import numpy as np
import requests
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_queries  # noqa: E402
from mv2_events import SYS, MODEL, URL  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
PHYS_GPU = 1                       # NVML index; not remapped by CUDA_VISIBLE_DEVICES


class PowerMeter:
    """Integrate NVML power draw on one physical GPU. Joules = integral of P dt. (Same method as
    component_energy_bench.py's meter, reimplemented here to stay free of run_eval's import side effects.)"""

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
        self._samples = []; self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(timeout=2.0)

    def joules(self):
        s = self._samples
        if len(s) < 2:
            return float("nan"), float("nan"), 0
        t = np.array([x[0] for x in s]); p = np.array([x[1] for x in s])
        return float(np.trapz(p, t)), float(p.mean()), len(s)

    def idle_watts(self, secs=8.0):
        p = []; t0 = time.time()
        while time.time() - t0 < secs:
            p.append(self._watts()); time.sleep(self.dt)
        return float(np.median(p))


def gen(query, num_predict=300):
    r = requests.post(URL, timeout=180, json={
        "model": MODEL, "stream": False, "format": "json",
        "options": {"temperature": 0.3, "num_predict": num_predict, "seed": 0},
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Query: {query}"}]})
    r.raise_for_status()
    d = r.json()
    return d.get("eval_count", 0), d.get("prompt_eval_count", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="queries to time")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "ablations", "mv2_energy.json"))
    a = ap.parse_args()

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qtexts = list(queries.values())[:a.n + 2]
    meter = PowerMeter()

    print(f"warming {MODEL} on physical GPU{PHYS_GPU} ...", flush=True)
    gen(qtexts[0]); gen(qtexts[1])                          # load + settle
    idle_w = meter.idle_watts(8.0)                          # model-loaded idle baseline
    print(f"idle (model loaded, no generation): {idle_w:.2f} W")

    rows = []
    for i, q in enumerate(qtexts[2:2 + a.n], 1):
        with meter:
            t0 = time.time()
            et, pt = gen(q)
            wall = time.time() - t0
        gross, mean_w, ns = meter.joules()
        net = gross - idle_w * wall
        rows.append({"wall": wall, "gross_j": gross, "net_j": net, "gen_tok": et,
                     "prompt_tok": pt, "mean_w": mean_w})
        if i % 10 == 0:
            print(f"  {i}/{a.n}  net {net:.1f} J  {et} tok  {wall:.2f}s  {mean_w:.0f}W", flush=True)

    net = np.array([r["net_j"] for r in rows]); wall = np.array([r["wall"] for r in rows])
    gtok = np.array([r["gen_tok"] for r in rows]); grossj = np.array([r["gross_j"] for r in rows])
    j_per_q = float(net.mean()); j_per_q_sd = float(net.std(ddof=1))
    j_per_tok = float(net.sum() / gtok.sum())
    peak_w = float(np.mean([r["mean_w"] for r in rows]))

    out = {"model": MODEL, "phys_gpu": PHYS_GPU, "n": len(rows), "idle_w": idle_w,
           "gen_j_per_query_net": j_per_q, "gen_j_per_query_sd": j_per_q_sd,
           "gen_j_per_query_gross": float(grossj.mean()),
           "gen_j_per_token_net": j_per_tok, "mean_gen_watts": peak_w,
           "mean_wall_s": float(wall.mean()), "mean_gen_tok": float(gtok.mean())}
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"\nqwen2.5:7b event decomposition, measured on physical GPU{PHYS_GPU}:")
    print(f"  {j_per_q:.1f} +/- {j_per_q_sd:.1f} J/query net  ({grossj.mean():.1f} J gross)")
    print(f"  {j_per_tok:.3f} J/token   {peak_w:.0f} W mean draw   {wall.mean():.2f} s/query")
    print(f"  -> full test set (2544 q): {j_per_q * 2544 / 1000:.1f} kJ = {j_per_q * 2544 / 3.6e6:.3f} kWh")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
