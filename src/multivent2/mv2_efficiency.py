"""Traditional IR/systems efficiency metrics for the MultiVENT 2.0 cascade: latency, throughput, and
the derived green-AI / cost-per-correct numbers.

We had measured joules but no measured *latency distribution* and no *throughput*, the two numbers
every IR efficiency paper reports (see reports/efficiency_metrics_review.md). This times every stage of
the cascade separately on real queries, warm, one query at a time (the serving regime), and reports
mean / median / p95 / p99 per tier -- tails matter here because escalation is exactly what inflates the
upper percentiles.

Stages timed (tier = sum of its stages):
  rank_A    sort the provided CLIP candidate scores -> top-10        [tier A]
  qenc      MiniLM encode of the query                               [tier B]
  capsim    caption-embedding dot product over the candidates        [tier B]
  fuse_B    RRF over the CLIP and caption rank lists + sort          [tier B]
  route     conf_features + ridge predict -- the router's own cost   [router overhead]
  llm_gen   Ollama event decomposition on GPU1                       [Full]
  evenc     MiniLM encode of the 3 event descriptions                [Full]
  evfuse    max-pool event sims + weighted RRF + sort                [Full]

HONEST SCOPE. Tier A here is only the ranking step: MultiVENT 2.0 ships a precomputed CLIP run, so the
CLIP text-tower encode and the search over 218K videos are not ours to time and tier A's latency below
is a *lower bound* on a deployed tier A. Everything in tiers B and Full is measured end to end. The
corpus-side caption embedding is a one-off offline cost and is excluded by design (it is amortized over
all queries, and every tier pays it equally).

Because escalation is a per-query decision, the routed system's latency is a *mixture*: this also emits
the latency percentiles as a function of the escalation fraction f, which is the form a router paper
should report.

CPU-only orchestration; the LLM runs on GPU1 (Whisper on GPU0 is never touched).

  CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_efficiency.py --n 300 --n-llm 40
"""
import os
import sys
import json
import time
import argparse
import concurrent.futures as cf
import numpy as np
import requests
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run, load_queries  # noqa: E402
from mv2_ab import load_captions  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402
from mv2_events import SYS, URL  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
RRF_K = 60
EVENTS = ["prequel", "during", "sequel"]
PCTS = (50, 90, 95, 99)


def pct(v):
    """-> {mean, sd, p50, p90, p95, p99, min, max} in milliseconds, from a seconds array."""
    a = np.asarray(v, float) * 1000.0
    d = {"mean_ms": float(a.mean()), "sd_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
         "min_ms": float(a.min()), "max_ms": float(a.max()), "n": int(len(a))}
    for p in PCTS:
        d[f"p{p}_ms"] = float(np.percentile(a, p))
    return d


def rrf(cands, scores, default=-1.0):
    order = sorted(cands, key=lambda v: -scores.get(v, default))
    rank = {v: i for i, v in enumerate(order)}
    return {v: 1.0 / (RRF_K + rank[v]) for v in cands}


def llm_gen(query, model, num_predict=300):
    r = requests.post(URL, timeout=300, json={
        "model": model, "stream": False, "format": "json",
        "options": {"temperature": 0.3, "num_predict": num_predict, "seed": 0},
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": f"Query: {query}"}]})
    r.raise_for_status()
    d = r.json()
    return json.loads(d["message"]["content"]), d.get("eval_count", 0)


def rel_at_10(qrels_q, ranked):
    return sum(1 for v in ranked[:10] if qrels_q.get(v, 0) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="queries timed for the CPU tiers")
    ap.add_argument("--n-llm", type=int, default=40, help="queries timed for the LLM tier")
    ap.add_argument("--model", default="qwen2.5:14b-instruct", help="decomposer for the Full tier")
    ap.add_argument("--emb", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4],
                    help="concurrency levels for the LLM throughput sweep")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads (recorded in the JSON)")
    ap.add_argument("--w-event", type=float, default=0.5, help="event RRF weight (matches mv2_full)")
    ap.add_argument("--co2-kg-per-kwh", type=float, default=0.475,
                    help="grid carbon intensity; default = IEA world average (documented, not measured)")
    ap.add_argument("--energy", default=os.path.join(ABL, "mv2_energy_qwen14b.json"))
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_efficiency.json"))
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    from sentence_transformers import SentenceTransformer

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    runA = load_run(os.path.join(DATA, "10pyscene_clip.json"))
    caps = load_captions(os.path.join(DATA, "qwen_captions_test.jsonl"))
    cap_ids = list(caps)
    z = np.load(os.path.join(DATA, f"capemb_{a.emb.split('/')[-1]}.npz"), allow_pickle=True)
    cap_emb = z["emb"]; assert list(z["ids"]) == cap_ids, "caption-embedding cache is stale"
    row = {c: i for i, c in enumerate(cap_ids)}
    model = SentenceTransformer(a.emb, device="cpu")
    qids = [q for q in runA if q in queries][:a.n]
    print(f"timing {len(qids)} queries | cand/query {np.mean([len(runA[q]) for q in qids]):.0f} "
          f"| torch threads {a.threads} | emb {a.emb}", flush=True)

    # a router to time: fit once on the shipped A->B cell, then time only its per-query decision
    ab = json.load(open(os.path.join(ABL, "mv2_ab_dense.json")))
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    fq = [q for q in ab["features"] if q in ab["per_query"]]
    rt = Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))]).fit(
        np.array([ab["features"][q] for q in fq]),
        np.array([ab["per_query"][q]["ndB"] - ab["per_query"][q]["ndA"] for q in fq]))

    model.encode(["warm up the encoder"], normalize_embeddings=True)          # warm
    for q in qids[:3]:                                                        # warm the caches
        c = list(runA[q]); cap_emb[np.array([row[v] for v in c])] @ np.zeros(cap_emb.shape[1])

    T = {k: [] for k in ("rank_A", "qenc", "capsim", "fuse_B", "route", "evenc", "evfuse")}
    relA, relB = [], []
    for qid in qids:
        cand_scores = runA[qid]
        cands = list(cand_scores)
        qr = qrels.get(qid, {})

        t0 = time.perf_counter()
        a_order = sorted(cands, key=lambda v: -cand_scores[v])
        top10_A = a_order[:10]
        T["rank_A"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        qvec = model.encode([queries[qid]], normalize_embeddings=True)[0]
        T["qenc"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        capm = cap_emb[np.array([row[v] for v in cands])]
        capsim = capm @ qvec
        T["capsim"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        clip_rrf = rrf(cands, cand_scores)
        cap_rrf = rrf(cands, dict(zip(cands, capsim.tolist())))
        b_scores = {v: clip_rrf[v] + cap_rrf[v] for v in cands}
        top10_B = sorted(cands, key=lambda v: -b_scores[v])[:10]
        T["fuse_B"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        f = conf_features(np.array([b_scores[v] for v in cands]))
        rt.predict(np.array([[f[k] for k in FEATURE_ORDER]]))
        T["route"].append(time.perf_counter() - t0)

        # Full-tier post-LLM stages, timed on every query with placeholder text of realistic length
        ev_text = ["a crowd gathers on a city street before the main event begins as police look on",
                   "the main event unfolds with people moving through the square amid smoke and noise",
                   "the aftermath shows damaged vehicles and responders attending to bystanders"]
        t0 = time.perf_counter()
        evec = model.encode(ev_text, normalize_embeddings=True)
        T["evenc"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        ev_sim = np.max(capm @ evec.T, axis=1)
        ev_rrf = rrf(cands, dict(zip(cands, ev_sim.tolist())))
        full = {v: b_scores[v] + a.w_event * ev_rrf[v] for v in cands}
        sorted(cands, key=lambda v: -full[v])[:10]
        T["evfuse"].append(time.perf_counter() - t0)

        relA.append(rel_at_10(qr, top10_A)); relB.append(rel_at_10(qr, top10_B))

    # ---- LLM stage: latency on real queries, then a throughput sweep over concurrency ----
    qtext = [queries[q] for q in qids[:a.n_llm + 2]]
    print(f"\nwarming {a.model} on GPU1 ...", flush=True)
    llm_gen(qtext[0], a.model); llm_gen(qtext[1], a.model)
    lat, toks = [], []
    for q in qtext[2:2 + a.n_llm]:
        t0 = time.perf_counter()
        _, nt = llm_gen(q, a.model)
        lat.append(time.perf_counter() - t0); toks.append(nt)
    T["llm_gen"] = lat
    print(f"  llm_gen: {1000*np.mean(lat):.0f} ms mean, {np.mean(toks):.0f} gen tok", flush=True)

    thr = {}
    for w in a.workers:
        batch = qtext[2:2 + max(8, 2 * w)]
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=w) as ex:
            list(ex.map(lambda q: llm_gen(q, a.model), batch))
        el = time.perf_counter() - t0
        thr[w] = {"qps": len(batch) / el, "n": len(batch), "wall_s": el}
        print(f"  workers={w}: {thr[w]['qps']:.3f} q/s over {len(batch)} queries", flush=True)

    stages = {k: pct(v) for k, v in T.items()}
    tierA = np.array(T["rank_A"])
    tierB = tierA + np.array(T["qenc"]) + np.array(T["capsim"]) + np.array(T["fuse_B"])
    post = np.array(T["evenc"]) + np.array(T["evfuse"])
    # Full = tier B + the LLM call + the post-LLM stages. The LLM sample is smaller than the CPU
    # sample, so pair them by bootstrap: draw an LLM latency per query from the measured sample.
    rng = np.random.default_rng(0)
    tierF = tierB + post + rng.choice(lat, size=len(tierB), replace=True)
    tiers = {"A_rank_only": pct(tierA), "B": pct(tierB), "Full": pct(tierF)}

    # routed mixture: escalate fraction f -> latency percentiles of the served population
    mix = {}
    for f in (0.0, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0):
        k = int(round(f * len(tierB)))
        v = np.concatenate([tierF[:k], tierB[k:]])
        mix[str(f)] = pct(v) | {"qps_serial": float(1.0 / v.mean())}

    e = json.load(open(a.energy)) if os.path.exists(a.energy) else {}
    j_llm = e.get("gen_j_per_query_net", float("nan"))
    kwh = j_llm / 3.6e6
    derived = {
        "energy_source": os.path.basename(a.energy), "j_llm_per_query": j_llm,
        "kwh_per_query": kwh, "gco2e_per_query": kwh * a.co2_kg_per_kwh * 1000.0,
        "co2_kg_per_kwh_assumed": a.co2_kg_per_kwh,
        "kwh_full_testset_2546q": kwh * 2546, "gco2e_full_testset_2546q": kwh * 2546 * a.co2_kg_per_kwh * 1000,
        "mean_rel_at10_A": float(np.mean(relA)), "mean_rel_at10_B": float(np.mean(relB)),
        # cost per useful item: joules spent per relevant video actually surfaced in the top 10
        "j_per_relevant_at10_B": float(22.35 / max(1e-9, np.mean(relB))),
        "j_per_relevant_at10_Full": float((22.35 + j_llm) / max(1e-9, np.mean(relB))),
    }

    out = {"config": {"n": len(qids), "n_llm": a.n_llm, "model": a.model, "emb": a.emb,
                      "torch_threads": a.threads, "w_event": a.w_event,
                      "mean_candidates_per_query": float(np.mean([len(runA[q]) for q in qids]))},
           "stages_ms": stages, "tiers_ms": tiers, "throughput_llm": thr,
           "serial_qps": {"A_rank_only": float(1 / tierA.mean()), "B": float(1 / tierB.mean()),
                          "Full": float(1 / tierF.mean())},
           "routed_mixture_by_f": mix, "derived": derived,
           "caveats": ["tier A excludes the CLIP text encode and the 218K-video search: MultiVENT 2.0 "
                       "ships a precomputed run, so tier A latency is a lower bound",
                       "corpus-side caption embedding is an offline one-off, excluded from all tiers",
                       "gCO2e uses an assumed grid intensity, not a measured one"]}
    json.dump(out, open(a.out, "w"), indent=2)

    print("\nper-stage latency (ms):")
    print(f"  {'stage':<10} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    for k, v in stages.items():
        print(f"  {k:<10} {v['mean_ms']:8.2f} {v['p50_ms']:8.2f} {v['p95_ms']:8.2f} {v['p99_ms']:8.2f}")
    print("\nper-tier latency (ms) and serial throughput:")
    for k, v in tiers.items():
        print(f"  {k:<12} mean {v['mean_ms']:9.1f}  p50 {v['p50_ms']:9.1f}  p95 {v['p95_ms']:9.1f}  "
              f"p99 {v['p99_ms']:9.1f}   {1000/v['mean_ms']:7.2f} q/s")
    print("\nrouted mixture (escalate fraction f):")
    for f, v in mix.items():
        print(f"  f={f:<5} mean {v['mean_ms']:8.1f} ms   p95 {v['p95_ms']:8.1f} ms   "
              f"p99 {v['p99_ms']:8.1f} ms   {v['qps_serial']:6.2f} q/s")
    print(f"\nderived: {derived['kwh_per_query']*1e6:.2f} mWh/query, "
          f"{derived['gco2e_per_query']:.3f} gCO2e/query (grid {a.co2_kg_per_kwh} kg/kWh), "
          f"{derived['j_per_relevant_at10_B']:.1f} J per relevant@10 at B -> "
          f"{derived['j_per_relevant_at10_Full']:.1f} J at Full")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
