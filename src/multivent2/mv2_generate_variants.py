"""Generate the query-variant pool for the pre-registered variant experiments.

Runs the source study's own toolkit (QueryGym) over the test queries: six reformulation methods,
--samples independent variants per method per query, temperature 0.6 throughout (the source study's
setting, overriding each method's own default), MuGI's adaptive_times pinned to 5 (the value its
concatenation code actually uses). Every variant is one reformulate() call; methods differ in how
many LLM calls that costs (GenQR 5, GenQR-Ensemble 10, MuGI 5, QA-Expand 3, Query2Doc 1,
Query2Exp 1, so 25 calls per query per full sample round).

Output is one JSONL line per (query, method, sample), resumable on rerun: existing keys are skipped,
so a crashed or rate-limited run continues where it stopped. The output file is data: it gets
committed once, then frozen.

  OPENAI_API_KEY=... python src/multivent2/mv2_generate_variants.py \
      --base-url https://api.anthropic.com/v1/ --model claude-haiku-4-5 --samples 5
"""
import os
import sys
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")

METHODS = ["genqr", "genqr_ensemble", "mugi", "qa_expand", "query2doc", "query2e"]
TEMPERATURE = 0.6  # the source study's setting for all methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    ap.add_argument("--model", required=True)
    ap.add_argument("--samples", type=int, default=5, help="variants per method per query")
    ap.add_argument("--queries", default=os.path.join(DATA, "multivent_2_test_queries.csv"))
    ap.add_argument("--out", default=os.path.join(DATA, "query_variants.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many queries")
    a = ap.parse_args()

    import querygym as qg
    from querygym import QueryItem
    from mv2_io import load_queries

    queries = load_queries(a.queries)
    qids = sorted(queries)
    if a.limit:
        qids = qids[:a.limit]

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["qid"], r["method"], r["sample"]))
        print(f"resuming: {len(done)} variants already present")

    llm_config = {"model": a.model}
    if a.base_url:
        llm_config["base_url"] = a.base_url

    reformulators = {}
    for meth in METHODS:
        params = {"temperature": TEMPERATURE}
        if meth == "mugi":
            params["adaptive_times"] = 5
        reformulators[meth] = qg.create_reformulator(meth, model=a.model, params=params,
                                                     llm_config=llm_config)

    jobs = [(q, meth, s) for q in qids for meth in METHODS for s in range(a.samples)
            if (q, meth, s) not in done]
    print(f"{len(jobs)} variants to generate "
          f"({len(qids)} queries x {len(METHODS)} methods x {a.samples} samples)")

    out_f = open(a.out, "a")
    n_ok = n_err = 0
    t0 = time.time()

    def gen(job):
        q, meth, s = job
        try:
            res = reformulators[meth].reformulate(QueryItem(qid=q, text=queries[q]))
            return {"qid": q, "method": meth, "sample": s, "variant": res.reformulated,
                    "model": a.model, "temperature": TEMPERATURE}
        except Exception as e:  # noqa: BLE001 - recorded and retried on the next resume pass
            return {"qid": q, "method": meth, "sample": s, "error": str(e)[:300]}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, rec in enumerate(ex.map(gen, jobs)):
            if "error" in rec:
                n_err += 1
                if n_err <= 5:
                    print("ERR", rec["qid"], rec["method"], rec["error"], flush=True)
            else:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(jobs) - i - 1) / rate / 3600
                print(f"{i+1}/{len(jobs)} ok={n_ok} err={n_err} "
                      f"{rate:.1f} var/s eta {eta:.1f} h", flush=True)
                out_f.flush()

    out_f.close()
    print(f"done: {n_ok} written, {n_err} errors (rerun to retry errors)")


if __name__ == "__main__":
    main()
