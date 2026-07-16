"""Full-tier event decomposition for MultiVENT 2.0 (the LLM step of the cost cascade).

For each test query, ask a local instruction model (qwen2.5:7b-instruct via Ollama on GPU1) to expand
the query into its event structure -- prequel / during / sequel visual descriptions -- the same
decomposition the original Q2E Full tier used (LLaMA-3.3-70B there; a 70B does not fit on a 15 GB A2,
so this is the feasible local stand-in and a lower bound on the Full-tier gain). Writes one JSONL row
per query with the three descriptions and token counts, incrementally, so it is resumable and the LLM
cost is paid once. CPU orchestration; the model runs on GPU1 (Whisper on GPU0 is untouched).

  CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_events.py            # all queries
  CUDA_VISIBLE_DEVICES="" python src/multivent2/mv2_events.py --limit 20 # smoke
"""
import os
import sys
import json
import argparse
import concurrent.futures as cf
import requests
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_queries  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:7b-instruct"

SYS = ("You expand a short news video search query into the event it describes, as three visual "
       "descriptions of what a video clip would show. Output STRICT JSON with keys "
       "\"prequel\", \"during\", \"sequel\". Each value is one concrete sentence describing the "
       "visible scene: prequel = the lead-up moments before the main event, during = the main event "
       "itself, sequel = the immediate aftermath. Describe people, places, and actions on screen. "
       "No commentary, JSON only.")


def generate(qid, query):
    r = requests.post(URL, timeout=180, json={
        "model": MODEL, "stream": False, "format": "json",
        "options": {"temperature": 0.3, "num_predict": 300, "seed": 0},
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": f"Query: {query}"}]})
    r.raise_for_status()
    d = r.json()
    obj = json.loads(d["message"]["content"])
    return {"qid": qid, "query": query,
            "prequel": str(obj.get("prequel", "")).strip(),
            "during": str(obj.get("during", "")).strip(),
            "sequel": str(obj.get("sequel", "")).strip(),
            "prompt_tok": d.get("prompt_eval_count", 0), "gen_tok": d.get("eval_count", 0),
            "dur_s": d.get("total_duration", 0) / 1e9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "events_qwen7b.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["qid"])
                except Exception:
                    pass
    todo = [(q, queries[q]) for q in queries if q not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"queries {len(queries)} | already done {len(done)} | to do {len(todo)} | workers {a.workers}")

    n_ok = n_err = ptok = gtok = 0
    with open(a.out, "a") as out, cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(generate, q, t): q for q, t in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            qid = futs[fut]
            try:
                row = fut.result()
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                n_ok += 1; ptok += row["prompt_tok"]; gtok += row["gen_tok"]
            except Exception as e:
                n_err += 1
                print(f"  ERR {qid}: {type(e).__name__} {e}", flush=True)
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok={n_ok} err={n_err}  "
                      f"tok(prompt/gen)={ptok}/{gtok}", flush=True)
    print(f"DONE ok={n_ok} err={n_err}  prompt_tok={ptok} gen_tok={gtok}  wrote {a.out}")


if __name__ == "__main__":
    main()
