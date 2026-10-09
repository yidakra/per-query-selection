"""Query-only LLM policy for the formulation decision: the main table's last empty cell.

The LLM sees the query and its thirty-one candidate formulations (the original plus five samples
from each of six QueryGym methods), numbered, each cut to its first 60 words, and answers with one
number. No retrieval result is shown, so this is a pre-retrieval policy like the language router in
`mv2_llm_router.py`: same model (qwen2.5 7B instruct, Q4_K_M, via Ollama), temperature 0. An answer
that names no candidate scores as the original query, the default.

Scored on the stored per-candidate nDCG@10 (`mv2_variant_features_full.json`) with the one-sided
event-grouped sign-flip test used for every other cell.

  python src/multivent2/mv2_llm_formulation.py --base-url http://localhost:11435/v1
"""
import os
import re
import sys
import json
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_queries  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from mv2_row_inference import group_stats  # noqa: E402
from mv2_variant_selection import METHODS, load_pool  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

PROMPT = """You choose the search query for a collection of news videos in many languages. The chosen query is matched against the videos' speech transcripts by a multilingual dense retriever. Below are an information need and {n} candidate queries for it. Pick the candidate that will retrieve the relevant videos best.

Information need: "{q}"

Candidates:
{cands}

Reply with only the number of the best candidate."""


def labels_for():
    return ["original"] + [f"{m}#{s}" for m in METHODS for s in range(5)]


def shorten(text, words=60):
    w = " ".join(text.split()).split(" ")
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


def ask(client, model, prompt, retries=3):
    import time
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=model, temperature=0, max_tokens=6, timeout=120,
                                               messages=[{"role": "user", "content": prompt}])
            return ((r.choices[0].message.content if r.choices else "") or "").strip()
        except Exception as e:  # noqa: BLE001 - transient endpoint errors are retried
            if attempt == retries - 1:
                print(f"  ask() failed: {str(e)[:120]}", flush=True)
                return None
            time.sleep(2 ** attempt)


def parse(text, n):
    m = re.search(r"\d+", text or "")
    if not m:
        return None
    k = int(m.group())
    return k if 0 <= k < n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default=os.path.join(ABL, "mv2_llm_formulation_choices.jsonl"))
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_llm_formulation.json"))
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key="ollama")

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    pool = load_pool(os.path.join(DATA, "query_variants.jsonl"), 5)
    feats = json.load(open(os.path.join(ABL, "mv2_variant_features_full.json")))
    nd = {(r["qid"], r["label"]): r["ndcg10"] for r in feats["rows"]}
    labels = labels_for()
    qids = sorted({r["qid"] for r in feats["rows"]})
    qids = [q for q in qids if all((q, lab) in nd for lab in labels)]

    def prompt_for(q):
        texts = [queries[q]] + [pool[(q, m, s)] for m in METHODS for s in range(5)]
        cands = "\n".join(f"{i}. {shorten(t)}" for i, t in enumerate(texts))
        return PROMPT.format(n=len(texts), q=queries[q], cands=cands)

    phash = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]
    done = {}
    if os.path.exists(a.cache):
        for line in open(a.cache):
            r = json.loads(line)
            if r["model"] == a.model and r["prompt_hash"] == phash:
                done[r["qid"]] = r
    todo = [q for q in qids if q not in done]
    print(f"{len(qids)} queries, {len(todo)} to ask", flush=True)

    def work(q):
        raw = ask(client, a.model, prompt_for(q))
        return {"qid": q, "model": a.model, "prompt_hash": phash, "raw": raw,
                "pick": parse(raw, len(labels))}

    # write each answer as it arrives, so one slow request never holds back the rest
    with open(a.cache, "a") as fh, ThreadPoolExecutor(a.workers) as ex:
        futures = [ex.submit(work, q) for q in todo]
        for i, fut in enumerate(as_completed(futures)):
            rec = fut.result()
            if rec["raw"] is None:
                continue                      # endpoint failure; a rerun re-asks it
            done[rec["qid"]] = rec
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)

    missing = [q for q in qids if q not in done]
    if missing:                     # never score a partial run as if it were the full one
        raise SystemExit(f"{len(missing)} queries have no answer (endpoint failures); rerun to "
                         f"ask them before scoring")
    picks = [done[q]["pick"] for q in qids]
    chosen = [labels[k] if k is not None else "original" for k in picks]
    sel = np.array([nd[(q, c)] for q, c in zip(qids, chosen)])
    orig = np.array([nd[(q, "original")] for q in qids])
    grp = event_groups(qids, qrels)
    p, obs, lo, hi = group_stats(sel - orig, grp)
    dist = {}
    for c in chosen:
        key = c.split("#")[0]
        dist[key] = dist.get(key, 0) + 1
    out = {"model": a.model, "prompt_hash": phash, "n_queries": len(qids),
           "default": float(orig.mean()), "achieved": float(sel.mean()), "gain": obs,
           "p_signflip_group": p, "ci95": [lo, hi],
           "off_menu": int(sum(k is None for k in picks)), "picks_by_method": dist}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
