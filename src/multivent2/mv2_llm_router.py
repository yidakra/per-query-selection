"""Zero-shot LLM router: the strongest query-only predictor the research plan names.

Category 1 of the plan's RQ2 (query and corpus information, no retrieval results): an instruction
LLM reads the query text alone and picks which evidence channel to search and which language to
ask in. This is the "you never tried the most natural approach" objection made concrete, and the
ModaRoute-style baseline reviewers will expect. Self-hosted qwen2.5-7b via Ollama, temperature 0,
one short prompt per decision, choices constrained to the option names.

Scored like every other selector: the chosen option's per-query nDCG against the default, with
group-level sign-flip tests. Retrieval runs already exist for every option, so the router costs
only the two prompts per query.

  python src/multivent2/mv2_llm_router.py --model qwen2.5:7b-instruct-q4_K_M
"""
import os
import re
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

from mv2_io import load_qrels, load_run, load_queries  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_qsd import event_groups  # noqa: E402
from mv2_variant_selection import rrf  # noqa: E402

LANGS = ["en", "zh", "ko", "ru", "ar"]
LANG_NAMES = {"en": "English", "zh": "Chinese", "ko": "Korean", "ru": "Russian", "ar": "Arabic"}
SUFFIX = {"en": "", "zh": "_qzh", "ko": "_qko", "ru": "_qru", "ar": "_qar"}
CHANNELS = ["speech", "screen_text", "both"]

CHANNEL_PROMPT = """You route search queries over a collection of news videos in many languages. A video can be found through what is spoken in it (speech transcript), what is written on screen (on-screen text), or both combined.

Query: "{q}"

Which evidence is most likely to contain this query's answer? Reply with exactly one word: speech, screen_text, or both."""

LANG_PROMPT = """You route search queries over a collection of news videos. The videos are in English, Chinese, Korean, Russian and Arabic. The query can be issued in its original English or translated into one of those languages before searching; asking in the language the relevant videos speak usually works best.

Query: "{q}"

In which language should this query be issued? Reply with exactly one word: English, Chinese, Korean, Russian, or Arabic."""


def ask(client, model, prompt):
    r = client.chat.completions.create(model=model, temperature=0, max_tokens=8,
                                       messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip().lower()


def parse_channel(text):
    for c in ("screen_text", "screen", "both", "speech"):
        if c in text:
            return "screen_text" if c == "screen" else c
    return None


def parse_lang(text):
    for code, name in LANG_NAMES.items():
        if name.lower() in text:
            return code
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--cache", default=os.path.join(ABL, "mv2_llm_router_choices.jsonl"))
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_llm_router.json"))
    a = ap.parse_args()
    from openai import OpenAI
    from mv2_row_inference import group_stats
    client = OpenAI(base_url=a.base_url, api_key="ollama")

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    qids = sorted(q for q in queries if q in qrels)

    # the cache is only reusable for the same model and the same prompts; a sidecar records both
    # and a mismatch moves the old cache aside instead of mixing choices from different setups
    import hashlib
    prompt_hash = hashlib.sha256((CHANNEL_PROMPT + LANG_PROMPT).encode()).hexdigest()[:12]
    meta_path = a.cache + ".meta.json"
    if os.path.exists(a.cache) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta.get("model") != a.model or meta.get("prompt_hash") != prompt_hash:
            stale = a.cache + f".stale-{meta.get('model','?').replace(':', '_')}-{meta.get('prompt_hash','?')}"
            os.replace(a.cache, stale)
            print(f"cache was for {meta.get('model')} / {meta.get('prompt_hash')}; moved to {stale}",
                  flush=True)
    json.dump({"model": a.model, "prompt_hash": prompt_hash}, open(meta_path, "w"))
    # strict provenance: only records that name this exact model and prompt version are reused;
    # anything without provenance is stale by definition and gets regenerated
    done = {}
    if os.path.exists(a.cache):
        for line in open(a.cache):
            r = json.loads(line)
            if r.get("model") == a.model and r.get("prompt_hash") == prompt_hash:
                done[r["qid"]] = r
    with open(a.cache, "a") as fh:
        for i, q in enumerate(qids):
            if q in done:
                continue
            rc = ask(client, a.model, CHANNEL_PROMPT.format(q=queries[q]))
            rl = ask(client, a.model, LANG_PROMPT.format(q=queries[q]))
            rec = {"qid": q, "model": a.model, "prompt_hash": prompt_hash,
                   "channel_raw": rc, "channel": parse_channel(rc),
                   "lang_raw": rl, "lang": parse_lang(rl)}
            done[q] = rec
            fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(qids)} routed", flush=True)

    # score the choices on the existing runs
    lang_runs = {L: load_run(os.path.join(DATA, f"asr_dense_bge-m3{SUFFIX[L]}.json")) for L in LANGS}
    ocr = load_run(os.path.join(DATA, "ocr_dense_bge-m3.json"))
    both = {q: rrf(lang_runs["en"][q], ocr[q]) for q in qids}
    chan_runs = {"speech": lang_runs["en"], "screen_text": ocr, "both": both}
    grp = np.asarray(event_groups(qids, qrels))
    out = {"model": a.model, "n_queries": len(qids)}

    for axis, runs, default, key in (("channel", chan_runs, "speech", "channel"),
                                     ("language", lang_runs, "en", "lang")):
        Y = {k: per_query_ndcg(qrels, {q: r[q] for q in qids}) for k, r in runs.items()}
        base = np.array([Y[default].get(q, 0.0) for q in qids])
        choices = [done[q][key] or default for q in qids]
        unparsed = sum(1 for q in qids if done[q][key] is None)
        routed = np.array([Y[c].get(q, 0.0) for q, c in zip(qids, choices)])
        oracle = np.array([max(Y[k].get(q, 0.0) for k in runs) for q in qids])
        opick = [max(runs, key=lambda k: Y[k].get(q, 0.0)) for q in qids]
        agree = float(np.mean([c == o for c, o in zip(choices, opick)]))
        p, obs, lo, hi = group_stats(routed - base, grp)
        dist = {k: int(sum(1 for c in choices if c == k)) for k in runs}
        out[axis] = {"default": float(base.mean()), "routed": float(routed.mean()),
                     "vs_default": float(100 * (routed.mean() - base.mean())),
                     "p": float(p), "ci95": [float(100 * lo), float(100 * hi)],
                     "oracle": float(oracle.mean()), "oracle_agreement": agree,
                     "choice_distribution": dist, "unparsed": unparsed}
        print(f"[{axis}] LLM router {routed.mean():.4f} vs default {base.mean():.4f} "
              f"({out[axis]['vs_default']:+.2f}, p={p:.4f}) | agrees with oracle {100*agree:.1f}% "
              f"| picks {dist} | unparsed {unparsed}", flush=True)

    # the joint policy: the router made both choices for every query, so the pair it implies is a
    # real policy and is scored as one, against the default pair and against each marginal alone
    ocr_runs = {L: load_run(os.path.join(DATA, f"ocr_dense_bge-m3{SUFFIX[L]}.json")) for L in LANGS}
    grid = {}
    for L in LANGS:
        grid[(L, "speech")] = lang_runs[L]
        grid[(L, "screen_text")] = ocr_runs[L]
        grid[(L, "both")] = {q: rrf(lang_runs[L][q], ocr_runs[L][q]) for q in qids}
    Yg = {k: per_query_ndcg(qrels, {q: r[q] for q in qids}) for k, r in grid.items()}
    base = np.array([Yg[("en", "speech")].get(q, 0.0) for q in qids])
    pair = [(done[q]["lang"] or "en", done[q]["channel"] or "speech") for q in qids]
    joint = np.array([Yg[pc].get(q, 0.0) for q, pc in zip(qids, pair)])
    lang_only = np.array([Yg[(done[q]["lang"] or "en", "speech")].get(q, 0.0) for q in qids])
    chan_only = np.array([Yg[("en", done[q]["channel"] or "speech")].get(q, 0.0) for q in qids])
    oracle_j = np.array([max(Yg[k].get(q, 0.0) for k in grid) for q in qids])
    p, obs, lo, hi = group_stats(joint - base, grp)
    p_l, _, _, _ = group_stats(joint - lang_only, grp)
    out["joint"] = {"default": float(base.mean()), "routed": float(joint.mean()),
                    "vs_default": float(100 * (joint.mean() - base.mean())),
                    "p": float(p), "ci95": [float(100 * lo), float(100 * hi)],
                    "vs_language_marginal": float(100 * (joint.mean() - lang_only.mean())),
                    "p_vs_language_marginal": float(p_l),
                    "vs_channel_marginal": float(100 * (joint.mean() - chan_only.mean())),
                    "oracle_joint": float(oracle_j.mean()),
                    "pair_distribution": {f"{L}|{c}": int(sum(1 for pc in pair if pc == (L, c)))
                                          for L in LANGS for c in CHANNELS if any(pc == (L, c) for pc in pair)}}
    print(f"[joint] LLM router pair {joint.mean():.4f} vs default {base.mean():.4f} "
          f"({out['joint']['vs_default']:+.2f}, p={p:.4f}) | vs language-only marginal "
          f"{out['joint']['vs_language_marginal']:+.2f} (p={p_l:.4f}) | vs channel-only marginal "
          f"{out['joint']['vs_channel_marginal']:+.2f} | joint oracle {oracle_j.mean():.4f}", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
