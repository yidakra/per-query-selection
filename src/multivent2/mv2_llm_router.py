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

# The three-way prompt lets the model refuse to choose ("both" is fixed fusion). The binary prompt
# removes that exit, so the reply is a real channel decision or an off-menu answer.
CHANNEL_PROMPT_BINARY = """You route search queries over a collection of news videos in many languages. A video can be found through what is spoken in it (speech transcript) or through what is written on screen (on-screen text). Only one of the two will be searched.

Query: "{q}"

Which evidence is more likely to contain this query's answer? Reply with exactly one word: speech or screen_text."""

LANG_PROMPT = """You route search queries over a collection of news videos. The videos are in English, Chinese, Korean, Russian and Arabic. The query can be issued in its original English or translated into one of those languages before searching; asking in the language the relevant videos speak usually works best.

Query: "{q}"

In which language should this query be issued? Reply with exactly one word: English, Chinese, Korean, Russian, or Arabic."""


def ask(client, model, prompt, retries=3):
    """One constrained prompt. Empty or missing content comes back as "" (the parsers then return
    None and the query scores as the default); transient client errors are retried with backoff,
    and a persistent failure also yields "" rather than aborting the batch."""
    import time
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(model=model, temperature=0, max_tokens=8,
                                               messages=[{"role": "user", "content": prompt}])
            content = r.choices[0].message.content if r.choices else None
            return (content or "").strip().lower()
        except Exception as e:  # noqa: BLE001 - transient endpoint errors are retried, then logged
            if attempt == retries - 1:
                print(f"  ask() failed after {retries} attempts: {str(e)[:120]}", flush=True)
                return None                 # endpoint failure, distinct from an empty answer
            time.sleep(2 ** attempt)


def parse_channel(text, options=tuple(CHANNELS)):
    """Map a reply to one of the options on the menu; anything else (including "both" when the menu
    is binary) is None, so an off-menu answer never counts as a decision."""
    if not text:
        return None
    for c in ("screen_text", "screen", "both", "speech"):
        if c in text:
            c = "screen_text" if c == "screen" else c
            return c if c in options else None
    return None


def parse_lang(text):
    if not text:
        return None
    for code, name in LANG_NAMES.items():
        if name.lower() in text:
            return code
    return None


def mean_or_none(x):
    """Mean of a non-empty array, else None, so an empty decided set never writes NaN to JSON."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if x.size else None


def two_sided_p(diff, grp, n_draws=2000, seed=0):
    """Two-sided group-level sign-flip test on the size-weighted mean difference: the fraction of
    sign-flipped draws whose absolute statistic reaches the observed absolute statistic."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float); grp = np.asarray(grp)
    groups = np.unique(grp)
    dg = np.array([diff[grp == g].mean() for g in groups])
    wg = np.array([(grp == g).sum() for g in groups], dtype=float); wg /= wg.sum()
    obs = abs(float((wg * dg).sum()))
    cnt = sum(1 for _ in range(n_draws)
              if abs(float((wg * dg * rng.choice([-1.0, 1.0], size=len(dg))).sum())) >= obs)
    return float((1 + cnt) / (n_draws + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--cache", default=os.path.join(ABL, "mv2_llm_router_choices.jsonl"))
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_llm_router.json"))
    ap.add_argument("--channel-binary", action="store_true",
                    help="channel axis only, with a two-option prompt (speech or screen_text, no "
                         "'both'); separate cache and output files")
    a = ap.parse_args()
    if a.channel_binary:            # its own artifacts, never mixed with the three-way run
        if a.cache == os.path.join(ABL, "mv2_llm_router_choices.jsonl"):
            a.cache = os.path.join(ABL, "mv2_llm_router_binary_choices.jsonl")
        if a.out == os.path.join(ABL, "mv2_llm_router.json"):
            a.out = os.path.join(ABL, "mv2_llm_router_binary.json")
    channel_prompt = CHANNEL_PROMPT_BINARY if a.channel_binary else CHANNEL_PROMPT
    channel_options = ("speech", "screen_text") if a.channel_binary else tuple(CHANNELS)
    need_lang = not a.channel_binary
    from openai import OpenAI
    from mv2_row_inference import group_stats
    client = OpenAI(base_url=a.base_url, api_key="ollama")

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    qids = sorted(q for q in queries if q in qrels)

    # the cache is only reusable for the same model and the same prompts; a sidecar records both
    # and a mismatch moves the old cache aside instead of mixing choices from different setups
    import hashlib
    prompt_hash = hashlib.sha256((channel_prompt + (LANG_PROMPT if need_lang else "")).encode()).hexdigest()[:12]
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
                done[r["qid"]] = r          # last record per qid wins; axes may be partially decided
    off_menu = {}                   # raw channel replies that named no option on the menu
    with open(a.cache, "a") as fh:
        for i, q in enumerate(qids):
            prev = done.get(q, {})
            if prev.get("channel") is not None and (prev.get("lang") is not None or not need_lang):
                continue
            rec = {"qid": q, "model": a.model, "prompt_hash": prompt_hash,
                   "channel_raw": prev.get("channel_raw"), "channel": prev.get("channel"),
                   "lang_raw": prev.get("lang_raw"), "lang": prev.get("lang")}
            if rec["channel"] is None:                       # re-ask only the missing axis
                rc = ask(client, a.model, channel_prompt.format(q=queries[q]))
                rec["channel_raw"], rec["channel"] = rc, parse_channel(rc, channel_options)
                if rc and rec["channel"] is None:
                    off_menu[rc] = off_menu.get(rc, 0) + 1
            if need_lang and rec["lang"] is None:
                rl = ask(client, a.model, LANG_PROMPT.format(q=queries[q]))
                rec["lang_raw"], rec["lang"] = rl, parse_lang(rl)
            if rec["channel"] is None and rec["lang"] is None:
                print(f"  {q}: no axis decided ({rec['channel_raw']!r}/{rec['lang_raw']!r}); "
                      f"not cached, a rerun will re-ask it", flush=True)
                continue
            if need_lang and (rec["channel"] is None or rec["lang"] is None):
                print(f"  {q}: one axis undecided; cached partially, a rerun re-asks that axis",
                      flush=True)
            done[q] = rec
            fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(qids)} routed", flush=True)

    # score the choices on the existing runs
    lang_runs = {L: load_run(os.path.join(DATA, f"asr_dense_bge-m3{SUFFIX[L]}.json")) for L in LANGS}
    ocr = load_run(os.path.join(DATA, "ocr_dense_bge-m3.json"))
    both = {q: rrf(lang_runs["en"][q], ocr[q]) for q in qids}
    chan_runs = {"speech": lang_runs["en"], "screen_text": ocr, "both": both}
    chan_runs = {k: chan_runs[k] for k in channel_options}
    grp = np.asarray(event_groups(qids, qrels))
    out = {"model": a.model, "n_queries": len(qids), "channel_prompt": "binary" if a.channel_binary else "three-way",
           "channel_options": list(channel_options)}

    axes = [("channel", chan_runs, "speech", "channel")]
    if need_lang:
        axes.append(("language", lang_runs, "en", "lang"))
    for axis, runs, default, key in axes:
        Y = {k: per_query_ndcg(qrels, {q: r[q] for q in qids}) for k, r in runs.items()}
        # only queries the router actually decided count as router decisions; the default is
        # compared on that same subset, and the excluded count is reported next to it
        decided = [q for q in qids if (done.get(q) or {}).get(key) is not None]
        unparsed = len(qids) - len(decided)
        dgrp = np.asarray([grp[qids.index(q)] for q in decided])
        base = np.array([Y[default].get(q, 0.0) for q in decided])
        choices = [done[q][key] for q in decided]
        routed = np.array([Y[c].get(q, 0.0) for q, c in zip(decided, choices)])
        oracle = np.array([max(Y[k].get(q, 0.0) for k in runs) for q in decided])
        opick = [max(runs, key=lambda k: Y[k].get(q, 0.0)) for q in decided]
        dist = {k: int(sum(1 for c in choices if c == k)) for k in runs}
        if decided:
            agree = float(np.mean([c == o for c, o in zip(choices, opick)]))
            p, obs, lo, hi = group_stats(routed - base, dgrp)
            p2 = two_sided_p(routed - base, dgrp)
            stats = {"vs_default": float(100 * (routed.mean() - base.mean())),
                     "p_greater": float(p), "p_two_sided": float(p2),
                     "ci95": [float(100 * lo), float(100 * hi)], "oracle_agreement": agree}
        else:                       # nothing was computed, so nothing may look like a real result
            stats = {"vs_default": None, "p_greater": None, "p_two_sided": None,
                     "ci95": None, "oracle_agreement": None}
        out[axis] = {"n_decided": len(decided), "unparsed": unparsed,
                     "default": mean_or_none(base), "routed": mean_or_none(routed),
                     "vs_default": stats["vs_default"],
                     "p_greater": stats["p_greater"], "p_two_sided": stats["p_two_sided"],
                     "ci95": stats["ci95"],
                     "oracle": mean_or_none(oracle), "oracle_agreement": stats["oracle_agreement"],
                     "choice_distribution": dist}
        if axis == "channel":
            out[axis]["off_menu_replies"] = off_menu       # this run's replies naming no option
        if not decided:
            print(f"[{axis}] no decided queries; statistics written as null", flush=True)
            continue
        print(f"[{axis}] LLM router {routed.mean():.4f} vs default {base.mean():.4f} "
              f"({out[axis]['vs_default']:+.2f}, p_two_sided={p2:.4f}) on {len(decided)} decided queries "
              f"| agrees with oracle {100*agree:.1f}% | picks {dist} | unparsed {unparsed}", flush=True)

    if not need_lang:               # channel-only run: no language choice, so no joint policy
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
        return

    # the joint policy: the router made both choices for every query, so the pair it implies is a
    # real policy and is scored as one, against the default pair and against each marginal alone
    ocr_runs = {L: load_run(os.path.join(DATA, f"ocr_dense_bge-m3{SUFFIX[L]}.json")) for L in LANGS}
    grid = {}
    for L in LANGS:
        grid[(L, "speech")] = lang_runs[L]
        grid[(L, "screen_text")] = ocr_runs[L]
        grid[(L, "both")] = {q: rrf(lang_runs[L][q], ocr_runs[L][q]) for q in qids}
    Yg = {k: per_query_ndcg(qrels, {q: r[q] for q in qids}) for k, r in grid.items()}
    dq = [q for q in qids if q in done and done[q]["lang"] is not None and done[q]["channel"] is not None]
    if not dq:
        out["joint"] = {"n_decided": 0, "unparsed": len(qids), "default": None, "routed": None,
                        "vs_default": None, "p_greater": None, "p_two_sided": None, "ci95": None,
                        "vs_language_marginal": None,
                        "p_greater_vs_language_marginal": None,
                        "p_two_sided_vs_language_marginal": None,
                        "vs_channel_marginal": None,
                        "oracle_joint": None, "pair_distribution": {}}
        print("[joint] no query decided on both axes; statistics written as null", flush=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
        return
    jgrp = np.asarray([grp[qids.index(q)] for q in dq])
    base = np.array([Yg[("en", "speech")].get(q, 0.0) for q in dq])
    pair = [(done[q]["lang"], done[q]["channel"]) for q in dq]
    joint = np.array([Yg[pc].get(q, 0.0) for q, pc in zip(dq, pair)])
    lang_only = np.array([Yg[(pc[0], "speech")].get(q, 0.0) for q, pc in zip(dq, pair)])
    chan_only = np.array([Yg[("en", pc[1])].get(q, 0.0) for q, pc in zip(dq, pair)])
    oracle_j = np.array([max(Yg[k].get(q, 0.0) for k in grid) for q in dq])
    p, obs, lo, hi = group_stats(joint - base, jgrp)
    p2 = two_sided_p(joint - base, jgrp)
    p_l, _, _, _ = group_stats(joint - lang_only, jgrp)
    p_l2 = two_sided_p(joint - lang_only, jgrp)
    out["joint"] = {"n_decided": len(dq), "unparsed": len(qids) - len(dq),
                    "default": float(base.mean()), "routed": float(joint.mean()),
                    "vs_default": float(100 * (joint.mean() - base.mean())),
                    "p_greater": float(p), "p_two_sided": float(p2),
                    "ci95": [float(100 * lo), float(100 * hi)],
                    "vs_language_marginal": float(100 * (joint.mean() - lang_only.mean())),
                    "p_greater_vs_language_marginal": float(p_l),
                    "p_two_sided_vs_language_marginal": float(p_l2),
                    "vs_channel_marginal": float(100 * (joint.mean() - chan_only.mean())),
                    "oracle_joint": float(oracle_j.mean()),
                    "pair_distribution": {f"{L}|{c}": int(sum(1 for pc in pair if pc == (L, c)))
                                          for L in LANGS for c in CHANNELS if any(pc == (L, c) for pc in pair)}}
    print(f"[joint] LLM router pair {joint.mean():.4f} vs default {base.mean():.4f} "
          f"({out['joint']['vs_default']:+.2f}, p_two_sided={p2:.4f}) | vs language-only marginal "
          f"{out['joint']['vs_language_marginal']:+.2f} (p_two_sided={p_l2:.4f}) | vs channel-only marginal "
          f"{out['joint']['vs_channel_marginal']:+.2f} | joint oracle {oracle_j.mean():.4f}", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
