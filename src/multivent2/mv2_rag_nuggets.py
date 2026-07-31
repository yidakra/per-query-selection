"""Multimodal RAG evaluation: does the routing policy that wins on nDCG@10 also win on answer quality?

MultiVENT 2.0 queries are event descriptors ("Super Bowl 2023 Philadelphia Eagles"), so the generation
task is event report writing, which is what nugget evaluation was built for. Protocol follows
github.com/Narabzad/QPP-4-RAG's nuggetizer: create atomic gold nuggets from the judged-relevant videos,
score each vital/okay, generate a report per retrieval policy, label each nugget
support/partial_support/not_support against that report, then score. The four prompts are reproduced
from that repo so our numbers are computed the same way; the judge is local qwen2.5:14b rather than
gpt-4o, so absolute levels are not comparable to their table, only policies to each other under a fixed
judge.

The multimodal question this exists to answer: a channel's ranking ability and its ability to ground an
answer are different things. The visual channel is our strongest cheap retriever and emits embeddings,
which no text generator can read; OCR is our weakest and emits text that a generator can use directly.
So evidence mode matters and both are run:

  --evidence own  each policy generates from its own channel's text (captions stand in for visual,
                  since the benchmark ships them). Tests whether ranking ability transfers to grounding.
  --evidence all  every policy generates from all available text for whichever documents it retrieved.
                  Isolates retrieval quality, the closest analogue of the text-RAG utility gap.

Phases are checkpointed to JSONL and resumable; a 400-query run over a local model takes hours.

  python src/multivent2/mv2_rag_nuggets.py --phase nuggets  --n 400
  python src/multivent2/mv2_rag_nuggets.py --phase generate --evidence all
  python src/multivent2/mv2_rag_nuggets.py --phase assign   --evidence all
  python src/multivent2/mv2_rag_nuggets.py --phase metrics  --evidence all
"""
import os, sys, json, ast, time, argparse, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run, load_queries  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
RAG = os.path.join(ABL, "rag")
os.makedirs(RAG, exist_ok=True)

# text channels available as generation evidence. "visual" has no text of its own, so the shipped
# qwen captions stand in for it -- otherwise the comparison is rigged against the visual channel.
TEXT_SOURCE = {"captions": "qwen_captions_test.jsonl",
               "asr": "asr_text.jsonl",
               "ocr": "ocr_text.jsonl"}

# policy -> (ranked-list file, channels whose text grounds the report under --evidence own).
# `None` means the channels vary per query and come from a picks file, which is how the router works.
# The routed and best-fixed runs are written by mv2_routed_run.py from the selector's out-of-fold
# decisions, so nothing here sees a label the nDCG numbers do not already allow.
POLICIES = {
    "visual":    ("10pyscene_clip.json",              ["captions"]),
    "asr_dense": ("asr_dense_bge-m3.json",            ["asr"]),
    "ocr":       ("10pyscene_paddleOCR_clip.json",    ["ocr"]),
    "bestfixed": ("bestfixed_dense_m3.json",          ["captions", "asr"]),
    "routed":    ("routed_dense_m3.json",             None),
}
PICKS = {"routed": "routed_dense_m3_picks.json"}
# selector channel names -> the text that channel can actually hand a generator
CHANNEL_TEXT = {"visual": "captions", "asr": "asr", "ocr": "ocr"}
ALL_CHANNELS = ["captions", "asr", "ocr"]
MAX_DOC_CHARS = 1500          # per document, keeps the context bounded on a 14b model
TOPK = 5                      # documents fed to the generator, matching QPP-4-RAG's RAG setting


def client(model, base):
    from openai import OpenAI
    return OpenAI(base_url=base, api_key="ollama"), model


def chat(cl, model, messages, retries=3, temperature=0.0):
    for a in range(retries):
        try:
            r = cl.chat.completions.create(model=model, messages=messages,
                                           temperature=temperature)
            return r.choices[0].message.content or ""
        except Exception as e:
            if a == retries - 1:
                print(f"    LLM error after {retries} tries: {e}", flush=True)
                return ""
            time.sleep(2 * (a + 1))
    return ""


def parse_list(text, n=None):
    """LLM is asked for a Pythonic list. Recover one even when it wraps it in prose or code fences."""
    if not text:
        return []
    t = text.strip()
    if "```" in t:
        parts = [p for p in t.split("```") if "[" in p]
        t = parts[0] if parts else t
        t = t.replace("python", "", 1).strip()
    i, j = t.find("["), t.rfind("]")
    if i >= 0 and j > i:
        frag = t[i:j + 1]
        try:
            v = ast.literal_eval(frag)
            if isinstance(v, list):
                out = [str(x).strip() for x in v]
                return out[:n] if n else out
        except Exception:
            # salvage quoted items
            import re
            items = re.findall(r'["\']([^"\']{2,})["\']', frag)
            if items:
                return items[:n] if n else items
    return []


def load_texts(names):
    out = {}
    for nm in names:
        p = os.path.join(DATA, TEXT_SOURCE[nm])
        if not os.path.exists(p):
            print(f"  WARNING missing {TEXT_SOURCE[nm]}, skipping channel {nm}", flush=True)
            continue
        d = {}
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("text", "").strip():
                    d[r["doc_id"]] = r["text"]
        out[nm] = d
        print(f"  {nm}: {len(d)} docs", flush=True)
    return out


def evidence_for(doc_ids, texts, channels):
    """Concatenate the chosen channels' text for these documents, labelled by source."""
    segs = []
    for did in doc_ids:
        bits = []
        for ch in channels:
            t = texts.get(ch, {}).get(did)
            if t:
                bits.append(f"({ch}) {t[:MAX_DOC_CHARS]}")
        if bits:
            segs.append(" ".join(bits))
    return segs


def sample_queries(qrels, judgments_path, n, seed=0):
    """Stratify by the language of each query's relevant videos so the cross-lingual split survives."""
    lang = {}
    with open(judgments_path) as f:
        for line in f:
            r = json.loads(line)
            if r["relevance"] > 0:
                lang.setdefault(r["query_id"], r["video_language"])
    by = collections.defaultdict(list)
    for q in sorted(qrels):
        by[lang.get(q, "unknown")].append(q)
    rng = np.random.RandomState(seed)
    total = sum(len(v) for v in by.values())
    picked = []
    for L, qs in sorted(by.items()):
        k = max(1, int(round(n * len(qs) / total)))
        idx = rng.permutation(len(qs))[:min(k, len(qs))]
        picked += [qs[i] for i in idx]
    rng.shuffle(picked)
    return picked[:n], lang


def jsonl_done(path, key="qid"):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r[key]] = r
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------- phase: gold nuggets
def phase_nuggets(a, cl, model):
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qids, _ = sample_queries(qrels, os.path.join(DATA, "multivent_2_test_judgments.jsonl"), a.n)
    texts = load_texts(ALL_CHANNELS)
    out_path = os.path.join(RAG, f"gold_nuggets_n{a.n}.jsonl")
    done = jsonl_done(out_path)
    print(f"nuggets: {len(qids)} queries, {len(done)} already done", flush=True)

    with open(out_path, "a", buffering=1) as fh:
        for i, q in enumerate(qids):
            if q in done:
                continue
            rel = [d for d, r in qrels[q].items() if r > 0]
            segs = evidence_for(rel[:8], texts, ALL_CHANNELS)
            if not segs:
                continue
            ctx = "\n".join(f"[{j+1}] {s}" for j, s in enumerate(segs))
            qt = queries.get(q, q)
            # creator prompt (QPP-4-RAG)
            msg = [{"role": "system", "content": "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."},
                   {"role": "user", "content": f"""Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process).  Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most 30 nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.

Search Query: {qt}
Context:
{ctx}
Search Query: {qt}
Initial Nugget List: []
Initial Nugget List Length: 0

Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings and only strings."""}]
            nug = parse_list(chat(cl, model, msg))[:30]
            if not nug:
                continue
            # scorer prompt (QPP-4-RAG): vital vs okay
            smsg = [{"role": "system", "content": "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query."},
                    {"role": "user", "content": f"""Based on the query, label each of the {len(nug)} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a "good" answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {qt}
Nugget List: {nug}

Only return the list of labels (List[str]). Do not explain."""}]
            lab = [x.lower() for x in parse_list(chat(cl, model, smsg), n=len(nug))]
            lab += ["okay"] * (len(nug) - len(lab))
            lab = ["vital" if "vital" in x else "okay" for x in lab[:len(nug)]]
            fh.write(json.dumps({"qid": q, "query": qt,
                                 "nuggets": [{"text": t, "importance": im}
                                             for t, im in zip(nug, lab)]}, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(qids)} ({sum(1 for _ in open(out_path))} written)", flush=True)
    print(f"wrote {out_path}", flush=True)


# ---------------------------------------------------------------- phase: generate reports
def phase_generate(a, cl, model):
    gold = jsonl_done(os.path.join(RAG, f"gold_nuggets_n{a.n}.jsonl"))
    if not gold:
        sys.exit("no gold nuggets yet: run --phase nuggets first")
    texts = load_texts(ALL_CHANNELS)
    runs = {}
    for pol, (fn, _) in POLICIES.items():
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            runs[pol] = load_run(p)
        else:
            print(f"  WARNING missing run {fn} for policy {pol}", flush=True)
    # per-query channel picks, for policies whose evidence is not fixed in advance
    picks = {}
    for pol, fn in PICKS.items():
        p = os.path.join(DATA, fn)
        if os.path.exists(p):
            picks[pol] = json.load(open(p))
        elif pol in runs:
            sys.exit(f"policy {pol} needs {fn}: run mv2_routed_run.py first")
    out_path = os.path.join(RAG, f"reports_n{a.n}_{a.evidence}.jsonl")
    done = {(r["qid"], r["policy"]) for r in jsonl_done(out_path, key="key").values()} \
        if os.path.exists(out_path) else set()
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line); done.add((r["qid"], r["policy"]))
                except Exception:
                    pass
    print(f"generate: {len(gold)} queries x {len(runs)} policies, {len(done)} already done", flush=True)

    with open(out_path, "a", buffering=1) as fh:
        n = 0
        for q, g in gold.items():
            for pol, run in runs.items():
                if (q, pol) in done or q not in run:
                    continue
                top = sorted(run[q], key=lambda d: -run[q][d])[:TOPK]
                if a.evidence != "own":
                    chans = ALL_CHANNELS
                elif POLICIES[pol][1] is not None:
                    chans = POLICIES[pol][1]
                else:
                    # the router chose this query's channels; ground it on exactly those, so a policy
                    # is never credited with evidence it did not retrieve
                    chans = [CHANNEL_TEXT[c] for c in picks[pol].get(q, "").split("+")
                             if c in CHANNEL_TEXT]
                segs = evidence_for(top, texts, chans)
                if not segs:
                    fh.write(json.dumps({"qid": q, "policy": pol, "report": "",
                                         "n_evidence": 0}, ensure_ascii=False) + "\n")
                    continue
                ctx = "\n".join(f"[{j+1}] {s}" for j, s in enumerate(segs))
                msg = [{"role": "system", "content": "You write factual event reports grounded only in the provided evidence."},
                       {"role": "user", "content": f"""Write a factual report about the event described by the query, using only the evidence below. Include concrete specifics (who, what, where, when, numbers, names) that the evidence supports. Do not speculate or add outside knowledge. Write 100-200 words of plain prose, no headings or lists.

Query: {g['query']}
Evidence:
{ctx}

Report:"""}]
                rep = chat(cl, model, msg, temperature=0.0)
                fh.write(json.dumps({"qid": q, "policy": pol, "report": rep,
                                     "n_evidence": len(segs)}, ensure_ascii=False) + "\n")
                n += 1
                if n % 20 == 0:
                    print(f"  {n} reports generated", flush=True)
    print(f"wrote {out_path}", flush=True)


# ---------------------------------------------------------------- phase: assign nuggets
def phase_assign(a, cl, model):
    gold = jsonl_done(os.path.join(RAG, f"gold_nuggets_n{a.n}.jsonl"))
    rep_path = os.path.join(RAG, f"reports_n{a.n}_{a.evidence}.jsonl")
    if not os.path.exists(rep_path):
        sys.exit("no reports yet: run --phase generate first")
    reports = []
    with open(rep_path) as f:
        for line in f:
            try:
                reports.append(json.loads(line))
            except Exception:
                pass
    out_path = os.path.join(RAG, f"assigned_n{a.n}_{a.evidence}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line); done.add((r["qid"], r["policy"]))
                except Exception:
                    pass
    print(f"assign: {len(reports)} reports, {len(done)} already done", flush=True)

    with open(out_path, "a", buffering=1) as fh:
        n = 0
        for r in reports:
            key = (r["qid"], r["policy"])
            if key in done or r["qid"] not in gold:
                continue
            g = gold[r["qid"]]
            nug = g["nuggets"]
            if not r["report"].strip():
                lab = ["not_support"] * len(nug)
            else:
                msg = [{"role": "system", "content": "You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage."},
                       {"role": "user", "content": f"""Based on the query and passage, label each of the {len(nug)} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {g['query']}
Passage: {r['report']}
Nugget List: {[x['text'] for x in nug]}

Only return the list of labels (List[str]). Do not explain."""}]
                raw = [x.lower().strip() for x in parse_list(chat(cl, model, msg), n=len(nug))]
                raw += ["not_support"] * (len(nug) - len(raw))
                lab = []
                for x in raw[:len(nug)]:
                    lab.append("support" if x == "support" else
                               ("partial_support" if "partial" in x else "not_support"))
            fh.write(json.dumps({"qid": r["qid"], "policy": r["policy"],
                                 "nuggets": [{"text": t["text"], "importance": t["importance"],
                                              "assignment": l} for t, l in zip(nug, lab)]},
                                ensure_ascii=False) + "\n")
            n += 1
            if n % 20 == 0:
                print(f"  {n} assigned", flush=True)
    print(f"wrote {out_path}", flush=True)


# ---------------------------------------------------------------- phase: metrics
def score(nuggets):
    vital = [n for n in nuggets if n["importance"] == "vital"]
    sv = sum(1 for n in vital if n["assignment"] == "support")
    sa = sum(1 for n in nuggets if n["assignment"] == "support")
    pv = sv + sum(0.5 for n in vital if n["assignment"] == "partial_support")
    pa = sa + sum(0.5 for n in nuggets if n["assignment"] == "partial_support")
    return {"strict_vital": sv / len(vital) if vital else 0.0,
            "strict_all": sa / len(nuggets) if nuggets else 0.0,
            "vital": pv / len(vital) if vital else 0.0,
            "all": pa / len(nuggets) if nuggets else 0.0}


def phase_metrics(a, *_):
    path = os.path.join(RAG, f"assigned_n{a.n}_{a.evidence}.jsonl")
    if not os.path.exists(path):
        sys.exit("nothing assigned yet")
    per = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            per[r["policy"]].append(score(r["nuggets"]))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    from mv2_ab import per_query_ndcg
    gold = jsonl_done(os.path.join(RAG, f"gold_nuggets_n{a.n}.jsonl"))
    nd = {}
    for pol, (fn, _) in POLICIES.items():
        p = os.path.join(DATA, fn)
        if os.path.exists(p) and pol in per:
            run = load_run(p)
            # average over the sampled queries only: ndcg10() means over every qid in qrels, which
            # would divide a 400-query sample by 2,546 and report near-zero for every policy
            pq = per_query_ndcg(qrels, {q: run[q] for q in gold if q in run})
            vals = [pq[q] for q in gold if q in pq]
            nd[pol] = float(np.mean(vals)) if vals else float("nan")
    rows = []
    for pol, vals in per.items():
        m = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        rows.append((pol, len(vals), nd.get(pol, float("nan")), m))
    rows.sort(key=lambda r: -r[2] if not np.isnan(r[2]) else 0)
    out = {"evidence": a.evidence, "n_queries": a.n,
           "rows": [{"policy": p, "n": n, "ndcg10": d, **m} for p, n, d, m in rows]}
    json.dump(out, open(os.path.join(RAG, f"metrics_n{a.n}_{a.evidence}.json"), "w"), indent=2)
    print(f"\nevidence={a.evidence}  (judge: local model; policies comparable to each other only)")
    print(f"{'policy':<12}{'n':>5}{'nDCG@10':>10}{'N_strict_v':>12}{'N_vital':>10}{'N_strict_a':>12}{'N_all':>8}")
    for p, n, d, m in rows:
        print(f"{p:<12}{n:>5}{d:>10.4f}{m['strict_vital']:>12.4f}{m['vital']:>10.4f}"
              f"{m['strict_all']:>12.4f}{m['all']:>8.4f}")
    if len(rows) > 1:
        by_nd = [r[0] for r in rows]
        print(f"\nranking by nDCG@10:{'':<10}{' > '.join(by_nd)}")
        flips = []
        for k in ("strict_vital", "vital", "strict_all", "all"):
            vals = {r[0]: r[3][k] for r in rows}
            if len(set(vals.values())) == 1:
                print(f"ranking by N_{k:<12} (all tied, uninformative)")
                continue
            order = [p for p, _ in sorted(vals.items(), key=lambda kv: -kv[1])]
            print(f"ranking by N_{k:<12} {' > '.join(order)}"
                  + ("   <-- differs from nDCG" if order != by_nd else ""))
            if order != by_nd:
                flips.append(k)
        print(f"\nUTILITY GAP on {', '.join(flips)}: the policy that ranks best does not generate best"
              if flips else "\nno utility gap: same ordering under retrieval and generation objectives")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["nuggets", "generate", "assign", "metrics"])
    ap.add_argument("--evidence", default="all", choices=["own", "all"])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model", default="qwen2.5:14b-instruct")
    ap.add_argument("--base", default="http://localhost:11434/v1")
    a = ap.parse_args()
    cl, model = client(a.model, a.base)
    {"nuggets": phase_nuggets, "generate": phase_generate,
     "assign": phase_assign, "metrics": phase_metrics}[a.phase](a, cl, model)


if __name__ == "__main__":
    main()
