"""Qualitative error analysis of channel selection.

The rest of the study measures whether a decision can be made. This asks where it goes wrong and
whether the failures have structure, using the judgment metadata the benchmark ships and never
otherwise uses: the relevant video's language, the event type, the production type, and which
modality the relevance came from.

Three questions, each answered by a breakdown rather than a single number.

1. Which queries does the decision matter for at all? A query whose channels tie carries no
   decision, and a selector cannot be blamed for it.
2. Where does the outcome-reading selector lose, and does the loss concentrate anywhere a
   practitioner could act on: a language, an event type, a production style, a modality of evidence?
3. What does a losing query look like? The largest individual losses are printed with their text so
   the failure can be read rather than inferred.

CPU only, reads the committed runs.

  python src/multivent2/mv2_error_analysis.py
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

META_FIELDS = ["video_language", "query_event_type", "video_type", "video_modality"]


def query_metadata(path):
    """One label per query per field, taken from its most relevant judged video. Ties on relevance
    go to the first seen, which is stable because the file order is stable."""
    best = {}
    for line in open(path):
        r = json.loads(line)
        q, rel = r["query_id"], int(r.get("relevance", 0))
        if rel <= 0:
            continue
        if q not in best or rel > best[q][0]:
            best[q] = (rel, {f: (r.get(f) or "unlabelled") for f in META_FIELDS})
    return {q: v[1] for q, v in best.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", default="asr_dense_bge-m3.json")
    ap.add_argument("--screen", default="ocr_dense_bge-m3.json")
    ap.add_argument("--tie", type=float, default=1e-9,
                    help="below this absolute difference the two channels are treated as tied")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_error_analysis.json"))
    a = ap.parse_args()

    from mv2_io import load_qrels, load_run, load_queries
    from mv2_ab import per_query_ndcg

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    qids = sorted(q for q in queries if q in qrels)
    meta = query_metadata(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))

    sp = load_run(os.path.join(DATA, a.speech))
    sc = load_run(os.path.join(DATA, a.screen))
    y_sp = per_query_ndcg(qrels, {q: sp[q] for q in qids})
    y_sc = per_query_ndcg(qrels, {q: sc[q] for q in qids})

    gain = {q: y_sc.get(q, 0.0) - y_sp.get(q, 0.0) for q in qids}      # screen text over speech
    tied = [q for q in qids if abs(gain[q]) <= a.tie]
    live = [q for q in qids if abs(gain[q]) > a.tie]
    screen_wins = [q for q in live if gain[q] > 0]

    print(f"{len(qids)} queries: {len(tied)} tied ({100*len(tied)/len(qids):.1f}%), "
          f"{len(live)} carry a decision, and screen text is the better channel on "
          f"{len(screen_wins)} of those ({100*len(screen_wins)/max(1,len(live)):.1f}%)", flush=True)

    # where the decision lives, by each metadata field
    breakdown = {}
    for field in META_FIELDS:
        rows = {}
        by = collections.defaultdict(list)
        for q in qids:
            by[meta.get(q, {}).get(field, "unlabelled")].append(q)
        for label, group in sorted(by.items(), key=lambda kv: -len(kv[1])):
            g_live = [q for q in group if abs(gain[q]) > a.tie]
            if len(group) < 20:                       # too small to read anything into
                continue
            rows[label] = {
                "queries": len(group),
                "tied_pct": 100 * (len(group) - len(g_live)) / len(group),
                "screen_wins_pct": (100 * sum(1 for q in g_live if gain[q] > 0) / len(g_live))
                                   if g_live else None,
                "speech_ndcg": float(np.mean([y_sp.get(q, 0.0) for q in group])),
                "screen_ndcg": float(np.mean([y_sc.get(q, 0.0) for q in group])),
                "headroom": float(np.mean([max(y_sp.get(q, 0.0), y_sc.get(q, 0.0)) for q in group])
                                  - np.mean([y_sp.get(q, 0.0) for q in group])),
            }
        breakdown[field] = rows
        print(f"\n[{field}]", flush=True)
        for label, r in rows.items():
            sw = "--" if r["screen_wins_pct"] is None else f"{r['screen_wins_pct']:.0f}%"
            print(f"  {label:24s} n={r['queries']:5d} tied={r['tied_pct']:4.0f}% "
                  f"screen wins={sw:>4s} speech={r['speech_ndcg']:.3f} "
                  f"screen={r['screen_ndcg']:.3f} headroom={100*r['headroom']:+.2f}", flush=True)

    # the queries where choosing screen text over speech would cost the most, and the reverse
    worst_screen = sorted(live, key=lambda q: gain[q])[:a.examples]
    best_screen = sorted(live, key=lambda q: -gain[q])[:a.examples]
    examples = {"speech_much_better": [], "screen_much_better": []}
    for key, group in (("speech_much_better", worst_screen), ("screen_much_better", best_screen)):
        for q in group:
            examples[key].append({
                "qid": q, "query": queries[q],
                "speech": round(y_sp.get(q, 0.0), 4), "screen": round(y_sc.get(q, 0.0), 4),
                **{f: meta.get(q, {}).get(f, "unlabelled") for f in META_FIELDS}})

    print("\n[examples] queries where speech beats screen text by the widest margin", flush=True)
    for e in examples["speech_much_better"][:6]:
        print(f"  {e['speech']:.3f} vs {e['screen']:.3f}  {e['video_language']:8s} "
              f"{e['query_event_type']:22s} {e['query'][:60]}", flush=True)
    print("\n[examples] queries where screen text beats speech by the widest margin", flush=True)
    for e in examples["screen_much_better"][:6]:
        print(f"  {e['screen']:.3f} vs {e['speech']:.3f}  {e['video_language']:8s} "
              f"{e['query_event_type']:22s} {e['query'][:60]}", flush=True)

    out = {"n_queries": len(qids), "tied": len(tied), "live": len(live),
           "screen_wins_of_live": len(screen_wins), "tie_threshold": a.tie,
           "speech_mean": float(np.mean([y_sp.get(q, 0.0) for q in qids])),
           "screen_mean": float(np.mean([y_sc.get(q, 0.0) for q in qids])),
           "breakdown": breakdown, "examples": examples}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
