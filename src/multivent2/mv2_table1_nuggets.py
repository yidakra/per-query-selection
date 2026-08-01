"""Nugget coverage for every row of Table 1, mixed from two judged runs per cell.

Arabzadeh et al.'s Table 1 reports each selector's end-to-end answer quality, not just its ranking. A
literal reading of that would need one judge pass per predictor per cell -- about thirty passes at
several GPU-hours each, which is not affordable and is also unnecessary.

A predictor row executes run A (visual alone) or run B (visual fused with the cell's channel) per query.
A report is written from a ranked list and scored against that query's gold nuggets, so the coverage of
a query is a property of the run it was routed to, not of the predictor that routed it. Judge run A and
run B once per cell and every row is a per-query mix:

    N(predictor) = mean_q [ N_B(q) if predictor escalated q else N_A(q) ]

Run A is visual alone and is the same in all three cells, so three runs cover the whole table: `visual`,
`cellB_asr_shipped`, `cellB_asr_dense`, `cellB_ocr` in the RAG arm's assignment file.

This is exact, not an approximation, and it is the same identity `mv2_recall_sidecar.py` uses for the
Recall@100 column. Two things it depends on, both checked below: the decisions must be the ones that
produced the row's nDCG, and the judged queries must be a subset of the routed ones.

The mix runs over the RAG arm's 395 judged queries, not all 2,546, so the nDCG reproduced here is the
subset's and will not equal the main table's. That is reported rather than hidden.

  python src/multivent2/mv2_table1_nuggets.py [--evidence all] [--tag _grouped]
"""
import os
import sys
import json
import argparse
import collections

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_rag_nuggets import score              # noqa: E402
from mv2_io import load_qrels, load_run        # noqa: E402
from mv2_ab import per_query_ndcg              # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
RAG = os.path.join(ABL, "rag")

# cell -> the policy in the RAG assignment file that IS that cell's B run. A is `visual` for all three.
CELL_B = {"ASR-shipped": "cellB_asr_shipped",
          "ASR-dense": "cellB_asr_dense",
          "OCR": "cellB_ocr"}
A_POLICY = "visual"
METRICS = ("vital", "strict_vital", "all", "strict_all")
TOL = 1e-6          # the nDCG check is a re-derivation of the same arithmetic, so it should be exact


def load_assigned(path):
    """policy -> qid -> per-query nugget metrics."""
    out = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["policy"]][r["qid"]] = score(r["nuggets"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="all", choices=["own", "all"],
                    help="which grounding protocol's judgments to mix. `all` isolates retrieval, which "
                         "is what the rest of Table 1 measures")
    ap.add_argument("--tag", default="_grouped")
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()

    assigned = load_assigned(os.path.join(RAG, f"assigned_n{a.n}_{a.evidence}.jsonl"))
    missing = [p for p in [A_POLICY] + list(CELL_B.values()) if p not in assigned]
    if missing:
        sys.exit(f"not judged yet: {', '.join(missing)}. Run mv2_rag_nuggets.py generate+assign "
                 f"after adding them to POLICIES.")

    table = json.load(open(os.path.join(ABL, f"mv2_qpp_table{a.tag}.json")))
    qsd = json.load(open(os.path.join(ABL, f"mv2_qsd{a.tag}.json")))
    bert = json.load(open(os.path.join(ABL, f"mv2_bertqpp{a.tag}.json")))
    p_bi = os.path.join(ABL, f"mv2_bertqpp_bi{a.tag}.json")
    bert_bi = json.load(open(p_bi)) if os.path.exists(p_bi) else {}
    p_qp = os.path.join(ABL, f"mv2_qsd_post{a.tag}.json")
    qsd_post = json.load(open(p_qp)) if os.path.exists(p_qp) else {}
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    SNAKE = {"ASR-shipped": "asr_shipped", "ASR-dense": "asr_dense", "OCR": "ocr"}
    qsd_k = "5_inv_dist" if not a.tag.endswith("_grouped") else "100_inv_dist"

    out = {}
    for cell, bpol in CELL_B.items():
        if cell not in table:
            continue
        qids = table[cell]["qids"]
        judged = [q for q in qids if q in assigned[A_POLICY] and q in assigned[bpol]]
        idx = {q: i for i, q in enumerate(qids)}
        print(f"\n=== {cell}: {len(judged)} of {len(qids)} queries judged")

        # per-query nDCG of the two runs on the judged subset, so the mix can be checked against the
        # row's own routed nDCG restricted to the same queries
        runA = load_run(os.path.join(DATA, "10pyscene_clip.json"))
        pqA = per_query_ndcg(qrels, {q: runA[q] for q in judged if q in runA})
        runB = load_run(os.path.join(DATA, f"{bpol}.json"))
        pqB = per_query_ndcg(qrels, {q: runB[q] for q in judged if q in runB})

        def mix(decisions):
            """Mean of each metric over the judged queries, taking B where the predictor escalated."""
            m = {k: [] for k in METRICS}
            nd = []
            for q in judged:
                b = decisions[idx[q]] == "1"
                s = assigned[bpol][q] if b else assigned[A_POLICY][q]
                for k in METRICS:
                    m[k].append(s[k])
                nd.append((pqB if b else pqA).get(q, 0.0))
            return {k: float(np.mean(v)) for k, v in m.items()} | {"ndcg10_subset": float(np.mean(nd))}

        rows = {}
        dec = table[cell]["decisions"]
        for sec in ("pre", "post"):
            for name, d in dec.get(sec, {}).items():
                rows[f"{sec}/{name}"] = mix(d)
        rows["pre/QSD_PRE"] = mix(qsd[SNAKE[cell]]["k"][qsd_k]["decisions"])
        rows["post/BERTQPP"] = mix(bert[SNAKE[cell]]["decisions"])
        if SNAKE[cell] in bert_bi and "decisions" in bert_bi[SNAKE[cell]]:
            rows["post/BERTQPP_BI"] = mix(bert_bi[SNAKE[cell]]["decisions"])
        if SNAKE[cell] in qsd_post and "decisions" in qsd_post[SNAKE[cell]]:
            rows["post/QSD_POST"] = mix(qsd_post[SNAKE[cell]]["decisions"])
        rows["ours"] = mix(dec["ours"])
        rows["oracle"] = mix(dec["oracle"])
        rows["_A_visual"] = mix("0" * len(qids))
        rows["_B_fused"] = mix("1" * len(qids))

        # the two endpoints must reproduce the judged runs' own coverage exactly, or the mixing is
        # indexed wrong somewhere and every row above is quietly wrong with it
        for endpoint, pol in (("_A_visual", A_POLICY), ("_B_fused", bpol)):
            for k in METRICS:
                direct = float(np.mean([assigned[pol][q][k] for q in judged]))
                d = abs(rows[endpoint][k] - direct)
                if d > TOL:
                    sys.exit(f"{cell} {endpoint} {k}: mixed {rows[endpoint][k]:.6f} against direct "
                             f"{direct:.6f}. The decision vectors are not aligned to `qids`.")
        print(f"  endpoints reproduce the judged runs exactly")
        print(f"  {'row':<22}{'nDCG@10':>9}{'N_vital':>9}{'N_str_v':>9}{'N_all':>8}{'N_str_a':>9}")
        for k in ("_A_visual", "_B_fused", "pre/QSD_PRE", "post/NQC", "post/BERTQPP",
                  "ours", "oracle"):
            if k in rows:
                r = rows[k]
                print(f"  {k:<22}{r['ndcg10_subset']:>9.4f}{r['vital']:>9.4f}"
                      f"{r['strict_vital']:>9.4f}{r['all']:>8.4f}{r['strict_all']:>9.4f}")
        out[cell] = {"n_judged": len(judged), "evidence": a.evidence, "rows": rows}

    dest = os.path.join(ABL, f"mv2_table1_nuggets{a.tag}.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
