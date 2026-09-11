"""Gold-split audit of the query-variant selection oracle.

`mv2_axis_goldsplit.py` audits an axis whose options have stored runs. The variant axis has none:
its 31 candidates per query are retrieved on the fly by `mv2_variant_selection.py`. This script
rebuilds those retrievals from the cached document embeddings, then hands the option set to the
same `audit()` protocol the channel and language axes went through, so all three oracles are
measured the same way.

The candidate pool is the pre-registered one: the original query plus six reformulation methods at
five samples each, restricted to the queries with a complete pool. The concatenate-everything
baseline is not a candidate and stays out, exactly as in the selection experiment.

Needs the GPU for the query encoder. Top-100 lists are enough for nDCG@10 and keep the whole option
set in memory at once.

  python src/multivent2/mv2_variant_goldsplit.py --gpu 0
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=os.path.join(DATA, "query_variants.jsonl"))
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--topk", type=int, default=100, help="enough for nDCG@10, small in memory")
    ap.add_argument("--gpu", type=int, default=0,
                    help="physical GPU index; the box now has a single shared device")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu keeps the shared GPU free; the retrieval maths is the same")
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_axis_goldsplit_variant.json"))
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    from mv2_io import load_qrels, load_queries
    from mv2_variant_selection import METHODS, load_pool, encode_and_search
    from mv2_axis_goldsplit import audit

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    pool = load_pool(a.variants, a.samples)

    qids = sorted(q for q in queries if q in qrels)
    complete = [q for q in qids
                if all((q, m, s) in pool for m in METHODS for s in range(a.samples))]
    labels = ["original"] + [f"{m}#{s}" for m in METHODS for s in range(a.samples)]
    print(f"{len(complete)}/{len(qids)} queries have the complete {len(METHODS)}x{a.samples} pool; "
          f"{len(labels)} options per query", flush=True)

    texts, owners = [], []
    for q in complete:
        for lab in labels:
            if lab == "original":
                texts.append(queries[q])
            else:
                m, s = lab.rsplit("#", 1)
                texts.append(pool[(q, m, int(s))])
            owners.append((q, lab))

    print(f"retrieving {len(texts)} candidates on the dense speech channel", flush=True)
    runs = encode_and_search(texts, os.path.join(_ROOT, "runs", "embcache", "asr_bge-m3.npz"),
                             "BAAI/bge-m3", topk=a.topk, device=a.device)

    by_label = {lab: {} for lab in labels}
    for i, (q, lab) in enumerate(owners):
        by_label[lab][q] = runs[i]
    pol_runs = [(lab, by_label[lab]) for lab in labels]

    print("axis variant", flush=True)
    rec = audit(qrels, pol_runs, complete, "original", a.seeds)
    rec = {"axis": "variant", "methods": list(METHODS), "samples": a.samples,
           "options": labels, "topk": a.topk, "device": a.device, **rec}
    json.dump(rec, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
