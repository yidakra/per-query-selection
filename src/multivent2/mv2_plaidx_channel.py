"""Build the speech channel with the SOTA systems' own retriever: translate-distill PLAID-X.

MMMORRF's text channels use HLTCOE's translate-distill ColBERT (PLAID-X) models, and the multilingual
checkpoint is public. This indexes the original-language ASR transcripts with that checkpoint and
searches the English queries against it, which is the strong systems' recipe for this exact setting:
no query-time translation, a retriever distilled for English-query CLIR over multilingual passages.

Documents are windowed to passages (MaxP at read-out): a document's score is the maximum over its
passages, the same aggregation the dense channel uses. Output is an ordinary run JSON, so the
selection experiment can face it unchanged.

Run inside .venv-plaidx with GPU1:

  CUDA_VISIBLE_DEVICES=1 .venv-plaidx/bin/python src/multivent2/mv2_plaidx_channel.py --smoke
  CUDA_VISIBLE_DEVICES=1 .venv-plaidx/bin/python src/multivent2/mv2_plaidx_channel.py
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
WORK = os.path.join(_ROOT, "runs", "plaidx")


def window(text, size=900, overlap=180):
    out = []
    step = size - overlap
    for s in range(0, max(len(text) - overlap, 1), step):
        w = text[s:s + size].strip()
        if w:
            out.append(w)
    return out or [text[:size]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hltcoe/plaidx-large-neuclir-mtd-mix-passages-mt5xxl-engeng")
    ap.add_argument("--text", default="asr_text.jsonl")
    ap.add_argument("--index-name", default="mv2_asr_plaidx")
    ap.add_argument("--k", type=int, default=2000, help="passages retrieved per query")
    ap.add_argument("--depth", type=int, default=1000, help="documents kept per query")
    ap.add_argument("--smoke", action="store_true", help="2000 docs, 20 queries, tiny index")
    ap.add_argument("--out", default=os.path.join(DATA, "asr_plaidx_mtd.json"))
    a = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "0"):
        sys.exit("refusing to run: set CUDA_VISIBLE_DEVICES to a nonzero physical GPU")

    from mv2_io import load_queries, load_qrels
    os.makedirs(WORK, exist_ok=True)
    tag = "_smoke" if a.smoke else ""
    name = a.index_name + tag

    # 1. collection: windowed passages, pid is the line number, mapping saved beside it
    coll_path = os.path.join(WORK, f"collection{tag}.tsv")
    map_path = os.path.join(WORK, f"pid2doc{tag}.json")
    if not os.path.exists(map_path):
        pid2doc = []
        n_docs = 0
        with open(os.path.join(DATA, a.text)) as f, open(coll_path, "w") as out:
            for line in f:
                r = json.loads(line)
                t = r.get("text", "").strip()
                if not t:
                    continue
                n_docs += 1
                if a.smoke and n_docs > 2000:
                    break
                for w in window(t):
                    out.write(f"{len(pid2doc)}\t" + w.replace("\t", " ").replace("\n", " ") + "\n")
                    pid2doc.append(r["doc_id"])
        json.dump(pid2doc, open(map_path, "w"))
        print(f"collection: {len(pid2doc)} passages from {n_docs} docs", flush=True)
    pid2doc = json.load(open(map_path))

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qids = sorted(queries)
    if a.smoke:
        qids = qids[:20]
    q_path = os.path.join(WORK, f"queries{tag}.tsv")
    with open(q_path, "w") as out:
        for i, q in enumerate(qids):
            out.write(f"{i}\t" + queries[q].replace("\t", " ").replace("\n", " ") + "\n")

    # 2. index + search
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    with Run().context(RunConfig(nranks=1, experiment="mv2", root=WORK)):
        config = ColBERTConfig(nbits=2, doc_maxlen=256, query_maxlen=48)
        indexer = Indexer(checkpoint=a.model, config=config)
        # PLAID-X splits indexing into three phases so shards can run separately; one host runs all
        indexer.prepare(name=name, collection=coll_path, overwrite=True)
        indexer.encode(name=name, collection=coll_path)
        indexer.finalize(name=name, collection=coll_path)
        searcher = Searcher(index=name, config=config)
        rankings = searcher.search_all(q_path, k=a.k)

    # 3. MaxP to documents, write the run
    run = {}
    for iq, hits in rankings.todict().items():
        best = {}
        for pid, _rank, score in hits:
            d = pid2doc[pid]
            if score > best.get(d, -1e9):
                best[d] = float(score)
        top = sorted(best.items(), key=lambda kv: -kv[1])[:a.depth]
        run[qids[int(iq)]] = dict(top)
    out_path = a.out + (".smoke" if a.smoke else "")
    json.dump(run, open(out_path, "w"))
    print(f"wrote {out_path}", flush=True)

    # 4. score
    import numpy as np
    sys.path.insert(0, HERE)
    from mv2_ab import per_query_ndcg
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    nd = per_query_ndcg(qrels, run)
    if nd:
        print(f"PLAID-X channel nDCG@10 over {len(nd)} queries: "
              f"{np.mean(list(nd.values())):.4f}", flush=True)


if __name__ == "__main__":
    main()
