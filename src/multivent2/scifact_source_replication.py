"""Beyond-video evidence-source selection on BEIR SciFact.

The two candidate sources are a paper's title and its abstract.  We retrieve the
same claim independently against each field with BM25, then ask whether a QPP
signal can choose the better field per claim.  This is deliberately a direct
source-choice test: both runs exist, so every predictor gets its strongest
source-aware form (abstract value minus title value).

The protocol mirrors the MultiVENT analysis where it matters:

* each scalar is oriented against the source gain strictly out of fold;
* queries sharing a relevant paper are kept in the same fold;
* the eleven collection-statistic predictors and ten score-distribution
  predictors are unchanged from ``mv2_qpp_predictors.py``;
* multi-feature corpus and score controls reveal whether a failure is specific
  to the analytic scalars or to the information family;
* all 1,109 labeled train+test claims are used as an analysis set.  This is not
  a submission to the BEIR test leaderboard.

Data source (official BEIR release):
https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from mv2_qpp_predictors import (PRE_RETRIEVAL, SCORE_ONLY, Index,  # noqa: E402
                                pre_retrieval_suite, score_only_suite)

DATA = os.path.join(ROOT, "data", "scifact", "scifact")
ABL = os.path.join(ROOT, "results", "ablations")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    """Small exact BM25 index; SciFact has only 5,183 documents."""

    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.n = len(texts)
        self.k1 = k1
        self.b = b
        self.lengths = np.zeros(self.n, dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for docno, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths[docno] = sum(counts.values())
            for term, tf in counts.items():
                postings[term].append((docno, tf))
        self.avgdl = float(self.lengths.mean())
        self.postings = {}
        self.idf = {}
        for term, values in postings.items():
            docnos = np.fromiter((x[0] for x in values), dtype=np.int32)
            tf = np.fromiter((x[1] for x in values), dtype=np.float32)
            self.postings[term] = (docnos, tf)
            df = len(values)
            self.idf[term] = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if posting is None:
                continue
            docnos, tf = posting
            norm = tf + self.k1 * (1.0 - self.b + self.b * self.lengths[docnos] / self.avgdl)
            out[docnos] += self.idf[term] * tf * (self.k1 + 1.0) / norm
        return out


def load_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def load_qrels() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for split in ("train", "test"):
        with open(os.path.join(DATA, "qrels", f"{split}.tsv")) as handle:
            next(handle)
            for line in handle:
                qid, docid, rel = line.rstrip().split("\t")
                out[qid][docid] = max(int(rel), out[qid].get(docid, 0))
    return dict(out)


def ndcg10(scores: np.ndarray, relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    top = np.argsort(-scores, kind="stable")[:10]
    dcg = sum(1.0 / math.log2(rank + 2.0)
              for rank, docno in enumerate(top) if int(docno) in relevant)
    ideal = sum(1.0 / math.log2(rank + 2.0)
                for rank in range(min(10, len(relevant))))
    return dcg / ideal


def event_groups(qids: list[str], qrels: dict[str, dict[str, int]]) -> np.ndarray:
    """Connected components under shared relevant papers."""
    parent = list(range(len(qids)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen: dict[str, int] = {}
    for i, qid in enumerate(qids):
        for docid, rel in qrels[qid].items():
            if rel <= 0:
                continue
            if docid in seen:
                union(i, seen[docid])
            else:
                seen[docid] = i
    roots = {}
    groups = []
    for i in range(len(qids)):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots)
        groups.append(roots[root])
    return np.asarray(groups, dtype=int)


def group_folds(groups: np.ndarray, seed: int, n_splits: int = 5):
    """Balanced, randomized group-disjoint folds (including mostly singleton groups)."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    shuffled = rng.permutation(len(uniq))
    order = shuffled[np.argsort(-counts[shuffled], kind="stable")]
    fold_groups = [set() for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)
    for pos in order:
        smallest = np.flatnonzero(fold_sizes == fold_sizes.min())
        fold = int(rng.choice(smallest))
        fold_groups[fold].add(int(uniq[pos]))
        fold_sizes[fold] += int(counts[pos])
    all_rows = np.arange(len(groups))
    for members in fold_groups:
        test = all_rows[np.isin(groups, list(members))]
        train = all_rows[~np.isin(groups, list(members))]
        yield train, test


def regressor():
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 4, 16))),
    ])


def repeated_oof(x: np.ndarray, gain: np.ndarray, groups: np.ndarray,
                 seeds: list[int]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    predictions = np.zeros((len(seeds), len(gain)), dtype=float)
    for si, seed in enumerate(seeds):
        for train, test in group_folds(groups, seed):
            fitted = regressor().fit(x[train], gain[train])
            predictions[si, test] = fitted.predict(x[test])
    return predictions.mean(axis=0)


def describe_route(raw: np.ndarray, pred: np.ndarray, title: np.ndarray,
                   abstract: np.ndarray) -> dict:
    gain = abstract - title
    decision = pred > 0
    tau = kendalltau(np.asarray(raw).ravel(), gain).statistic
    if not np.isfinite(tau):
        tau = 0.0
    return {
        "tau_raw": float(tau),
        "routed_ndcg10": float(np.where(decision, abstract, title).mean()),
        "choose_abstract_fraction": float(decision.mean()),
        "degenerate": bool(decision.all() or not decision.any()),
    }


def overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    aa = set(np.argsort(-a, kind="stable")[:k].tolist())
    bb = set(np.argsort(-b, kind="stable")[:k].tolist())
    return len(aa & bb) / max(1, len(aa | bb))


def grouped_inference(diff: np.ndarray, groups: np.ndarray, seed: int = 41,
                      draws: int = 10000) -> dict:
    """Cluster bootstrap CI and cluster sign-flip p-value for a paired mean."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in uniq]
    boot = np.empty(draws, dtype=float)
    for i in range(draws):
        sample = rng.integers(0, len(members), len(members))
        vals = [diff[members[j]] for j in sample]
        boot[i] = float(np.concatenate(vals).mean())
    sums = np.asarray([diff[idx].sum() for idx in members])
    observed = abs(float(diff.mean()))
    perm = np.empty(draws, dtype=float)
    denom = len(diff)
    for i in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(sums))
        perm[i] = abs(float(np.dot(signs, sums) / denom))
    return {
        "mean": float(diff.mean()),
        "bootstrap_95ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "signflip_p_two_sided": float((1 + np.sum(perm >= observed)) / (draws + 1)),
        "draws": draws,
    }


def markdown(payload: dict) -> str:
    s = payload["summary"]
    lines = [
        "# SciFact evidence-source selection replication",
        "",
        "BM25 retrieves each of 1,109 labeled scientific claims independently against paper titles and abstracts. Queries sharing a relevant paper stay in the same fold. Every selector is repeated five-fold cross-fitted over five randomized group partitions; there is no evaluation-label threshold sweep.",
        "",
        "| fixed / selector | nDCG@10 | vs best fixed | chooses abstract | tau |",
        "|---|---:|---:|---:|---:|",
        f"| title only | {s['title_ndcg10']:.4f} | -- | 0.0% | -- |",
        f"| abstract only | {s['abstract_ndcg10']:.4f} | -- | 100.0% | -- |",
        f"| corpus-stat multi-feature control | {s['corpus_control']['routed_ndcg10']:.4f} | {100*(s['corpus_control']['routed_ndcg10']-s['best_fixed']):+.2f} | {100*s['corpus_control']['choose_abstract_fraction']:.1f}% | {s['corpus_control']['tau_pred']:+.3f} |",
        f"| score-distribution multi-feature control | {s['score_control']['routed_ndcg10']:.4f} | {100*(s['score_control']['routed_ndcg10']-s['best_fixed']):+.2f} | {100*s['score_control']['choose_abstract_fraction']:.1f}% | {s['score_control']['tau_pred']:+.3f} |",
        f"| oracle | {s['oracle_ndcg10']:.4f} | {100*(s['oracle_ndcg10']-s['best_fixed']):+.2f} | {100*s['oracle_choose_abstract_fraction']:.1f}% | +1.000 |",
        "",
        f"Analytic corpus-statistic predictors above the best fixed source by >0.0005: **{s['pre_above_fixed']}/{len(PRE_RETRIEVAL)}** (degenerate {s['pre_degenerate']}/{len(PRE_RETRIEVAL)}). Score-only predictors: **{s['score_above_fixed']}/{len(SCORE_ONLY)}** (degenerate {s['score_degenerate']}/{len(SCORE_ONLY)}).",
        "",
        f"The score control's paired gain is {100*s['score_control_inference']['mean']:+.2f} nDCG (group bootstrap 95% CI {100*s['score_control_inference']['bootstrap_95ci'][0]:+.2f} to {100*s['score_control_inference']['bootstrap_95ci'][1]:+.2f}; two-sided group sign-flip p={s['score_control_inference']['signflip_p_two_sided']:.4g}).",
        "",
        "This uses train and test labels together as a cross-fitted analysis set, not as a BEIR leaderboard submission. The source-aware differences are deliberately generous to both predictor families: each scalar sees its value on both fields before choosing.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--inference-draws", type=int, default=10000)
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    corpus = load_jsonl(os.path.join(DATA, "corpus.jsonl"))
    queries = {row["_id"]: row["text"] for row in load_jsonl(os.path.join(DATA, "queries.jsonl"))}
    qrels = load_qrels()
    qids = sorted(qrels, key=lambda x: int(x))
    docids = [row["_id"] for row in corpus]
    docno = {docid: i for i, docid in enumerate(docids)}
    relevant = [{docno[d] for d, rel in qrels[q].items() if rel > 0} for q in qids]
    groups = event_groups(qids, qrels)

    titles = [row.get("title", "") for row in corpus]
    abstracts = [row.get("text", "") for row in corpus]
    print(f"building BM25: {len(corpus)} papers, {len(qids)} labeled claims, "
          f"{len(np.unique(groups))} relevance groups", flush=True)
    title_bm25, abstract_bm25 = BM25(titles), BM25(abstracts)
    title_index = Index(titles, tokenizer=tokenize)
    abstract_index = Index(abstracts, tokenizer=tokenize)

    title_scores, abstract_scores = [], []
    title_ndcg, abstract_ndcg = [], []
    pre_diff = {name: [] for name in PRE_RETRIEVAL}
    score_diff = {name: [] for name in SCORE_ONLY}
    pre_control, score_control = [], []

    for i, qid in enumerate(qids):
        query = queries[qid]
        tokens = tokenize(query)
        ts, ab = title_bm25.scores(query), abstract_bm25.scores(query)
        title_scores.append(ts)
        abstract_scores.append(ab)
        title_ndcg.append(ndcg10(ts, relevant[i]))
        abstract_ndcg.append(ndcg10(ab, relevant[i]))

        pt = pre_retrieval_suite(tokens, title_index)
        pa = pre_retrieval_suite(tokens, abstract_index)
        st = score_only_suite(ts, len(tokens))
        sa = score_only_suite(ab, len(tokens))
        for name in PRE_RETRIEVAL:
            pre_diff[name].append(pa[name] - pt[name])
        for name in SCORE_ONLY:
            score_diff[name].append(sa[name] - st[name])
        pre_control.append([pt[name] for name in PRE_RETRIEVAL]
                           + [pa[name] for name in PRE_RETRIEVAL])
        score_control.append([st[name] for name in SCORE_ONLY]
                             + [sa[name] for name in SCORE_ONLY]
                             + [overlap(ts, ab, 10), overlap(ts, ab, 100), len(tokens)])
        if (i + 1) % 200 == 0:
            print(f"  retrieved {i + 1}/{len(qids)}", flush=True)

    title_ndcg = np.asarray(title_ndcg)
    abstract_ndcg = np.asarray(abstract_ndcg)
    gain = abstract_ndcg - title_ndcg
    best_fixed = max(float(title_ndcg.mean()), float(abstract_ndcg.mean()))
    fixed_scores = abstract_ndcg if abstract_ndcg.mean() >= title_ndcg.mean() else title_ndcg

    pre_rows = {}
    for name, raw in pre_diff.items():
        raw = np.asarray(raw)
        pred = repeated_oof(raw, gain, groups, seeds)
        pre_rows[name] = describe_route(raw, pred, title_ndcg, abstract_ndcg)
    score_rows = {}
    for name, raw in score_diff.items():
        raw = np.asarray(raw)
        pred = repeated_oof(raw, gain, groups, seeds)
        score_rows[name] = describe_route(raw, pred, title_ndcg, abstract_ndcg)

    pre_pred = repeated_oof(np.asarray(pre_control), gain, groups, seeds)
    score_pred = repeated_oof(np.asarray(score_control), gain, groups, seeds)

    def control_row(pred):
        choose = pred > 0
        return {
            "routed_ndcg10": float(np.where(choose, abstract_ndcg, title_ndcg).mean()),
            "choose_abstract_fraction": float(choose.mean()),
            "tau_pred": float(kendalltau(pred, gain).statistic),
        }

    pc, sc = control_row(pre_pred), control_row(score_pred)
    score_routed = np.where(score_pred > 0, abstract_ndcg, title_ndcg)
    margin = 5e-4
    summary = {
        "n_queries": len(qids),
        "n_relevance_groups": int(len(np.unique(groups))),
        "title_ndcg10": float(title_ndcg.mean()),
        "abstract_ndcg10": float(abstract_ndcg.mean()),
        "best_fixed": best_fixed,
        "gain_sd": float(gain.std()),
        "title_better_fraction": float((gain < 0).mean()),
        "abstract_better_fraction": float((gain > 0).mean()),
        "tied_fraction": float((gain == 0).mean()),
        "oracle_ndcg10": float(np.maximum(title_ndcg, abstract_ndcg).mean()),
        "oracle_choose_abstract_fraction": float((gain > 0).mean()),
        "pre_above_fixed": sum(v["routed_ndcg10"] > best_fixed + margin for v in pre_rows.values()),
        "score_above_fixed": sum(v["routed_ndcg10"] > best_fixed + margin for v in score_rows.values()),
        "pre_degenerate": sum(v["degenerate"] for v in pre_rows.values()),
        "score_degenerate": sum(v["degenerate"] for v in score_rows.values()),
        "corpus_control": pc,
        "score_control": sc,
        "score_control_inference": grouped_inference(
            score_routed - fixed_scores, groups, draws=args.inference_draws),
    }
    payload = {
        "protocol": {
            "dataset": "BEIR SciFact, train+test qrels as one cross-fitted analysis set",
            "sources": ["title BM25", "abstract BM25"],
            "metric": "nDCG@10",
            "folds": "5-fold relevance-component-disjoint, averaged over seeds " + str(seeds),
            "selector": "OOF ridge predicts abstract-minus-title gain; choose abstract iff positive",
            "above_fixed_margin": margin,
        },
        "summary": summary,
        "pre_retrieval": pre_rows,
        "score_only": score_rows,
    }
    os.makedirs(ABL, exist_ok=True)
    with open(os.path.join(ABL, "scifact_source_replication.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    md = markdown(payload)
    with open(os.path.join(ABL, "scifact_source_replication.md"), "w") as handle:
        handle.write(md)
    print(md, flush=True)


if __name__ == "__main__":
    main()
