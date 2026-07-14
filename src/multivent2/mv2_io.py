"""Load MultiVENT 2.0 eval files.

Formats, taken from the HF dataset `hltcoe/MultiVENT2.0`:
- queries CSV: columns `Query_id, query`.
- qrels JSONL: one line per judged (query, video), with `query_id`, `doc_id`, `relevance` (graded
  0-3), plus `video_language` / `video_type` / `video_modality` / `query_event_type` used by the
  official per-split scores.
- run JSON: `{query_id: {doc_id: score}}`.
"""
import csv
import json
import jsonlines


def load_queries(path):
    with open(path) as f:
        return {row["Query_id"]: row["query"] for row in csv.DictReader(f)}


def load_qrels(path):
    """-> {query_id: {doc_id: relevance}}. Also returns the per-line split fields for later use."""
    qrels, meta = {}, {}
    with jsonlines.open(path) as reader:
        for line in reader:
            qid, did = line["query_id"], line["doc_id"]
            qrels.setdefault(qid, {})[did] = int(line["relevance"])
            meta.setdefault(qid, {})[did] = {k: line.get(k, "") for k in
                                             ("video_language", "video_type", "video_modality",
                                              "query_event_type")}
    return qrels, meta


def load_run(path):
    """-> {query_id: {doc_id: score}}."""
    with open(path) as f:
        return json.load(f)
