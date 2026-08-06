"""Replicate source selection on MSR-VTT-1kA using already-cached retrieval components.

The MultiVENT 2.0 paper experiment asks whether a predictor can decide when to add a document-side
evidence source. This script asks two binary versions on a second collection: keep the visual run or
fuse it with the query-to-caption run (the exact escalation analogue), and choose visual-only versus
caption-only (the direct source-selection analogue). It evaluates the same eleven analytic
pre-retrieval predictors and ten score-only predictors, plus a multi-feature positive control. No
retrieval or model inference is run; the two MSR-VTT encoders and ASR/no-ASR conditions were reproduced
and cached previously.

MSR-VTT has one query phrasing per information need, so there is no event-duplicate grouping analogue.
All predictor calibration remains out of fold under shuffled five-fold CV.
"""

import json
import os
import sys

import numpy as np
from datasets import load_from_disk
from scipy.stats import kendalltau
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "evaluation"))

from mv2_qpp_predictors import (PRE_RETRIEVAL, SCORE_ONLY, Index,  # noqa: E402
                                pre_retrieval_suite, score_only_suite)
from oracle_router_headroom import LADDER, fuse, per_query          # noqa: E402
from router_hetero import comps_msrvtt                              # noqa: E402
from retrieve import FEATURE_ORDER, conf_features                   # noqa: E402

ABL = os.path.join(ROOT, "results", "ablations")
DATASET = os.path.join(
    ROOT, "data", "MSR-VTT-1kA",
    "Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_{setting}",
)
CELLS = [(encoder, setting) for encoder in ("multiclip", "internvideo2")
         for setting in ("noASR", "ASR")]
TASKS = {
    "add-caption": LADDER["B_noevents"],
    "choose-source": ["query_vs_captions"],
}


def model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 12))),
    ])


def document_texts(setting):
    """One lexical surrogate per video, matching the caption source used by the added channel."""
    ds = load_from_disk(DATASET.format(setting=setting))
    by_video = {}
    for row in ds:
        parts = []
        if isinstance(row.get("frame2video_caption"), str):
            parts.append(row["frame2video_caption"])
        parts.extend(x for x in (row.get("frame_captions") or []) if isinstance(x, str))
        if setting == "ASR" and isinstance(row.get("asr"), dict):
            for key in ("translated", "translated_llm", "original"):
                value = row["asr"].get(key)
                if isinstance(value, str) and value and value != "Not Generated":
                    parts.append(value)
                    break
        by_video.setdefault(row["video_id"], " ".join(parts))
    return list(by_video.values())


def overlap_features(visual, caption, depth):
    a = set(np.argsort(-visual, kind="stable")[:depth])
    b = set(np.argsort(-caption, kind="stable")[:depth])
    return len(a & b) / max(1, len(a | b))


def route(raw, gain, cheap, expensive, cv):
    raw = np.asarray(raw, dtype=float)
    tau = 0.0 if np.allclose(raw.std(), 0) else float(kendalltau(raw, gain).statistic)
    if np.allclose(raw.std(), 0):
        pred = np.zeros(len(raw))
    else:
        pred = cross_val_predict(model(), raw.reshape(-1, 1), gain, cv=cv)
    decision = pred > 0
    return {
        "tau": tau,
        "routed_ndcg10": float(np.where(decision, expensive, cheap).mean()),
        "frac_escalated": float(decision.mean()),
        "degenerate": bool(decision.all() or not decision.any()),
    }


def one_cell(encoder, setting, index, task):
    queries, target, components = comps_msrvtt(encoder, setting)
    visual_scores = components["query_vs_video"]
    caption_scores = components["query_vs_captions"]
    cheap_run = fuse(components, LADDER["A_visual"])
    expensive_run = fuse(components, TASKS[task])
    cheap = per_query(cheap_run, target)[0].numpy()
    expensive = per_query(expensive_run, target)[0].numpy()
    gain = expensive - cheap
    cv = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(queries))))

    pre_raw = {name: [] for name in PRE_RETRIEVAL}
    score_raw = {name: [] for name in SCORE_ONLY}
    control_features = []
    for i, query in enumerate(queries):
        tokens = query.lower().split()
        pre = pre_retrieval_suite(tokens, index)
        score = score_only_suite(visual_scores[i].numpy(), len(tokens))
        for name in PRE_RETRIEVAL:
            pre_raw[name].append(pre[name])
        for name in SCORE_ONLY:
            score_raw[name].append(score[name])

        visual = visual_scores[i].numpy()
        caption = caption_scores[i].numpy()
        vf = conf_features(np.sort(visual)[::-1])
        cf = conf_features(np.sort(caption)[::-1])
        control_features.append(
            [vf[name] for name in FEATURE_ORDER]
            + [cf[name] for name in FEATURE_ORDER]
            + [overlap_features(visual, caption, 10), overlap_features(visual, caption, 100)]
        )

    pre = {name: route(values, gain, cheap, expensive, cv)
           for name, values in pre_raw.items()}
    score = {name: route(values, gain, cheap, expensive, cv)
             for name, values in score_raw.items()}
    control_pred = cross_val_predict(model(), np.asarray(control_features), gain, cv=cv)
    control_decision = control_pred > 0
    fixed = max(float(cheap.mean()), float(expensive.mean()))
    margin = 5e-4

    def summary(rows):
        return {
            "above_fixed": sum(v["routed_ndcg10"] > fixed + margin for v in rows.values()),
            "n": len(rows),
            "degenerate": sum(v["degenerate"] for v in rows.values()),
            "max_abs_tau": max(abs(v["tau"]) for v in rows.values()),
        }

    return {
        "encoder": encoder,
        "setting": setting,
        "task": task,
        "n_queries": len(queries),
        "cheap_visual": float(cheap.mean()),
        "option_b": float(expensive.mean()),
        "best_fixed": fixed,
        "gain_mean": float(gain.mean()),
        "gain_sd": float(gain.std()),
        "frac_helped": float((gain > 1e-9).mean()),
        "frac_hurt": float((gain < -1e-9).mean()),
        "pre": pre,
        "score": score,
        "pre_summary": summary(pre),
        "score_summary": summary(score),
        "control": {
            "tau": float(kendalltau(control_pred, gain).statistic),
            "routed_ndcg10": float(np.where(control_decision, expensive, cheap).mean()),
            "frac_escalated": float(control_decision.mean()),
        },
        "oracle": {
            "routed_ndcg10": float(np.where(gain > 0, expensive, cheap).mean()),
            "frac_escalated": float((gain > 0).mean()),
        },
    }


def markdown(results):
    lines = [
        "# MSR-VTT source-selection replication",
        "",
        "Two tasks: add-caption is visual versus visual+caption fusion; choose-source is visual-only versus caption-only. All calibration is five-fold out of fold; no test-label operating-fraction sweep.",
        "",
        "| task / encoder / evidence | option A | option B | control | oracle | pre above / deg. | score above / deg. | max |tau| pre / score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in results.items():
        ps, ss = row["pre_summary"], row["score_summary"]
        lines.append(
            f"| {key} | {row['cheap_visual']:.4f} | {row['option_b']:.4f} | "
            f"{row['control']['routed_ndcg10']:.4f} | {row['oracle']['routed_ndcg10']:.4f} | "
            f"{ps['above_fixed']}/{ps['n']} / {ps['degenerate']} | "
            f"{ss['above_fixed']}/{ss['n']} / {ss['degenerate']} | "
            f"{ps['max_abs_tau']:.3f} / {ss['max_abs_tau']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    indices = {}
    results = {}
    for encoder, setting in CELLS:
        if setting not in indices:
            texts = document_texts(setting)
            indices[setting] = Index(texts)
            print(f"{setting}: lexical surrogate has {indices[setting].n_docs} documents", flush=True)
        for task in TASKS:
            key = f"{task}/{encoder}/{setting}"
            results[key] = one_cell(encoder, setting, indices[setting], task)
            row = results[key]
            print(f"{key}: fixed={row['best_fixed']:.4f}, "
                  f"control={row['control']['routed_ndcg10']:.4f}, "
                  f"pre={row['pre_summary']['above_fixed']}/11, "
                  f"score={row['score_summary']['above_fixed']}/10", flush=True)

    payload = {
        "protocol": "five-fold OOF; visual versus visual+caption and visual versus caption; 5e-4 above-fixed margin",
        "results": results,
    }
    json_path = os.path.join(ABL, "mv2_msrvtt_source_replication.json")
    md_path = os.path.join(ABL, "mv2_msrvtt_source_replication.md")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(md_path, "w") as f:
        f.write(markdown(results))
    print(markdown(results), end="")
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
