"""Compare the original 14B nugget assignments with an independent judge.

Gold nuggets and generated reports remain fixed. Only the support/partial/not-support assignment step
is repeated, which isolates assignment sensitivity from generation variance. The secondary model can
be either a same-family capacity check (Qwen2.5 7B) or a cross-family judge (Gemma 3). The paper's
downstream claim depends on routed versus best-fixed differences under both grounding rules, so those
are the only two policies compared here.
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAG = os.path.join(ROOT, "results", "ablations", "rag")
METRICS = ("vital", "strict_vital", "all", "strict_all")


def nugget_score(nuggets):
    vital = [n for n in nuggets if n["importance"] == "vital"]
    supported_vital = sum(n["assignment"] == "support" for n in vital)
    supported_all = sum(n["assignment"] == "support" for n in nuggets)
    partial_vital = supported_vital + 0.5 * sum(
        n["assignment"] == "partial_support" for n in vital
    )
    partial_all = supported_all + 0.5 * sum(
        n["assignment"] == "partial_support" for n in nuggets
    )
    return {
        "strict_vital": supported_vital / len(vital) if vital else 0.0,
        "strict_all": supported_all / len(nuggets) if nuggets else 0.0,
        "vital": partial_vital / len(vital) if vital else 0.0,
        "all": partial_all / len(nuggets) if nuggets else 0.0,
    }


def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["policy"] in {"routed", "bestfixed"}:
                rows[(row["qid"], row["policy"])] = row["nuggets"]
    return rows


def paired_p(values, samples=10000):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(0)
    exceed = 0
    for _ in range(samples):
        null = float((values * rng.choice((-1.0, 1.0), size=len(values))).mean())
        exceed += abs(null) >= observed
    return (exceed + 1) / (samples + 1)


def compare_one(evidence, tag, model):
    primary = load(os.path.join(RAG, f"assigned_n400_{evidence}.jsonl"))
    secondary = load(os.path.join(RAG, f"assigned_n400_{evidence}_{tag}.jsonl"))
    qids = sorted(
        q for q in {key[0] for key in primary} & {key[0] for key in secondary}
        if all((q, policy) in primary and (q, policy) in secondary
               for policy in ("routed", "bestfixed"))
    )
    if not qids:
        raise RuntimeError(f"no complete paired assignments for evidence={evidence}")

    scores = {judge: {policy: {} for policy in ("routed", "bestfixed")}
              for judge in ("qwen14b", tag)}
    labels_primary, labels_secondary = [], []
    for qid in qids:
        for policy in ("routed", "bestfixed"):
            a = primary[(qid, policy)]
            b = secondary[(qid, policy)]
            scores["qwen14b"][policy][qid] = nugget_score(a)
            scores[tag][policy][qid] = nugget_score(b)
            by_text = {n["text"]: n["assignment"] for n in b}
            for nugget in a:
                if nugget["text"] in by_text:
                    labels_primary.append(nugget["assignment"])
                    labels_secondary.append(by_text[nugget["text"]])

    result = {
        "evidence": evidence,
        "n_queries": len(qids),
        "secondary_model": model,
        "assignment_exact_agreement": float(np.mean(np.asarray(labels_primary) == labels_secondary)),
        "assignment_cohen_kappa": float(cohen_kappa_score(labels_primary, labels_secondary)),
        "judges": {},
    }
    for judge in ("qwen14b", tag):
        jr = {"policies": {}, "routed_minus_bestfixed": {}}
        for policy in ("routed", "bestfixed"):
            jr["policies"][policy] = {
                metric: float(np.mean([scores[judge][policy][q][metric] for q in qids]))
                for metric in METRICS
            }
        for metric in METRICS:
            delta = np.asarray([
                scores[judge]["routed"][q][metric] - scores[judge]["bestfixed"][q][metric]
                for q in qids
            ])
            jr["routed_minus_bestfixed"][metric] = {
                "delta": float(delta.mean()),
                "paired_permutation_p": float(paired_p(delta)),
                "win": int((delta > 0).sum()),
                "tie": int((delta == 0).sum()),
                "loss": int((delta < 0).sum()),
            }
        result["judges"][judge] = jr

    for policy in ("routed", "bestfixed"):
        for metric in METRICS:
            x = [scores["qwen14b"][policy][q][metric] for q in qids]
            y = [scores[tag][policy][q][metric] for q in qids]
            result.setdefault("per_query_judge_correlation", {}).setdefault(policy, {})[metric] = (
                float(pearsonr(x, y).statistic) if np.std(x) and np.std(y) else None
            )
    return result


def markdown(results, tag):
    lines = [
        "# Second-judge comparison",
        "",
        "Gold nuggets and reports are fixed; only nugget-support assignment changes.",
        "",
        "| evidence | judge | routed vital | fixed vital | delta | paired p | exact label agreement | kappa |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        for judge in ("qwen14b", tag):
            values = row["judges"][judge]
            delta = values["routed_minus_bestfixed"]["vital"]
            agreement = row["assignment_exact_agreement"] if judge == tag else None
            kappa = row["assignment_cohen_kappa"] if judge == tag else None
            lines.append(
                f"| {row['evidence']} | {judge} | {values['policies']['routed']['vital']:.4f} | "
                f"{values['policies']['bestfixed']['vital']:.4f} | {delta['delta']:+.4f} | "
                f"{delta['paired_permutation_p']:.4f} | "
                f"{agreement:.3f} | {kappa:.3f} |" if agreement is not None else
                f"| {row['evidence']} | {judge} | {values['policies']['routed']['vital']:.4f} | "
                f"{values['policies']['bestfixed']['vital']:.4f} | {delta['delta']:+.4f} | "
                f"{delta['paired_permutation_p']:.4f} | -- | -- |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-tag", default="qwen7b")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--output-stem", default=None,
                        help="artifact basename; defaults to the legacy name for qwen7b and a "
                             "judge-tagged name for every other judge")
    args = parser.parse_args()
    results = [compare_one(evidence, args.judge_tag, args.model) for evidence in ("all", "own")]
    payload = {"secondary_model": args.model, "judge_tag": args.judge_tag, "results": results}
    stem = args.output_stem or (
        "mv2_rag_judge_comparison" if args.judge_tag == "qwen7b"
        else f"mv2_rag_judge_comparison_{args.judge_tag}"
    )
    json_path = os.path.join(ABL := os.path.dirname(RAG), stem + ".json")
    md_path = os.path.join(ABL, stem + ".md")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    text = markdown(results, args.judge_tag)
    with open(md_path, "w") as f:
        f.write(text)
    print(text, end="")
    print(f"wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
