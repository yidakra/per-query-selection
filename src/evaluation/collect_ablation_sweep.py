"""Collect the representative MultiVENT size/frame ablation sweep into tables.

For each variant tag it loads the 4 cached text components, fuses them with inverse-entropy
(the text-only "Q2E - Video" quantity), and reports NDCG@10. The 38B/70B/Funiform_16 anchor
point comes from the already-completed `multivent_textonly_noASR` run.

Usage: python src/evaluation/collect_ablation_sweep.py
Writes results/ablations/sweep_{vlm,llm,frame}.md and sweep_all.json. Tags with no
metrics.json yet are reported as 'pending' so this can be run mid-sweep.
"""
import os, sys, json
import torch, torch.nn.functional as F
from collections import defaultdict

OFF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "q2e_official"))
sys.path.insert(0, OFF)
from datasets import load_from_disk  # noqa: E402
os.chdir(OFF)
from src.eval.evaluation import retrieval_score  # noqa: E402
from src.eval.fusion_score import fusion_inverse_entropy  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
RUNS = f"{_ROOT}/runs"
OUT = f"{_ROOT}/results/ablations"
TEXT4 = ["query_vs_captions", "prequel_vs_captions", "during_vs_captions", "sequel_vs_captions"]

# tag -> (dataset local name, sweep, x-axis label)
ANCHOR = "multivent_textonly_noASR"
ANCHOR_DS = "data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"
VARIANTS = {
  "mv_vlm_internvl1b_noASR": ("Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_1B_Funiform_16_noASR", "vlm", "InternVL-1B"),
  "mv_vlm_internvl8b_noASR": ("Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_8B_Funiform_16_noASR", "vlm", "InternVL-8B"),
  ANCHOR:                    (None, "vlm", "InternVL-38B"),
  "mv_llm_llama1b_noASR":    ("Q2E_MultiVENT_LLAMA_1B_InternVL_38B_Funiform_16_noASR", "llm", "LLaMA-1B"),
  "mv_llm_llama8b_noASR":    ("Q2E_MultiVENT_LLAMA_8B_InternVL_38B_Funiform_16_noASR", "llm", "LLaMA-8B"),
  # anchor also serves LLM sweep as LLaMA-3.3-70B and frame sweep as Funiform-16
  "mv_frame_f2_noASR":       ("Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_2_noASR", "frame", "Funiform-2"),
  "mv_frame_f64_noASR":      ("Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_64_noASR", "frame", "Funiform-64"),
}
# anchor participates in all three sweeps
ANCHOR_IN = {"vlm": "InternVL-38B", "llm": "LLaMA-3.3-70B", "frame": "Funiform-16"}
# sort keys for x-axis
ORDER = {
  "vlm": ["InternVL-1B", "InternVL-8B", "InternVL-38B"],
  "llm": ["LLaMA-1B", "LLaMA-8B", "LLaMA-3.3-70B"],
  "frame": ["Funiform-2", "Funiform-16", "Funiform-64"],
}


def target_for(ds_name):
    ds = load_from_disk(f"data/MultiVENT/{ds_name}")
    queries, vids, sq, sv = [], [], set(), set()
    for r in ds:
        if r["query"] not in sq: sq.add(r["query"]); queries.append(r["query"])
        if r["video_id"] not in sv: sv.add(r["video_id"]); vids.append(r["video_id"])
    vindex = {v: i for i, v in enumerate(vids)}
    q2vi = defaultdict(list)
    for r in ds: q2vi[r["query"]].append(vindex[r["video_id"]])
    t = torch.zeros((len(queries), len(vids)))
    for i, q in enumerate(queries): t[i, q2vi[q]] = 1
    return t.bool()


def ndcg_for(tag, ds_name):
    cdir = os.path.join(RUNS, tag, "cache")
    if not all(os.path.exists(os.path.join(cdir, f"{p}.pt")) for p in TEXT4):
        return None
    comps = {p: torch.load(os.path.join(cdir, f"{p}.pt")) for p in TEXT4}
    tgt = target_for(ds_name)
    res = [F.softmax(comps[p], dim=0) for p in TEXT4]
    Q, V = res[0].shape
    sm = fusion_inverse_entropy(None, res, TEXT4, range(Q), range(V))
    return retrieval_score(sm, tgt)["NDCG"]


def main():
    # anchor NDCG (also its dataset name for target)
    anchor_ndcg = ndcg_for(ANCHOR, os.path.basename(ANCHOR_DS))
    points = defaultdict(dict)  # sweep -> {label: ndcg or None}
    for sweep, label in ANCHOR_IN.items():
        points[sweep][label] = anchor_ndcg
    for tag, (ds_name, sweep, label) in VARIANTS.items():
        if tag == ANCHOR:
            continue
        points[sweep][label] = ndcg_for(tag, ds_name)

    allout = {}
    os.makedirs(OUT, exist_ok=True)
    for sweep in ["vlm", "llm", "frame"]:
        rows = []
        for label in ORDER[sweep]:
            v = points[sweep].get(label)
            rows.append((label, v))
        allout[sweep] = {lbl: (round(v, 4) if v is not None else None) for lbl, v in rows}
        title = {"vlm": "VLM size (InternVL)", "llm": "LLM size (LLaMA)", "frame": "Frame count (Funiform)"}[sweep]
        md = [f"## {title} ablation — MultiVENT text-only (Q2E − Video), noASR NDCG@10\n",
              "| variant | NDCG@10 |", "|---|---:|"]
        for lbl, v in rows:
            md.append(f"| {lbl} | {'%.2f' % v if v is not None else '_pending_'} |")
        open(os.path.join(OUT, f"sweep_{sweep}.md"), "w").write("\n".join(md) + "\n")
        print("\n".join(md)); print()
    json.dump(allout, open(os.path.join(OUT, "sweep_all.json"), "w"), indent=2)
    print("[written]", os.path.join(OUT, "sweep_all.json"))


if __name__ == "__main__":
    main()
