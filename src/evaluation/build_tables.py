"""Build reproduced-vs-reported comparison tables from runs/<tag>/metrics.json.

Selects the specific component-subsets that correspond to the paper's Table 1 rows and
Table 5 component ablation, and writes:
  results/main_tables/reproduced_vs_reported.csv
  results/main_tables/reproduced_vs_reported.md
"""
import json, os, csv, glob

RUNS = "/home/ubuntu/q2e_repro/runs"
OUT = "/home/ubuntu/q2e_repro/results/main_tables"
os.makedirs(OUT, exist_ok=True)

ALL5 = ["query_vs_video", "query_vs_captions", "prequel_vs_captions",
        "during_vs_captions", "sequel_vs_captions"]
TEXT4 = ALL5[1:]

# Paper targets (NDCG, R1, R5, R10, MRR, MAP) — from reproduction_spec.md Table 1.
PAPER = {
 ("MSR-VTT-1kA","multiclip","baseline"):     dict(NDCG=59.72,R1=43.52,R5=69.05,R10=76.88,MRR=0.54,MAP=54.27),
 ("MSR-VTT-1kA","multiclip","Q2E"):          dict(NDCG=61.51,R1=44.52,R5=71.26,R10=79.40,MRR=0.56,MAP=55.84),
 ("MSR-VTT-1kA","multiclip","Q2E+ASR"):      dict(NDCG=63.59,R1=46.23,R5=73.37,R10=81.71,MRR=0.58,MAP=57.83),
 ("MSR-VTT-1kA","internvideo2","baseline"):  dict(NDCG=66.07,R1=52.56,R5=73.07,R10=80.10,MRR=0.62,MAP=61.62),
 ("MSR-VTT-1kA","internvideo2","Q2E"):       dict(NDCG=67.16,R1=53.47,R5=73.57,R10=82.11,MRR=0.62,MAP=62.47),
 ("MSR-VTT-1kA","internvideo2","Q2E+ASR"):   dict(NDCG=69.53,R1=56.28,R5=76.58,R10=83.72,MRR=0.65,MAP=65.06),
 ("MultiVENT","multiclip","baseline"):       dict(NDCG=75.34,R1=9.83,R5=44.32,R10=70.82,MRR=0.92,MAP=86.33),
 ("MultiVENT","multiclip","Q2E"):            dict(NDCG=80.04,R1=10.24,R5=46.71,R10=75.76,MRR=0.95,MAP=89.42),
 ("MultiVENT","multiclip","Q2E+ASR"):        dict(NDCG=83.24,R1=10.32,R5=49.00,R10=79.60,MRR=0.95,MAP=91.20),
 ("MultiVENT","internvideo2","baseline"):    dict(NDCG=50.43,R1=5.60,R5=28.92,R10=49.12,MRR=0.68,MAP=63.77),
 ("MultiVENT","internvideo2","Q2E"):         dict(NDCG=69.15,R1=9.54,R5=40.88,R10=63.40,MRR=0.92,MAP=83.96),
 ("MultiVENT","internvideo2","Q2E+ASR"):     dict(NDCG=76.10,R1=10.24,R5=44.94,R10=70.79,MRR=0.95,MAP=88.09),
}
# Table 5 component ablation (MultiVENT multiclip) NDCG no-audio / audio
PAPER_ABL = {
 ("Q2E Full","noASR"):80.04, ("Q2E Full","ASR"):83.24,
 ("Q2E -Video","noASR"):64.83, ("Q2E -Video","ASR"):73.96,
 ("Q2E -Query","noASR"):78.78, ("Q2E -Query","ASR"):81.54,
 ("Q2E -Events","noASR"):79.02, ("Q2E -Events","ASR"):81.75,
}


def load(tag):
    p = os.path.join(RUNS, tag, "metrics.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def pick(data, aggregation, params):
    if data is None:
        return None
    ps = sorted(params)
    for r in data["records"]:
        if r["aggregation"] == aggregation and sorted(r["params"]) == ps:
            return r["metrics"]
    return None


def fmt(v):
    return "" if v is None else f"{v:.2f}"


def main():
    rows = []
    # dataset, encoder, tag prefix
    for dataset, dtag in [("MSR-VTT-1kA","msrvtt"), ("MultiVENT","multivent")]:
        for enc in ["multiclip","internvideo2"]:
            no = load(f"{dtag}_{enc}_noASR")
            asr = load(f"{dtag}_{enc}_ASR")
            avail_no = no["meta"]["available_components"] if no else []
            has_video = "query_vs_video" in avail_no
            configs = {
              # Paper's video-only baseline is the single query_vs_video component run
              # through the SAME softmax->min-max normalization pipeline as fused Q2E
              # (not the raw dot-products). For a single component all fusion operators
              # are numerically identical, so inv_entropy == mean == max here. Matches
              # paper R@5 to 2 decimals (raw does not).
              "baseline": (no, "inv_entropy", ["query_vs_video"]) if has_video else (None,None,None),
              "Q2E": (no, "inv_entropy", ALL5 if has_video else TEXT4),
              "Q2E+ASR": (asr, "inv_entropy", ALL5 if has_video else TEXT4),
            }
            for setting,(d,agg,params) in configs.items():
                paper = PAPER.get((dataset,enc,setting))
                if d is None:
                    rep = None
                else:
                    rep = pick(d, agg, params)
                row = {"dataset":dataset,"encoder":enc,"setting":setting,
                       "components":"+".join(p.replace("_vs_captions","").replace("query_vs_video","video") for p in params) if params else "",
                       "no_video_mode": (not has_video)}
                for m in ["NDCG","R1","R5","R10","MRR","MAP"]:
                    row[f"rep_{m}"] = None if rep is None else rep.get(m)
                    row[f"paper_{m}"] = None if paper is None else paper.get(m)
                    if rep is not None and paper is not None and paper.get(m) is not None:
                        row[f"gap_{m}"] = round(rep.get(m) - paper.get(m), 2)
                    else:
                        row[f"gap_{m}"] = None
                rows.append(row)

    # write CSV
    cols = ["dataset","encoder","setting","components","no_video_mode",
            "rep_NDCG","paper_NDCG","gap_NDCG","rep_R1","paper_R1","gap_R1",
            "rep_R5","paper_R5","gap_R5","rep_R10","paper_R10","gap_R10",
            "rep_MRR","paper_MRR","gap_MRR","rep_MAP","paper_MAP","gap_MAP"]
    with open(os.path.join(OUT,"reproduced_vs_reported.csv"),"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)

    # markdown (NDCG focus)
    lines = ["# Reproduced vs Reported (headline: NDCG@10, torchmetrics)","",
             "| Dataset | Encoder | Setting | rep NDCG | paper NDCG | ΔNDCG | rep R@1 | paper R@1 | rep R@10 | paper R@10 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['encoder']} | {r['setting']} | "
                     f"{fmt(r['rep_NDCG'])} | {fmt(r['paper_NDCG'])} | {fmt(r['gap_NDCG'])} | "
                     f"{fmt(r['rep_R1'])} | {fmt(r['paper_R1'])} | {fmt(r['rep_R10'])} | {fmt(r['paper_R10'])} |")

    # Component ablation (MultiVENT multiclip)
    lines += ["","## Component ablation — MultiVENT / MultiCLIP (NDCG)","",
              "| Config | components | rep noASR | paper noASR | rep ASR | paper ASR |",
              "|---|---|---|---|---|---|"]
    abl_map = {
      "Q2E Full": ALL5,
      "Q2E -Video": TEXT4,
      "Q2E -Query": [p for p in ALL5 if p not in ("query_vs_video","query_vs_captions")],
      "Q2E -Events": ["query_vs_video","query_vs_captions"],
    }
    no = load("multivent_multiclip_noASR"); asr = load("multivent_multiclip_ASR")
    for cfg,params in abl_map.items():
        rn = pick(no,"inv_entropy",params); ra = pick(asr,"inv_entropy",params)
        lines.append(f"| {cfg} | {'+'.join(p.replace('_vs_captions','').replace('query_vs_video','video') for p in params)} | "
                     f"{fmt(rn['NDCG'] if rn else None)} | {fmt(PAPER_ABL.get((cfg,'noASR')))} | "
                     f"{fmt(ra['NDCG'] if ra else None)} | {fmt(PAPER_ABL.get((cfg,'ASR')))} |")

    open(os.path.join(OUT,"reproduced_vs_reported.md"),"w").write("\n".join(lines)+"\n")
    print("\n".join(lines))
    print(f"\n[written] {OUT}/reproduced_vs_reported.{{csv,md}}")


if __name__ == "__main__":
    main()
