"""Table 1: every predictor as a selector, laid out the way Arabzadeh et al. lay theirs out.

Their Table 1 (SIGIR '26) reports, for each QPP method, the end-to-end performance of executing the
option that method selected -- grouped Original / Pre-retrieval / Post-retrieval / Oracle, with a rule
that reads at a glance: underline anything that beats the Original row, bold the best in each section.
Their pre-retrieval block is almost entirely underlined. That is their headline.

This emits the same shape for our selection problem, where the options are evidence channels rather
than query variants. The point of matching their layout exactly is that the comparison then needs no
prose: the pre-retrieval block that is underlined in their table is bare in ours.

Reads what is already computed and writes the table. It runs no experiments, so it is safe to re-run
and cheap; the columns it cannot fill yet are marked rather than quietly dropped (see COVERAGE below).

  python src/multivent2/mv2_table1.py [--tag _grouped]
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
ABL = os.path.join(_ROOT, "results", "ablations")

# their method order, so the two tables can be read side by side. Names on the left are theirs; ours
# on the right, None where we have no equivalent and the row is reported as a gap rather than omitted.
PRE = [("IDF_avg", "IDF_avg"), ("IDF_max", "IDF_max"), ("IDF_sum", "IDF_sum"), ("IDF_std", "IDF_std"),
       ("ICTF_avg", "avgICTF"), ("SCQ_avg", "SCQ_avg"), ("SCQ_max", "SCQ_max"), ("SCQ_sum", "SCQ_sum"),
       ("SCS_apx", "SCS_1"), ("SCS_full", "SCS_2"),
       ("QL", "QL"), ("QSD_pre", "QSD_PRE"),
       # DM appears in their Table 1 and nowhere in the reference repository, and its row is
       # numerically identical to their Original row in all eight columns. Left unresolved rather
       # than guessed at; one for the authors.
       ("DM", None)]
POST = [("RSD", "RSD"), ("clarity", "CLARITY_NA"), ("NQC", "NQC"), ("NQC_norm", "NQC_norm"),
        ("sigma_max", "sigma_max"), ("sigma_0.5", "sigma_x0.5"), ("SMV", "SMV"), ("SMV_norm", "SMV_norm"),
        ("WIG", "WIG"), ("WIG_norm", "WIG_norm"), ("max", "max"),
        ("QSD_post", None), ("BERTQPP", "BERTQPP"), ("BERTQPP_bi", "BERTQPP_BI")]

CELLS = ["ASR-shipped", "ASR-dense", "OCR"]

# what each column costs to fill. Recall@100 needs the per-channel A/B runs rebuilt and re-scored
# (CPU, cheap). The nugget columns look like they need a judge pass per selected run, which at ~19 h on
# one GPU for a five-policy arm would put a row-per-predictor version out of reach. They do not: a
# predictor row executes run A or run B per query, and a report's coverage is a property of the run it
# was written from, so the whole column mixes from two judged runs per cell (mv2_table1_nuggets.py).
# Run A is visual alone and shared by all three cells, so four judged runs fill every row.
COVERAGE = {"nDCG@10": "complete", "tau": "complete",
            "Recall@100": "complete -- mv2_recall_sidecar.py reproduces all 3 cells' stored nDCG "
                          "exactly, and QSD_pre / BERTQPP are re-scored from their own predictions",
            "N_all": "complete -- mixed per query from the judged A and B runs of each cell, over the "
                     "395 queries of the RAG arm rather than all 2,546; both endpoints reproduce the "
                     "judged runs exactly or the mix aborts",
            "N_strict": "complete -- same mix, strict support (see mv2_rag_nuggets.py for the "
                        "difference between all and strict)"}


def load(tag):
    with open(os.path.join(ABL, f"mv2_qpp_table{tag}.json")) as f:
        table = json.load(f)
    # Arabzadeh et al. give BERT-QPP in both flavours and Jingfen asked for both. They fail differently,
    # which is the reason to carry two rows: the cross-encoder orders well and cannot decide, the
    # bi-encoder cannot order at all.
    bert, bert_bi = {}, {}
    for suffix, dest in (("", bert), ("_bi", bert_bi)):
        p = os.path.join(ABL, f"mv2_bertqpp{suffix}{tag}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            raw = json.load(f)
        # that file keys its cells differently from the main table
        for k, cell in (("asr_shipped", "ASR-shipped"), ("asr_dense", "ASR-dense"), ("ocr", "OCR")):
            if k in raw:
                dest[cell] = (raw[k]["routed_ndcg10"], raw[k]["tau"],
                              raw[k].get("routed_recall100"), raw[k].get("frac_escalated"))
    # QSD_pre reports at the best k for the split, per qpp_baselines.md: k=5 under a plain split where
    # the duplicate IS the answer, k=100 under event grouping where many neighbours are needed to
    # average the noise out. That reversal is itself the leakage evidence, so the k is not a free knob.
    qsd, qsd_k = {}, "5_inv_dist" if not tag.endswith("_grouped") else "100_inv_dist"
    p = os.path.join(ABL, f"mv2_qsd{tag}.json")
    if os.path.exists(p):
        with open(p) as f:
            raw = json.load(f)
        for k, cell in (("asr_shipped", "ASR-shipped"), ("asr_dense", "ASR-dense"), ("ocr", "OCR")):
            if k in raw and qsd_k in raw[k]["k"]:
                v = raw[k]["k"][qsd_k]
                qsd[cell] = (v["routed_ndcg10"], v["tau"], v.get("routed_recall100"),
                             v.get("frac_escalated"))
    return table, bert, bert_bi, qsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_grouped", help="'_grouped' for event-grouped folds")
    a = ap.parse_args()
    table, bert, bert_bi, qsd = load(a.tag)
    cells = [c for c in CELLS if c in table]

    # The row every other row is measured against. Their Original is the unmodified query: the default
    # action when you do no selection. Ours is therefore the best FIXED policy, not the cheap channel.
    # This matters more than it sounds. Against the cheap channel, a predictor that degenerates to
    # "always fuse" gets underlined in the ASR-dense cell purely because fusion beats visual-only --
    # the mark would be reporting that fusion works, not that the predictor selected anything. Against
    # the best fixed policy a degenerate predictor scores exactly zero improvement, which is the truth.
    original = {c: max(table[c]["cheap"], table[c]["uniform"]) for c in cells}

    def get(section, ours_name, cell):
        if ours_name is None:
            return None
        if ours_name == "CLARITY_NA":
            return "n/a"
        if ours_name == "BERTQPP":
            return bert.get(cell)
        if ours_name == "BERTQPP_BI":
            return bert_bi.get(cell)
        if ours_name == "QSD_PRE":
            return qsd.get(cell)
        v = table[cell].get(section, {}).get(ours_name)
        return tuple(v) if v else None

    # Nugget coverage per row, mixed from two judged runs per cell by mv2_table1_nuggets.py. Absent
    # until that has run, in which case the two columns stay empty rather than being dropped.
    nug = {}
    p = os.path.join(ABL, f"mv2_table1_nuggets{a.tag}.json")
    if os.path.exists(p):
        nug = json.load(open(p))

    def nug_for(key, c):
        r = nug.get(c, {}).get("rows", {}).get(key)
        return (r["all"], r["strict_all"]) if r else None

    rows = []
    def orec(c):
        rc, ru = table[c].get("recall_cheap"), table[c].get("recall_uniform")
        if rc is None:
            return None
        return ru if table[c]["uniform"] >= table[c]["cheap"] else rc
    def okey(c):
        return "_B_fused" if table[c]["uniform"] >= table[c]["cheap"] else "_A_visual"
    rows.append(("Original", "best fixed policy (no selection)",
                 {c: (original[c], None, orec(c)) for c in cells}, {c: okey(c) for c in cells}))
    rows.append(("", "visual only", {c: (table[c]["cheap"], None, table[c].get("recall_cheap"))
                                     for c in cells}, "_A_visual"))
    rows.append(("", "uniform fusion (best w)",
                 {c: (table[c]["uniform"], None, table[c].get("recall_uniform")) for c in cells},
                 "_B_fused"))
    for i, (their, ours) in enumerate(PRE):
        rows.append(("Pre-retrieval" if i == 0 else "", their,
                     {c: get("pre", ours, c) for c in cells},
                     f"pre/{ours}" if ours else None))
    for i, (their, ours) in enumerate(POST):
        rows.append(("Post-retrieval" if i == 0 else "", their,
                     {c: get("post", ours, c) for c in cells},
                     f"post/{ours}" if ours and ours != "CLARITY_NA" else None))
    rows.append(("Ours", "k-way channel selector", {c: tuple(table[c]["ours"]) for c in cells},
                 "ours"))
    rows.append(("Oracle", "route by true gain", {c: tuple(table[c]["oracle"]) for c in cells},
                 "oracle"))

    # section bests, for the bold rule
    best = {}
    for sec in ("Pre-retrieval", "Post-retrieval"):
        members, cur = [], None
        for cat, _, vals, _k in rows:
            if cat in ("Pre-retrieval", "Post-retrieval", "Original", "Ours", "Oracle"):
                cur = cat
            if cur == sec:
                members.append(vals)
        for c in cells:
            vs = [m[c][0] for m in members if isinstance(m.get(c), tuple) and m[c][0] is not None]
            best[(sec, c)] = max(vs) if vs else None

    NCOL = 5 if nug else 3
    BLANK = " | ".join(["—"] * (NCOL - 1))

    def cellstr(v, sec, c, key):
        if v is None:                       # every sub-column, or the row loses its alignment
            return f"*n.i.* | {BLANK}"
        if v == "n/a":
            return f"n/a | {BLANK}"
        nd, tau = v[0], v[1]
        rec = v[2] if len(v) > 2 else None
        s = f"{nd:.4f}"
        # EPS, not zero. A predictor that degenerates to "always escalate" lands on the fixed policy
        # give or take float noise, and a 3e-5 difference marked as an improvement would misreport the
        # main result. 5e-4 is half a hundredth of an nDCG point: below anything we would ever claim.
        if nd > original[c] + 5e-4:                       # their underline rule
            s = f"<u>{s}</u>"
        if best.get((sec, c)) is not None and abs(nd - best[(sec, c)]) < 1e-12:
            s = f"**{s}**"
        s += (f" | {tau:+.3f}" if tau is not None else " | —")
        s += (f" | {rec:.4f}" if rec is not None else " | —")
        if nug:
            n = nug_for(key.get(c) if isinstance(key, dict) else key, c) if key else None
            s += (f" | {n[0]:.4f} | {n[1]:.4f}" if n else " | — | —")
        return s

    head = "nDCG@10 | τ | R@100" + (" | N_all | N_strict" if nug else "")
    L = ["| Category | Method | " + " | ".join(f"{c} {head}" for c in cells) + " |",
         "|" + "---|" * (2 + NCOL * len(cells))]
    sec = "Original"
    for cat, name, vals, key in rows:
        if cat:
            sec = cat
        L.append(f"| {cat} | `{name}` | " +
                 " | ".join(cellstr(vals.get(c), sec, c, key) for c in cells) + " |")

    # Which rows never actually choose. A predictor whose routed nDCG lands on the always-escalate or
    # never-escalate policy made one decision for all 2,546 queries, so its nDCG reports that fixed
    # policy and not the predictor. Worth naming: a row can carry a healthy tau and still be degenerate
    # here -- BERTQPP correlates at +0.24 and escalates every query -- which is exactly the gap between
    # correlation and decision quality that Arabzadeh et al.'s two metric families are there to expose.
    degen, tally, sec = [], {}, None
    for cat, name, vals, _k in rows:
        if cat:
            sec = cat
        if sec == "Original":           # the fixed policies ARE the degenerate ones; not a finding
            continue
        for c in cells:
            v = vals.get(c)
            if not isinstance(v, tuple) or v[0] is None or len(v) < 4 or v[3] is None:
                continue                # no recorded escalation fraction -> we do not claim either way
            tally.setdefault((sec, c), [0, 0])[1] += 1
            if v[3] in (0.0, 1.0):
                degen.append(f"{name}/{c} ({'always' if v[3] else 'never'} fuse)")
                tally[(sec, c)][0] += 1
    summary = [f"{s} / {c}: {d}/{n}" for (s, c), (d, n) in tally.items() if d]

    out = "\n".join(L)
    print(out)
    print("\nunderline = beats the Original row; bold = best in section; "
          "`n/a` = undefined for this channel; *not implemented* = we have no equivalent predictor")
    if degen:
        print(f"\ndegenerate (one decision for every query): {len(degen)} cells")
        for s in summary:
            print(f"  {s}")
    print("\ncolumn coverage:")
    for k, v in COVERAGE.items():
        print(f"  {k:<12} {v}")

    dest = os.path.join(ABL, f"mv2_table1{a.tag}.md")
    with open(dest, "w") as f:
        f.write("# Table 1 (paper layout)\n\n" + out + "\n\n"
                "underline = beats the Original row; bold = best in section; `n/a` = undefined for "
                "this channel; *n.i.* = no equivalent predictor implemented.\n")
        if degen:
            f.write(f"\n## Degenerate cells\n\nCounted from each predictor's recorded escalation "
                    f"fraction, degenerate meaning exactly 0 or exactly 1: the same decision for all "
                    f"{table[cells[0]]['n']} queries, so the number in the cell reports a fixed "
                    f"policy and not the predictor. A row can carry a healthy tau and still be "
                    f"degenerate, which is the gap between correlation and decision quality. "
                    f"Denominators cover the rows that have a fraction on record, so `clarity`, `DM` "
                    f"and `QSD_post` are excluded rather than counted as non-degenerate.\n\n"
                    + "\n".join(f"- {s}" for s in summary)
                    + "\n\nFull list: " + "; ".join(degen) + ".\n")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
