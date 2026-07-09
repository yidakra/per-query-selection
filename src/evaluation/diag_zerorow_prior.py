#!/usr/bin/env python
"""Is the tier-C subset oracle's gain real, or is it harvesting a zeros-row artifact?

The oracle selects a subset of paraphrases per query. When it selects NONE for an event type
(which it does for 45% of queries), build_comps_from_selection writes a ZERO row for that query.

That is NOT the same as dropping the component. fuse() applies softmax over dim=0 (ACROSS
QUERIES), so a zero row contributes

    exp(0 - logsumexp_q comps[e][q, v]) = 1 / Z_e[v]

which is non-uniform across videos v. It is a query-independent VIDEO PRIOR: videos that score
low against every query's events get a LARGE 1/Z, and vice versa. An oracle free to switch that
prior on and off per query could gain nDCG without selecting a single useful paraphrase.

So the headline "+2.00 over Fixed-Full from paraphrase selection" is only meaningful if it
survives removing the artifact. Three arms, all on the paracache tensors so fp16 noise cancels:

  Z (as-reported)  empty subset -> zero row               [what the oracle optimised]
  O (omit)         empty subset -> component dropped from that query's fusion
  P (prior-only)   ALL event subsets forced empty         [pure artifact, no paraphrases at all]

If P alone lands near Fixed-Full, the artifact is the story and the oracle is not about
paraphrases. If O retains most of Z's gain, selection is real.

We also report the oracle restricted to queries that keep >=1 paraphrase, since for keep-none
queries "selection" is vacuous.

CPU-only.
"""
import os, sys, json, argparse, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from tracking import track                                            # noqa: E402
import tierC_selection_oracle as O                                    # noqa: E402

REPO = "/home/ubuntu/q2e_repro"


def fused_row_subset(rows, D):
    """fuse() for one query over ONLY the components present in `rows` (true omission)."""
    out = None
    for p, x in rows.items():
        s = torch.exp(x - D[p])
        ent = -(s * torch.log2(s + 1e-6)).sum()
        w = 1.0 / (ent + 1e-6)
        out = w * s if out is None else out + w * s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="noASR")
    a = ap.parse_args()

    with track(f"tierC-zerorow-diag-{a.setting}", gpu_ids=[], tags=["tierC", "diagnostic"],
               config={"setting": a.setting}) as tr:
        queries, video_ids, target, comps, P, valid, nd_pub, delta = O.load_cell(a.setting)
        T = len(queries)
        D = O.frozen_softmax_parts(comps)
        sel = json.load(open(f"{REPO}/results/ablations/tierC_selection_oracle_{a.setting}.json"))["selection"]
        assert len(sel) == T

        ndF = O.per_query(O.fuse(comps, O.LADDER["C_full"]), target)[0].numpy()
        ndB = O.per_query(O.fuse(comps, O.LADDER["B_noevents"]), target)[0].numpy()

        base = ["query_vs_video", "query_vs_captions"]
        zero = torch.zeros_like(comps["query_vs_captions"][0])

        zZ, zO, zP = np.zeros(T), np.zeros(T), np.zeros(T)
        keep_any = np.zeros(T, dtype=bool)

        for q in range(T):
            rel = target[q].numpy().astype(np.float64)
            chosen = {e: [] for e in O.EVENTS}
            for e, p in sel[q]:
                chosen[e].append(p)
            keep_any[q] = any(len(v) for v in chosen.values())

            rows_z, rows_o = {p: comps[p][q] for p in base}, {p: comps[p][q] for p in base}
            for e in O.EVENTS:
                if chosen[e]:
                    r = torch.stack([P[e][q, p] for p in chosen[e]]).max(dim=0).values
                    rows_z[e] = r
                    rows_o[e] = r
                else:
                    rows_z[e] = zero          # artifact retained
                    # rows_o: component omitted entirely

            zZ[q] = O.ndcg10(O.fused_row(rows_z, D).numpy(), rel)
            zO[q] = O.ndcg10(fused_row_subset(rows_o, D).numpy(), rel)

            rows_p = {p: comps[p][q] for p in base}
            for e in O.EVENTS:
                rows_p[e] = zero
            zP[q] = O.ndcg10(O.fused_row(rows_p, D).numpy(), rel)

        f = lambda x: x.mean() * 100
        print(f"\n### zeros-row artifact diagnostic  MultiVENT {a.setting}  (T={T})\n")
        print(f"  Fixed-B                       {f(ndB):.2f}")
        print(f"  Fixed-Full (all paraphrases)  {f(ndF):.2f}")
        print(f"  P prior-only (ALL events empty, zero rows, NO paraphrases) {f(zP):.2f}"
              f"   [vs Full {f(zP)-f(ndF):+.2f}]")
        print(f"  Z oracle, empty->zero row  (as reported)                   {f(zZ):.2f}"
              f"   [vs Full {f(zZ)-f(ndF):+.2f}]")
        print(f"  O oracle, empty->component omitted                         {f(zO):.2f}"
              f"   [vs Full {f(zO)-f(ndF):+.2f}]")

        kn = ~keep_any
        print(f"\n  queries keeping >=1 paraphrase: {keep_any.sum()}/{T} "
              f"({100*keep_any.mean():.1f}%)")
        if keep_any.any():
            print(f"    on those: Full {f(ndF[keep_any]):.2f} | Z {f(zZ[keep_any]):.2f} "
                  f"| O {f(zO[keep_any]):.2f}")
        if kn.any():
            print(f"  queries keeping NONE ({kn.sum()}): Full {f(ndF[kn]):.2f} | "
                  f"Z(=prior) {f(zZ[kn]):.2f} | O(=Fixed-B) {f(zO[kn]):.2f} | "
                  f"Fixed-B {f(ndB[kn]):.2f}")
            # O on a keep-none query omits all 3 events -> should equal Fixed-B exactly
            d = np.abs(zO[kn] - ndB[kn]).max()
            print(f"    check: on keep-none queries, O == Fixed-B?  max|diff| = {d:.2e}")

        share = (f(zZ) - f(zO)) / (f(zZ) - f(ndF)) * 100 if f(zZ) != f(ndF) else float("nan")
        print(f"\n  => the zeros-row artifact accounts for {share:.0f}% of the oracle's "
              f"reported gain over Fixed-Full.")

        out = {"setting": a.setting, "T": T,
               "fixedB": f(ndB), "fixedFull": f(ndF),
               "prior_only": f(zP), "oracle_zerorow": f(zZ), "oracle_omit": f(zO),
               "pct_keep_any": float(100 * keep_any.mean()),
               "artifact_share_of_gain_pct": float(share)}
        of = f"{REPO}/results/ablations/tierC_zerorow_diag_{a.setting}.json"
        json.dump(out, open(of, "w"), indent=2)
        print(f"\nwrote {of}")
        tr.summary(out)


if __name__ == "__main__":
    main()
