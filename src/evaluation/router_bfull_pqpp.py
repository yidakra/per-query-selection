#!/usr/bin/env python
"""Can a PQPP-style signal resurrect the B->Full escalation the paper declared dead?

`router_findings.md` §2/§4 kill B->Full on two grounds: the oracle prize is small (~+2.55) AND
in-sample -- a gold-split shows it does not survive held-out labels (optimism ~3.0, out-of-sample
NEGATIVE). But that verdict used only *retrieval-confidence* features (tier-B score distribution).
The JHU discussion pointed at PQPP (Poesina et al., 2025): predict, from a *prompt*, whether a
generative step will pay off -- before running it. The Full tier's generative step is exactly the
~30 LLaMA-70B calls that decompose the query into prequel/during/sequel events. So the fair question
is: does a signal about the GENERATION -- not the retrieval -- predict real, out-of-sample B->Full
gain that retrieval confidence could not?

We separate features by CASCADE LEGALITY, because that decides whether a positive result is even
usable for routing:

  PROMPT       (legal)   -- from the query text alone, before any generation. This is PQPP proper:
                           predict decomposability from the prompt. If THIS works, routing works.
  GENERATION   (semi)    -- from the generated event paraphrases, but BEFORE scoring them against the
                           gallery. You have paid the LLM but not the 17x scoring. Weak routing use,
                           strong analysis: does the generation's own content foretell its value?
  POST-SCORING (illegal) -- confidence of the Full-tier retrieval / how much events moved the ranking.
                           You have paid the entire Full cost. This is the CEILING: if even this
                           cannot predict out-of-sample gain, B->Full is truly unpredictable.

Two evaluations per feature set, both on MultiVENT (the only cells where event decomposition is
meaningful and multi-gold enough to split):
  (1) OOF nested gap + Kendall tau / Pearson rho + permutation, on the FULL gold labels.
  (2) THE ACID TEST -- gold-split: order queries by a ridge trained on one gold half's B->Full gain,
      grade the realized gap on the OTHER half. This is what killed the oracle (§4); a learned
      predictor must survive it for the signal to be real rather than label noise.

CPU-only; reads cached tensors + the dataset's generated event fields.
"""
import os, sys, re, json, numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from router_hetero import comps_multivent, mk, gap_at, MV  # noqa: E402
from oracle_router_headroom import fuse, per_query, LADDER  # noqa: E402
from router_cascade_exp import conf_feats  # noqa: E402
from tierC_goldsplit_oracle import split_golds  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from sklearn.model_selection import cross_val_predict, KFold  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

RNG = np.random.default_rng(0)
FRACS = np.arange(0.05, 1.0, 0.05)
GOLD_SEEDS = 20
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{_ROOT}/results/ablations/router_bfull_pqpp.json"

# Small news-event lexicon (MultiVENT is Wikipedia news events). Used only as a cheap prompt cue for
# "is this a temporally decomposable event"; not tuned.
EVENT_KW = {"earthquake", "attack", "protest", "fire", "explosion", "flood", "crash", "storm", "war",
            "election", "riot", "collapse", "eruption", "shooting", "rally", "hurricane", "tornado",
            "bombing", "strike", "accident", "disaster", "landslide", "tsunami", "wildfire", "clash",
            "ceremony", "match", "game", "concert", "launch", "parade", "festival", "race", "flooding",
            "earthquakes", "protests", "attacks", "wildfires", "eruptions"}
TEMPORAL_KW = {"during", "after", "before", "when", "while", "following", "amid", "as", "then", "later"}

PROMPT_COLS = ["P_charlen", "P_wordlen", "P_ndigit", "P_hasyear", "P_ncaps", "P_ncomma",
               "P_eventkw", "P_tempkw"]
GEN_COLS = ["G_nonempty", "G_nparas", "G_paras_per_event", "G_genchars", "G_paralen_mean",
            "G_lexdiv", "G_qoverlap"]
POST_COLS = ["S_Fentropy", "S_Fmargin12", "S_Ftop5mass", "S_l1move", "S_top1changed", "S_klBF"]
FEATURE_SETS = {"prompt": PROMPT_COLS, "generation": GEN_COLS, "postscore": POST_COLS,
                "prompt+gen": PROMPT_COLS + GEN_COLS, "all": PROMPT_COLS + GEN_COLS + POST_COLS}


def prompt_feats(q):
    toks = q.split()
    ltok = [t.strip(".,;:!?").lower() for t in toks]
    return {"P_charlen": len(q), "P_wordlen": len(toks),
            "P_ndigit": sum(c.isdigit() for c in q),
            "P_hasyear": 1.0 if re.search(r"\b(19|20)\d{2}\b", q) else 0.0,
            "P_ncaps": sum(1 for t in toks if t[:1].isupper()),
            "P_ncomma": q.count(","),
            "P_eventkw": sum(1 for t in ltok if t in EVENT_KW),
            "P_tempkw": sum(1 for t in ltok if t in TEMPORAL_KW)}


def gen_feats(pre, dur, seq, q):
    events = [e or [] for e in (pre, dur, seq)]
    nonempty = [e for e in events if len(e) > 0]
    allp = [p for e in events for p in e]
    qtok = set(q.lower().split())
    toks_all = [t for p in allp for t in p.lower().split()]

    def jac(p):
        pt = set(p.lower().split())
        return len(pt & qtok) / max(1, len(pt | qtok))
    return {"G_nonempty": float(len(nonempty)), "G_nparas": float(len(allp)),
            "G_paras_per_event": len(allp) / max(1, len(nonempty)),
            "G_genchars": float(sum(len(p) for p in allp)),
            "G_paralen_mean": float(np.mean([len(p.split()) for p in allp])) if allp else 0.0,
            "G_lexdiv": len(set(toks_all)) / max(1, len(toks_all)),
            "G_qoverlap": float(np.mean([jac(p) for p in allp])) if allp else 0.0}


def post_feats(fB_row, fF_row):
    pb = torch.softmax(fB_row.double(), 0); pf = torch.softmax(fF_row.double(), 0)
    cf = conf_feats(fF_row)
    return {"S_Fentropy": cf["entropy"], "S_Fmargin12": cf["margin12"], "S_Ftop5mass": cf["top5mass"],
            "S_l1move": float((pf - pb).abs().sum()),
            "S_top1changed": 1.0 if int(pf.argmax()) != int(pb.argmax()) else 0.0,
            "S_klBF": float((pf * ((pf + 1e-12).log() - (pb + 1e-12).log())).sum())}


def event_map(ds_name):
    """query -> (prequel, during, sequel) lists, from the dataset's generated fields (per query)."""
    ds = load_from_disk(f"{_ROOT}/data/MultiVENT/{ds_name}")
    m = {}
    for r in ds:
        if r["query"] not in m:
            m[r["query"]] = (r["prequel"], r["during"], r["sequel"])
    return m


def per_q(comps, tier, tgt):
    return per_query(fuse(comps, LADDER[tier]), tgt)[0].numpy()


def nested_gap(X, h):
    """OOF nested frontier gap (NDCG points), same protocol as router_hetero.cell()."""
    n = len(h)
    outer = KFold(5, shuffle=True, random_state=1)
    gaps = []
    for tr, te in outer.split(np.arange(n)):
        m = mk().fit(X[tr], h[tr])
        gh_tr = cross_val_predict(mk(), X[tr], h[tr], cv=KFold(5, shuffle=True, random_state=2))
        f_star = max(FRACS, key=lambda f: gap_at(f, gh_tr, h[tr]))
        gaps.append(gap_at(f_star, m.predict(X[te]), h[te]) * 100)
    gaps = np.array(gaps)
    return float(gaps.mean()), float(gaps.std(ddof=1) / np.sqrt(len(gaps)))


def goldsplit_learned(X, comps, target):
    """THE ACID TEST. Per seed: split golds; train a ridge (OOF) on one half's B->Full gain, choose f
    on that half, grade the realized gap on the OTHER half. Positive here = signal survives label
    noise. Also returns the ORACLE's out-of-sample gap (ghat = true half-A gain) for reference."""
    learned, oracle_oos, oracle_ins = [], [], []
    for s in range(GOLD_SEEDS):
        rs = np.random.RandomState(700 + s)
        A, B = split_golds(target, rs)
        hA = per_q(comps, "C_full", A) - per_q(comps, "B_noevents", A)
        hB = per_q(comps, "C_full", B) - per_q(comps, "B_noevents", B)
        # learned: ridge predicts hA from features (OOF), graded on hB
        ghat = cross_val_predict(mk(), X, hA, cv=KFold(5, shuffle=True, random_state=0))
        f_star = max(FRACS, key=lambda f: gap_at(f, ghat, hA))
        learned.append(gap_at(f_star, ghat, hB) * 100)
        # oracle: order by true hA, graded on hB (this is what §4 showed goes negative)
        fo = max(FRACS, key=lambda f: gap_at(f, hA, hA))
        oracle_oos.append(gap_at(fo, hA, hB) * 100)
        oracle_ins.append(max(gap_at(f, hB, hB) for f in FRACS) * 100)
    return (float(np.mean(learned)), float(np.std(learned)),
            float(np.mean(oracle_oos)), float(np.mean(oracle_ins)))


def cell(setting):
    ds_name = MV[setting][1]
    queries, target, comps = comps_multivent(setting)
    emap = event_map(ds_name)

    fB = fuse(comps, LADDER["B_noevents"]); fF = fuse(comps, LADDER["C_full"])
    ndB = per_query(fB, target)[0].numpy(); ndF = per_query(fF, target)[0].numpy()
    h = ndF - ndB                                   # B->Full gain on full golds

    rows = []
    for i, q in enumerate(queries):
        pre, dur, seq = emap[q]
        r = {}
        r.update(prompt_feats(q)); r.update(gen_feats(pre, dur, seq, q))
        r.update(post_feats(fB[i], fF[i]))
        rows.append(r)
    df = pd.DataFrame(rows)

    print(f"\n### MultiVENT {setting}  (T={len(h)}, B->Full mean gain {h.mean()*100:+.2f}, "
          f"sd {h.std()*100:.2f})")
    res = {"setting": setting, "T": len(h), "mean_gain": float(h.mean() * 100),
           "sd_gain": float(h.std() * 100), "feature_sets": {}}
    for tag, cols in FEATURE_SETS.items():
        X = df[cols].values
        ghat = cross_val_predict(mk(), X, h, cv=KFold(5, shuffle=True, random_state=0))
        tau = float(kendalltau(ghat, h).statistic)
        rho = float(np.corrcoef(ghat, h)[0, 1])
        perm = np.array([kendalltau(RNG.permutation(ghat), h).statistic for _ in range(2000)])
        p_tau = float((1 + (perm >= tau).sum()) / (1 + len(perm)))
        ng, ngs = nested_gap(X, h)
        lm, lsd, oor, oin = goldsplit_learned(X, comps, target)
        res["feature_sets"][tag] = {"tau": tau, "rho": rho, "p_tau": p_tau,
                                    "nested_gap": ng, "nested_sem": ngs,
                                    "goldsplit_learned_oos": lm, "goldsplit_learned_sd": lsd,
                                    "oracle_oos": oor, "oracle_insample": oin}
        star = "*" if p_tau < 0.05 else " "
        acid = "SURVIVES" if lm - lsd / np.sqrt(GOLD_SEEDS) > 0 else "dies"
        print(f"  {tag:11s} tau={tau:+.3f}{star} (p={p_tau:.3f})  nested_gap={ng:+.2f}+/-{ngs:.2f}  "
              f"| ACID gold-split OOS={lm:+.2f}+/-{lsd:.2f} [{acid}]  (oracle OOS {oor:+.2f})")
    return res


def main():
    out = {"gold_seeds": GOLD_SEEDS, "cells": {}}
    for setting in ["noASR", "ASR"]:
        out["cells"][setting] = cell(setting)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    print("\n[verdict] a feature set RESURRECTS B->Full only if its gold-split OOS gap is > 0 (beats "
          "the oracle's negative OOS) AND its legality permits routing (prompt = usable).")


if __name__ == "__main__":
    main()
