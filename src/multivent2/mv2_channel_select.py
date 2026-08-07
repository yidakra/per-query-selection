"""Per-query channel *selection* on MultiVENT 2.0: pick which channel combination to trust for each
query, instead of the binary escalate/don't of the pairwise cells.

Motivation: the best-single-channel-per-query oracle sits at 0.4777 against 0.3134 for the best fixed
channel (`mv2_channels.py`) -- global fusion throws away a third of the stack's value. The pairwise
router already converts a slice of that per pair; this asks whether a K-way selector can do it across
the whole policy set.

Protocol, matching the repo's discipline:
- Policies are all non-empty subsets of the channels, fused with unit-weight RRF (the same `fuse` as
  mv2_channels.py). Every policy here is cheap -- ranked lists are shipped and the dense channel costs
  ~2 ms/query -- so this is an accuracy claim, not a cost claim: the decision uses only score
  distributions that are already computed either way.
- Features: each channel's own confidence features (conf_features) plus cross-channel agreement
  (top-10/top-100 candidate overlap per channel pair). All observable before choosing.
- Selection: multi-target ridge predicts per-query nDCG for every policy, out-of-fold; argmax picks.
- Headline: NESTED gap -- outer 5-fold, the best *fixed* policy is chosen on the training fold too, so
  neither side of the comparison peeks (the gold-split lesson from router_findings.md section 5).
- Significance: permutation test on the selection assignment, 2000 perms.

CPU-only. Channels overridable as in mv2_channels.py, e.g.:

  python src/multivent2/mv2_channel_select.py --channel asr=asr_dense_bge-m3.json --cell-tag _dense_m3
"""
import os
import sys
import json
import argparse
import itertools
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, KFold

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels, load_run  # noqa: E402
from mv2_ab import per_query_ndcg  # noqa: E402
from mv2_channels import CHANNELS, fuse  # noqa: E402
from retrieve import conf_features, FEATURE_ORDER  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")
RNG = np.random.default_rng(0)


def mk():
    return Pipeline([("sc", StandardScaler()),
                     ("m", RidgeCV(alphas=np.logspace(-2, 3, 12), alpha_per_target=True))])


def overlap(a_scores, b_scores, k):
    ta = set(sorted(a_scores, key=lambda v: -a_scores[v])[:k])
    tb = set(sorted(b_scores, key=lambda v: -b_scores[v])[:k])
    return len(ta & tb) / k


def features(runs, qids):
    """Per-query feature vector: conf_features of every channel + pairwise top-K overlap."""
    names = sorted(runs)
    order, rows = None, []
    for q in qids:
        row, labels = [], []
        for n in names:
            f = conf_features(np.array(sorted(runs[n][q].values(), reverse=True)))
            row += [f[k] for k in FEATURE_ORDER]
            labels += [f"{n}:{k}" for k in FEATURE_ORDER]
        for a, b in itertools.combinations(names, 2):
            for k in (10, 100):
                row.append(overlap(runs[a][q], runs[b][q], k))
                labels.append(f"overlap{k}:{a}+{b}")
        order = order or labels
        rows.append(row)
    return np.array(rows), order


def nested_selection(X, Y, model=mk, splits=None):
    """Outer 5-fold: fit the selector on train, pick the best fixed policy on train, compare on test.
    Returns (gap mean, gap sem, per-fold detail, per-query achieved and nested-fixed arrays)."""
    if splits is None:
        splits = KFold(5, shuffle=True, random_state=1).split(X)
    gaps, folds = [], []
    ach_q = np.zeros(len(Y))
    fix_q = np.zeros(len(Y))
    for tr, te in splits:
        m = model().fit(X[tr], Y[tr])
        sel = np.argmax(m.predict(X[te]), axis=1)
        achieved = Y[te, sel].mean()
        fixed_j = int(np.argmax(Y[tr].mean(axis=0)))
        fixed = Y[te, fixed_j].mean()
        ach_q[te] = Y[te, sel]
        fix_q[te] = Y[te, fixed_j]
        gaps.append(100 * (achieved - fixed))
        folds.append({"achieved": float(achieved), "fixed": float(fixed), "fixed_policy": fixed_j})
    return (float(np.mean(gaps)), float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))), folds,
            ach_q, fix_q)


def group_signflip_p(diff, grp, n_draws=2000, rng=None):
    """One-sided sign-flip test of mean(diff) > 0 at the event-group level.

    Queries within an event are near-duplicates, so query-level permutation overstates the evidence;
    the exchangeable unit is the group. Per-group mean differences are size-weighted so the tested
    statistic stays the overall per-query mean, and signs are flipped per group.
    """
    rng = rng or RNG
    diff = np.asarray(diff, dtype=float)
    grp = np.asarray(grp)
    groups = np.unique(grp)
    dg = np.array([diff[grp == g].mean() for g in groups])
    wg = np.array([(grp == g).sum() for g in groups], dtype=float)
    wg /= wg.sum()
    obs = float((wg * dg).sum())
    cnt = 0
    for _ in range(n_draws):
        s = rng.choice([-1.0, 1.0], size=len(dg))
        if float((wg * dg * s).sum()) >= obs:
            cnt += 1
    return float((1 + cnt) / (n_draws + 1)), obs


def load_cell(channel_overrides=()):
    """Everything a selection experiment needs for one cell: policies, targets Y, features X."""
    channels = dict(CHANNELS)
    for spec in channel_overrides:
        name, _, fn = spec.partition("=")
        channels[name] = fn

    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    runs = {n: load_run(os.path.join(DATA, fn)) for n, fn in channels.items()}
    qids = sorted(set.intersection(set(qrels), *[set(r) for r in runs.values()]))

    names = sorted(runs)
    policies = []
    for k in range(1, len(names) + 1):
        for c in itertools.combinations(names, k):
            policies.append("+".join(c))
    pq = {}
    for pol in policies:
        w = {n: (1.0 if n in pol.split("+") else 0.0) for n in names}
        pq[pol] = per_query_ndcg(qrels, fuse(runs, w, qids))
    qids = [q for q in qids if all(q in pq[p] for p in policies)]
    Y = np.array([[pq[p][q] for p in policies] for q in qids])
    X, feat_order = features(runs, qids)
    return channels, names, policies, qids, X, Y, feat_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", action="append", default=[], metavar="NAME=FILE",
                    help="add or override a channel, e.g. asr=asr_dense_bge-m3.json")
    ap.add_argument("--cell-tag", default="", help="suffix for the output JSON")
    ap.add_argument("--group-cv", action="store_true",
                    help="split by event group instead of by query. MultiVENT 2.0 carries several "
                         "phrasings of the same event, so a plain split puts near-duplicates on both "
                         "sides of the fold boundary and the selector can memorise event -> best policy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(ABL, f"mv2_channel_select{a.cell_tag}.json")

    channels, names, policies, qids, X, Y, feat_order = load_cell(a.channel)
    print(f"channels: {names}  queries: {len(qids)}")
    if a.group_cv:
        from sklearn.model_selection import GroupKFold
        from mv2_qsd import event_groups
        qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
        grp = event_groups(qids, qrels)
        print(f"grouped CV: {len(set(grp))} event groups over {len(qids)} queries")
        cv_pool = list(GroupKFold(5).split(X, groups=grp))
        cv_nested = list(GroupKFold(5).split(X, groups=grp))
    else:
        cv_pool = KFold(5, shuffle=True, random_state=0)
        cv_nested = None
    print(f"policy means: " + "  ".join(f"{p}={Y[:, j].mean():.4f}" for j, p in enumerate(policies)))
    print(f"features: {X.shape[1]} ({len(names)} channels x {len(FEATURE_ORDER)} + overlaps)")

    # pooled out-of-fold selection: diagnostics, histogram, significance
    Yhat = cross_val_predict(mk(), X, Y, cv=cv_pool)
    sel = np.argmax(Yhat, axis=1)
    achieved = Y[np.arange(len(Y)), sel]
    best_fixed_j = int(np.argmax(Y.mean(axis=0)))
    oracle = Y.max(axis=1)
    hist = {policies[j]: int((sel == j).sum()) for j in range(len(policies))}

    ng, ngs, folds, ach_q, fix_q = nested_selection(X, Y, splits=cv_nested)

    if a.group_cv:
        # queries cluster by event, so query-level permutation overstates the evidence: the
        # exchangeable unit is the group. One-sided sign-flip tests at the group level, for the
        # pooled selection gain and for the nested gap itself.
        p_pooled, _ = group_signflip_p(achieved - Y[:, best_fixed_j], grp)
        p_nested, _ = group_signflip_p(ach_q - fix_q, grp)
        sig = {"p_signflip_group_pooled": p_pooled, "p_signflip_group_nested": p_nested,
               "signflip_draws": 2000}
        sig_str = (f"group sign-flip p: pooled "
                   + ("< 5e-4" if p_pooled <= 1 / 2001 else f"= {p_pooled:.4f}")
                   + ", nested gap "
                   + ("< 5e-4" if p_nested <= 1 / 2001 else f"= {p_nested:.4f}"))
    else:
        perm_stats = []
        for _ in range(2000):
            p = RNG.permutation(len(sel))
            perm_stats.append(Y[np.arange(len(Y)), sel[p]].mean())
        perm_stats = np.array(perm_stats)
        p_perm = float((1 + (perm_stats >= achieved.mean()).sum()) / 2001)
        sig = {"p_perm_query_pooled": p_perm, "perm_draws": 2000}
        sig_str = ("query-permutation p " +
                   ("< 5e-4" if p_perm <= 1 / 2001 else f"= {p_perm:.4f}") +
                   " (pooled selection mean; anti-conservative without event grouping)")

    print(f"\nbest fixed policy   {policies[best_fixed_j]:<22} {Y[:, best_fixed_j].mean():.5f}")
    print(f"selected (oof)      {'':<22} {achieved.mean():.5f}  {sig_str}")
    print(f"oracle (per-query best policy)         {oracle.mean():.5f}")
    print(f"NESTED selection gap vs best-fixed-on-train: {ng:+.2f} +/- {ngs:.2f} nDCG")
    print("picks: " + "  ".join(f"{k}={v}" for k, v in sorted(hist.items(), key=lambda x: -x[1])))

    json.dump({"channels": channels, "policies": policies, "n_queries": len(qids),
               "policy_means": {p: float(Y[:, j].mean()) for j, p in enumerate(policies)},
               "best_fixed": policies[best_fixed_j],
               "selected_oof": float(achieved.mean()), **sig,
               "oracle_best_policy": float(oracle.mean()),
               "nested_gap": ng, "nested_sem": ngs, "folds": folds, "group_cv": bool(a.group_cv),
               "picks": hist, "feature_order": feat_order},
              open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
