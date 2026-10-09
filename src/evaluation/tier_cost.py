"""Single source of truth for per-tier retrieval cost, in MEASURED joules per query.

SCOPE: the ORIGINAL Q2E pipeline only. The MultiVENT 2.0 cascade is a different system with its own
cost profile -- tiers A and B there are CPU-only and cost ~0.01 J and ~1 J per query. Its numbers live
in `src/multivent2/mv2_efficiency.py` and the first-phase efficiency report (in the git history). Do not import these joules
into MultiVENT 2.0 reporting.

Supersedes the component-count proxy `{A_visual:1, -Events:2, Full:5}` -> normalized
`{0.2, 0.4, 1.0}`, which rated all five similarity components at unit cost. They differ by
up to **68x**. Costs here are the measured marginal GPU energy per query (NVML @5Hz on
physical GPU1, integrated, minus warm idle); see `results/ablations/cost_model_findings.md`
and `component_energy{,_tierA}.json`.

    tier A_visual (query_vs_video, as shipped)             6.83 J
    tier -Events  (A + query_vs_captions)                 22.35 J
    Full          (A + query_vs_captions + 3 event comps) 1418.36 J

"As shipped" for tier A includes the black-image ViT-H forward that the published nDCG
actually paid for (bypassable to 0.83 J; see the cost model). The Full total counts only GPU
similarity; it excludes the ~30 LLaMA-70B generations/query that ONLY Full additionally pays.

INVARIANCE. Every frontier point's cost is affine in the escalated fraction `f`:
`cost(f) = cost(A) + f*(cost(B) - cost(A))`, under ANY per-tier cost assignment. So a
cost-matched baseline is an `f`-matched baseline in either unit, and every routing GAP
(nested-CV, permutation-tested) is unchanged by swapping the proxy for joules -- only the
x-axis positions and labels move. Verified: regenerating the frontier JSONs changes only the
`cost` field, byte-for-byte elsewhere. See cost_model_findings.md and router_findings.md.
"""

# measured marginal energy per query, J, as-shipped
JOULES = {"A_visual": 6.83, "-Events": 22.35, "Full": 1418.36}
JOULES_FULL = JOULES["Full"]

# normalized to Full = 1.0 -- same convention as the retired proxy, so figures stay comparable.
# {A_visual: 0.00482, -Events: 0.01576, Full: 1.0}
COST_NORM = {t: JOULES[t] / JOULES_FULL for t in JOULES}

LADDER = ["A_visual", "-Events", "Full"]

# retired component-count proxy, kept only for reference / axis annotations
PROXY_NORM = {"A_visual": 0.2, "-Events": 0.4, "Full": 1.0}

COST_UNIT = "measured J/query, normalized to Full=1.0"
