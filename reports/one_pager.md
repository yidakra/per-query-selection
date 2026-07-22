# Adaptive Q2E — one-page summary

**Per-query adaptive routing over LLM query-expansion cost tiers for zero-shot multilingual
text-to-video retrieval.** One clean claim: *spend expensive LLM compute only on the queries that
benefit, decided from signals the cheap tier already produced, and trace the whole accuracy–compute
frontier instead of one operating point.*

## Motivation

LLM query expansion/decomposition (Q2E and successors) reliably improves retrieval but pays a large,
**fixed** cost on **every** query — ~30 LLaMA-3.3-70B generations/query and a measured **1,418 J/query**
for the full tier, a **68×** cost spread over the cheap tier. The benefit, though, is highly uneven:
across our data **17–23% of queries are actively *hurt*** by the expensive signal, and many others gain
nothing. Paying full price uniformly is wasteful. If we can predict *per query* whether escalation will
pay — using only cheap-tier features already computed — we spend compute where it helps and move the
efficiency–effectiveness frontier.

## Research questions and the evidence we have

| RQ | Question | Evidence (status) |
|---|---|---|
| **RQ1** | Is escalation benefit predictable from the **query text** (Adaptive-RAG's premise) or from the query×corpus **retrieval interaction**? | **Not in the text.** A TF-IDF+LogReg router over the query string scores at/below the majority-class prior. Signal lives in the cheap tier's **retrieval confidence**. *(supported)* |
| **RQ2** | Which cheap-tier features predict per-query gain, and does gain-regression beat classical QPP? | Out-of-fold ridge on top-1 score, margins, entropy, softmax mass, A/B-disagreement orders queries by true gain; beats NQC/WIG/Clarity baselines. *(supported)* |
| **RQ3** | What does routing buy on the **accuracy–compute** plane, in standard cost terms? | **Same accuracy for 24–58% less escalation cost**, or **+0.24 to +1.92 nDCG at equal cost**, in the sub-B budget region. *(supported)* |
| **RQ4** | Does per-query gain **heterogeneity** drive how much routing wins? | `sd(per-query gain)` predicts the achieved gap at **Spearman ρ = +0.943** (n=6). This is the differentiator from Adaptive-RAG. *(supported)* |
| **RQ5** | Does query **extension + concatenation** (one enriched query) reduce the heterogeneity routing exploits — is robust fusion a **substitute for** or **complement to** routing? | **Complement, not substitute.** On MultiVENT 2.0, concatenating the LLM events beats decompose-and-fuse (**+1.57 vs +0.81 nDCG** at 14B, same weight) and is lower-variance at every matched weight — but only ~5–10% lower, so the router keeps its signal (concat B→Full τ +0.042 vs +0.035). *(supported)* |

## Results (all nDCG@10)

**Tiers, nested so escalation is free — cost is measured, not assumed.**

| tier | components | LLM calls/q | measured cost |
|---|---|---|---|
| A (visual) | `query_vs_video` | 0 | 6.83 J |
| B (+captions) | `+ query_vs_captions` | 0 | 22.35 J |
| Full (+events) | `+ {prequel,during,sequel}_vs_captions` | ~30 | 1,418 J |

Cost law: `E(N) = 3167 + 0.912·N` J, R²=0.9994 (held-out extrapolation within 1.3%). Escalating a
fraction `f` is affine in `f` under *any* per-component cost, so every routing gap below is unit-invariant.

**Routing quality — the router orders queries by true gain (out-of-fold, permutation-tested).**

| cell | ρ(pred,true) | perm p | nested gap vs cost-matched random |
|---|---|---|---|
| MultiVENT noASR | +0.164 | .0065 | **+0.73 ± 0.22** |
| MultiVENT ASR | +0.233 | .0005 | **+1.68 ± 0.25** |
| **MultiVENT 2.0** (2,546 test q, graded multi-gold) | τ=+0.127 | .0005 | **+2.23 ± 0.18** (APGR 0.217, CPT₅₀ 0.24) |

MultiVENT 2.0 (the community benchmark our JHU collaborators built): tier A = 0.30364 (matches the
official evaluator to the last digit), tier B = 0.36052 (+5.69), `sd(gain)` = 24.58, **43% helped / 23%
hurt**. CPT₅₀ = 0.24 → the router captures half the caption-tier improvement by escalating only **24%**
of queries.

**Pareto reframe — the effectiveness hook.**

| cell | Fixed-A | Fixed-B | +nDCG at equal cost | cost cut at equal accuracy |
|---|---|---|---|---|
| MultiVENT/mCLIP/noASR | 67.71 | 74.28 | **+1.23** | 32% |
| MultiVENT/mCLIP/ASR | 67.71 | 78.31 | **+1.92** | 24% |
| MSR-VTT/IV2/ASR | 66.00 | 68.86 | +1.06 | **58%** |

**Paired negatives — these are the *shape* of the frontier (why it stops at B), not failures.**

- **Full tier is correctly declined.** B→Full oracle gap is small (~+2.55) and does **not** survive a
  held-out gold-split (optimism +8.4); price is **63–177× tier B**. On MSR-VTT/IV2/noASR Full is
  *literally* Pareto-dominated (67.11 < Fixed-B 67.52). On MultiVENT 2.0 the null replicates: B→Full
  τ=+0.002 (p=.44), gap +0.15 (ns); a **7B→14B** decomposer moves routability only .002→.035 while
  energy nearly doubles (143.5→259.7 J/q) — a size *trend* arguing 70B won't fix it either.
- **"Tier C" (paraphrase selection) does not exist.** Oracle shows +2.00 headroom; three independent
  achievable estimates all *lose* (−1.49 to −1.81). Gold-split proves optimism +4.89 = label noise.

**Positioning vs SOTA (MultiVENT 2.0 test, graded nDCG@10).** Our tiers are a deliberately *cheap
visual/caption cascade*: tier A (CLIP) = 0.304 (matches the benchmark's mCLIP baseline), tier B
(+captions) = 0.36. The strong systems — **MMMORRF 0.586** (SigLIP + PLAID-X dense over ASR + OCR,
weighted RRF), **CLaMR 0.585** (late-interaction VLM), **OmniEmbed 0.753** (in-domain fine-tuned
omni-backbone) — get their lift from **ASR + OCR retrieval channels our cascade never touches**, at
far higher cost. So the honest framing is *not* "we beat SOTA": the router is a **method-agnostic
frontier layer** that spends compute per query within a tier stack. It optimizes the accuracy–compute
frontier in the ~0.30–0.38 band; demonstrating it over a strong fusion stack (bolt the router onto
MMMORRF's modality tiers) is the next step, not a reproduction of Q2E's own numbers.

## What we borrowed vs. what is ours

**Borrowed:** Q2E decomposition + inverse-entropy fusion + tier structure (Dipta & Ferraro, AACL 2025,
[arXiv:2506.10202](https://arxiv.org/abs/2506.10202)); the query-complexity-routing *premise*
(Adaptive-RAG) — which we test and **partly refute** (signal is not in the query text); classical QPP
predictors (Clarity/WIG/NQC) as routing baselines; router cost-quality metrics **APGR, CPT₅₀/₈₀**
(RouteLLM/RouterBench); **MultiVENT 2.0** benchmark and the SOTA baselines to position against
(**MMMORRF**, **CLAMR**); measured GPU joules via NVML/CodeCarbon.

**Ours:** (1) **cheap-tier-features-*only* escalation-gain regression** — the expensive tier is never
invoked to decide, which no prior routing work does; (2) the **heterogeneity → routing-value law**
(ρ=0.943); (3) a **measured-joules** cost model over LLM-decomposition tiers; (4) a set of **paired
negative results** (Full tier, Tier C) established with nested-CV + gold-split rigor.
