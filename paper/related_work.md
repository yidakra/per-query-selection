# Related work & positioning (verified bibliography)

Assembled from a literature sweep (2024–2026). **Verification status** marks how each entry was
checked: **[V]** = title/abstract confirmed by direct fetch of the arXiv/venue page in our own
sessions; **[a]** = reported by a research subagent that fetched it, not re-confirmed here; **[?]** =
cited from search snippets only — **verify the arXiv ID and claims before it enters a submitted
bibliography.** IDs are future-dated relative to some tooling (it is 2026); treat any **[a]/[?]** ID
as provisional.

## The one-line positioning

> The abstract idea — a per-query router that predicts the benefit of escalating to a costlier
> retrieval tier and escalates only some queries — is **now crowded** in text IR/RAG (2024–2026).
> Our defensible contribution is the **intersection nobody occupies**: cheap-tier-features-*only*
> escalation-**gain** regression (the expensive tier is never invoked to decide), over **LLM-
> decomposition cost tiers in multimodal text-to-video retrieval**, with **measured joules**,
> **nested-CV + gold-split rigor**, and a set of **paired negative results**. We must cite the twins
> below head-on and stake that delta, and re-anchor away from Adaptive-RAG onto the
> reranking-cutoff / adaptive-computation / QPP-routing lineage.

## Positioning table — us vs. the closest work

| Work | Domain | Routes over | Decision signal | Gain vs. classify | Cost axis | OOS / correction rigor |
|---|---|---|---|---|---|---|
| **Ours** | **multimodal T2V** | **visual→caption→LLM-event tiers** | **cheap-tier confidence only** | **regress nDCG uplift** | **measured joules** | **nested-CV + gold-split + (planned) BH-FDR** |
| Adaptive Re-Ranking (Genc, Korukluoglu, Allan 2026) [V] | text IR | BM25→MiniLM→BGE rerank tiers | learned router, latency-aware utility labels | classify tier | latency (measured) | oracle analysis; no gold-split |
| Cost-Aware Query Routing (Mishra 2026) [a] | RAG | retrieval-depth tiers (k) | per-query router | classify tier | cost proxy | empirical, single-author |
| LTRR (Kim & Diaz 2025) [V] | RAG retrievers | which retriever | query + retriever features | **regress utility gain** (pointwise/pairwise) | — | LiveRAG eval |
| TARG (Wang et al. 2025) [a] | RAG gating | retrieve or not | token entropy / logit margin / small-N variance | threshold gate | acc–eff frontier | training-free |
| Adaptive-RAG (Jeong et al. NAACL 2024) [V] | RAG | no/single/multi-step | **query text** classifier | classify complexity | step-count proxy | none |
| Selective Query Processing (Chifu, Déjean, Mothe et al. TOIS 2025) [V] | text IR | sparse vs dense/hybrid | QPP predictors | classify method | — | **finds only marginal gains, poor cross-collection generalization** |
| Breaking Flat / MRSQ-PP (Santra, Basuchowdhuri, Ganguly 2026) [V] | text IR | which ranker per query | QPP models | select best ranker | — | shows per-ranker selection is *harder* than classic QPP |
| RPP (Tian, Ganguly, Macdonald 2026) [V] | RAG | context vs no-context | retriever+reader+doc features | **predict gain delta** | — | linear reg on NQ |
| ModaRoute (Dela Rosa 2024) [a] | multimodal video | which modalities to search | LLM router | select modality set | modality count (41% ↓) | ~86.5% routing acc |
| MetaEmbed (Xiao et al. 2025, ICLR'26 Oral) [V] | multimodal retrieval | # meta-tokens (embedding budget) | Matryoshka multi-vector | continuous budget knob | test-time compute | SOTA MMEB/ViDoRe |

**Reading of the table.** Every ingredient of ours exists somewhere — gain regression (LTRR, RPP),
cheap-confidence gating (TARG), per-query tier routing (Genc, Mishra), QPP-for-routing (Chifu,
Breaking Flat), multimodal routing (ModaRoute), test-time compute knobs (MetaEmbed). **No row
combines them in our cell**: cheap-only gain regression over an *LLM-decomposition* cost cascade in
*video*, graded on measured energy with gold-split rigor. That intersection is the paper.

## Bibliography by theme

### Our base method & the collaborators' assets
- **Q2E** — the base retrieval method (LLM query→event decomposition + CLIP + ColBERT captions). arXiv 2506.10202 [a].
- **MultiVENT 2.0** — Kriz, Sanders, Etter, Murray, … **Yang, Van Durme**. *A Massive Multilingual Benchmark for Event-Centric Video Retrieval.* arXiv 2410.11619 [V]. 218K+ news videos, 3,906 event queries, modalities = visual/audio/embedded-text/metadata; multilingual. **Authored by our JHU collaborators.** Our top scale-up target.
- **MMMORRF** — Samuel, DeGenaro, … **Yates, Yang**, … Kriz. *Multimodal Multilingual Modularized Reciprocal Rank Fusion.* arXiv 2503.20698 [V]. Modality-aware weighted RRF; +81% nDCG@20 over multimodal encoders, +37% over single-modality; on MultiVENT 2.0 + TVR; "effective and efficient." **Also our collaborators' system** — the efficiency-aware baseline to route over and beat.

### Adaptive retrieval / per-query cost routing (the twins — cite head-on)
- **Adaptive Re-Ranking** — Genc, Korukluoglu, Allan 2026. arXiv 2606.25249 [V]. Closest competitor; per-query routing over 3 rerank tiers, latency-aware utility labels; 1.15–53× lower median latency. Classifies; we regress gain.
- **Cost-Aware Query Routing in RAG** — Mishra 2026. arXiv 2606.02581 [a]. Per-query retrieval-depth tiers under cost objective.
- **LTRR: Learning To Rank Retrievers for LLMs** — Kim & Diaz 2025/2026 (SIGIR LiveRAG). arXiv 2506.13743 [V]. Regress per-query retriever utility gain; pairwise+XGBoost best. The gain-regression precedent.
- **TARG: Training-Free Adaptive Retrieval Gating** — Wang et al. 2025. arXiv 2511.09803 [a]. Cheap-confidence threshold gate — our feature family. Required "confidence-gate" baseline.
- **Adaptive-RAG** — Jeong et al. NAACL 2024. arXiv 2403.14403 [V]. Query-text complexity classifier. Cite but **do not anchor on it**.
- **MoR: Mixture of Sparse, Dense, and Human Retrievers** — EMNLP 2025. arXiv 2506.15862 [a]; **RouterRetriever** (mixture-of-retrievers landscape).
- **RouteLLM** — Ong et al. 2024. arXiv 2406.18665 [a]. LLM cost-quality router; source of **APGR / CPT** metrics.
- **RouterBench** — Hu et al. ICML 2024. arXiv 2403.12031 [a]. Routing benchmark; **area-under-cost-quality-curve** metric.
- **FrugalGPT** — Chen, Zaharia, Zou 2023. arXiv 2305.05176 [a]. Founding cost-aware LLM cascade.

### Reranking-cutoff / adaptive computation (our TRUE lineage for the negatives)
- **Ranked List Truncation for LLM-based Re-Ranking** — Meng, Arabzadeh, Askari, Aliannejadi, de Rijke. SIGIR 2024. arXiv 2404.18185 [a]. **The template for our B→Full negative:** fixed-k hard to beat, learned methods show no clear advantage, large oracle gap. Cite prominently.
- **Adaptive Re-Ranking with a Corpus Graph (GAR)** — MacAvaney et al. CIKM 2022. arXiv 2208.08942 [a]. Budgeted adaptive reranking; IR reviewers demand it.
- **AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking** — 2025. arXiv 2505.18512 [?]. Per-query adaptive rerank compute on a Pareto frontier.
- **Mixture-of-Depths** — Raposo et al. 2024. arXiv 2404.02258 [a]. Token-level conditional compute; the fine-grained analog of our per-query allocation.
- Cutoff line: **Choppy** (Bahri, SIGIR 2020), **BiCut** (Lien 2019) — QPP-for-thresholding precedents.

### QPP frontier (our framing's home)
- **iQPP** — Poesina, Ionescu, Mothe. SIGIR 2023. arXiv 2302.10126 [V]. First image-QPP benchmark; Kendall τ / Pearson; predictors don't generalize across scenarios (mirrors our transfer finding).
- **PQPP** — Poesina, Costache, Chifu, Mothe, Ionescu. CVPR 2025. arXiv 2406.04746 [?]. Joint text-to-image prompt+query PP.
- **VQPP** — Lutu, Poesina, Ionescu. 2026. arXiv 2602.17814 [V]. First video-QPP benchmark; **does not do routing** — our opening.
- **Breaking Flat / MRSQ-PP** — Santra, Basuchowdhuri, Ganguly 2026. arXiv 2601.17359 [V]. Per-query ranker selection = routing; harder than classic QPP. Our most direct QPP-side prior art.
- **Beyond Correlations: A Downstream Evaluation Framework for QPP** — Santra, Basuchowdhuri, Ganguly 2026. arXiv 2601.17339 [V]. **Kendall/Pearson do not measure decision utility** — undercuts our current τ/ρ protocol and hands us a downstream one (QPP-weighted fusion, +4.5%).
- **Selective Query Processing** — Chifu, Déjean, Mothe, Garouani, Ortiz, Ullah. TOIS 2025. arXiv 2504.01101 [V]. QPP-for-compute-routing; **"only marginal gains, poor cross-collection generalization"** — the bar we must beat.
- **RPP** — Tian, Ganguly, Macdonald 2026. arXiv 2601.14546 [V]. Predicts the RAG-vs-no-context gain delta — differential QPP by another name.
- **QPP-GenRE** — Meng, Arabzadeh, Aliannejadi, de Rijke. TOIS 2025. arXiv 2404.01012 [a]. LLM-judgment QPP; reconstructs any IR metric; SOTA TREC-DL.
- **BERT-QPP** — Arabzadeh, Khodabakhsh, Bagheri. CIKM 2021 [a]. Canonical supervised QPP; the "supervised differential predictor" baseline template.
- **Group-QPP / qppBERT-PL** — Chen et al. ECIR 2022 (arXiv 2204.11489) [a]; Datta et al. ECIR 2022 [a]. Groupwise/listwise relative QPP.
- **QPP for Neural IR: Are We There Yet?** — Faggioli et al. ECIR 2023. arXiv 2302.09947 [a]. Classical predictors degrade on neural rankers — motivation.
- Classical predictors to run as baselines: **Clarity** (Cronen-Townsend 2002), **WIG** (Zhou & Croft 2007), **NQC** (Shtok 2012), **SMV**, **UEF**, **σ_max / n(σ%)**.
- Per-query eval metric: **sMARE / sARE** (Faggioli et al. IRJ 2022) [?].

### Efficiency methodology & measurement (bulletproofing)
- **The Efficiency Misnomer** — Dehghani et al. ICLR 2022. arXiv 2110.12894 [a]. Report ≥2 cost indicators; the attack on any single-number efficiency claim.
- **MLPerf Power** — Tschand et al. HPCA 2025. arXiv 2410.12032 [a]. Energy-measurement standard to align our NVML methodology to.
- **Cost-of-Pass** — 2025. arXiv 2504.13359 [?]. Economic frontier metric (expected cost per correct answer).
- **Predicting Efficiency/Effectiveness Trade-offs (dense vs sparse strategy selection)** — Arabzadeh et al. CIKM 2021. arXiv 2109.10739 [a]. **The IR-native precedent for per-query strategy selection** — must-cite.
- **Cawley & Talbot** — JMLR 2010 [a]. Nested-CV / selection-bias; justifies our protocol.
- **Towards Reliable Testing for Multiple IR System Comparisons** — ECIR 2025. arXiv 2501.03930 [?]. Wilcoxon + BH-FDR for our multiplicity problem.
- **Green IR** — Scells, Zuccon et al. SIGIR 2022 [a]. The green-IR framing.

### Multimodal / video retrieval SOTA (base modernization — verify [?] before citing)
- **CLAMR** — arXiv 2506.06144 [a]. Contextualized multimodal late interaction (frames+ASR+OCR+metadata) on MultiVENT 2.0. Replaces our plain caption-ColBERT tier.
- **VidVec** — arXiv 2602.08099 [?]; **Qwen3-VL-Embedding** — arXiv 2601.04720 [?]. Modern retriever tiers to add — **verify both**.
- **GRAM** — ICLR 2025. arXiv 2412.11959 [a]. Gramian multi-modality alignment.
- **MAGMaR 2026 shared task** — arXiv 2606.12295 [?]. Concluded text/caption + LLM reasoning beats operating on video directly — re-validates the Q2E thesis.

### Multilingual video retrieval (the moat)
- **C2KD** (ICASSP 2023, arXiv 2210.03625) [a]; **LanguageBind** (ICLR 2024, arXiv 2310.01852) [a]; **mCLIP** (ACL 2023) [a]. No benchmark covers Ukrainian or Kyrgyz — a small Ru/Uk/Ky eval slice is novel data.

## Immediate to-do before submission
1. Re-verify every **[a]/[?]** arXiv ID page-by-page (esp. the 2026 IDs and VidVec/Qwen3-VL-Embedding/Cost-of-Pass/AcuRank/sMARE).
2. Confirm PQPP's exact venue/ID (2406.04746 vs a CVPR page).
3. Pull BibTeX for all **[V]** entries first (they are safe).
