# Efficiency / cost metrics for adaptive-retrieval & routing papers — lit review

Purpose: answer the supervision ask ("you didn't measure efficiency with traditional metrics — do a
lit review to find what people use, since we do routing"). Resolves "traditional metrics" into two
vocabularies: (a) the **IR/systems** efficiency vocabulary (latency, throughput/QPS, index/memory,
FLOPs) and (b) the **cost–quality-tradeoff** vocabulary that routing/cascade/adaptive-retrieval papers
standardized (call-rate, cost/query, area-under-the-cost-quality-curve, deferral/risk-coverage). We
already have most of (b); the gaps are mostly in (a) and a couple of (b)'s presentation conventions.

## 1. Master metric table

| Metric | Definition / formula | Papers | Captures |
|---|---|---|---|
| Avg #retrievals (steps) / query | mean count of retrieval/gen calls per query | Adaptive-RAG (2403.14403), Self-RAG (2310.11511), FLARE (2305.06983), Mallen (2212.10511) | compute ∝ how often the expensive path fires; the adaptive-retrieval analog of our escalation fraction |
| Retrieval-call rate / "% queries that retrieve" | fraction of queries that trigger retrieval at all | FLARE, Self-RAG, Mallen 2022, Adaptive Gating (2511.09803) | selectivity of the "when to act" policy |
| Escalation / deferral fraction (f) | fraction routed to the strong/expensive model | RouteLLM (2406.18665), FrugalGPT (2305.05176), Tabi (EuroSys'23) | **our f**; drives cost |
| % calls to the strong model | same as f, phrased for a 2-model router | RouteLLM, RouterBench (2403.12031) | cost proxy at fixed per-call price |
| Cost per query ($/query) | avg dollar/token-priced inference cost per query | FrugalGPT, RouterBench, RadialRouter (2506.03880) | monetary efficiency; canonical x-axis |
| PGR — Performance Gap Recovered | (perf(router)−perf(weak))/(perf(strong)−perf(weak)) | RouteLLM | quality normalized between cheap/expensive endpoints |
| APGR — Average PGR | area under the call–performance curve, averaged over budget | RouteLLM | single-number cost-quality summary — **we report this** |
| CPT(x) — Call-Performance Threshold | min % strong-model calls to reach PGR=x% (CPT50, CPT80) | RouteLLM | budget to hit a quality target — **we report CPT50/CPT80** |
| AIQ — Average Improvement in Quality | area under the absolute cost–quality (acc vs $) curve | RouterBench | APGR's absolute-axis sibling |
| Cost–quality Pareto frontier | quality vs cost across operating points; report non-dominated frontier | FrugalGPT, RouterBench, RouteLLM, Tabi, IR eff.–eff. papers | the field's signature *figure* |
| Cost reduction @ iso-quality | "% cost saved to match model X" (FrugalGPT: match GPT-4 at ≤98% less) | FrugalGPT, RouteLLM (3.66×) | fix quality, report savings |
| Quality retained @ iso-cost | quality at a fixed budget fraction (acc at 20% strong-calls) | RouteLLM, RouterBench, Tabi | fix cost, report quality |
| Accuracy-at-budget | quality vs an explicit compute/$/latency budget | RouterBench, PROTEUS (2601.19402), Tabi (SLO) | budget-constrained operating point |
| Cost per correct answer / acc-per-$ | total cost ÷ #correct (or quality÷cost) | RouterArena (2510.00202), RadialRouter | normalizes spend by *useful* output |
| Query latency (mean/median/p95/p99) | wall-clock per query; central tendency **and tails** | ColBERT (SIGIR'20), PLAID (2205.09707), BEIR (2104.08663) | responsiveness; tails matter under SLOs |
| Throughput / QPS | queries ÷ total wall-clock | ColBERT/PLAID, ANN benchmarks, IR eff. special issue | system capacity; standard IR axis |
| Latency–throughput (QPS-latency) curve | latency vs offered QPS load | ANN/vector-search benchmarks, PLAID | behavior under load/contention |
| FLOPs / FPO per query | float ops to produce one result | Green AI (1907.10597), ColBERT (~4 orders < cross-encoder) | hardware-independent compute |
| Index size / memory footprint | bytes to store index/embeddings (ColBERTv2 16–25 GiB, 6–10× smaller) | ColBERT/ColBERTv2 (2112.01488), PLAID, BEIR | storage/deploy cost of retrieval side |
| Energy / query (J or kWh) | measured GPU+CPU energy per query | Green AI, Strubell (1906.02243), ML CO2 (1910.09700), CodeCarbon, Henderson (2002.05651) | direct energy efficiency — **we report joules/query** |
| Carbon (gCO₂e) | energy(kWh) × grid intensity × PUE | Strubell'19, ML CO2 Impact, CodeCarbon | environmental cost; expected alongside energy |
| Router / overhead cost | extra latency/compute the router adds per query | FrugalGPT, Tabi, RouteLLM | whether routing machinery is worth it — **we report 345 µs/query** |
| Risk–coverage curve & AURC | risk (error on answered) vs coverage; AURC = area under it | selective-prediction lit.; NeurIPS'24 flaws-in-selective-classification | routing ≈ selective escalation; principled acc-vs-defer tradeoff |

## 2. Expected vs nice-to-have

**Expected ("traditional metrics") — a routing/adaptive-retrieval paper reads as incomplete without:**
1. **Query latency as a distribution** — mean/median **and a tail (p95/p99)**, per operating point. Our affine fit (83.6s + 0.049s/len) is a good *model* but not the expected reporting form.
2. **Throughput / QPS** — the single most standard IR efficiency number. Current gap.
3. **Cost–quality Pareto curve** (accuracy vs cost/latency/energy), router points overlaid on fixed-tier baselines + oracle. The canonical figure. We have the ingredients (APGR/CPT/f/energy); must plot it.
4. **Compute/query in a hardware-independent unit** — FLOPs/FPO (Green AI). We have estimates; report per tier.
5. **Escalation fraction f + cost consequence** — have it; frame as "% calls to expensive tier" (RouteLLM).
6. **Router overhead** — have 345 µs/query; keep.
7. **APGR + CPT** — have them; the modern routing-specific standard; satisfies "we do routing."

**Nice-to-have:** AIQ (RouterBench); risk–coverage / AURC (routing as selective prediction); carbon gCO₂e + kWh (trivial from joules); index/memory footprint; cost-per-correct-answer / acc-per-$; QPS-latency-under-load curve.

## 3. Mapping: our measurements → standard vocabulary

| Standard metric | Have it? | As | Action |
|---|---|---|---|
| % calls to expensive tier | yes | escalation fraction f | rename "% strong-tier calls" vs RouteLLM |
| Cost/query (compute) | partial | joules, tokens, FLOPs | consolidate one x-axis unit |
| Cost/query ($) | gap | — | add if any priced API; else "self-hosted, cost = energy/FLOPs" |
| APGR | yes | APGR | keep |
| CPT(x) | yes | CPT50/80 | keep |
| AIQ | no | — | optional add |
| Cost–quality Pareto figure | ingredients | f, energy, APGR | **produce the plot** vs baselines + oracle |
| Cost reduction @ iso-quality | derivable | from CPT | report "X% energy saved to match all-expensive-tier" |
| Quality @ iso-cost | derivable | from curve | report "quality at f=20%" |
| Latency mean/median | fit only | 83.6s+0.049s/len | **also report measured mean/median per tier** |
| Latency p95/p99 | gap | — | add tails |
| Throughput / QPS | gap | — | **add** |
| FLOPs/query | yes | estimates | report per tier |
| Index/memory | gap | — | add embedding/index size if space |
| Energy/query (J) | yes | NVML/CodeCarbon | keep (explicit gpu_ids caveat) |
| kWh / gCO₂e | trivial | from joules | convert for green-AI readers |
| Router overhead | yes | 345 µs/query | keep |
| Risk–coverage + AURC | gap | — | optional but strong |
| Cost per correct answer | gap | — | cheap add |

## 4. Gaps to fill (flagged)

1. **Throughput / QPS** — most conspicuous omission; every IR efficiency paper reports it.
2. **Per-query latency as an empirical distribution (mean/median + p95/p99), not just the affine fit** — tails matter because escalation inflates upper percentiles.
3. **The cost–quality Pareto frontier as an actual figure** with baselines + oracle bound.
4. **Cost-per-correct-answer / accuracy-per-joule** — normalizes spend by useful output; rhetorically strong.
5. **Deferral / risk–coverage curve (AURC)** — routing = selective escalation; principled presentation; currently absent.
6. **Carbon (gCO₂e) + kWh** — derive from joules if energy is a headline axis.
7. **Index / memory footprint** of the video-retrieval side — standard in IR; secondary for a router paper.

**Net:** our routing-side metrics (APGR, CPT, f, overhead, energy) are already in the field's modern
vocabulary. The supervisor's "traditional metrics" gap is almost entirely on the **IR/systems side**:
add **QPS/throughput**, **measured latency mean+p95/p99**, the **plotted cost-quality Pareto curve**,
and **cost-per-correct-answer**; optionally **AURC**, **gCO₂e/kWh**, **index size**.

## Citations

Adaptive-RAG (2403.14403, NAACL'24); Self-RAG (2310.11511, ICLR'24); FLARE (2305.06983, EMNLP'23);
Mallen "When Not to Trust LMs" (2212.10511, ACL'23); Adaptive Gating (2511.09803); RouteLLM
(2406.18665, ICLR'25); RouterBench (2403.12031); FrugalGPT (2305.05176); Tabi (EuroSys'23);
RadialRouter (2506.03880); RouterArena (2510.00202); PROTEUS (2601.19402); ColBERT (SIGIR'20);
ColBERTv2 (2112.01488); PLAID (2205.09707); BEIR (2104.08663); MTEB (2210.07316); Green AI
(1907.10597); Strubell (1906.02243, ACL'19); ML CO2 Impact (1910.09700); experiment-impact-tracker
(2002.05651, JMLR'20); selective-classification evaluation (NeurIPS'24).
