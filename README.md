# per-query-selection

Code and results for a study of **adaptive retrieval decisions in multilingual video search**: does
deciding which evidence channel to search, which query formulation to run and which language to ask
in require evidence from the candidate rankings, or is evidence available before retrieval enough?
Experiments run on MultiVENT 2.0, with robustness checks on MSR-VTT-1kA and MVEB.

`reports/task_definition.md` defines the decisions, collections, judgments, query sets, evaluation
protocol and predictor families, and is the place to start. `reports/evidence.md` records every
result with its caveats and the artifact it comes from.

## Paper results and where they come from

Every number in the paper is computed by a script in `src/multivent2/` and stored in
`results/ablations/`. Each script's docstring gives its exact command line.

| Result in the paper | Script | Artifact |
|---|---|---|
| Main table, channel decision: learner over retrieval-outcome features (0.4131, +7.59) | `mv2_channel_select.py` | `mv2_channel_select_dense_m3_grouped.json` |
| Main table, channel decision: same learner over corpus statistics (+0.74), and the best single pre- and post-retrieval predictor | `mv2_qpp_prechannel.py --single` | `mv2_qpp_prechannel_single.json` |
| Main table, rewriting decision (outcome; with and without method indicator) | `mv2_variant_select_learned.py` | `mv2_variant_select_learned_centred{,_nomethod}.json` |
| Main table, rewriting decision (corpus statistics) | `mv2_variant_corpus_features.py` | `mv2_variant_select_learned_corpus.json` |
| Main table, language decision (outcome / corpus statistics) | `mv2_language_select_learned.py` | `mv2_language_select_learned{,_corpus}.json` |
| Single predictors on the channel cells (0 of 33, equivalence, score-only, BERT-QPP) | `mv2_qpp_table.py`, `mv2_row_inference.py` | `mv2_row_inference.json`, `mv2_table1_nested_grouped.md` |
| Single predictors on the language decision | `mv2_axis_inference.py` | `mv2_axis_inference.json` |
| Single predictors and fusion baselines on the rewriting decision | `mv2_variant_selection.py` | `mv2_variant_selection_full.json` |
| Fusing all languages | `mv2_plan_additions.py` | `mv2_plan_additions.json` |
| Answer quality of the rewriting picks | `mv2_rag_nuggets.py`, `mv2_variant_nugget_tests.py` | `rag/variant_vs_original_tests.json` |
| Robustness: translated, caption and union indexes | `mv2_qpp_table.py`, `mv2_row_inference.py` | `mv2_row_inference_{en,supcap,supcap27b,supcap_plus}.json` |
| Robustness: seven query formulations | `mv2_variant_task_a.py` | `mv2_row_inference_va_*.json` |
| Robustness: MSR-VTT-1kA | `mv2_msrvtt_source_replication.py` | `mv2_msrvtt_source_replication*.json` |
| Robustness: MVEB | `mv2_mveb_selection.py` | `mv2_mveb_selection.json` |
| Robustness: stronger speech channel | `mv2_rerank_channel.py`, `mv2_plaidx_channel.py`, `mv2_channel_select.py` | `mv2_channel_select_{rr,plaidx,plaidx_ocrm3}_grouped.json` |
| Query-only LLM router | `mv2_llm_router.py` | `mv2_llm_router{,_binary}.json` |
| Oracle audit (split labels) | `mv2_axis_goldsplit.py`, `mv2_variant_goldsplit.py` | `mv2_axis_goldsplit_*.json` |

Inputs that are generated rather than shipped: query rewritings (`mv2_generate_variants.py`, six
QueryGym methods, five samples each, Qwen2.5-7B-Instruct at temperature 0.6) and query translations
(`mv2_translate_queries.py`, NLLB-200 600M).

## Data

Place the MultiVENT 2.0 test release (queries, judgments, speech and on-screen text, shipped runs) in
`data/multivent2/`. `data/` is not version-controlled. The two caption sets from Qwen3.5-9B and
Qwen3.5-27B used in the robustness checks were provided by their authors and are not redistributed;
the scripts that read them are included.

## Environment

```bash
uv venv --python 3.10 .venv-eval
source .venv-eval/bin/activate
uv pip install -r env/requirements-eval.txt
```

`environment/` records the hardware and package versions the results were produced with.
Experiments log to Weights & Biases offline by default (`src/evaluation/tracking.py`).

## Layout

- `src/multivent2/`: the experiments behind the paper.
- `src/evaluation/`: shared evaluation code, tracking, and the earlier Q2E reproduction.
- `results/ablations/`: result artifacts (JSON) and per-experiment notes.
- `reports/`: task definition, evidence record, predictor notes, figures.
The project's first phase (adaptive routing over a reproduced Q2E pipeline) had its own README,
reports and server scripts. The paper does not depend on them, and they remain in the git history.
