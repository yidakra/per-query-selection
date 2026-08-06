## LLM size (LLaMA) ablation: MultiVENT text-only (Q2E − Video), noASR NDCG@10

| variant | NDCG@10 |
|---|---:|
| LLaMA-1B | 61.18 |
| LLaMA-8B | 62.47 |
| LLaMA-3.3-70B | 64.83 |

Monotonic in model size: event-decomposition quality improves with the LLM that
generates the paraphrases (+1.29 from 1B→8B, +2.36 from 8B→70B). The 1B artifact
required the `Q2E_EVENT_MAXPARAS=32` cap (pre-cap max 270 degenerate paraphrases/event
vs 8B's 35) to be evaluable; see `repro_log.md`. Text-only (Q2E − Video) so the number
is encoder-independent and isolates the LLM's contribution.
