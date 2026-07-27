# Routing across modality channels: channel value is per-query, weak or strong

This doc covers the **shipped provided channels**. The ASR list here is the benchmark's weak
CLIP-text scoring of the transcripts; a real dense retriever over the same transcripts is a separate
result in `mv2_dense_channel_findings.md`, and it flips ASR's sign without changing the routing
conclusion. Read the two together: the invariant is the per-query spread, not whether a channel's
average effect is positive or negative.

Our cascade is a cheap visual+caption stack (nDCG@10 0.30–0.37), while the strong systems on MultiVENT
2.0 (MMMORRF 0.586, CLaMR 0.585) get their lift from speech and on-screen-text channels we were not
using. Those channels do not have to be built: the benchmark ships them as ranked lists next to the
CLIP run we already use — `ranked_lists/whisperASR_clip.json` and
`ranked_lists/10pyscene_paddleOCR_clip.json` (84 MB each, same 2,546 queries × 1,000 candidates). So a
MMMORRF-style weighted RRF over visual + ASR + OCR is reproducible here on CPU at zero retrieval cost.
`src/multivent2/mv2_channels.py`.

## 1. Each channel alone, and fused the usual way

| | nDCG@10 |
|---|---|
| visual (CLIP) | **0.30364** |
| ASR (Whisper transcripts) | 0.26665 |
| OCR (PaddleOCR) | 0.12226 |
| visual + ASR (RRF, best fixed weight 0.5) | 0.27955 |
| visual + ASR + OCR (RRF, best fixed weights) | 0.25937 |
| visual + OCR | 0.24448 |
| ASR + OCR | 0.14173 |

**With these shipped lists, every fusion is worse than the visual channel alone.** This is not a
weighting failure — the sweep over ASR ∈ {0.5, 1, 1.5, 2} × OCR ∈ {0, 0.25, 0.5, 1} never beats
0.30364; the best point (ASR 0.5, OCR 0) is 0.27955. Uniform RRF hands a fixed share of the vote to a
channel that, for most queries, has nothing useful to say, and the noise it injects costs more than the
signal it adds. (Swap the weak ASR list for the dense retriever and this table changes: uniform
visual+ASR then reaches 0.3408, above visual alone. The per-query spread and the routing gain on top
survive the swap. See `mv2_dense_channel_findings.md`.)

## 2. The reason: channel usefulness is a per-query property

| escalation cell | mean Δ | sd | helped | hurt |
|---|---|---|---|---|
| visual → +ASR | −2.41 | 24.06 | 27% | 35% |
| visual → +OCR | −5.92 | 19.41 | 14% | 42% |
| visual → +ASR+OCR | −4.43 | 24.89 | 24% | 38% |

The spread is roughly **ten times** the mean effect. A silent protest clip has no speech to match; a
news studio segment is almost all speech. Averaging over that distribution is what destroys the channel.

The ceiling this heterogeneity implies is large: an oracle that picks the single best channel per query
scores **0.44022**, against 0.30364 for the best fixed channel — a **+13.7 point** headroom. Treat that
as an upper bound, not a promise: this project has twice seen oracle headroom evaporate under a real
predictor (tier C, B→Full), which is exactly why the next section is the one that matters.

## 3. It survives a real predictor

Out-of-fold ridge on the visual channel's own confidence features (score margins, entropy, top-k mass —
all free, the visual channel has already run). Escalate the top-*f* queries by predicted gain.

| cell | τ (perm p) | nested gap vs cost-matched random | cheap only | uniform fusion | **routed** | oracle |
|---|---|---|---|---|---|---|
| visual → +ASR | **+0.217** (p=.0005) | **+3.07 ± 0.38** | 0.30364 | 0.27955 | **0.32206** @ f=0.55 | 0.36530 |
| visual → +ASR+OCR | — | — | 0.30364 | 0.25937 | **0.31432** @ f=0.35 | 0.35854 |
| visual → +OCR | — | — | 0.30364 | 0.24448 | 0.30364 @ **f=0.00** | 0.33047 |

Three things to take from this table:

1. **Routing turns a harmful channel into a useful one.** ASR fused everywhere costs 2.41 points; ASR
   fused where the router says costs nothing and gains **+1.84 over visual-only, +4.25 over uniform
   fusion**. The nested-CV gap of **+3.07 ± 0.38** is the largest routing gap anywhere in this project
   (previous best: +2.23 for the caption tier on the same queries, +1.68 on MultiVENT v1).
2. **τ = +0.217 is also our strongest predictability.** Compare +0.127 (caption tier) and +0.002/+0.035
   (LLM expansion tier at 7B/14B). The routing signal is much stronger across channels than across
   expansion tiers.
3. **The router correctly refuses OCR.** Its optimum is f = 0.00 — escalate nothing. OCR is too weak
   (0.122 alone) for the predictor to find a subset worth paying for, so the honest outcome is a
   decline, not a manufactured gain. A router that only ever finds reasons to spend would be suspicious;
   this one declines a channel and declines the LLM expansion tier.

CPT₅₀/CPT₈₀ are undefined for these cells: PGR normalizes between the cheap and expensive endpoints, and
here the "expensive" endpoint is *worse* than the cheap one. Report APGR (0.332 for ASR) and the nested
gap instead, and say why.

## 4. Why this reframes the project

The work so far routes between **LLM query-expansion tiers**, where the honest conclusion was mostly
negative — the Full tier is expensive, barely helps, and is close to unroutable. Routing between
**modality channels** is the same question asked where the answer is loud:

- larger effect (+3.07 nested gap vs +2.23 and ~0),
- reproducible on CPU from shipped artifacts, no LLM generation cost at all,
- and it speaks directly to what the strong systems on this benchmark do — MMMORRF's contribution *is*
  a weighted fusion over these modality channels. Our claim becomes: those weights should be per query,
  not global.

Caveat to keep stating: the absolute numbers are low because these are the benchmark's cheap provided
channels, not MMMORRF's retrievers. We have since built a stronger one. A bge-m3 dense retriever over
the raw transcripts (`features/test/whisper_asr.zip`) scores 0.3134, above visual, and we re-ran this
routing test on top of it. Routing still pays (+5.09 over visual-only, +1.37 over the best fixed weight,
τ +0.170), so the conclusion here does not depend on the channel being weak. Details, and the
cross-lingual reason the first encoder failed, are in `mv2_dense_channel_findings.md`.
