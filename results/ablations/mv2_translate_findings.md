# Translating the speech channel: what it bought, and where

The question was narrow. MMMORRF (0.586) and OmniEmbed (0.753) sit well above our channels, and part of
their recipe is translate-distill dense retrieval per channel. We retrieved speech in the original
language with a multilingual encoder instead. So: how much of the gap is the translation step alone?

All 109,488 ASR transcripts went through NLLB-200-1.3B to English (20.3 h on one A2), then back through
the same bge-m3 encoder (65 min, 209,381 windows). Everything else held fixed: same encoder, same
fusion, same queries. `mv2_translate_corpus.py`, `mv2_per_language.py`.

## Aggregate

| policy | nDCG@10 |
|---|---|
| visual only | 0.3036 |
| dense ASR, original language | 0.3134 |
| dense ASR, translated | **0.3332** |
| visual + ASR fused, original | 0.3372 |
| visual + ASR fused, translated | **0.3452** |

Two points on the speech channel, one on the fusion. Real, and much smaller than the distance to
MMMORRF. Translation is not what separates us from them.

## Per language

Queries are English. The language belongs to the video. And 22.5% of queries have relevant videos in
more than one language, so tagging each query with a single language throws information away. Instead
the judgments are restricted one language at a time: keep the judgments whose `video_language` is L,
keep the queries that still have something relevant to find, score the unchanged ranked lists against
that reduced qrels. A query can appear in several rows. That is correct: it really does have relevant
material in each.

| language | queries | visual | ASR orig | ASR +MT | Δ channel | fused orig | fused +MT | Δ fused |
|---|---|---|---|---|---|---|---|---|
| english | 790 | 0.4755 | 0.4349 | 0.3739 | **−0.0610** | 0.4823 | 0.4454 | −0.0369 |
| chinese | 729 | 0.1222 | 0.1835 | 0.1957 | +0.0122 | 0.1699 | 0.1744 | +0.0045 |
| korean | 627 | 0.1045 | 0.2238 | 0.2619 | +0.0381 | 0.2106 | 0.2424 | +0.0318 |
| russian | 563 | 0.2809 | 0.3415 | 0.3870 | +0.0454 | 0.3742 | 0.3989 | +0.0247 |
| arabic | 345 | 0.3446 | 0.1509 | 0.2564 | **+0.1055** | 0.2349 | 0.2809 | +0.0461 |
| spanish | 205 | 0.2783 | 0.4865 | 0.4576 | −0.0288 | 0.3974 | 0.4125 | +0.0152 |

Japanese (7), Ukrainian (6), Cantonese (1) and Malay (1) are in the JSON and too thin to read.

**Arabic is where the translation earns its keep.** 0.1509 to 0.2564, a 70% relative jump, and it moves
Arabic from the worst-served language on the speech channel to mid-table. Russian and Korean gain
usefully too.

**Chinese barely moves.** +0.0122 on the channel, +0.0045 fused, on the second-largest language in the
set. Whatever is wrong with Chinese here is not a retrieval-language problem, so translate-distill will
not fix it either. Chinese also has the weakest visual channel of any language (0.1222).

An earlier version of this note guessed the ASR was the bottleneck. It is not. `mv2_asr_quality.py`
audits the transcripts of every judged document by language: Chinese has 99.3% coverage, a median of 186
content units, which is the densest of the large languages, and a degenerate-repetition rate of 7.2%,
which is *lower* than English at 10.8%. Whisper handled Chinese. (The first pass of that audit said the
opposite, 45% degenerate on a median of 3 tokens, because splitting on whitespace is meaningless for a
script written without spaces. Worth remembering before anyone reads a length statistic off this corpus
again.)

Which leaves Chinese unexplained, and it is more honest to leave it there than to reach for a second
guess. Both its channels are weak, its relevant-documents-per-query is 4.02 against English's 4.38 so
the ideal DCG is comparable, and Korean sits right next to it on transcript quality and visual weakness
while gaining three times as much from translation.

The audit did settle something else. Arabic has the least reliable transcripts in the set, 32.8% in the
wrong script and 15.8% degenerate, and Arabic is the language translation helps most. The gain tracks
how badly the original-language channel was being served, not how far the language sits from English.

**English gets worse, and that is not a bug.** English documents were passed through untouched, and 35.3%
of the corpus comes out byte-identical, with spot checks confirming English transcripts are unchanged. The
drop is contention. Once Russian and Arabic and Korean transcripts read as English, they compete for
English queries and push English relevant documents down the same ranked list. The per-language subtask
restricts the qrels to English relevant documents, so that reshuffling shows up as a loss. Spanish loses
for the same reason and to a smaller degree, being closer to English already.

So the aggregate +0.02 is a net of two opposing movements, not a uniform lift.

## What this implies

Translating the whole corpus is the wrong unit of decision. It helps Arabic a lot, Russian and Korean
somewhat, Chinese barely, and it costs English. A system that translated only the documents that benefit
would keep the +0.1055 on Arabic and give back none of the −0.0610 on English.

That is the same argument this project makes about channels, one level down: the choice is per item, and
a global setting averages a real gain against a real loss and reports the difference. We have not built
the per-document version, and I would not claim it without measuring it. But the shape of the table is
hard to read any other way.

## Does a better channel make routing redundant?

The obvious objection to this project is that per-query routing only pays while the channels are weak,
and that anyone who invests in the channels gets the gain for free. The ladder now has four rungs: the
original dense channel, the translated one, the translated one reranked with a multilingual
cross-encoder (`mv2_rerank_channel.py`, bge-reranker-v2-m3 over the top 100, single-channel nDCG
0.3332 to 0.3515), and the translated transcripts retrieved with HLTCOE's released translate-distill
PLAID-X checkpoint (`mv2_plaidx_channel.py`), the retriever family MMMORRF names as its core, at
0.3673 alone. Same selector, same 30 features, same event-grouped folds each time, one channel
replaced by a better version of itself.

| cell | best fixed policy | selected, out-of-fold | nested gap | single-channel picks |
|---|---|---|---|---|
| dense ASR, original language | 0.3372 (asr+visual) | 0.4131 | +7.59 ± 1.01 | 72% |
| dense ASR, translated | 0.3428 (asr+visual) | 0.4229 | +8.01 ± 1.01 | 76% |
| dense ASR, translated + reranked | 0.3527 (asr+visual) | 0.4319 | +8.68 ± 0.73 | 78% |
| translated + PLAID-X translate-distill | 0.3678 (**asr alone**) | 0.4419 | **+7.41 ± 0.69** | 86% |

The gap does not vanish. Across a 3.1-point climb in the fixed baseline it stays inside a
+7.4 to +8.7 band, rising over the first three rungs and easing at the top one, with the group
sign-flip p below 5e-4 at every rung. There is no trend toward zero, which is what the objection
needs. That is what you would expect if routing exploits *variation* in which channel suits which
query rather than the average weakness of any one channel: making speech better does not make it
better for the queries it was already wrong for. Two other movements at the top rung point the same
way. The best fixed policy becomes the speech channel alone, because global RRF fusion with the
visual channel now costs 2.6 points (0.3418 against 0.3678), and the share of queries answered from a
single channel rises along the whole ladder, 72% to 86%. The stronger the channels, the more the
global-fusion default the benchmark's systems use turns into the thing selection replaces.

This is not proof that the gain survives arbitrarily strong channels. MMMORRF's full system is still
above ours, and only their run files can settle that. But the ladder now ends at a channel built with
their own released retriever, and the objection has no measured rung to stand on.

## Caveats

- Fusion weights come from each cell's own sweep (original: asr 1.0; translated: asr 2.0), chosen on the
  full test set. Both fused columns are therefore best-case fixed policies, which is how the one-pager
  reports uniform fusion elsewhere. The channel columns have no such tuning.
- NLLB-200-1.3B, greedy decoding, no distillation. MMMORRF's translate-distill trains the retriever on
  translated pairs, which is a stronger recipe than translate-then-encode. This measures the cheap half.
- Per-language nDCG on a restricted qrels is not comparable to the aggregate number, since the ideal DCG
  differs. Compare within a column.
