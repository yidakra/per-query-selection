# A real dense channel over the ASR transcripts

MultiVENT 2.0 ships `whisperASR_clip.json`, a ranked list built by scoring Whisper transcripts with
CLIP's text tower. That tower was trained to match images and truncates at 77 tokens, so it throws away
most of a transcript that runs 747 characters at the median. MMMORRF gets its lift from running an
actual text retriever over the same transcripts. The benchmark also ships the raw transcripts, so we
can do that ourselves and then re-test routing on a channel worth routing over.

`mv2_extract_text.py` pulls the transcripts out of the feature archives (109,488 docs with speech,
98,805 with on-screen text). `mv2_dense_channel.py` encodes them with a multilingual bi-encoder,
splits long ones into 1,200-character windows with 200 overlap, and scores a document as its best
window.

## Two encoders, very different outcomes

| channel | nDCG@10 |
|---|---|
| shipped `whisperASR_clip` | 0.2666 |
| dense, multilingual-e5-base | 0.1248 |
| dense, BAAI/bge-m3 | **0.3134** |
| shipped visual CLIP (for reference) | 0.3036 |

e5-base loses to the channel it was supposed to replace, by a lot. bge-m3 beats it and edges past the
visual channel too.

## The e5 failure is cross-lingual, and only cross-lingual

Grouping queries by the language spoken in their relevant videos:

| language | queries | dense e5 | dense m3 | shipped ASR | visual |
|---|---|---|---|---|---|
| english | 577 | 0.3967 | 0.4293 | 0.2850 | 0.5222 |
| chinese | 539 | 0.0414 | 0.2485 | 0.2827 | 0.1673 |
| korean | 463 | 0.0309 | 0.2450 | 0.1805 | 0.1362 |
| russian | 442 | 0.0360 | 0.3264 | 0.2750 | 0.3057 |
| arabic | 330 | 0.0326 | 0.1889 | 0.2720 | 0.3806 |
| spanish | 195 | 0.1316 | 0.4930 | 0.3448 | 0.2965 |
| **all** | 2,546 | 0.1248 | 0.3134 | 0.2666 | 0.3036 |

On English, e5 beats the shipped channel by 11 points. Everywhere else it falls to near zero. Queries
are English throughout, so English rows ask the encoder for monolingual matching and the rest ask it to
align an English query against a foreign-language transcript.

Nothing is missing from the input. 97-100% of relevant docs in every language have transcripts, and
Russian ones average 1,088 characters. Recall confirms it is not a ranking-order problem either:
R@1000 is 0.360 for e5 against 0.578 for the shipped channel, so the documents never enter the pool.
e5-base is multilingual in the sense of handling many languages one at a time. Its training pairs are
largely monolingual and English-to-foreign retrieval is a known weak spot. About two thirds of the
relevant videos here are non-English, so that weak spot eats the channel.

bge-m3 is trained with cross-lingual objectives and holds 0.19-0.49 across all six languages. It still
loses to the shipped channel on Chinese and Arabic, which is worth chasing later.

This also explains a design choice in MMMORRF that is easy to read past. Their dense component is
PLAID-X with translate-distill, a model built for *cross-language* retrieval rather than a merely
multilingual one. On this benchmark that distinction decides the whole result.

## What the stronger channel does to routing

| | shipped ASR channel | dense bge-m3 channel |
|---|---|---|
| cheap (visual only) | 0.3036 | 0.3036 |
| best uniform fusion | 0.2796 | 0.3408 |
| routed | 0.3221 @ f=0.55 | **0.3545** @ f=0.70 |
| gain vs cheap | +1.84 | **+5.09** |
| gain vs uniform | +4.25 | +1.37 |
| per-query gain: mean / sd | −2.40 / 24.1 | +3.72 / 23.1 |
| helps / hurts | 27% / 35% | 34% / 28% |
| oracle, best single channel per query | 0.4402 | 0.4777 |

The routing gain against visual-only nearly triples. But the story behind it changes, and the one-pager
needs updating for that. With the shipped channel, uniform fusion *lost* 2.4 points and routing was
repairing a channel that was net-harmful. With bge-m3, uniform fusion helps by 3.7 and routing adds
1.37 on top of the best weight setting we could find.

So "uniform fusion of an informative channel can be worse than ignoring it" no longer describes ASR.
It still describes OCR exactly: uniform fusion loses 5.92 there and the router declines the channel
outright at f=0.00, same as it declined the LLM expansion tier.

The claim that survives both cases is the one about spread, not sign. Per-query gain has sd 23.1 while
its mean is 3.72, roughly six times its own average effect, and the channel helps 34% of queries while
hurting 28%. That heterogeneity is what routing converts into a gain, whether the mean happens to be
positive or negative. It is also the version that matches the ρ=0.943 heterogeneity law from
`router_findings.md`.

The best-single-channel oracle at 0.4777, against 0.3134 for the best fixed channel, says there is a
lot left on the table for a per-query policy.

## Cost

Encoding 175,716 windows with bge-m3 took 67 minutes on one A2, which is a one-off corpus-side cost
that every policy pays equally. At query time the channel costs 1.3 ms to encode and 0.7 ms to search
109,488 documents, so it sits in the same cost class as the existing cheap tiers rather than anywhere
near the 9.4 s LLM stage.

## Loose ends

Chinese and Arabic still favour the shipped channel, and I have not looked at why. A translate-distill
model is the obvious next thing to try, since that is what the systems above us use. The OCR text is
extracted but no dense channel has been built over it yet.

Artifacts: `mv2_channels_dense_m3.json`, `mv2_channels_dense_e5.json`, and the per-cell router files
tagged `_dense_m3` / `_dense_e5`. Runs with substituted channels need `--cell-tag`, otherwise they
overwrite the shipped-channel cells that other analyses read.
