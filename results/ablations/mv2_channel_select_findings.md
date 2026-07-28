# K-way channel selection: choosing evidence per query beats every fixed policy

The pairwise cells ask "escalate or not" for one channel at a time. This asks the K-way question:
given visual, ASR and OCR, pick the channel *subset* to trust for each query. The selector is a
multi-target ridge over 30 cheap features (each channel's confidence features plus pairwise top-10/100
candidate overlap), predicting per-query nDCG for all 7 unit-weight RRF policies, argmax out-of-fold.
Every channel here is cheap, so the claim is about accuracy. The decision reads only score
distributions that are computed either way. `mv2_channel_select.py`.

The headline number is nested: the best fixed policy is chosen on the training fold too, so neither
side of the comparison peeks. Permutation p is the floor of 2000 shuffles in all three cells.

| cell | best fixed policy | selected (oof) | nested gap | perm p |
|---|---|---|---|---|
| shipped channels | visual, 0.30364 | **0.36640** | **+6.20 ± 0.29** | .0005 |
| dense ASR (bge-m3) | asr+visual, 0.33715 | **0.41714** | **+8.09 ± 0.65** | .0005 |
| dense ASR + dense OCR | asr+visual, 0.33715 | 0.41502 | +7.83 ± 1.01 | .0005 |

These are the largest gaps in the project. The previous best was +3.07 (pairwise channel routing);
the best fixed *or routed* number on these channels was 0.3545. The picks histogram is the telling
part: in the dense cell the selector takes a single channel for 71% of queries (asr 986, visual 815)
and the asr+visual fusion for 25% (644). Fusing everything everywhere is exactly what it learns not
to do.

## The per-query-best oracle does not survive a gold split

The selection cells report an oracle of 0.4669 (shipped) / 0.5181 (dense). Before anyone divides by
it: the section-5 gold-split test says it is mostly fitting capacity. Pick each query's best policy on
one half of its golds, grade on the other half (`mv2_select_goldsplit.py`, 5 seeds, 1,760 splittable
queries at 5.25 golds/query):

| cell | in-sample | out-of-sample | fixed (chosen on A) | optimism | oos oracle vs fixed |
|---|---|---|---|---|---|
| shipped | 0.3788 | 0.2325 | 0.2453 | **+14.63** | −1.28 |
| dense ASR | 0.4112 | 0.2591 | 0.2524 | **+15.21** | +0.68 |
| dense ASR+OCR | 0.4169 | 0.2573 | 0.2524 | **+15.96** | +0.49 |

Out of sample the oracle roughly ties the best fixed policy. The 14–16 points of apparent headroom
are the oracle memorising which videos this half of the labels happens to mark relevant. This is the
tier-C optimism result at larger scale, and it again tracks the spread in option quality rather than
the option count.

The learned selector is immune by construction: its features never touch labels and every prediction
is out-of-fold. The nested protocol even picks its fixed baseline on train. A pooled learner beating a
per-query oracle fed noisy labels is the same non-paradox as section 5. The +6 to +8 is real; the
oracle's extra 10 points were never there. "% of oracle captured" stays retired.

## Dense OCR barely helps, and the reason is the modality

bge-m3 over the raw on-screen text scores 0.13295 against 0.1223 for the shipped CLIP-tower list.
Compare ASR, where the same swap bought 4.7 points (0.2666 to 0.3134). The encoder-decides-everything
story from the dense-ASR finding does not transfer; OCR text on this benchmark carries little signal
however it is scored. Two small consolations. The dense scorer turns the visual→+OCR cell from
"router declines at f=0.00" into a marginal purchase (+0.40 at f=0.50). And in the all-dense selection
cell OCR-containing policies are picked for 352 queries. Neither moves the aggregate: adding dense OCR
to the selector changes 0.41714 into 0.41502, a wash. OCR keeps its role as the channel uniform
fusion should fear (fusing it uniformly still loses 5.9 points) and routing should mostly skip.

## Scope and loose ends

- Policies are unit-weight RRF subsets. Per-policy weight tuning would grow the policy set and
  probably the gap; it also multiplies the ways to overfit, so it waits for a reason.
- The features deliberately use every channel's score distribution. For channel selection that is
  legal by design; it would not be legal for a cost cascade, where the expensive tier's scores do not
  exist before the decision.
- A classification head (predict the argmax directly) and richer disagreement features are untried.
- The selector's absolute numbers still sit on cheap channels. MMMORRF-class dense retrieval per
  channel plus this selector is the obvious composition, and nothing in the protocol changes.

Artifacts: `mv2_channel_select{,_dense_m3,_all_dense_m3}.json`,
`mv2_select_goldsplit{,_dense_m3,_all_dense_m3}.json`, `mv2_channels_all_dense_m3.json`, the pairwise
cells `mv2_chan_visual_to_{asr,ocr,all}_all_dense_m3.json`, and the OCR ranked list
`data/multivent2/ocr_dense_bge-m3.json` (874 s to encode on one A2).
