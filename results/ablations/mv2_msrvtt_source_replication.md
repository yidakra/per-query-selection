# MSR-VTT source-selection replication

Two tasks: add-caption is visual versus visual+caption fusion; choose-source is visual-only versus caption-only. All calibration is five-fold out of fold; no test-label operating-fraction sweep.

| task / encoder / evidence | option A | option B | control | oracle | pre above / deg. | score above / deg. | max |tau| pre / score |
|---|---:|---:|---:|---:|---:|---:|---:|
| add-caption/multiclip/noASR | 0.5972 | 0.6094 | 0.6081 | 0.6130 | 0/11 / 5 | 0/10 / 7 | 0.052 / 0.052 |
| choose-source/multiclip/noASR | 0.5972 | 0.4787 | 0.6009 | 0.6737 | 0/11 / 9 | 0/10 / 9 | 0.065 / 0.062 |
| add-caption/multiclip/ASR | 0.5972 | 0.6195 | 0.6182 | 0.6209 | 0/11 / 6 | 0/10 / 9 | 0.066 / 0.050 |
| choose-source/multiclip/ASR | 0.5972 | 0.5273 | 0.6298 | 0.7009 | 0/11 / 4 | 1/10 / 9 | 0.044 / 0.078 |
| add-caption/internvideo2/noASR | 0.6600 | 0.6752 | 0.6753 | 0.6811 | 0/11 / 9 | 4/10 / 2 | 0.040 / 0.112 |
| choose-source/internvideo2/noASR | 0.6600 | 0.4787 | 0.6610 | 0.7157 | 0/11 / 9 | 0/10 / 9 | 0.089 / 0.086 |
| add-caption/internvideo2/ASR | 0.6600 | 0.6886 | 0.6886 | 0.6939 | 0/11 / 8 | 0/10 / 3 | 0.037 / 0.115 |
| choose-source/internvideo2/ASR | 0.6600 | 0.5273 | 0.6759 | 0.7355 | 0/11 / 11 | 0/10 / 10 | 0.046 / 0.077 |
