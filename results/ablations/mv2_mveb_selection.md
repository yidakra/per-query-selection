# MVEB per-query system selection

Choose between supervision's two first-stage systems per query. Single-gold identity judgments, five-fold out-of-fold calibration, no retrieval run here.

| pool | n | base | 01mv | best fixed | oracle | pre above | score above | control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| acaps | 665 | 0.4446 | 0.3801 | 0.4446 | 0.5164 | 1/11 | 0/10 | +0.0000 (p=1.000) |
| anet | 4884 | 0.6715 | 0.6054 | 0.6715 | 0.7127 | 0/11 | 0/10 | -0.0004 (p=0.404) |
| didemo | 999 | 0.5615 | 0.5851 | 0.5851 | 0.6529 | 0/11 | 0/10 | +0.0030 (p=0.372) |
| mrvmteb | 879 | 0.6969 | 0.6913 | 0.6969 | 0.7655 | 3/11 | 1/10 | +0.0050 (p=0.377) |
| vatex | 1000 | 0.7773 | 0.7928 | 0.7928 | 0.8462 | 2/11 | 6/10 | +0.0092 (p=0.025) |
| vgga | 696 | 0.3567 | 0.3178 | 0.3567 | 0.4269 | 0/11 | 0/10 | +0.0065 (p=0.258) |
| vggv | 696 | 0.9553 | 0.9686 | 0.9686 | 0.9833 | 0/11 | 0/10 | +0.0013 (p=0.552) |
