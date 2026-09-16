# MVEB per-query system selection

Choose between supervision's two first-stage systems per query. Single-gold identity judgments, five-fold out-of-fold calibration, no retrieval run here.

| pool | n | base | 01mv | best fixed | oracle | pre above | score above | control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mrvmteb | 879 | 0.6969 | 0.6913 | 0.6969 | 0.7655 | -- | 1/10 | +0.0050 (p=0.377) |
| didemo | 999 | 0.5615 | 0.5851 | 0.5851 | 0.6529 | -- | 0/10 | +0.0030 (p=0.372) |
