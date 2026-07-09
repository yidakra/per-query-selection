#!/bin/bash
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"   # repo root, wherever it is checked out
f=$REPO/data/models/InternVideo2/InternVideo2-stage2_1b-224p-f4.pt
# wait until file is >= 2.8GB and wget finished
while true; do
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -ge 2800000000 ] && ! pgrep -f "ziyjiang/InternVideo2-1B" >/dev/null; then break; fi
  sleep 15
done
# quick load-verification of state_dict keys
cd $REPO
source .venv-eval/bin/activate
python3 - <<'PY'
import torch
ck=torch.load("data/models/InternVideo2/InternVideo2-stage2_1b-224p-f4.pt", map_location="cpu")
sd = ck.get("model", ck.get("module", ck)) if isinstance(ck,dict) else ck
keys=list(sd.keys()) if isinstance(sd,dict) else []
print("n_keys",len(keys))
print("has vision_encoder:", any("vision_encoder" in k for k in keys))
print("has text_encoder:", any("text_encoder" in k for k in keys))
print("sample:", keys[:3])
PY
echo "IV2_DOWNLOADED $(stat -c%s "$f")" >> $REPO/data/raw/iv2_dl.log
