#!/bin/bash
cd /home/ubuntu/q2e_repro
source .venv-eval/bin/activate
export CUDA_VISIBLE_DEVICES=1
f=runs/msrvtt_multiclip_noASR/cache/query_vs_video.pt
while [ ! -f "$f" ]; do sleep 30; done
sleep 5
python3 - <<'PY'
import torch, sys
sys.path.insert(0,"external/q2e_official")
from src.eval.evaluation import retrieval_score
from datasets import load_from_disk
from collections import defaultdict
m=torch.load("runs/msrvtt_multiclip_noASR/cache/query_vs_video.pt")
ds=load_from_disk("data/MSR-VTT-1kA/Q2E_MSRVTT-1kA_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR")
q=[];v=[];sq=set();sv=set()
for r in ds:
    if r["query"] not in sq: sq.add(r["query"]); q.append(r["query"])
    if r["video_id"] not in sv: sv.add(r["video_id"]); v.append(r["video_id"])
vi={x:i for i,x in enumerate(v)}
q2vi=defaultdict(list)
for r in ds: q2vi[r["query"]].append(vi[r["video_id"]])
import torch as t
tgt=t.zeros(len(q),len(v))
for i,qq in enumerate(q): tgt[i,q2vi[qq]]=1
met=retrieval_score(m, tgt.bool())
print("EARLY BASELINE (MSR-VTT MultiCLIP raw query_vs_video):")
print({k:round(x,2) for k,x in met.items()})
print("paper baseline NDCG=59.72 R1=43.52 R5=69.05 R10=76.88")
PY
