"""Translate the test queries into the corpus's top non-English languages.

The route-by-language question needs query-side language variants: the corpus documents were
translated to English once (mv2_translate_corpus.py, NLLB-1.3B, document side); this goes the other
way and renders each English query in the languages the videos actually speak. Queries are one
sentence each, so the distilled 600M model carries them; a seeded 50-query back-translation sample
is written next to the output so the quality concession is inspectable rather than asserted.

  python src/multivent2/mv2_translate_queries.py --gpu 1
"""
import os
import sys
import csv
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

MODEL = "facebook/nllb-200-distilled-600M"
# top non-English corpus languages by relevant-document count (see mv2_per_language.json)
LANGS = {"zho_Hans": "zh", "kor_Hang": "ko", "rus_Cyrl": "ru", "arb_Arab": "ar"}
SPOT_N = 50


def translate(texts, tok, model, src, tgt, batch=32):
    import torch
    out = []
    tok.src_lang = src
    bos = tok.convert_tokens_to_ids(tgt)
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=4, max_new_tokens=256)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  {min(s + batch, len(texts))}/{len(texts)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from mv2_io import load_queries

    queries = load_queries(os.path.join(DATA, "multivent_2_test_queries.csv"))
    qids = sorted(queries)
    texts = [queries[q] for q in qids]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).half().cuda().eval()

    import numpy as np
    spot_idx = sorted(np.random.default_rng(0).choice(len(qids), SPOT_N, replace=False).tolist())
    spot = {"model": MODEL, "n": SPOT_N, "languages": {}}

    for code, short in LANGS.items():
        print(f"translating {len(texts)} queries -> {code}", flush=True)
        tr = translate(texts, tok, model, "eng_Latn", code, a.batch)
        out_path = os.path.join(DATA, f"queries_{short}.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Query_id", "query"])
            for q, t in zip(qids, tr):
                w.writerow([q, t])
        print(f"wrote {out_path}", flush=True)

        back = translate([tr[i] for i in spot_idx], tok, model, code, "eng_Latn", a.batch)
        spot["languages"][short] = [
            {"qid": qids[i], "en": texts[i], "translated": tr[i], "back": b}
            for i, b in zip(spot_idx, back)]

    spot_path = os.path.join(ABL, "mv2_query_translation_spotcheck.json")
    json.dump(spot, open(spot_path, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {spot_path}")


if __name__ == "__main__":
    main()
