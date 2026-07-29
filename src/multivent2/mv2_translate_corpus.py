"""Translate the non-English ASR transcripts to English so the dense channel can retrieve them with an
English query in a single language space. This is the 'translate' half of MMMORRF's translate-distill:
bge-m3 handles English->English well, and the earlier per-language split showed the whole loss on this
channel is English-query -> foreign-transcript alignment, worst on Chinese and Arabic.

Source language is routed by dominant Unicode script, which is exact for the four non-Latin languages in
this pool (Chinese CJK, Arabic, Korean Hangul, Russian Cyrillic). Latin-script docs are left untouched:
they are English or the small Spanish slice, both of which the untranslated channel already handles.

Long transcripts are split into ~sentence-sized pieces under NLLB's window and rejoined after. Output is
a drop-in replacement for asr_text.jsonl with the same doc_id keys.

  python src/multivent2/mv2_translate_corpus.py --model facebook/nllb-200-distilled-1.3B
"""
import os
import re
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")

# dominant-script -> NLLB source code. Only the scripts that actually occur here. Compiled character
# classes so routing is a C-level regex count, not a per-char Python loop (the latter takes ~10 min over
# the 109k-doc corpus and re-runs on every restart).
SCRIPT_RE = [
    ("kor_Hang", re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")),
    ("arb_Arab", re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")),
    ("rus_Cyrl", re.compile(r"[Ѐ-ӿ]")),
    ("zho_Hans", re.compile(r"[一-鿿㐀-䶿]")),   # CJK; 7 JA docs fold in harmlessly
]


def route_lang(text):
    """Dominant non-Latin script -> NLLB src code, or None to leave the text as English. Script is
    uniform within a transcript, so a prefix sample decides it."""
    s = text[:2000]
    denom = max(1, len(s) - s.count(" "))
    best, best_n = None, 0
    for code, rx in SCRIPT_RE:
        n = len(rx.findall(s))
        if n > best_n:
            best, best_n = code, n
    # require a real presence of the script, not a stray loanword in an English transcript
    return best if best_n >= 0.10 * denom else None


def pieces(text, size=380):
    """Split into <=size-char pieces on sentence-ish boundaries so NLLB never truncates mid-window."""
    text = " ".join(text.split())
    if len(text) <= size:
        return [text]
    parts, cur = [], ""
    for tok in re.split(r"(?<=[.!?。！？؟\n])\s+", text):
        while len(tok) > size:                       # a single very long run with no boundary
            parts.append(tok[:size]); tok = tok[size:]
        if len(cur) + len(tok) + 1 <= size:
            cur = (cur + " " + tok).strip()
        else:
            if cur:
                parts.append(cur)
            cur = tok
    if cur:
        parts.append(cur)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="asr")
    ap.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-tok", type=int, default=512)
    ap.add_argument("--doc-chunk", type=int, default=200, help="checkpoint to disk every this many docs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    in_path = os.path.join(DATA, f"{a.which}_text.jsonl")
    out_path = a.out or os.path.join(DATA, f"{a.which}_text_en.jsonl")

    docs = []
    with open(in_path) as f:
        for line in f:
            d = json.loads(line)
            if d["text"].strip():
                docs.append((d["doc_id"], d["text"]))
    if a.limit:
        docs = docs[:a.limit]

    # bucket work by source language, keep English docs aside untouched
    by_lang, passthrough = {}, {}
    for did, txt in docs:
        code = route_lang(txt)
        if code is None:
            passthrough[did] = txt
        else:
            by_lang.setdefault(code, []).append((did, txt))
    print(f"{len(docs)} docs: {len(passthrough)} kept as English, "
          + ", ".join(f"{c} {len(v)}" for c, v in sorted(by_lang.items())), flush=True)

    # resume: any doc_id already in the output file is done and is skipped
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
        print(f"resume: {len(done_ids)} docs already in {os.path.basename(out_path)}", flush=True)

    out = open(out_path, "a", buffering=1)                       # line-buffered append
    for did, txt in passthrough.items():
        if did not in done_ids:
            out.write(json.dumps({"doc_id": did, "text": txt}, ensure_ascii=False) + "\n")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    eng_id = tok.convert_tokens_to_ids("eng_Latn")

    def translate(texts):
        res = []
        for s in range(0, len(texts), a.batch):
            enc = tok(texts[s:s + a.batch], return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_tok).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=eng_id,
                                     max_length=a.max_tok, num_beams=1)
            res.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return res

    t0, n_done = time.time(), 0
    for code, items in sorted(by_lang.items()):
        tok.src_lang = code
        pend = [(did, txt) for did, txt in items if did not in done_ids]
        print(f"  {code}: {len(pend)} to translate ({len(items) - len(pend)} already done)", flush=True)
        # checkpoint every doc-chunk so a crash at hour 14 keeps hours 0-13
        for cs in range(0, len(pend), a.doc_chunk):
            sub = pend[cs:cs + a.doc_chunk]
            flat, owner = [], []
            for i, (did, txt) in enumerate(sub):
                for p in pieces(txt):
                    flat.append(p); owner.append(i)
            outp = translate(flat)
            joined = {i: [] for i in range(len(sub))}
            for pos, t in zip(owner, outp):
                joined[pos].append(t)
            for i, (did, _) in enumerate(sub):
                out.write(json.dumps({"doc_id": did, "text": " ".join(joined[i]).strip(), "mt": True},
                                     ensure_ascii=False) + "\n")
            n_done += len(sub)
            el = time.time() - t0
            print(f"  {code}: {cs + len(sub)}/{len(pend)} docs | {n_done} total | "
                  f"{n_done / max(1e-6, el) * 60:.0f} docs/min | {el / 3600:.2f} h", flush=True)
    out.close()
    print(f"wrote {out_path} in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
