"""Is the speech channel's Chinese problem an ASR problem? A transcript-side audit.

`mv2_translate_findings.md` records that Chinese gained almost nothing from translating the corpus
(+0.0122 on the channel, +0.0045 fused) while Arabic gained +0.1055, and guesses that the ASR itself is
the bottleneck. That was a guess with a hedge on it. This measures the transcripts directly, per
language, over the judged documents (the ones with a `video_language` label):

  coverage      how often the document has any transcript at all
  length        characters and whitespace tokens, median, since a truncated transcript retrieves badly
  script        what writing system the text is actually in, which catches the failure mode where
                whisper transcribes into the wrong language or emits romanisation
  degenerate    the repetition hallucination whisper falls into on music and B-roll, measured as the
                share of the text taken by its single most frequent token

Script is the load-bearing one. If Chinese transcripts come out in Latin script, the multilingual
encoder is matching English queries against pinyin and the retrieval number says nothing about the
video. If they come out in Han script at normal length, the ASR is fine and the Chinese gap lives
somewhere else.

CPU-only.

  python src/multivent2/mv2_asr_quality.py
"""
import os
import sys
import json
import argparse
import collections
import unicodedata

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mv2_io import load_qrels  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(_ROOT, "data", "multivent2")
ABL = os.path.join(_ROOT, "results", "ablations")

MIN_DOCS = 25          # languages thinner than this are reported but not read as evidence

# the script each language should mostly be written in, for the "wrong script" check
EXPECTED = {"chinese": "han", "cantonese": "han", "japanese": "han", "korean": "hangul",
            "russian": "cyrillic", "ukrainian": "cyrillic", "arabic": "arabic",
            "english": "latin", "spanish": "latin", "portuguese": "latin", "french": "latin",
            "german": "latin", "malay": "latin", "indonesian": "latin", "hindi": "devanagari"}


def script_of(ch):
    """Coarse writing-system bucket for one character. Unicode block names are stable enough for this."""
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    for key, bucket in (("CJK", "han"), ("HIRAGANA", "kana"), ("KATAKANA", "kana"),
                        ("HANGUL", "hangul"), ("CYRILLIC", "cyrillic"), ("ARABIC", "arabic"),
                        ("DEVANAGARI", "devanagari"), ("LATIN", "latin"), ("HEBREW", "hebrew"),
                        ("THAI", "thai"), ("GREEK", "greek")):
        if name.startswith(key):
            return bucket
    return "other"


# scripts written without spaces between words. Splitting these on whitespace gives a handful of huge
# "tokens", which makes both the length and the repetition measure meaningless -- a Chinese transcript
# scores 3 tokens and looks degenerate no matter what it says. Count characters for these instead.
CONTINUOUS = {"han", "kana", "thai"}


def profile(text):
    counts = collections.Counter()
    for ch in text:
        b = script_of(ch)
        if b:
            counts[b] += 1
        # japanese mixes kana and han; fold kana into han so the expectation table stays simple
    if counts.get("kana"):
        counts["han"] += counts.pop("kana")
    total = sum(counts.values())
    dom = counts.most_common(1)[0][0] if total else None

    # unit of analysis follows the script, so lengths and repetition are comparable across languages
    if dom in CONTINUOUS:
        units = [c for c in text if script_of(c) in CONTINUOUS]
        # repetition in scriptio continua shows up as a repeated phrase, not a repeated character, so
        # measure it over character 4-grams; a single character recurring is just normal frequency
        grams = ["".join(units[i:i + 4]) for i in range(max(0, len(units) - 3))]
    else:
        units = text.split()
        grams = units
    top = collections.Counter(grams).most_common(1)[0][1] if grams else 0
    return {"dominant": dom, "n_alpha": total, "n_chars": len(text), "n_tokens": len(units),
            "repeat_share": top / len(grams) if grams else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ABL, "mv2_asr_quality.json"))
    ap.add_argument("--relevant-only", action="store_true",
                    help="restrict to documents judged relevant, not merely judged")
    a = ap.parse_args()

    qrels, meta = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    # doc -> language, from whichever judgment mentions it; a document has one language
    lang_of = {}
    for q in meta:
        for d, m in meta[q].items():
            if a.relevant_only and qrels[q].get(d, 0) <= 0:
                continue
            if m["video_language"]:
                lang_of.setdefault(d, m["video_language"])
    print(f"{len(lang_of)} judged documents carry a language label")

    asr = {}
    with open(os.path.join(DATA, "asr_text.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["doc_id"] in lang_of:
                asr[r["doc_id"]] = r.get("text", "")

    by = collections.defaultdict(list)
    for d, L in lang_of.items():
        by[L].append(d)

    rows, out = [], {"n_docs": len(lang_of), "relevant_only": bool(a.relevant_only), "languages": {}}
    for L, docs in by.items():
        present = [d for d in docs if asr.get(d, "").strip()]
        profs = [profile(asr[d]) for d in present]
        exp = EXPECTED.get(L)
        wrong = [p for p in profs if exp and p["dominant"] and p["dominant"] != exp]
        cell = {
            "n_docs": len(docs),
            "coverage": len(present) / len(docs) if docs else 0.0,
            "median_chars": float(np.median([p["n_chars"] for p in profs])) if profs else 0.0,
            "median_tokens": float(np.median([p["n_tokens"] for p in profs])) if profs else 0.0,
            "expected_script": exp,
            "wrong_script_share": len(wrong) / len(profs) if profs else 0.0,
            "degenerate_share": (sum(1 for p in profs if p["repeat_share"] > 0.5) / len(profs)
                                 if profs else 0.0),
            "dominant_scripts": dict(collections.Counter(
                p["dominant"] for p in profs).most_common(4)),
        }
        out["languages"][L] = cell
        rows.append((L, cell))

    rows.sort(key=lambda r: -r[1]["n_docs"])
    hdr = (f"{'language':<12} {'docs':>7} {'has ASR':>8} {'med chars':>10} {'med toks':>9} "
           f"{'wrong script':>13} {'degenerate':>11}  scripts")
    print("\n" + hdr)
    print("-" * (len(hdr) + 20))
    for L, c in rows:
        flag = "" if c["n_docs"] >= MIN_DOCS else "  (thin)"
        sc = " ".join(f"{k}:{v}" for k, v in c["dominant_scripts"].items() if k)
        print(f"{L:<12} {c['n_docs']:>7} {c['coverage']:>7.1%} {c['median_chars']:>10.0f} "
              f"{c['median_tokens']:>9.0f} {c['wrong_script_share']:>12.1%} "
              f"{c['degenerate_share']:>10.1%}  {sc}{flag}")

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
