"""Extract raw Whisper ASR transcripts and PaddleOCR on-screen text from the MultiVENT 2.0 feature
archives into one JSONL per modality, keyed by the doc_id the qrels and ranked lists use.

Archive layout: a zip of ~1,100 tar.gz shards, each holding ~100 per-video CSVs. Shard `000003`
contains `./000003/200.m4a.csv` ... `299.m4a.csv`, i.e. the file stem is a *global* video index, and
that integer is the doc_id -- so no separate mapping table is needed.

  ASR csv: video_id,text                              -> one row per video
  OCR csv: video_id,image_id,ocr_nbr,bbox,text        -> many rows per frame, many frames per video

OCR text is deduplicated: the same chyron or ticker is detected on dozens of consecutive frames, and
repeating it inflates term frequencies and pushes the useful text out of the encoder's window. We keep
first occurrence order and drop exact repeats.

  python src/multivent2/mv2_extract_text.py --which asr
  python src/multivent2/mv2_extract_text.py --which ocr
"""
import os
import io
import csv
import sys
import json
import zipfile
import tarfile
import argparse
from concurrent.futures import ProcessPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(_ROOT, "data", "multivent2")
RAW = os.path.join(DATA, "raw", "features", "test")
ARCHIVE = {"asr": os.path.join(RAW, "whisper_asr.zip"),
           "ocr": os.path.join(RAW, "paddle_ocr.zip")}

csv.field_size_limit(1 << 30)


def _doc_id(member_name):
    """'./000003/247.m4a.csv' or './000001/0.mp4_frames/0_frame_81.png.csv' -> '247' / '0'."""
    parts = member_name.strip("./").split("/")
    if len(parts) < 2:
        return None
    stem = parts[1]
    for suffix in (".m4a.csv", ".mp4_frames", ".m4a", ".mp4"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def _rows(blob):
    # a handful of the shipped CSVs carry stray NUL bytes, which csv refuses outright
    return csv.DictReader(io.StringIO(blob.decode("utf8", "replace").replace("\x00", "")))


def _shard(args):
    """Read one tar.gz shard -> {doc_id: text}. Runs in a worker process."""
    zip_path, name, which = args
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(name)
    out = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as t:
        for m in t:
            if not m.isfile() or not m.name.endswith(".csv"):
                continue
            did = _doc_id(m.name)
            if did is None:
                continue
            fh = t.extractfile(m)
            if fh is None:
                continue
            try:
                blob = fh.read()
            except Exception:
                continue
            if which == "asr":
                txt = " ".join((r.get("text") or "").strip() for r in _rows(blob))
                if txt.strip():
                    out[did] = txt.strip()
            else:
                seen, keep = set(), out.setdefault(did, [])
                for r in _rows(blob):
                    s = (r.get("text") or "").strip()
                    if s and s not in seen:
                        seen.add(s)
                        keep.append(s)
    if which == "ocr":
        out = {k: " ".join(v) for k, v in out.items() if v}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["asr", "ocr"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    zip_path = ARCHIVE[a.which]
    out_path = a.out or os.path.join(DATA, f"{a.which}_text.jsonl")
    with zipfile.ZipFile(zip_path) as z:
        shards = sorted(n for n in z.namelist() if n.endswith(".tar.gz"))
    print(f"{a.which}: {len(shards)} shards in {os.path.basename(zip_path)}")

    texts, done = {}, 0
    with ProcessPoolExecutor(a.workers) as ex:
        for part in ex.map(_shard, [(zip_path, n, a.which) for n in shards], chunksize=4):
            # OCR splits one video's frames across shards in a few cases; concatenate rather than
            # overwrite so no frame text is silently dropped.
            for k, v in part.items():
                texts[k] = (texts[k] + " " + v) if k in texts else v
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(shards)} shards, {len(texts)} docs", flush=True)

    with open(out_path, "w") as f:
        for did, txt in sorted(texts.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
            f.write(json.dumps({"doc_id": did, "text": txt}, ensure_ascii=False) + "\n")

    lens = sorted(len(t) for t in texts.values())
    print(f"wrote {out_path}: {len(texts)} docs, "
          f"chars median {lens[len(lens)//2]}, p90 {lens[int(.9*len(lens))]}, max {lens[-1]}")

    # coverage check against the judged pool -- also confirms the doc_id convention is right
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mv2_io import load_qrels  # noqa: E402
    qrels, _ = load_qrels(os.path.join(DATA, "multivent_2_test_judgments.jsonl"))
    judged = {d for q in qrels.values() for d in q}
    rel = {d for q in qrels.values() for d, r in q.items() if r > 0}
    print(f"judged docs covered: {len(judged & set(texts))}/{len(judged)}  "
          f"relevant docs covered: {len(rel & set(texts))}/{len(rel)}")


if __name__ == "__main__":
    main()
