"""PILOT: fetch a small, language-balanced sample of MultiVENT YouTube videos, extract the
exact frame cache the MultiCLIP encoder consumes, and measure availability/attrition + disk.

Faithful path (no custom frame math): download mp4 -> call the repo's own
`MultiCLIP.vision_embedder.process_video` (CPU: decord sample + transform) which writes
`data/models/MultiCLIP/clip_frames/{max_frames}/{video_id}.npy` -> delete the mp4.
That .npy is byte-identical to what the GPU encode step will later load, so validating it =
confirming decord could read the yt-dlp file (checked here via shape + non-zero variance).

GPU-free (safe to run alongside the ablation sweep on GPU1). Run from repo root with venv.

Usage: python src/evaluation/pilot_fetch_videos.py [N_PER_LANG]   (default 6 -> ~30 videos)
Writes results/video_repro/pilot_report.json.
"""
import os, sys, json, time, subprocess, shutil, glob
from collections import defaultdict, OrderedDict

N_PER_LANG = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAX_FRAMES = 12  # MultiCLIP default (cfg.max_frames); cache dir is per-frame-count
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, wherever it is checked out
REPO = f"{_ROOT}/external/q2e_official"
DS = f"{_ROOT}/data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"
OUT = f"{_ROOT}/results/video_repro"
TMP_VIDEO_DIR = os.path.join(REPO, "data", "MultiVENT", "_pilot_videos_tmp")
YTDLP = shutil.which("yt-dlp") or os.path.expanduser("~/q2e_repro/.venv-eval/bin/yt-dlp")

os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP_VIDEO_DIR, exist_ok=True)

sys.path.insert(0, REPO)
os.chdir(REPO)  # so CLIP_FRAME_DIR="data/models/MultiCLIP/clip_frames" resolves under repo
from datasets import load_from_disk  # noqa: E402
import numpy as np  # noqa: E402


def select_videos():
    ds = load_from_disk(DS)
    seen = OrderedDict()
    for r in ds:
        vid = r["video_id"]
        if vid in seen:
            continue
        md = r["metadata"] if isinstance(r["metadata"], dict) else {}
        seen[vid] = {"video_id": vid, "url": md.get("video_URL", ""), "lang": md.get("language", "?")}
    by_lang = defaultdict(list)
    for v in seen.values():
        by_lang[v["lang"]].append(v)
    picks = []
    for lang in sorted(by_lang):
        picks.extend(by_lang[lang][:N_PER_LANG])
    return picks, {l: len(vs) for l, vs in by_lang.items()}


def download(url, vid):
    """Low-res mp4 (enough for 224px frames). Returns (ok, err, seconds, bytes)."""
    dest = os.path.join(TMP_VIDEO_DIR, f"{vid}.mp4")
    for f in glob.glob(os.path.join(TMP_VIDEO_DIR, f"{vid}.*")):
        os.remove(f)
    t0 = time.time()
    cmd = [YTDLP, "--no-playlist", "--no-warnings", "-q",
           "-f", "b[height<=360][ext=mp4]/b[height<=360]/b[ext=mp4]/b",
           "--merge-output-format", "mp4", "-o", dest, url]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "timeout", time.time() - t0, 0
    dt = time.time() - t0
    if p.returncode != 0 or not os.path.exists(dest):
        err = (p.stderr or "").strip().splitlines()
        reason = err[-1][:160] if err else f"rc={p.returncode}"
        # classify common cases
        low = " ".join(err).lower()
        if "private" in low: reason = "PRIVATE"
        elif "removed" in low or "not available" in low or "no longer" in low: reason = "REMOVED"
        elif "geo" in low or "country" in low: reason = "GEOBLOCK"
        elif "sign in" in low or "age" in low or "confirm your age" in low: reason = "AGE_RESTRICTED"
        return False, reason, dt, 0
    return True, None, dt, os.path.getsize(dest)


def extract_frames(vid):
    """Call the repo's own process_video to write the .npy cache. Returns (ok, err, npy_bytes)."""
    import importlib
    ve = importlib.import_module("src.eval.MultiCLIP.vision_embedder")
    ve.video_path = TMP_VIDEO_DIR
    ve.cfg.max_frames = MAX_FRAMES
    npy = f"{ve.CLIP_FRAME_DIR}/{MAX_FRAMES}/{vid}.npy"
    if os.path.exists(npy):
        os.remove(npy)
    try:
        ve.process_video(vid)  # decord read + transform + np.save (CPU)
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:140]}", 0
    if not os.path.exists(npy):
        return False, "no_npy_written", 0
    arr = np.load(npy)
    if float(np.std(arr)) < 1e-6:
        return False, f"degenerate_frames std={float(np.std(arr)):.2e} shape={arr.shape}", os.path.getsize(npy)
    return True, f"shape={tuple(arr.shape)} std={float(np.std(arr)):.3f}", os.path.getsize(npy)


def main():
    picks, lang_totals = select_videos()
    print(f"[pilot] {len(picks)} videos selected ({N_PER_LANG}/lang). Dataset lang totals: {lang_totals}")
    print(f"[pilot] yt-dlp: {YTDLP}")
    rows = []
    for i, v in enumerate(picks, 1):
        vid, url, lang = v["video_id"], v["url"], v["lang"]
        ok_dl, err_dl, dt, nbytes = download(url, vid)
        rec = {"video_id": vid, "lang": lang, "url": url, "download_ok": ok_dl,
               "download_err": err_dl, "download_s": round(dt, 1), "mp4_bytes": nbytes}
        if ok_dl:
            ok_fr, info, npy_bytes = extract_frames(vid)
            rec.update({"frames_ok": ok_fr, "frames_info": info, "npy_bytes": npy_bytes})
            mp4 = os.path.join(TMP_VIDEO_DIR, f"{vid}.mp4")
            if os.path.exists(mp4):
                os.remove(mp4)  # stream-and-delete: never accumulate videos
        else:
            rec.update({"frames_ok": False, "frames_info": None, "npy_bytes": 0})
        rows.append(rec)
        tag = "OK " if rec["frames_ok"] else ("DL-FAIL" if not ok_dl else "FR-FAIL")
        print(f"  [{i:2d}/{len(picks)}] {lang:8s} {vid:12s} {tag}  "
              f"{'%.1fs'%dt:>7s} {(nbytes/1e6):5.1f}MB  {rec.get('frames_info') or err_dl}")

    # summary
    per_lang = defaultdict(lambda: {"n": 0, "dl_ok": 0, "fr_ok": 0})
    npy_sizes, dl_times = [], []
    for r in rows:
        s = per_lang[r["lang"]]; s["n"] += 1
        s["dl_ok"] += int(r["download_ok"]); s["fr_ok"] += int(r["frames_ok"])
        if r["frames_ok"]:
            npy_sizes.append(r["npy_bytes"]); dl_times.append(r["download_s"])
    n = len(rows); fr_ok = sum(r["frames_ok"] for r in rows)
    avg_npy = (sum(npy_sizes) / len(npy_sizes)) if npy_sizes else 0
    avg_dl = (sum(dl_times) / len(dl_times)) if dl_times else 0
    total_videos = sum(lang_totals.values())
    summary = {
        "n_pilot": n, "frames_ok": fr_ok, "success_rate": round(fr_ok / n, 3) if n else 0,
        "per_language": {l: dict(v) for l, v in per_lang.items()},
        "avg_npy_bytes": int(avg_npy), "avg_download_s": round(avg_dl, 1),
        "dataset_total_videos": total_videos,
        "projected_full": {
            "expected_available": int(round((fr_ok / n) * total_videos)) if n else 0,
            "projected_npy_gb": round(avg_npy * total_videos * (fr_ok / n) / 1e9, 1) if n else 0,
            "projected_download_hours": round(avg_dl * total_videos / 3600, 1) if n else 0,
        },
        "max_frames": MAX_FRAMES,
    }
    out = {"summary": summary, "rows": rows}
    json.dump(out, open(os.path.join(OUT, "pilot_report.json"), "w"), indent=2)
    print("\n=== PILOT SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("[written]", os.path.join(OUT, "pilot_report.json"))
    # tidy tmp dir
    for f in glob.glob(os.path.join(TMP_VIDEO_DIR, "*")):
        try: os.remove(f)
        except OSError: pass


if __name__ == "__main__":
    main()
