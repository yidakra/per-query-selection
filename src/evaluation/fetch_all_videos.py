"""Full MultiVENT video fetch: download every source video, extract the MultiCLIP frame
cache (num_of_frames from the dataset = 16) via the repo's own process_video, delete the mp4.

- GPU-free (decord + torchvision on CPU) -> safe alongside the ablation sweep on GPU1.
- Resumable: skips any video whose {CLIP_FRAME_DIR}/{F}/{vid}.npy already exists.
- Disk-safe: stream-and-delete; mp4s never accumulate.
- 403-hardened: retries across yt-dlp player clients (default -> android -> tv -> ios).
- Writes results/video_repro/fetch_manifest.json  (video_id -> ok / reason) so the later
  GPU encode knows exactly which videos are available.

Run from repo root with the venv active. Frame count is read from the dataset.
"""
import os, sys, json, time, subprocess, shutil, glob
from collections import defaultdict, OrderedDict

REPO = "/home/ubuntu/q2e_repro/external/q2e_official"
DS = "/home/ubuntu/q2e_repro/data/MultiVENT/Q2E_MultiVENT_LLAMA_3.3_70B_InternVL_38B_Funiform_16_noASR"
OUT = "/home/ubuntu/q2e_repro/results/video_repro"
TMP_VIDEO_DIR = os.path.join(REPO, "data", "MultiVENT", "_fetch_videos_tmp")
MANIFEST = os.path.join(OUT, "fetch_manifest.json")
YTDLP = shutil.which("yt-dlp") or os.path.expanduser("~/q2e_repro/.venv-eval/bin/yt-dlp")

os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP_VIDEO_DIR, exist_ok=True)
sys.path.insert(0, REPO)
os.chdir(REPO)
from datasets import load_from_disk  # noqa: E402
import numpy as np  # noqa: E402

# player-client ladder for YouTube 403 recovery; ignored by non-YT extractors
CLIENT_LADDER = [None, "android", "tv", "ios"]


def build_targets():
    ds = load_from_disk(DS)
    F = int(ds["num_of_frames"][0])
    seen = OrderedDict()
    for r in ds:
        vid = r["video_id"]
        if vid in seen:
            continue
        md = r["metadata"] if isinstance(r["metadata"], dict) else {}
        seen[vid] = {"video_id": vid, "url": md.get("video_URL", ""), "lang": md.get("language", "?")}
    return list(seen.values()), F


def download(url, vid, client):
    dest = os.path.join(TMP_VIDEO_DIR, f"{vid}.mp4")
    for f in glob.glob(os.path.join(TMP_VIDEO_DIR, f"{vid}.*")):
        try: os.remove(f)
        except OSError: pass
    cmd = [YTDLP, "--no-playlist", "--no-warnings", "-q",
           "--retries", "5", "--fragment-retries", "5", "--sleep-requests", "1",
           "-f", "b[height<=360][ext=mp4]/b[height<=360]/b[ext=mp4]/b",
           "--merge-output-format", "mp4", "-o", dest]
    if client:
        cmd += ["--extractor-args", f"youtube:player_client={client}"]
    cmd += [url]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
    except subprocess.TimeoutExpired:
        return False, "timeout", 0
    if p.returncode == 0 and os.path.exists(dest):
        return True, None, os.path.getsize(dest)
    err = (p.stderr or "").strip().splitlines()
    low = " ".join(err).lower()
    reason = err[-1][:160] if err else f"rc={p.returncode}"
    if "private" in low: reason = "PRIVATE"
    elif "removed" in low or "no longer" in low or "terminated" in low or "not available" in low: reason = "REMOVED"
    elif "403" in low or "forbidden" in low: reason = "HTTP403"
    elif "geo" in low or "not available in your country" in low: reason = "GEOBLOCK"
    elif "age" in low or "sign in to confirm" in low: reason = "AGE_RESTRICTED"
    elif "no video could be found" in low: reason = "NO_MEDIA"
    return False, reason, 0


def fetch_one(url, vid):
    """Try the client ladder; retry the ladder only for 403-ish failures."""
    last = "unknown"
    for client in CLIENT_LADDER:
        ok, reason, nbytes = download(url, vid, client)
        if ok:
            return True, (client or "default"), nbytes
        last = reason
        if reason not in ("HTTP403", "AGE_RESTRICTED", "timeout"):
            break  # PRIVATE/REMOVED/NO_MEDIA won't be fixed by another client
    return False, last, 0


def extract_frames(vid, F, ve):
    ve.video_path = TMP_VIDEO_DIR
    ve.cfg.max_frames = F
    npy = f"{ve.CLIP_FRAME_DIR}/{F}/{vid}.npy"
    try:
        ve.process_video(vid)
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"
    if not os.path.exists(npy):
        return False, "no_npy"
    arr = np.load(npy)
    if float(np.std(arr)) < 1e-6:
        try: os.remove(npy)
        except OSError: pass
        return False, "degenerate_frames"
    return True, f"{tuple(arr.shape)}"


def main():
    targets, F = build_targets()
    import importlib
    ve = importlib.import_module("src.eval.MultiCLIP.vision_embedder")
    cache_dir = f"{ve.CLIP_FRAME_DIR}/{F}"
    os.makedirs(cache_dir, exist_ok=True)
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    print(f"[fetch] {len(targets)} videos, F={F} frames, cache={cache_dir}", flush=True)

    t_start = time.time()
    counts = defaultdict(int)
    for i, t in enumerate(targets, 1):
        vid, url, lang = t["video_id"], t["url"], t["lang"]
        npy = os.path.join(cache_dir, f"{vid}.npy")
        if os.path.exists(npy):
            counts["cached"] += 1
            continue
        prev = manifest.get(vid)
        if prev and not prev.get("ok") and prev.get("reason") in ("PRIVATE", "REMOVED", "NO_MEDIA", "GEOBLOCK"):
            counts["skip_perm_fail"] += 1  # don't re-hammer permanently-gone videos on resume
            continue
        ok_dl, via, nbytes = fetch_one(url, vid)
        if ok_dl:
            ok_fr, info = extract_frames(vid, F, ve)
            mp4 = os.path.join(TMP_VIDEO_DIR, f"{vid}.mp4")
            if os.path.exists(mp4):
                os.remove(mp4)
            manifest[vid] = {"ok": ok_fr, "lang": lang, "via": via, "reason": None if ok_fr else info}
            counts["ok" if ok_fr else "frame_fail"] += 1
        else:
            manifest[vid] = {"ok": False, "lang": lang, "via": None, "reason": via}
            counts[via if via in ("PRIVATE","REMOVED","GEOBLOCK","HTTP403","NO_MEDIA","AGE_RESTRICTED","timeout") else "dl_fail"] += 1
        if i % 25 == 0 or i == len(targets):
            json.dump(manifest, open(MANIFEST, "w"), indent=2)
            done = sum(1 for v in manifest.values() if v.get("ok")) + counts["cached"]
            rate = (time.time() - t_start) / max(1, i)
            eta = rate * (len(targets) - i) / 60
            print(f"  [{i}/{len(targets)}] ok_total={done} counts={dict(counts)} "
                  f"eta~{eta:.0f}min", flush=True)

    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    n_ok = sum(1 for v in manifest.values() if v.get("ok")) + counts["cached"]
    # per-language availability
    per_lang = defaultdict(lambda: {"ok": 0, "fail": 0})
    for t in targets:
        v = manifest.get(t["video_id"])
        cached = os.path.exists(os.path.join(cache_dir, f"{t['video_id']}.npy"))
        if cached or (v and v.get("ok")):
            per_lang[t["lang"]]["ok"] += 1
        else:
            per_lang[t["lang"]]["fail"] += 1
    summary = {"total": len(targets), "frames_ok": n_ok, "F": F,
               "counts": dict(counts), "per_language": {k: dict(v) for k, v in per_lang.items()},
               "elapsed_min": round((time.time() - t_start) / 60, 1)}
    json.dump({"summary": summary}, open(os.path.join(OUT, "fetch_summary.json"), "w"), indent=2)
    print("\n=== FETCH SUMMARY ===")
    print(json.dumps(summary, indent=2), flush=True)
    for f in glob.glob(os.path.join(TMP_VIDEO_DIR, "*")):
        try: os.remove(f)
        except OSError: pass


if __name__ == "__main__":
    main()
