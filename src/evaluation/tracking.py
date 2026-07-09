#!/usr/bin/env python
"""Experiment + energy tracking: Weights & Biases and CodeCarbon, in one context manager.

Usage:
    from tracking import track
    with track("perparaphrase", config={"setting": "noASR"}, gpu_ids=[1]) as t:
        ...
        t.log({"doc_slot": j, "slot_seconds": dt})
        t.summary({"ndcg": 61.18})

DESIGN NOTES (this machine is shared -- read before trusting any number):

* GPU energy.  CodeCarbon reads *whole-device* power via NVML; it cannot attribute draw to a
  process.  GPU0 permanently hosts an unrelated whisper server.  So every tracked job MUST pass
  the GPU it actually uses (we pin ours to GPU1 via CUDA_VISIBLE_DEVICES=1), otherwise whisper's
  idle draw is silently billed to our experiments.  `gpu_ids` are PHYSICAL/NVML indices, which
  are NOT remapped by CUDA_VISIBLE_DEVICES -- inside a job pinned to GPU1, torch sees cuda:0 but
  codecarbon must be told [1].  CPU-only jobs pass gpu_ids=[].

* CPU energy.  No Intel RAPL powercap interface is exposed here, so CodeCarbon cannot measure
  CPU energy and falls back to a TDP-based ESTIMATE (Xeon Silver 4314, 135 W) scaled by process
  CPU utilisation.  CPU figures are therefore modelled, not measured.  We record which mode was
  used in the run summary so no one later reads an estimate as a measurement.

* tracking_mode="process" keeps CPU/RAM accounting to this process rather than the whole box.
  It does NOT change the whole-device GPU caveat above.

* Carbon intensity.  CodeCarbon geolocates by IP to pick a grid intensity unless
  CODECARBON_COUNTRY_ISO is set.  That is an outbound request.  Set the env var to avoid it.

W&B defaults to offline (writes to runs/wandb/) unless credentials exist or WANDB_MODE is set,
so nothing leaves the box by accident -- this repo is private and pre-publication.
Run `wandb login` once to enable online sync, then `wandb sync runs/wandb/offline-*`.
"""
import os, sys, time, json, socket, subprocess, contextlib

REPO = "/home/ubuntu/q2e_repro"
PROJECT = os.environ.get("WANDB_PROJECT", "adaptive-q2e")
ENERGY_DIR = os.path.join(REPO, "results", "energy")


def _wandb_mode():
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    netrc = os.path.expanduser("~/.netrc")
    if os.path.exists(netrc):
        try:
            if "api.wandb.ai" in open(netrc).read():
                return "online"
        except OSError:
            pass
    return "offline"


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _rapl_available():
    import glob
    return bool(glob.glob("/sys/class/powercap/intel-rapl*"))


class _Tracker:
    def __init__(self, run, cc, meta):
        self._run, self._cc, self.meta = run, cc, meta
        self._t0 = time.time()

    def log(self, d, step=None):
        if self._run is not None:
            self._run.log(d, step=step)

    def summary(self, d):
        self.meta.setdefault("summary", {}).update(d)
        if self._run is not None:
            for k, v in d.items():
                self._run.summary[k] = v


@contextlib.contextmanager
def track(name, config=None, gpu_ids=None, project=PROJECT, tags=None):
    """gpu_ids: PHYSICAL NVML indices this job uses. [] for CPU-only. None = refuse to guess."""
    if gpu_ids is None:
        raise ValueError("tracking.track() requires explicit gpu_ids (use [] for CPU-only jobs). "
                         "CodeCarbon bills WHOLE-DEVICE GPU power; guessing would charge this run "
                         "for the whisper server on GPU0.")
    os.makedirs(ENERGY_DIR, exist_ok=True)
    cfg = dict(config or {})
    rapl = _rapl_available()
    cfg.update({"git_sha": _git_sha(), "host": socket.gethostname(),
                "gpu_ids_tracked": gpu_ids, "cpu_energy_mode": "rapl" if rapl else "tdp_estimate",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "grid_region": os.environ.get("CODECARBON_COUNTRY_ISO", "NLD"),
                "grid_intensity_gco2_kwh": 267.622,   # codecarbon bundled NLD annual mix
                "grid_intensity_source": "codecarbon bundled annual energy mix (not live)"})

    run = None
    mode = _wandb_mode()
    try:
        import wandb
        os.environ.setdefault("WANDB_DIR", os.path.join(REPO, "runs"))
        run = wandb.init(project=project, name=name, config=cfg, mode=mode,
                         tags=tags or [], reinit=True)
        print(f"[wandb] mode={mode} run={name}", flush=True)
    except Exception as e:
        print(f"[wandb] disabled ({e})", flush=True)

    cc = None
    iso = os.environ.get("CODECARBON_COUNTRY_ISO", "NLD")
    try:
        # OfflineEmissionsTracker, not EmissionsTracker: the online tracker geolocates by an
        # outbound IP request to pick a grid carbon intensity, and in codecarbon 3.x it does not
        # even accept country_iso_code. The offline tracker takes the region directly and makes
        # no network call. Intensity for NLD comes from codecarbon's bundled annual energy-mix
        # table (267.6 gCO2e/kWh) -- an annual average, NOT live grid intensity.
        from codecarbon import OfflineEmissionsTracker
        kw = dict(project_name=project, experiment_id=name, output_dir=ENERGY_DIR,
                  output_file="emissions.csv", log_level="error", measure_power_secs=15,
                  tracking_mode="process", save_to_file=True, country_iso_code=iso)
        # ALWAYS pass gpu_ids, including the empty list. `if gpu_ids:` would treat [] as falsy,
        # omit the key, and codecarbon would then track EVERY GPU on the box -- silently billing
        # CPU-only runs for the whisper server idling on GPU0 (measured: 0.16 Wh in 19 s).
        kw["gpu_ids"] = gpu_ids
        cc = OfflineEmissionsTracker(**kw)
        cc.start()
        print(f"[codecarbon] region={iso} gpu_ids={gpu_ids or 'none (CPU-only)'} "
              f"cpu={'RAPL' if rapl else 'TDP estimate'}", flush=True)
    except Exception as e:
        # Do not silently proceed. A long GPU run that finishes with no emissions record is a
        # wasted run when the whole point was to measure it. Set CC_OPTIONAL=1 to downgrade.
        if os.environ.get("CC_OPTIONAL") == "1":
            print(f"[codecarbon] disabled ({e})", flush=True)
        else:
            if run is not None:
                run.finish(exit_code=1)
            raise RuntimeError(
                f"codecarbon failed to start ({e}). Emissions would go unrecorded. "
                f"Fix it, or set CC_OPTIONAL=1 to run without energy tracking.") from e

    meta = {"name": name, "config": cfg}
    t = _Tracker(run, cc, meta)
    t0 = time.time()
    try:
        yield t
    finally:
        wall = time.time() - t0
        em = None
        if cc is not None:
            try:
                em = cc.stop()   # kg CO2eq
            except Exception as e:
                print(f"[codecarbon] stop failed ({e})", flush=True)
        energy = {}
        if em is not None:
            d = cc.final_emissions_data
            energy = {
                "emissions_kgco2e": float(em),
                "energy_kwh_total": float(d.energy_consumed),
                "energy_kwh_gpu": float(d.gpu_energy),
                "energy_kwh_cpu": float(d.cpu_energy),
                "energy_kwh_ram": float(d.ram_energy),
                "duration_s": float(d.duration),
                "country": d.country_name,
                "cpu_energy_mode": "rapl" if rapl else "tdp_estimate",
            }
            print(f"[codecarbon] {em*1000:.2f} g CO2eq | {d.energy_consumed*1000:.1f} Wh "
                  f"(gpu {d.gpu_energy*1000:.1f} Wh, cpu {d.cpu_energy*1000:.1f} Wh"
                  f"{' [ESTIMATED]' if not rapl else ''})", flush=True)
        meta["wall_s"] = wall
        meta["energy"] = energy
        with open(os.path.join(ENERGY_DIR, "runs.jsonl"), "a") as f:
            f.write(json.dumps(meta) + "\n")
        if run is not None:
            for k, v in energy.items():
                run.summary[k] = v
            run.summary["wall_s"] = wall
            run.finish()
