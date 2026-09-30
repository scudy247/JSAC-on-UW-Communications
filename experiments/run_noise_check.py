"""Content check of candidate ambient-noise recordings (noise census, checkpoint 2).

For every file in the config: download it (if a URL) into $UWSB_DATA_DIR, stream it in
blocks of `block_s`, band-pass it to each library-channel band that fits below Nyquist,
and compute model-free impulsiveness statistics per block (uwsb.noise.impulsiveness).
No tail-index estimates (decided 2026-09-30; they wait for estimation/tail_index.py).

  python -m experiments.run_noise_check --config experiments/configs/b_noise_check.yaml --dry-run
  python -m experiments.run_noise_check --config ... --n-workers 4

--dry-run downloads nothing: it lists files, sizes (HTTP HEAD), destinations, free disk
and a runtime estimate from a timing of the statistics on synthetic data.
Idempotent: files already present with the expected size are not downloaded again; a
results dir (keyed by config hash) that already holds summary.json is not recomputed.
"""

# Pin BLAS threads before numpy is imported: parallelism only via --n-workers.
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
import sys
import time
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from uwsb.noise.impulsiveness import BandpassStream, block_stats
from uwsb.runtime import config_hash, data_dir, result_dir, run_metadata

CHUNK_BYTES = 8 * 1024 * 1024


# --- file handling -----------------------------------------------------------
def local_path(entry: dict) -> Path:
    if "path" in entry:
        return Path(entry["path"])
    return data_dir() / "noise_census" / entry["id"] / entry["url"].rsplit("/", 1)[-1]


def remote_size_bytes(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def download(url: str, dest: Path, size_bytes: int, log) -> None:
    if dest.exists() and dest.stat().st_size == size_bytes:
        log(f"present, skipping download: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    t0_s = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, CHUNK_BYTES)
    got = part.stat().st_size
    if got != size_bytes:
        raise IOError(f"{url}: got {got} bytes, expected {size_bytes}")
    part.rename(dest)
    log(f"downloaded {size_bytes / 1e6:.1f} MB in {time.time() - t0_s:.0f} s: {dest}")


# --- analysis ----------------------------------------------------------------
def analyse_file(args):
    """Stream one file; return per-band lists of per-block statistics."""
    entry, cfg = args
    import soundfile as sf

    path = local_path(entry)
    block_s = float(cfg["block_s"])
    warmup_s = float(cfg["warmup_s"])
    k_sigma = float(cfg["k_sigma"])
    clip_level = float(cfg["clip_level"])
    with sf.SoundFile(str(path)) as f:
        fs_hz = float(f.samplerate)
        if f.channels != 1:
            raise ValueError(f"{path}: expected mono, got {f.channels} channels")
        bands = {name: tuple(cfg["bands_hz"][name]) for name in entry["bands"]
                 if cfg["bands_hz"][name][1] < fs_hz / 2.0}
        skipped = [name for name in entry["bands"] if name not in bands]
        filters = {name: BandpassStream(fs_hz, b) for name, b in bands.items()}
        per_band = {name: [] for name in bands}
        n_warm = int(round(warmup_s * fs_hz))
        n_clip = 0
        n_total = 0
        t_block_s = 0.0
        for block in f.blocks(blocksize=int(round(block_s * fs_hz)), dtype="float64"):
            n_clip += int(np.sum(np.abs(block) >= clip_level))
            start = n_warm if n_total == 0 else 0
            n_total += block.size
            for name, bp in filters.items():
                y = bp(block)[start:]
                if y.size < int(0.5 * block_s * fs_hz):
                    continue                       # short final block
                s = block_stats(y, k_sigma=k_sigma)
                s["t_start_s"] = t_block_s
                per_band[name].append(s)
            t_block_s += block.size / fs_hz
    return {
        "id": entry["id"], "file": path.name, "fs_hz": fs_hz,
        "duration_s": n_total / fs_hz, "clip_frac": n_clip / max(n_total, 1),
        "bands_skipped_above_nyquist": skipped, "blocks": per_band,
    }


def summarise(res: dict) -> dict:
    out = {k: res[k] for k in ("id", "file", "fs_hz", "duration_s", "clip_frac",
                               "bands_skipped_above_nyquist")}
    out["bands"] = {}
    for name, blocks in res["blocks"].items():
        kurt = np.array([b["excess_kurtosis"] for b in blocks])
        ratio = np.array([b["exceed_ratio"] for b in blocks])
        out["bands"][name] = {
            "n_blocks": len(blocks),
            "excess_kurtosis_p10_p50_p90": np.percentile(kurt, [10, 50, 90]).tolist(),
            "exceed_ratio_p10_p50_p90": np.percentile(ratio, [10, 50, 90]).tolist(),
        }
    return out


def plot(results: list, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for res in results:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for name, blocks in res["blocks"].items():
            t_min = [b["t_start_s"] / 60.0 for b in blocks]
            axes[0].semilogy(t_min, [max(b["exceed_ratio"], 1e-3) for b in blocks],
                             ".-", label=name)
            axes[1].plot(t_min, [b["excess_kurtosis"] for b in blocks], ".-", label=name)
        axes[0].axhline(1.0, color="k", lw=0.8)
        axes[0].set_ylabel("exceedance ratio vs Gaussian")
        axes[1].set_ylabel("excess kurtosis")
        axes[1].set_xlabel("time in file [min]")
        axes[0].legend()
        axes[0].set_title(f"{res['id']}: {res['file']} (fs = {res['fs_hz']:.0f} Hz)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{res['id']}_{Path(res['file']).stem}.png", dpi=120)
        plt.close(fig)


# --- dry run -----------------------------------------------------------------
def time_stats_per_msample_s(cfg: dict) -> float:
    """Seconds per 1e6 samples for one band (filter + block_stats), measured here."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4_000_000)
    bp = BandpassStream(96_000.0, (20_000.0, 30_000.0))
    t0_s = time.perf_counter()
    block_stats(bp(x), k_sigma=float(cfg["k_sigma"]))
    return (time.perf_counter() - t0_s) / 4.0


def dry_run(cfg: dict, n_workers: int) -> None:
    total_bytes = 0
    missing_bytes = 0
    print(f"{'id':12s} {'size MB':>9s}  destination")
    for e in cfg["files"]:
        dest = local_path(e)
        size = remote_size_bytes(e["url"]) if "url" in e else dest.stat().st_size
        total_bytes += size
        have = dest.exists() and dest.stat().st_size == size
        missing_bytes += 0 if have else size
        print(f"{e['id']:12s} {size / 1e6:9.1f}  {dest}{'  (present)' if have else ''}")
    free_bytes = shutil.disk_usage(data_dir() if data_dir().exists()
                                   else data_dir().parent).free
    print(f"total {total_bytes / 1e9:.2f} GB, to download {missing_bytes / 1e9:.2f} GB, "
          f"free disk {free_bytes / 1e12:.2f} TB")
    s_per_ms = time_stats_per_msample_s(cfg)

    def file_cpu_s(e):
        n_bands = sum(cfg["bands_hz"][b][1] < e["expected_fs_hz"] / 2.0 for b in e["bands"])
        return s_per_ms * e["expected_duration_s"] * e["expected_fs_hz"] * n_bands / 1e6

    per_file_s = [file_cpu_s(e) for e in cfg["files"]]
    cpu_s = sum(per_file_s)
    wall_s = max(cpu_s / max(n_workers, 1), max(per_file_s))   # one file = one task
    print(f"statistics: {s_per_ms:.3f} s per 1e6 samples per band (measured); "
          f"≈ {cpu_s / 60:.1f} CPU-min total, ≈ {wall_s / 60:.1f} min wall with "
          f"{n_workers} workers, longest single file ≈ {max(per_file_s) / 60:.1f} min "
          f"(decoding and download time not included)")
    print(f"results dir: {result_dir(cfg)}")


# --- main ----------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.dry_run:
        dry_run(cfg, args.n_workers)
        return 0

    out_dir = result_dir(cfg)
    if (out_dir / "summary.json").exists():
        print(f"already complete: {out_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"config hash {config_hash(cfg)}, n_workers {args.n_workers}")
    for e in cfg["files"]:
        if "url" in e:
            download(e["url"], local_path(e), remote_size_bytes(e["url"]), log)

    with Pool(args.n_workers) as pool:
        results = pool.map(analyse_file, [(e, cfg) for e in cfg["files"]])

    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps(run_metadata(), indent=2))
    (out_dir / "blocks.json").write_text(json.dumps(results, indent=1))
    plot(results, out_dir)
    summary = [summarise(r) for r in results]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done: {out_dir}")
    log_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
