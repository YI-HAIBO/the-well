# -*- coding: utf-8 -*-
"""
extract_slice_multi.py -- multi-file variant of extract_slice.py.

Fetches [0:n_traj, 0:n_steps, ...] from the FIRST `--n-files` registry files of a
split and concatenates trajectories. Needed because the shear_flow TEST split files
contain only 4 trajectories each (train files contain 32), so a single file is
insufficient for a representative held-out set.

Usage:
  python extract_slice_multi.py --file train --n-traj 16 --n-steps 50 --n-files 1 --out ./data/slice_train_m.npz
  python extract_slice_multi.py --file test  --n-traj 4  --n-steps 50 --n-files 4 --out ./data/slice_test_m.npz
"""
from __future__ import annotations
import argparse, io, os, subprocess, sys, time, yaml
import numpy as np

CURL = r"C:\Windows\System32\curl.exe"


def log(msg):
    print(msg, flush=True)


def read_offsets(url):
    import fsspec, h5py
    last = None
    for attempt in range(8):
        try:
            fs = fsspec.filesystem("http", block_size=1024 * 1024)
            with fs.open(url, "rb") as f:
                with h5py.File(f, "r") as h5:
                    out = {}
                    for name in ("t0_fields/pressure", "t0_fields/tracer", "t1_fields/velocity"):
                        d = h5[name]
                        assert d.id.get_offset() is not None, "%s not contiguous" % name
                        out[name] = (int(d.id.get_offset()), tuple(d.shape), str(d.dtype))
                    return out
        except Exception as e:
            last = e
            log("  offsets attempt %d failed: %s" % (attempt, str(e)[:100]))
            time.sleep(3)
    raise RuntimeError("offsets read failed: %s" % last)


def fetch_range(url, start, end, n_expected, chunk=32 * 1024 * 1024, n_workers=4):
    """Parallel chunked byte-range fetch. Splits [start,end] into ~32MB chunks,
    downloads concurrently with curl -r, and concatenates in order. Robust against
    long single-connection drops and much faster than a serial 400MB+ fetch."""
    import concurrent.futures, tempfile

    def dl(rng):
        s, e = rng
        fd, tmp = tempfile.mkstemp(suffix=".r")
        os.close(fd)
        try:
            r = subprocess.run([CURL, "-sS", "--create-dirs",
                                "--retry", "10", "--retry-all-errors", "--retry-delay", "2",
                                "-r", "%d-%d" % (s, e), "-o", tmp, url],
                               capture_output=True)
            if r.returncode != 0:
                raise RuntimeError("curl rc=%d: %s" % (r.returncode, r.stderr.decode("utf-8", "replace")[:200]))
            with open(tmp, "rb") as f:
                return f.read()
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    t0 = time.time()
    ranges = []
    pos = start
    while pos <= end:
        e = min(pos + chunk - 1, end)
        ranges.append((pos, e))
        pos = e + 1
    if n_workers > 1 and len(ranges) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            parts = list(ex.map(dl, ranges))
    else:
        parts = [dl(r) for r in ranges]
    data = b"".join(parts)
    got = len(data)
    if got != n_expected:
        raise RuntimeError("short read: got %d want %d" % (got, n_expected))
    return data, time.time() - t0


def slice_field(url, name, off, shape, n_traj, n_steps):
    """Fetch [0:nt, 0:ns, ...] for a contiguous field; nt/ns clamped to file shape."""
    full_traj, full_steps = shape[0], shape[1]
    nt = min(n_traj, full_traj)
    ns = min(n_steps, full_steps)
    inner = 1
    for s in shape[2:]:
        inner *= s
    start_el = 0
    end_el = nt * ns * inner
    start = off + start_el * 4
    end = off + end_el * 4 - 1
    nbytes = end_el * 4
    log("  %s [f%d]: range %d-%d (%.1f MB) nt=%d ns=%d" %
        (name, _CUR_FILE, start, end, nbytes / 1e6, nt, ns))
    raw, dt = fetch_range(url, start, end, nbytes)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(nt, ns, *shape[2:])
    return arr


_CUR_FILE = 0


def main():
    global _CUR_FILE
    ap = argparse.ArgumentParser(description="Extract shear_flow slice via byte-range (multi-file)")
    ap.add_argument("--file", choices=["train", "valid", "test"], default="train")
    ap.add_argument("--n-traj", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--n-files", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from the_well.utils.download import WELL_REGISTRY
    reg = yaml.safe_load(io.open(WELL_REGISTRY, encoding="utf-8"))
    urls = reg["shear_flow"][args.file][: args.n_files]
    log("split=%s n_files=%d" % (args.file, len(urls)))

    parts = {"pressure": [], "tracer": [], "velocity": []}
    total_traj = 0
    t_start = time.time()
    for fi, url in enumerate(urls):
        _CUR_FILE = fi
        log("== file %d: %s" % (fi, url))
        offsets = read_offsets(url)
        log("  offsets:")
        for k, (off, shape, dt) in offsets.items():
            log("    %s offset=%d shape=%s dtype=%s" % (k, off, shape, dt))
        for field in ("pressure", "tracer"):
            key = "t0_fields/%s" % field
            arr = slice_field(url, key, offsets[key][0], offsets[key][1],
                              args.n_traj, args.n_steps)
            parts[field].append(arr)
        key = "t1_fields/velocity"
        varr = slice_field(url, key, offsets[key][0], offsets[key][1],
                           args.n_traj, args.n_steps)
        parts["velocity"].append(varr)
        total_traj += parts["pressure"][-1].shape[0]

    log("fetched all in %.1fs" % (time.time() - t_start))
    pressure = np.concatenate(parts["pressure"], axis=0)
    tracer = np.concatenate(parts["tracer"], axis=0)
    velocity = np.concatenate(parts["velocity"], axis=0)
    log("  combined: pressure%s velocity%s total_traj=%d" %
        (pressure.shape, velocity.shape, total_traj))
    log("  pressure mean=%.4f std=%.4f" % (float(pressure.mean()), float(pressure.std())))
    log("  tracer   mean=%.4f std=%.4f" % (float(tracer.mean()), float(tracer.std())))
    log("  velocity mean=%.4f std=%.4f" % (float(velocity.mean()), float(velocity.std())))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, pressure=pressure, tracer=tracer, velocity=velocity,
                        n_traj=pressure.shape[0], n_steps=pressure.shape[1],
                        n_files=args.n_files,
                        H=pressure.shape[2], W=pressure.shape[3],
                        src_url=urls[0], src_split=args.file)
    log("saved -> %s" % args.out)
    log("  npz size=%.1f MB" % (os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    sys.exit(main())
