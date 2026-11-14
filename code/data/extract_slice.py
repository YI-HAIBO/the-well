# -*- coding: utf-8 -*-
"""
extract_slice.py -- robust extraction of a small contiguous slice of a shear_flow
hdf5 file from the Flatiron SDSC host, WITHOUT downloading the 13 GB file.

Method:
  1. Open the file over HTTP with fsspec (metadata only; verified reliable for
     the small header reads) and read each dataset's byte offset + shape.
  2. The datasets are CONTIGUOUS (no chunking/compression) -> the bytes of any
     [traj0:traj1, t0:t1] slice occupy one contiguous byte range.
  3. Fetch that range with `curl -r start-end --retry 10 --retry-all-errors`
     (robust against drops; exact-length verified) and np.frombuffer it.

Output: an .npz with pressure/tracer/velocity slices (float32) + shape/grid meta.

Usage:
  python extract_slice.py --file train --n-traj 4 --n-steps 20 --out ./data/slice_train.npz
  python extract_slice.py --file test  --n-traj 4 --n-steps 20 --out ./data/slice_test.npz
"""
from __future__ import annotations
import argparse, io, os, subprocess, sys, time, yaml
import numpy as np

CURL = r"C:\Windows\System32\curl.exe"


def log(msg):
    print(msg, flush=True)


def read_offsets(url):
    """Return {field: (offset, shape, dtype)} by reading h5 metadata over HTTP."""
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


def fetch_range(url, start, end, n_expected):
    """curl -r with resumable retries via a temp file (avoids stdout concatenation
    bugs when --retry re-requests a dropped range); assert exact length."""
    import tempfile
    t0 = time.time()
    fd, tmp = tempfile.mkstemp(suffix=".range")
    os.close(fd)
    try:
        r = subprocess.run([CURL, "-sS", "--create-dirs",
                            "--retry", "10", "--retry-all-errors", "--retry-delay", "2",
                            "-r", "%d-%d" % (start, end), "-o", tmp, url],
                           capture_output=True)
        if r.returncode != 0:
            raise RuntimeError("curl rc=%d: %s" % (r.returncode, r.stderr.decode("utf-8", "replace")[:200]))
        with open(tmp, "rb") as f:
            data = f.read()
        got = len(data)
        if got != n_expected:
            raise RuntimeError("short read: got %d want %d" % (got, n_expected))
        return data, time.time() - t0
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def slice_field(url, name, off, shape, n_traj, n_steps, axis_meta):
    """Fetch [0:n_traj, 0:n_steps, ...] for a contiguous field."""
    # field layout: (32, 200, *spatial, V) for velocity, else (32,200,*spatial)
    n_traj_full, n_steps_full = shape[0], shape[1]
    spatial = tuple(shape[2:])
    # bytes per element
    # velocity has extra trailing dim; compute per-trajectory-stride in elements
    inner = 1
    for s in shape[2:]:
        inner *= s
    step_elems = inner                 # elements per (traj, step)
    start_el = 0
    end_el = n_traj * n_steps * inner
    start = off + start_el * 4
    end = off + end_el * 4 - 1
    nbytes = end_el * 4
    log("  %s: range %d-%d (%.1f MB)" % (name, start, end, nbytes / 1e6))
    raw, dt = fetch_range(url, start, end, nbytes)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(n_traj, n_steps, *spatial)
    return arr


def main():
    ap = argparse.ArgumentParser(description="Extract shear_flow slice via byte-range")
    ap.add_argument("--file", choices=["train", "valid", "test"], default="train")
    ap.add_argument("--n-traj", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from the_well.utils.download import WELL_REGISTRY
    reg = yaml.safe_load(io.open(WELL_REGISTRY, encoding="utf-8"))
    url = reg["shear_flow"][args.file][0]
    log("file=%s  url=%s" % (args.file, url))

    offsets = read_offsets(url)
    log("offsets:")
    for k, (off, shape, dt) in offsets.items():
        log("  %s offset=%d shape=%s dtype=%s" % (k, off, shape, dt))

    t0 = time.time()
    pressure = slice_field(url, "t0_fields/pressure", offsets["t0_fields/pressure"][0],
                           offsets["t0_fields/pressure"][1], args.n_traj, args.n_steps, None)
    tracer = slice_field(url, "t0_fields/tracer", offsets["t0_fields/tracer"][0],
                         offsets["t0_fields/tracer"][1], args.n_traj, args.n_steps, None)
    velocity = slice_field(url, "t1_fields/velocity", offsets["t1_fields/velocity"][0],
                           offsets["t1_fields/velocity"][1], args.n_traj, args.n_steps, None)
    log("fetched all in %.1fs" % (time.time() - t0))

    # sanity vs stats.yaml
    log("  pressure mean=%.4f std=%.4f" % (float(pressure.mean()), float(pressure.std())))
    log("  tracer   mean=%.4f std=%.4f" % (float(tracer.mean()), float(tracer.std())))
    log("  velocity mean=%.4f std=%.4f" % (float(velocity.mean()), float(velocity.std())))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, pressure=pressure, tracer=tracer, velocity=velocity,
                        n_traj=args.n_traj, n_steps=args.n_steps,
                        H=pressure.shape[2], W=pressure.shape[3],
                        src_url=url, src_split=args.file)
    log("saved -> %s" % args.out)
    # report file size
    log("  npz size=%.1f MB" % (os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    sys.exit(main())
