# -*- coding: utf-8 -*-
"""Fetch the exact data subsets used by the paper, on the instabox sandbox, via
byte-range reads over HTTP (no full-file download). Replicates extract_slice.py:
  shear_flow file[0] -> tracer[0:4, 0:20, :, :]   (native 256x512)
  turb file[0]       -> density[0:8, 0:20, :, :]  (native 128x384)
Then downsamples both to 64x64 and saves a compact npz for the B3 experiment.
"""
import io, os, subprocess, time, yaml
import numpy as np, fsspec, h5py

BASE = os.path.expanduser("~/well_b3")
os.makedirs(BASE, exist_ok=True)
HOST = "https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/datasets"

def reg():
    from the_well.utils import download as d
    return yaml.safe_load(io.open(d.WELL_REGISTRY, encoding="utf-8"))

def offsets(url):
    for attempt in range(8):
        try:
            fs = fsspec.filesystem("http", block_size=1024 * 1024)
            with fs.open(url, "rb") as f, h5py.File(f, "r") as h5:
                out = {}
                for name in ("t0_fields/tracer", "t0_fields/density"):
                    if name in h5:
                        d = h5[name]
                        out[name] = (int(d.id.get_offset()), tuple(d.shape), str(d.dtype))
                return out
        except Exception as e:
            print("  offsets attempt fail: %s" % str(e)[:120], flush=True)
            time.sleep(3)
    raise RuntimeError("offsets fail")

def fetch_range(url, start, end, nbytes):
    t0 = time.time()
    r = subprocess.run(["curl", "-sS", "--retry", "8", "--retry-all-errors", "--retry-delay", "2",
                        "-r", "%d-%d" % (start, end), "-o", os.path.join(BASE, "_tmp.raw"), url],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("curl rc=%d %s" % (r.returncode, r.stderr.decode()[:150]))
    with open(os.path.join(BASE, "_tmp.raw"), "rb") as f:
        data = f.read()
    if len(data) != nbytes:
        raise RuntimeError("short read %d want %d" % (len(data), nbytes))
    return data, time.time() - t0

def slice_field(url, off, shape, nt, ns):
    inner = 1
    for s in shape[2:]:
        inner *= s
    nbytes = nt * ns * inner * 4
    start = off
    end = off + nbytes - 1
    print("  range %d-%d (%.1f MB)" % (start, end, nbytes / 1e6), flush=True)
    raw, dt = fetch_range(url, start, end, nbytes)
    return np.frombuffer(raw, dtype=np.float32).reshape(nt, ns, *shape[2:])

R = reg()

# --- shear tracer ---
sur = R["shear_flow"]["train"][0]
print("shear:", sur, flush=True)
off = offsets(sur)["t0_fields/tracer"]
print("  tracer offset=%d shape=%s" % (off[0], off[1]), flush=True)
tracer = slice_field(sur, off[0], off[1], 4, 20)
print("  tracer mean=%.4f std=%.4f" % (tracer.mean(), tracer.std()), flush=True)

# --- turb density ---
tur = R["turbulent_radiative_layer_2D"]["train"][0]
print("turb:", tur, flush=True)
off = offsets(tur)["t0_fields/density"]
print("  density offset=%d shape=%s" % (off[0], off[1]), flush=True)
density = slice_field(tur, off[0], off[1], 8, 20)
print("  density mean=%.4f std=%.4f" % (density.mean(), density.std()), flush=True)

# --- downsample to 64x64 (block-mean, matching gap_fix loaders) ---
def down64(A):
    n, t, H, W = A.shape
    if (H % 4 == 0 and W % 8 == 0):
        return A.reshape(n, t, H // 4, 4, W // 8, 8).mean(axis=(3, 5))
    if (H % 2 == 0 and W % 6 == 0):
        return A.reshape(n, t, H // 2, 2, W // 6, 6).mean(axis=(3, 5))
    raise ValueError("cannot downsample %s" % (A.shape,))

tracer64 = down64(tracer)   # 256x512 -> 64x64
density64 = down64(density) # 128x384 -> 64x64
print("tracer64", tracer64.shape, "density64", density64.shape, flush=True)

np.savez_compressed(os.path.join(BASE, "data64.npz"),
                    shear=tracer64.astype(np.float32), turb=density64.astype(np.float32))
print("saved data64.npz", os.path.getsize(os.path.join(BASE, "data64.npz")) / 1e6, "MB", flush=True)
