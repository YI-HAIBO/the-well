# -*- coding: utf-8 -*-
"""Fetch a gray_scott_reaction_diffusion subset (field A) via byte-range reads,
downsample 128x128 -> 64x64, save gray64.npz (n=4 traj, 20 steps)."""
import io, os, subprocess, time, yaml
import numpy as np, fsspec, h5py

BASE = os.path.expanduser("~/well_b3")
os.makedirs(BASE, exist_ok=True)

def reg():
    from the_well.utils import download as d
    return yaml.safe_load(io.open(d.WELL_REGISTRY, encoding="utf-8"))

url = reg()["gray_scott_reaction_diffusion"]["train"][0]
print("url:", url, flush=True)

# get offset of t0_fields/A (contiguous)
for attempt in range(8):
    try:
        fs = fsspec.filesystem("http", block_size=1024 * 1024)
        with fs.open(url, "rb") as f, h5py.File(f, "r") as h5:
            d = h5["t0_fields/A"]
            off = int(d.id.get_offset()); shape = tuple(d.shape)
            print("A offset=%d shape=%s dtype=%s" % (off, shape, d.dtype), flush=True)
            break
    except Exception as e:
        print("  offsets fail: %s" % str(e)[:120], flush=True)
        time.sleep(3)

nt, ns = 4, 20
inner = shape[2] * shape[3]
nbytes = nt * ns * inner * 4
start = off; end = off + nbytes - 1
print("  range %d-%d (%.2f MB)" % (start, end, nbytes / 1e6), flush=True)
r = subprocess.run(["curl", "-sS", "--retry", "8", "--retry-all-errors", "--retry-delay", "2",
                    "-r", "%d-%d" % (start, end), "-o", os.path.join(BASE, "_gs.raw"), url],
                   capture_output=True)
if r.returncode != 0:
    raise RuntimeError("curl rc=%d %s" % (r.returncode, r.stderr.decode()[:150]))
raw = open(os.path.join(BASE, "_gs.raw"), "rb").read()
print("  got %d bytes (want %d)" % (len(raw), nbytes), flush=True)
assert len(raw) == nbytes
A = np.frombuffer(raw, dtype=np.float32).reshape(nt, ns, *shape[2:])
print("  A mean=%.4f std=%.4f" % (A.mean(), A.std()), flush=True)
# downsample 128x128 -> 64x64
A64 = A.reshape(nt, ns, 128 // 2, 2, 128 // 2, 2).mean(axis=(3, 5))
print("  A64", A64.shape, "mean=%.4f std=%.4f" % (A64.mean(), A64.std()), flush=True)
np.savez_compressed(os.path.join(BASE, "gray64.npz"), gray=A64.astype(np.float32))
print("saved gray64.npz", os.path.getsize(os.path.join(BASE, "gray64.npz")) / 1e6, "MB", flush=True)
