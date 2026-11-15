# -*- coding: utf-8 -*-
"""Fetch active_matter + euler_multi_quadrants_periodicBC subsets (2D scalar)
via byte-range reads, downsample to 64x64, save active64.npz / euler64.npz."""
import io, os, subprocess, time, yaml
import numpy as np, fsspec, h5py

BASE = os.path.expanduser("~/well_b3")
os.makedirs(BASE, exist_ok=True)

def reg():
    from the_well.utils import download as d
    return yaml.safe_load(io.open(d.WELL_REGISTRY, encoding="utf-8"))

def get_off(url, name):
    for attempt in range(8):
        try:
            fs = fsspec.filesystem("http", block_size=1024 * 1024)
            with fs.open(url, "rb") as f, h5py.File(f, "r") as h5:
                d = h5[name]
                return int(d.id.get_offset()), tuple(d.shape)
        except Exception as e:
            print("  offsets fail %s: %s" % (name, str(e)[:100]), flush=True)
            time.sleep(3)
    raise RuntimeError("offsets fail")

def fetch(url, off, shape, nt, ns, out):
    inner = shape[2] * shape[3]
    nbytes = nt * ns * inner * 4
    start, end = off, off + nbytes - 1
    print("  %s: range %d-%d (%.1f MB)" % (out, start, end, nbytes / 1e6), flush=True)
    r = subprocess.run(["curl", "-sS", "--retry", "8", "--retry-all-errors", "--retry-delay", "2",
                        "-r", "%d-%d" % (start, end), "-o", os.path.join(BASE, out + ".raw"), url],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("curl rc=%d %s" % (r.returncode, r.stderr.decode()[:150]))
    raw = open(os.path.join(BASE, out + ".raw"), "rb").read()
    assert len(raw) == nbytes, "short read %d want %d" % (len(raw), nbytes)
    return np.frombuffer(raw, dtype=np.float32).reshape(nt, ns, *shape[2:])

R = reg()

# active_matter concentration 256x256 -> 64x64 (/4 /4)
url = R["active_matter"]["train"][0]
print("active_matter:", url, flush=True)
off, shape = get_off(url, "t0_fields/concentration")
print("  conc offset=%d shape=%s" % (off, shape), flush=True)
c = fetch(url, off, shape, 3, 20, "_am")
c64 = c.reshape(3, 20, 256 // 4, 4, 256 // 4, 4).mean(axis=(3, 5))
print("  c64", c64.shape, "mean=%.4f std=%.4f" % (c64.mean(), c64.std()), flush=True)
np.savez_compressed(os.path.join(BASE, "active64.npz"), active=c64.astype(np.float32))

# euler density 512x512 -> 64x64 (/8 /8)
url = R["euler_multi_quadrants_periodicBC"]["train"][0]
print("euler:", url, flush=True)
off, shape = get_off(url, "t0_fields/density")
print("  density offset=%d shape=%s" % (off, shape), flush=True)
e = fetch(url, off, shape, 4, 20, "_eu")
e64 = e.reshape(4, 20, 512 // 8, 8, 512 // 8, 8).mean(axis=(3, 5))
print("  e64", e64.shape, "mean=%.4f std=%.4f" % (e64.mean(), e64.std()), flush=True)
np.savez_compressed(os.path.join(BASE, "euler64.npz"), euler=e64.astype(np.float32))

print("done", flush=True)
