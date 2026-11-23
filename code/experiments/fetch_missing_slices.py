# -*- coding: utf-8 -*-
"""Fetch the 6 missing 64x64 system slices (gray/euler/visco/rb/ac/ps) directly
from the official The Well SDSC source, via HTTP byte-range reads over the HDF5
file (NO full download). Produces the npz files that sol4_routing.py and
routing_regularizer_by_family.py expect under ~/well_b3/.

Protocol mirrors fetch_well_data.py / fetch_gray.py in the repo:
  - each system: take 1 train HDF5 (from the_well WELL_REGISTRY), read the first
    NTraj trajectories and first NTIM time steps of the system's primary field,
    downsample native resolution -> 64x64 (block-mean), save <key>64.npz.
Primary fields per system (confirmed via HDF5 header probe):
  gray   : first time-varying field in t0_fields (Gray-Scott A)
  euler  : t0_fields/density
  visco  : t0_fields/pressure
  rb     : t0_fields/pressure
  ac     : t0_fields/pressure
  ps     : t0_fields/velocity (first dim_varying field)
The Well files are compressed (chunked); a 64x64 slice of ~4 traj x 20 steps is
small, but HDF5 random reads over http pull whole chunks. To stay tractable we
read a bounded prefix of the field and extract the needed samples.

Usage:  python fetch_missing_slices.py [--limit-systems gray,ps]
"""
import io, os, sys, time, argparse
import urllib.request
import numpy as np, h5py, yaml, fsspec
from the_well.utils import download as d

BASE = os.path.expanduser("~/well_b3")
HOST = "https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/datasets"

# key -> (registry_name, primary_field, split). Field chosen from header probe.
PLAN = {
    "gray":  ("gray_scott_reaction_diffusion",  "A",       "train"),  # Gray-Scott field A (128x128)
    "euler": ("euler_multi_quadrants_periodicBC", "density", "train"),
    "visco": ("viscoelastic_instability",         "pressure", "train"),
    "rb":    ("rayleigh_benard",                  "pressure", "train"),
    "ac":    ("acoustic_scattering_inclusions",   "pressure", "train"),
    "ps":    ("planetswe",                        "height",  "train"),
}

R = yaml.safe_load(io.open(d.WELL_REGISTRY, encoding="utf-8"))

def down64(A):
    """Block-mean downsample native (n,t,H,W) -> 64x64 (exact, no resampling)."""
    n, t, H, W = A.shape
    if H == 64 and W == 64:
        return A.astype(np.float32)
    if H % 64 == 0 and W % 64 == 0:
        fh, fw = H // 64, W // 64
        return A.reshape(n, t, 64, fh, 64, fw).mean(axis=(3, 5)).astype(np.float32)
    raise ValueError(f"cannot block-mean {H}x{W} to 64x64 (must be divisible by 64)")

def pick_field(h5, tgroup, hint):
    """Return the time-varying field name in tgroup matching hint (or first)."""
    f = h5[tgroup]
    fn = list(f.attrs.get("field_names", []))
    print(f"   {tgroup} fields: {fn}")
    if hint in fn and f[hint].attrs.get("time_varying"):
        return hint
    for cand in [hint, "tracer", "A", "density", "pressure", "height", "velocity", "buoyancy", "speed_of_sound"]:
        for n_ in fn:
            if cand in n_ and f[n_].attrs.get("time_varying"):
                return n_
    for n_ in fn:
        if f[n_].attrs.get("time_varying"):
            return n_
    return fn[0]

def fetch_one(key, NTraj, NTIM):
    regname, field_hint, split = PLAN[key]
    url = R[regname][split][0]
    out = os.path.join(BASE, f"{key}64.npz")
    print(f"=== {key} <- {regname} field~{field_hint} [{split}] ===")
    print(f"   url: {url}")
    fs, _ = fsspec.url_to_fs(url, cache_type="blockcache", block_size=8*1024*1024)
    t0 = time.time()
    with fs.open(url, "rb") as f, h5py.File(f, "r") as h5:
        print(f"   opening header {time.time()-t0:.1f}s")
        fgroup = "t0_fields" if "t0_fields" in h5 else "t1_fields"
        field = pick_field(h5, fgroup, field_hint)
        dset = h5[fgroup][field]
        shape = dset.shape
        print(f"   field={field} shape={shape} dtype={dset.dtype} chunks={dset.chunks} "
              f"compression={dset.compression} time_varying={dset.attrs.get('time_varying')}")
        # sample up to NTraj trajectories, NTIM time steps (batch read; blockcache -> fast)
        N = min(NTraj, max(1, shape[0]))
        T = min(NTIM, max(4, shape[1]))
        print(f"   reading dset[:{N}, :{T}] ...")
        t1 = time.time()
        arr = dset[:N, :T].astype(np.float32)   # (N, T, H, W)
        print(f"   batch read {arr.shape} in {time.time()-t1:.1f}s")
        A64 = down64(arr)
        print(f"   -> {A64.shape}; mean={A64.mean():.4f} std={A64.std():.4f} ({time.time()-t0:.1f}s)")
    np.savez_compressed(out, **{key: A64})
    print(f"   saved {out} ({os.path.getsize(out)/1e6:.1f} MB)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-systems", default="")
    ap.add_argument("--ntraj", type=int, default=4)
    ap.add_argument("--ntim", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(BASE, exist_ok=True)
    want = [k for k in args.limit_systems.split(",") if k] or list(PLAN.keys())
    for key in want:
        fetch_one(key, args.ntraj, args.ntim)
    print("\nAll requested slices done. Missing from sol4 set:",
          [k for k in PLAN if not os.path.exists(os.path.join(BASE, f"{k}64.npz"))])

if __name__ == "__main__":
    main()
