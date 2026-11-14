# -*- coding: utf-8 -*-
"""
download_well_subset.py -- download a SMALL subset of The Well from the official
Flatiron SDSC host (sdsc-users.flatironinstitute.org) using curl, mirroring the
directory layout of the_well.utils.download.well_download:
    <base>/datasets/<dataset>/stats.yaml
    <base>/datasets/<dataset>/data/<split>/<file>.hdf5

Usage:
  python download_well_subset.py --base-path ./data --dataset shear_flow \
      --splits train,test --n 2

This fetches only the first N files per split (v2.0 registry) -- a deliberate
smoke subset. Use --n all to fetch everything (NOT recommended here).
"""
from __future__ import annotations
import argparse, io, os, subprocess, sys, time, yaml

CURL = r"C:\Windows\System32\curl.exe"


def main():
    ap = argparse.ArgumentParser(description="Download small The Well subset (SDSC)")
    ap.add_argument("--base-path", default="./data")
    ap.add_argument("--dataset", default="shear_flow")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--n", default=2, help="files per split (or 'all')")
    ap.add_argument("--registry", default=None)
    args = ap.parse_args()

    import the_well.utils.download as d
    registry_path = args.registry or d.WELL_REGISTRY
    with io.open(registry_path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)

    ds = args.dataset
    assert ds in reg, "unknown dataset %s; have %s" % (ds, list(reg.keys()))

    base = os.path.abspath(os.path.expanduser(args.base_path))
    n = args.n
    total = 0
    t0 = time.time()

    def run(curl_cmd):
        nonlocal total
        print("  curl ->", curl_cmd[-1].split("/")[-1], flush=True)
        r = subprocess.run(curl_cmd)
        if r.returncode != 0:
            print("  !! curl rc=%d" % r.returncode, flush=True)
        return r.returncode

    # stats.yaml
    stat_url = reg[ds]["stats"]
    stat_path = os.path.join(base, "datasets", ds, "stats.yaml")
    os.makedirs(os.path.dirname(stat_path), exist_ok=True)
    run([CURL, "--create-dirs", "--continue-at", "-", "-sS", "-o", stat_path, stat_url])
    total += 1

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        urls = reg[ds][split]
        if n != "all":
            urls = urls[: int(n)]
        outdir = os.path.join(base, "datasets", ds, "data", split)
        os.makedirs(outdir, exist_ok=True)
        print("Downloading %s/%s  n=%d" % (ds, split, len(urls)), flush=True)
        for url in urls:
            fn = os.path.join(outdir, os.path.basename(url))
            if os.path.exists(fn) and os.path.getsize(fn) > 0:
                print("  exists, skip:", os.path.basename(fn), flush=True)
                continue
            rc = run([CURL, "--create-dirs", "--continue-at", "-", "-sS",
                      "-o", fn, url])
            total += 1
            if rc != 0:
                print("  FAILED:", url, flush=True)
    print("DONE: %d files in %.1fs  base=%s" % (total, time.time() - t0, base), flush=True)


if __name__ == "__main__":
    sys.exit(main())
