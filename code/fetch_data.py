# -*- coding: utf-8 -*-
"""Fetch the one-click reproduction data from the GitHub release.

Downloads the official The-Well slices (repro-data-v1 release) into ./data/
and reassembles the split turb archive. No credentials needed — release
assets are public.

Usage:  python fetch_data.py [train|test|turb|all]
"""
import os, sys, time, urllib.request, urllib.parse

REPO = "YI-HAIBO/the-well"
TAG = "repro-data-v1"
BASE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.github.com/repos/%s/releases/tags/%s" % (REPO, TAG)

def api_json():
    req = urllib.request.Request(API)
    req.add_header("Accept", "application/vnd.github+json")
    for a in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return __import__("json").loads(r.read().decode())
        except Exception as e:
            print("api retry", a, e); time.sleep(5 * (a + 1))
    raise RuntimeError("cannot reach release API")

def download(url, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("exists", out); return
    for a in range(4):
        try:
            with urllib.request.urlopen(url, timeout=1800) as r:
                open(out, "wb").write(r.read())
            print("downloaded", os.path.basename(out)); return
        except Exception as e:
            print("dl retry", a, e); time.sleep(8 * (a + 1))
    raise RuntimeError("download failed " + out)

def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    rel = api_json()
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)

    plan = {
        "train": [("slice_train.npz", "data/slice_train.npz")],
        "test":  [("slice_test.npz", "data/slice_test.npz")],
    }
    for k in ("train", "test"):
        if what in ("all", k):
            for name, relpath in plan[k]:
                if name in assets:
                    download(assets[name], os.path.join(BASE, relpath))
                else:
                    print("WARN", name, "not in release yet")

    if what in ("all", "turb"):
        parts = sorted(n for n in assets if n.startswith("turb_rad_layer.hdf5.part_"))
        dst = os.path.join(BASE, "turb_rad_layer.hdf5")
        if len(parts) and not os.path.exists(dst):
            print("reassembling turb from", len(parts), "parts ...")
            with open(dst, "wb") as f:
                for p in parts:
                    print("  +", p)
                    download(assets[p], os.path.join(BASE, "data", p))
                    f.write(open(os.path.join(BASE, "data", p), "rb").read())
            print("reassembled", dst)
        else:
            print("turb:", "ready" if os.path.exists(dst) else "parts not all uploaded yet")

if __name__ == "__main__":
    main()
