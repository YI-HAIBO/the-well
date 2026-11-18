# -*- coding: utf-8 -*-
"""Resumable chunked download of the full official shear_flow train HDF5 (~13.4 GB).

Writes data/shear_flow_train_full.hdf5 + data/shear_dl_state.json (resume point).
Usage: python dl_shear_full.py
"""
import os, json, time, urllib.request, sys

URL = ("https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/"
       "datasets/shear_flow/data/train/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5")
BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
OUT = os.path.join(BASE, "data", "shear_flow_train_full.hdf5")
STATE = os.path.join(BASE, "data", "shear_dl_state.json")
CHUNK = 32 * 1024 * 1024  # 32 MB

def http_len():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])

def load_state(total):
    if os.path.exists(STATE):
        with open(STATE) as f:
            s = json.load(f)
        if s.get("total") == total and os.path.exists(OUT) and os.path.getsize(OUT) == s.get("offset"):
            return s["offset"]
    return 0

def main():
    total = http_len()
    off = load_state(total)
    mode = "ab" if off else "wb"
    print(f"total={total/1e9:.2f} GB, resume offset={off/1e9:.2f} GB", flush=True)
    with open(OUT, mode) as f:
        while off < total:
            end = min(off + CHUNK, total) - 1
            req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{end}"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=180) as r:
                        data = r.read()
                    break
                except Exception as e:
                    print("retry", attempt, e, flush=True)
                    time.sleep(3 * (attempt + 1))
            else:
                print("giving up this round", flush=True)
                sys.exit(1)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            off += len(data)
            with open(STATE, "w") as sf:
                json.dump({"total": total, "offset": off}, sf)
            print(f"{off/1e9:.2f}/{total/1e9:.2f} GB ({off*100//total}%)", flush=True)
    print("DOWNLOAD COMPLETE", flush=True)

if __name__ == "__main__":
    main()
