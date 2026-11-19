# -*- coding: utf-8 -*-
"""Multi-threaded parallel range download of the full official shear train HDF5.

The single-stream download to this mirror was capped at ~tens of KB/s
(a 13.4 GB file never made progress). This splits the file into N parallel
range streams and aggregates them, which usually beats the per-connection
throttle by an order of magnitude.

Resumable: writes data/shear_flow_train_full.hdf5 + data/shear_dl_state.json.
Usage: python dl_shear_full_mt.py [NTHREADS]
"""
import os, json, time, sys, threading, urllib.request

URL = ("https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/"
       "datasets/shear_flow/data/train/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5")
BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
OUT = os.path.join(BASE, "data", "shear_flow_train_full.hdf5")
STATE = os.path.join(BASE, "data", "shear_dl_state.json")

state_lock = threading.Lock()
total_bytes = 0

def http_len():
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])

def stream(start, end, chunk=4 * 1024 * 1024, fh=None):
    global total_bytes
    off = start
    handle = fh or open(OUT, "r+b")
    try:
        while off <= end:
            e = min(off + chunk - 1, end)
            req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{e}"})
            data = None
            for _ in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    break
                except Exception:
                    time.sleep(3)
            if data is None:
                raise RuntimeError(f"range {off}-{e} failed after retries")
            handle.seek(off)
            handle.write(data)
            handle.flush()
            off += len(data)
            with state_lock:
                total_bytes += len(data)
                print(f"total {total_bytes/1e9:.2f} GB  [stream {start}-{end} -> {off}]", flush=True)
    finally:
        handle.close()

def main():
    nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    total = http_len()
    print(f"total={total/1e9:.2f} GB, {nthreads} parallel range streams", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # pre-allocate so threads can seek-write
    with open(OUT, "wb") as f:
        f.truncate(total)
    size = total // nthreads
    streams = [(i * size, min((i + 1) * size - 1, total - 1)) for i in range(nthreads)]
    threads = [threading.Thread(target=stream, args=(s, e)) for s, e in streams]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0
    with open(STATE, "w") as sf:
        json.dump({"total": total, "done": total, "nthreads": nthreads, "seconds": round(dt, 1)}, sf)
    print(f"DONE {total/1e9:.2f} GB in {dt:.0f}s = {total/1e9/dt:.3f} GB/s", flush=True)

if __name__ == "__main__":
    main()
