# -*- coding: utf-8 -*-
"""Multi-threaded resumable download of the FULL official shear train HDF5
from the HuggingFace mirror (hf-mirror.com), which serves at ~4 MB/s per
stream vs ~0.3 MB/s aggregate on the flatiron mirror.

Target: data/datasets/shear_flow/data/train/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5
        (the well's official directory layout; 13.44 GB, 32 train trajectories)
Downloaded to <target>.part with per-stream resume state in
data/shear_dl_state_hf.json, then atomically renamed on completion.

Usage: python dl_shear_hf.py [NTHREADS]
"""
import os, json, time, sys, threading, urllib.request

URL = ("https://hf-mirror.com/datasets/polymathic-ai/shear_flow/resolve/main/"
       "data/train/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5")
TOTAL = 13438550016  # verified via HEAD
BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
FINAL = os.path.join(BASE, "data", "datasets", "shear_flow", "data", "train",
                     "shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5")
OUT = FINAL + ".part"
STATE = os.path.join(BASE, "data", "shear_dl_state_hf.json")

state_lock = threading.Lock()
print_lock = threading.Lock()

def load_state():
    if os.path.exists(STATE):
        try:
            return {int(k): int(v) for k, v in json.load(open(STATE)).items()}
        except Exception:
            pass
    return {}

def save_state(st):
    with state_lock:
        with open(STATE, "w") as f:
            json.dump({str(k): v for k, v in st.items()}, f)

def stream(i, start, end, chunk, st):
    off = int(st.get(i, start))
    if off > end:
        return
    handle = open(OUT, "r+b")
    t0 = time.time()
    try:
        while off <= end:
            e = min(off + chunk - 1, end)
            req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{e}"})
            data = None
            for _ in range(6):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    break
                except Exception:
                    time.sleep(3)
            if data is None:
                raise RuntimeError(f"stream {i} range {off}-{e} failed after retries")
            handle.seek(off)
            handle.write(data)
            handle.flush()
            off += len(data)
            st[i] = off
            if off % (64 * 1024 * 1024) < len(data) or off > end:
                save_state(st)
            dt = time.time() - t0
            done = off - start
            with print_lock:
                print(f"stream {i:02d}  {done/1e6:.0f}/{(end-start+1)/1e6:.0f} MB "
                      f"({done/1e6/max(dt,1e-6):.1f} MB/s)", flush=True)
    except Exception as ex:
        save_state(st)
        with print_lock:
            print(f"stream {i} stop at {off}: {ex}", flush=True)
    finally:
        handle.close()
        save_state(st)

def main():
    nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    print(f"total={TOTAL/1e9:.2f} GB from hf-mirror, {nthreads} streams", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if not os.path.exists(OUT):
        with open(OUT, "wb") as f:
            f.truncate(TOTAL)
    st = load_state()
    size = TOTAL // nthreads
    ranges = [(i * size, min((i + 1) * size - 1, TOTAL - 1)) for i in range(nthreads)]
    threads = [threading.Thread(target=stream, args=(i, s, e, 1024 * 1024, st))
               for i, (s, e) in enumerate(ranges)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    save_state(st)
    all_done = all(int(st.get(i, ranges[i][0])) > ranges[i][1] for i in range(nthreads))
    if all_done:
        sz = os.path.getsize(OUT)
        if sz == TOTAL:
            os.replace(OUT, FINAL)
            print(f"DONE {TOTAL/1e9:.2f} GB in {time.time()-t0:.0f}s -> {FINAL}", flush=True)
        else:
            print(f"size mismatch: {sz} != {TOTAL}; keeping .part for inspection", flush=True)
    else:
        covered = sum(min(int(st.get(i, ranges[i][0])), ranges[i][1] + 1) for i in range(nthreads))
        print(f"PARTIAL {covered/1e9:.2f}/{TOTAL/1e9:.2f} GB; re-run to resume", flush=True)

if __name__ == "__main__":
    main()
