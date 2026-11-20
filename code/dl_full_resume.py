# -*- coding: utf-8 -*-
"""Resumable multi-stream range download of the full official shear train HDF5.

Designed for CI: a clean runner is NOT subject to the per-IP throttle that
crawls on this mirror (~0.3 MB/s locally), so a public runner can pull the
13.4 GB file in a fraction of that time. The file is large for a single
GitHub artifact, so this script is *resumable by on-disk byte count*:

  - data/shear_flow_train_full.hdf5 : the target, pre-allocated to total size
  - data/shear_dl_state.json        : {stream_id: completed_offset}

On each invocation it (re)computes N parallel ranges, then for each range
downloads only the still-missing bytes (from the last persisted offset to
the range end), flushing 1 MB chunks. Because offsets are absolute, repeated
invocations never re-download already-fetched bytes. CI re-dispatches the
workflow until all streams reach their end; a final check confirms the whole
file is covered and then runs the 32-trajectory full-data re-test.

Usage:
  python code/dl_full_resume.py [NTHREADS] [GB_BUDGET]
    NTHREADS  parallel range streams (default 32)
    GB_BUDGET stop after this many NEW bytes in this invocation (default 0=all)
"""
import os, json, time, sys, threading, urllib.request

URL = ("https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/"
       "datasets/shear_flow/data/train/shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5")
TOTAL = 13421804672
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
OUT = os.path.join(BASE, "data", "shear_flow_train_full.hdf5")
STATE = os.path.join(BASE, "data", "shear_dl_state.json")

STATE_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()

def load_state():
    if os.path.exists(STATE):
        try:
            return {int(k): int(v) for k, v in json.load(open(STATE)).items()}
        except Exception:
            pass
    return {}

def save_state(st):
    with STATE_LOCK:
        with open(STATE, "w") as f:
            json.dump({str(k): v for k, v in st.items()}, f)

class Budget:
    def __init__(self, gb):
        self.limit = int(gb * 1e9) if gb > 0 else float("inf")
        self.used = 0
        self.lock = threading.Lock()
    def take(self, n):
        with self.lock:
            if self.used + n > self.limit:
                return False
            self.used += n
            return True

def stream(i, start, end, chunk, st, budget):
    off = int(st.get(i, start))
    if off > end:
        return
    handle = open(OUT, "r+b")
    try:
        while off <= end:
            e = min(off + chunk - 1, end)
            if not budget.take(e - off + 1):
                break
            req = urllib.request.Request(URL, headers={"Range": f"bytes={off}-{e}"})
            data = None
            for _ in range(6):
                try:
                    with urllib.request.urlopen(req, timeout=180) as r:
                        data = r.read()
                    break
                except Exception:
                    time.sleep(4)
            if data is None:
                raise RuntimeError(f"stream {i} range {off}-{e} failed after retries")
            handle.seek(off)
            handle.write(data)
            handle.flush()
            off += len(data)
            st[i] = off
            save_state(st)
            with PRINT_LOCK:
                print(f"stream {i:02d}  {off/1e6:.0f}/{(end+1)/1e6:.0f} MB  "
                      f"(new {budget.used/1e9:.2f} GB this run)", flush=True)
    except Exception as ex:
        save_state(st)
        with PRINT_LOCK:
            print(f"stream {i} stop at off={off}: {ex}", flush=True)
    finally:
        handle.close()
        save_state(st)

def main():
    nthreads = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    gb = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if not os.path.exists(OUT):
        with open(OUT, "wb") as f:
            f.truncate(TOTAL)
    st = load_state()
    size = TOTAL // nthreads
    ranges = [(i * size, min((i + 1) * size - 1, TOTAL - 1)) for i in range(nthreads)]
    budget = Budget(gb)
    t0 = time.time()
    threads = [threading.Thread(target=stream, args=(i, s, e, 1024 * 1024, st, budget))
               for i, (s, e) in enumerate(ranges)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    save_state(st)
    all_done = all(int(st.get(i, ranges[i][0])) > ranges[i][1] for i in range(nthreads))
    covered = sum(min(int(st.get(i, ranges[i][0])), ranges[i][1] + 1) for i in range(nthreads))
    print(f"[dl_full_resume] {nthreads} streams, new this run {budget.used/1e9:.2f} GB "
          f"in {time.time()-t0:.0f}s; file coverage {covered/1e9:.2f}/{TOTAL/1e9:.2f} GB "
          f"({100*covered/TOTAL:.1f}%); {'COMPLETE' if all_done else 'incomplete -> re-run to resume'}",
          flush=True)
    sys.exit(0 if all_done else 3)

if __name__ == "__main__":
    main()
