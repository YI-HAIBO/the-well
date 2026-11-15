# -*- coding: utf-8 -*-
"""C1 + C3) Larger-resolution (256x256) training and online-deployment inference
throughput, on NATIVE-resolution shear flow data re-fetched via byte-range."""
import numpy as np, torch, torch.nn as nn, os, json, time, subprocess, io, yaml
import fsspec, h5py

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
BASE = os.path.expanduser("~/well_b3")

# --- re-fetch native shear tracer [0:4, 0:20] (256x512) ---
from the_well.utils import download as d
reg = yaml.safe_load(io.open(d.WELL_REGISTRY, encoding="utf-8"))
url = reg["shear_flow"]["train"][0]
fs = fsspec.filesystem("http", block_size=1024 * 1024)
with fs.open(url, "rb") as f, h5py.File(f, "r") as h:
    dd = h["t0_fields/tracer"]
    off = int(dd.id.get_offset()); shape = tuple(dd.shape)
nt, ns = 4, 20
inner = shape[2] * shape[3]
nbytes = nt * ns * inner * 4
r = subprocess.run(["curl", "-sS", "--retry", "8", "--retry-all-errors", "--retry-delay", "2",
                    "-r", "%d-%d" % (off, off + nbytes - 1), "-o", os.path.join(BASE, "_sh.raw"), url],
                   capture_output=True)
assert r.returncode == 0, r.stderr.decode()[:150]
raw = open(os.path.join(BASE, "_sh.raw"), "rb").read()
assert len(raw) == nbytes, (len(raw), nbytes)
T = np.frombuffer(raw, dtype=np.float32).reshape(nt, ns, 256, 512)
print("native tracer", T.shape, "mean=%.4f std=%.4f" % (T.mean(), T.std()), flush=True)

def d128(A): return A.reshape(4, 20, 256 // 2, 2, 512 // 4, 4).mean(axis=(3, 5))
def d256(A): return A.reshape(4, 20, 256 // 1, 1, 512 // 2, 2).mean(axis=(3, 5))
S128, S256 = d128(T), d256(T)
print("S128", S128.shape, "S256", S256.shape, flush=True)

def mk_xy(A, n_in=4):
    n, st, H, W = A.shape
    X, Y = [], []
    for t in range(st - n_in):
        X.append(A[:, t:t+n_in]); Y.append(A[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

def std(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s

class MoE_lite(nn.Module):
    def __init__(self, ch=16, in_ch=4):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.e1 = nn.Conv2d(ch, ch, 1); self.e2 = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(4, 2); self.desc = nn.Parameter(torch.randn(4))
    def forward(self, x):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[None]), -1)
        h = g[0,0]*self.e1(h) + g[0,1]*self.e2(h)
        return self.out(h)

def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float('inf')

res = {}

# ---------- C1: train at 256x256 (native), report VRMSE + per-step time ----------
X256, Y256 = (std(a) for a in mk_xy(S256))
X2 = torch.tensor(X256); Y2 = torch.tensor(Y256)
m2 = MoE_lite(); opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3); lf = nn.MSELoss()
t0 = time.time()
for s in range(40): opt2.zero_grad(); lf(m2(X2), Y2).backward(); opt2.step()
per_step = (time.time() - t0) / 40
m2.eval()
with torch.no_grad():
    pred = m2(X2[:16]).numpy()
v256 = vrmse(pred, Y256[:16])
res["C1_256"] = {"per_step_s": per_step, "vrmse_256": v256}
print("C1 256x256: %.2f s/step, 40 steps done, VRMSE=%.3f" % (per_step, v256), flush=True)

# ---------- C3: inference throughput at 64/128/256 ----------
X64, Y64 = (std(a) for a in mk_xy(T.reshape(4, 20, 256, 512).mean(axis=(2,3)) if False else np.load(os.path.join(BASE, "data64.npz"))["shear"]))
X128, Y128 = (std(a) for a in mk_xy(S128))
thr = {}
for name, Xs, Ys in [("64", X64, Y64), ("128", X128, Y128), ("256", X256, Y256)]:
    m = MoE_lite()
    xb = torch.tensor(Xs[:8]); yb = torch.tensor(Ys[:8])
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for s in range(30): o.zero_grad(); lf(m(xb), yb).backward(); o.step()
    m.eval()
    with torch.no_grad():
        for _ in range(3): m(xb)
        t0 = time.time()
        for _ in range(20): m(xb[:1])
        lat1 = (time.time() - t0) / 20
        t0 = time.time()
        for _ in range(20): m(xb)
        lat8 = (time.time() - t0) / 20
    thr[name] = {"lat_ms_b1": lat1*1000, "lat_ms_b8": lat8*1000,
                 "pred_per_s_b1": 1.0/lat1, "pred_per_s_b8": 8.0/lat8}
    print("C3 %s: batch1 %.2f ms (%.1f/s), batch8 %.2f ms (%.1f/s)" %
          (name, lat1*1000, 1.0/lat1, lat8*1000, 8.0/lat8), flush=True)
res["C3_throughput"] = thr

out = os.path.join(BASE, "c1c3_results.json")
json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
print("Saved ->", out, flush=True)
