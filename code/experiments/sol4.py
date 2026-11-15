# -*- coding: utf-8 -*-
"""#11 256^2 capacity-compensates-resolution; #29 real param count + ch=64;
#30 10-seed main in-domain; #3/#23 8-system real-gate routing confusion + delta CI."""
import numpy as np, torch, torch.nn as nn, os, json, time

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
BASE = os.path.expanduser("~/well_b3")

# ---------- 256^2 native shear (reuse _sh.raw) ----------
raw = open(os.path.join(BASE, "_sh.raw"), "rb").read()
T = np.frombuffer(raw, dtype=np.float32).reshape(4, 20, 256, 512)
S256 = T.reshape(4, 20, 256, 1, 512 // 2, 2).mean(axis=(3, 5))
print("native->256x256", S256.shape, "mean=%.4f std=%.4f" % (S256.mean(), S256.std()), flush=True)

def mk_xy(A, n_in=4):
    n, st, H, W = A.shape; X, Y = [], []
    for t in range(st - n_in):
        X.append(A[:, t:t+n_in]); Y.append(A[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

def std(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s

def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float('inf')

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

def params(m): return int(sum(p.numel() for p in m.parameters()))

X256, Y256 = (std(a) for a in mk_xy(S256))
X2 = torch.tensor(X256); Y2 = torch.tensor(Y256)

print("\n===== #11: 256^2 capacity sweep (40 steps, lr=1e-3) =====", flush=True)
cap = {}
for ch in [8, 16, 32]:
    torch.manual_seed(0)
    m = MoE_lite(ch=ch); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
    t0 = time.time()
    for s in range(40): opt.zero_grad(); lf(m(X2), Y2).backward(); opt.step()
    pt = (time.time() - t0) / 40
    m.eval()
    with torch.no_grad(): pred = m(X2[:16]).numpy()
    v = vrmse(pred, Y256[:16])
    cap[ch] = {"params": params(m), "vrmse_256": v, "s_per_step": pt}
    print("  ch=%2d params=%5d  VRMSE=%.3f  %.2f s/step" % (ch, cap[ch]["params"], v, pt), flush=True)

print("\n===== #29: ch=64 large-capacity point =====", flush=True)
big = {}
d64 = np.load(os.path.join(BASE, "data64.npz"))["shear"]
X64b, Y64b = (std(a) for a in mk_xy(d64))
X64t = torch.tensor(X64b); Y64t = torch.tensor(Y64b)
for (name, Xt, Yt, n_ev) in [("64x64", X64t, Y64t, len(X64b)), ("256x256", X2, Y2, 16)]:
    torch.manual_seed(0)
    m = MoE_lite(ch=64); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
    t0 = time.time()
    for s in range(40): opt.zero_grad(); lf(m(Xt), Yt).backward(); opt.step()
    pt = (time.time() - t0) / 40
    m.eval()
    with torch.no_grad(): pred = m(Xt[:n_ev]).numpy()
    big[name] = {"params": params(m), "vrmse": vrmse(pred, Yt[:n_ev].numpy()), "s_per_step": pt}
    print("  ch=64 @ %s params=%d VRMSE=%.3f  %.2f s/step" % (name, big[name]["params"], big[name]["vrmse"], pt), flush=True)

print("\n===== #30: 10-seed in-domain shear 64^2 (ch=32, lr=1e-3, 60 steps) =====", flush=True)
# paper main config: hidden 32, lr=1e-3, 60 steps
X, Y = (std(a) for a in mk_xy(d64))
perm = np.random.RandomState(0).permutation(len(X)); ntr = int(0.8*len(X))
Xtr = torch.tensor(X[perm[:ntr]]); Ytr = torch.tensor(Y[perm[:ntr]])
Xev = torch.tensor(X[perm[ntr:]]); Yev = Y[perm[ntr:]]
print("train=%d eval=%d" % (ntr, len(Xev)), flush=True)
seeds = []
for seed in range(10):
    torch.manual_seed(seed)
    m = MoE_lite(ch=32); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
    for s in range(60): opt.zero_grad(); lf(m(Xtr), Ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad(): pred = m(Xev).numpy()
    seeds.append(vrmse(pred, Yev))
seeds = np.array(seeds)
from math import sqrt
tstat = (seeds.mean() - 1.0) / (seeds.std(ddof=1) / sqrt(len(seeds))) if seeds.std(ddof=1) > 0 else float('nan')
print("VRMSE per seed:", np.round(seeds, 3).tolist(), flush=True)
print("mean=%.3f std=%.3f median=%.3f IQR=[%.3f,%.3f] min=%.3f max=%.3f" % (
    seeds.mean(), seeds.std(), np.median(seeds), np.percentile(seeds,25), np.percentile(seeds,75),
    seeds.min(), seeds.max()), flush=True)
print("one-sample t (vs 1.0) = %.2f  df=9" % tstat, flush=True)

print("\n===== #3/#23: 8-system joint gate routing (8 experts) =====", flush=True)
def load(name):
    if name == "shear": return np.load(os.path.join(BASE, "data64.npz"))["shear"]
    if name == "turb":  return np.load(os.path.join(BASE, "data64.npz"))["turb"]
    if name == "gray":  return np.load(os.path.join(BASE, "gray64.npz"))["gray"]
    if name == "euler": return np.load(os.path.join(BASE, "euler64.npz"))["euler"]
    if name == "visco": return np.load(os.path.join(BASE, "visco64.npz"))["visco"]
    if name == "rb":    return np.load(os.path.join(BASE, "rb64.npz"))["rb"]
    if name == "ac":    return np.load(os.path.join(BASE, "ac64.npz"))["ac"]
    if name == "ps":    return np.load(os.path.join(BASE, "ps64.npz"))["ps"]
SYSTEMS = ["shear", "turb", "gray", "euler", "visco", "rb", "ac", "ps"]
DAT = {}
for s in SYSTEMS:
    X, Y = (std(a) for a in mk_xy(load(s).astype(np.float32)))
    n = len(X); perm = np.random.RandomState(0).permutation(n); ntr = int(0.8*n)
    DAT[s] = (X[perm[:ntr]], Y[perm[:ntr]], X[perm[ntr:]], Y[perm[ntr:]])
class JointMoE(nn.Module):
    def __init__(self, ch=16, in_ch=4, n_experts=8, desc_dim=8, n_sys=8):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 1), nn.GELU(), nn.Conv2d(ch, ch, 1)) for _ in range(n_experts)])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(desc_dim, n_experts)
        self.desc = nn.Parameter(torch.randn(n_sys, desc_dim))
    def forward(self, x, desc_idx):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[desc_idx]), -1)
        eh = torch.stack([e(h) for e in self.experts], 1)
        w = g[:, :, None, None, None]
        return self.out((eh * w).sum(1))
idx = {s: i for i, s in enumerate(SYSTEMS)}
torch.manual_seed(0)
jm = JointMoE(n_experts=8, n_sys=8)
Xs, Ys, Ds = [], [], []
for s in SYSTEMS:
    X, Y, _, _ = DAT[s]
    Xs.append(X); Ys.append(Y); Ds.append(np.full(len(X), idx[s], dtype=np.int64))
Xc = torch.tensor(np.concatenate(Xs, 0)); Yc = torch.tensor(np.concatenate(Ys, 0))
Dc = torch.tensor(np.concatenate(Ds, 0))
opt = torch.optim.AdamW(jm.parameters(), lr=1e-3); lf = nn.MSELoss()
for st in range(40): opt.zero_grad(); lf(jm(Xc, Dc), Yc).backward(); opt.step()
jm.eval()
with torch.no_grad():
    conf = np.zeros((8, 8)); n_hold = np.zeros(8)
    for s in SYSTEMS:
        di = idx[s]
        g = torch.softmax(jm.gate(jm.desc[di]), -1).numpy()
        conf[di] = g; n_hold[di] = len(DAT[s][2])
print("per-system gate weights (rows=system, cols=expert):")
for s in SYSTEMS:
    print("    %-6s " % s + " ".join("%.2f" % w for w in conf[idx[s]]))
major = conf.argmax(1)
from collections import Counter
cnt = Counter(major.tolist())
dedicated = sum(1 for e in major if cnt[e] == 1)
print("majority expert per system:", major.tolist())
print("experts used (distinct):", len(cnt), "/ 8  systems dedicated: %d/8" % dedicated)
shared_experts = {e for e in major if cnt[e] > 1}
sample_share = sum(n_hold[idx[s]] for s in SYSTEMS if major[idx[s]] in shared_experts) / n_hold.sum()
delta_sys = (8 - dedicated) / 8.0
rng = np.random.RandomState(1)
bs = []
for b in range(300):
    draws = [int(np.argmax(np.random.multinomial(1, conf[idx[s]]))) for s in SYSTEMS]
    cc = Counter(draws)
    sh = sum(n_hold[idx[s]] for s in SYSTEMS if cc[draws[idx[s]]] > 1)
    bs.append(sh / n_hold.sum())
bs = np.array(bs)
ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
print("delta (system-level, shared-majority) = %.3f" % delta_sys)
print("delta (sample-level) = %.3f  95%%CI=[%.3f, %.3f]" % (sample_share, ci[0], ci[1]))

out = {"11_capacity_256": cap, "29_large_capacity": big,
       "30_seeds": {"vrmse": seeds.tolist(), "mean": float(seeds.mean()), "std": float(seeds.std()),
                     "median": float(np.median(seeds)), "q1": float(np.percentile(seeds,25)),
                     "q3": float(np.percentile(seeds,75)), "min": float(seeds.min()), "max": float(seeds.max()),
                     "t_vs_baseline": float(tstat)},
       "3_23_routing": {"confusion": conf.tolist(), "majority": major.tolist(),
                        "experts_used": len(cnt), "systems_dedicated": int(dedicated),
                        "delta_system": float(delta_sys), "delta_sample": float(sample_share),
                        "delta_ci": [float(ci[0]), float(ci[1])]}}
json.dump(out, open(os.path.join(BASE, "sol4_results.json"), "w"), indent=2, ensure_ascii=False)
print("\nSaved ->", os.path.join(BASE, "sol4_results.json"), flush=True)