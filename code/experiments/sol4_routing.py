# -*- coding: utf-8 -*-
"""#3/#23 only: 8-system joint gate routing confusion + delta CI (8 experts)."""
import numpy as np, torch, torch.nn as nn, os, json, time
from collections import Counter
torch.set_num_threads(4)
BASE = os.path.expanduser("~/well_b3")
def mk_xy(A, n_in=4):
    n, st, H, W = A.shape; X, Y = [], []
    for t in range(st - n_in):
        X.append(A[:, t:t+n_in]); Y.append(A[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)
def std(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s
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
out = {"confusion": conf.tolist(), "majority": major.tolist(), "experts_used": len(cnt),
       "systems_dedicated": int(dedicated), "delta_system": float(delta_sys),
       "delta_sample": float(sample_share), "delta_ci": ci, "n_experts": 8}
json.dump(out, open(os.path.join(BASE, "sol4_routing.json"), "w"), indent=2, ensure_ascii=False)
print("Saved ->", os.path.join(BASE, "sol4_routing.json"))