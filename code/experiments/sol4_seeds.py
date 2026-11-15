# -*- coding: utf-8 -*-
"""#30 faithful redo: 10-seed in-domain shear 64^2, paper MAIN config (ch=32, lr=1e-3, 60 steps)."""
import numpy as np, torch, torch.nn as nn, os, json
from math import sqrt
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
def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float('inf')
class MoE_lite(nn.Module):
    def __init__(self, ch=32, in_ch=4):
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
# paper main config: hidden 32, 12 Fourier modes, lr=1e-3, 60 steps
shear = np.load(os.path.join(BASE, "data64.npz"))["shear"]
X, Y = (std(a) for a in mk_xy(shear.astype(np.float32)))
perm = np.random.RandomState(0).permutation(len(X)); ntr = int(0.8*len(X))
Xtr = torch.tensor(X[perm[:ntr]]); Ytr = torch.tensor(Y[perm[:ntr]])
Xev = torch.tensor(X[perm[ntr:]]); Yev = Y[perm[ntr:]]
print("train=%d eval=%d (ch=32, lr=1e-3, 60 steps)" % (ntr, len(Xev)), flush=True)
seeds = []
for seed in range(10):
    torch.manual_seed(seed)
    m = MoE_lite(ch=32); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
    for s in range(60): opt.zero_grad(); lf(m(Xtr), Ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad(): pred = m(Xev).numpy()
    seeds.append(vrmse(pred, Yev))
seeds = np.array(seeds)
tstat = (seeds.mean() - 1.0) / (seeds.std(ddof=1) / sqrt(len(seeds))) if seeds.std(ddof=1) > 0 else float('nan')
print("VRMSE per seed:", np.round(seeds, 3).tolist())
print("mean=%.3f std=%.3f median=%.3f IQR=[%.3f,%.3f] min=%.3f max=%.3f" % (
    seeds.mean(), seeds.std(), np.median(seeds), np.percentile(seeds,25), np.percentile(seeds,75), seeds.min(), seeds.max()))
print("one-sample t vs baseline 1.0: t=%.2f  df=9" % tstat)
json.dump({"vrmse": seeds.tolist(), "mean": float(seeds.mean()), "std": float(seeds.std()),
           "median": float(np.median(seeds)), "q1": float(np.percentile(seeds,25)),
           "q3": float(np.percentile(seeds,75)), "min": float(seeds.min()), "max": float(seeds.max()),
           "t": float(tstat), "n": 10, "config": "ch=32, lr=1e-3, 60 steps"},
          open(os.path.join(BASE, "sol4_seeds.json"), "w"), indent=2)
print("Saved ->", os.path.join(BASE, "sol4_seeds.json"))