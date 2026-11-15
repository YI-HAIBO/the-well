# -*- coding: utf-8 -*-
"""B1) Third-system zero-shot cross-system generalization (validates T2 with a
third, more distant operator family: gray_scott_reaction_diffusion).

For each per-system model (trained 60 steps, matching the paper's setup) we
measure the zero-shot VRMSE on every OTHER system, plus the in-domain VRMSE.
The mean baseline corresponds to VRMSE = 1.0.
"""
import numpy as np, torch, torch.nn as nn, os, json

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
BASE = os.path.expanduser("~/well_b3")

z = np.load(os.path.join(BASE, "data64.npz"))
g = np.load(os.path.join(BASE, "gray64.npz"))
DATA = {
    "shear": z["shear"].astype(np.float32),
    "turb":  z["turb"].astype(np.float32),
    "gray":  g["gray"].astype(np.float32),
}

def mk_xy(T, n_in=4):
    n, st, H, W = T.shape
    X, Y = [], []
    for t in range(st - n_in):
        X.append(T[:, t:t+n_in]); Y.append(T[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

def std(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s

# build (X, Y) per system, per-sample normalized
XY = {}
for k, T in DATA.items():
    X, Y = (std(a) for a in mk_xy(T))
    XY[k] = (X.astype(np.float32), Y.astype(np.float32))

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

def train_eval(train_key, eval_key, steps=60, lr=1e-3, seed=0):
    Xtr, Ytr = XY[train_key]
    Xev, Yev = XY[eval_key]
    torch.manual_seed(seed)
    m = MoE_lite()
    Xt_ = torch.tensor(Xtr); Yt_ = torch.tensor(Ytr)
    opt = torch.optim.AdamW(m.parameters(), lr=lr); lf = nn.MSELoss()
    for s in range(steps):
        opt.zero_grad(); lf(m(Xt_), Yt_).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(Xev)).numpy()
    return vrmse(pred, Yev)

results = {}
for tr in ["shear", "turb", "gray"]:
    row = {}
    for ev in ["shear", "turb", "gray"]:
        v = train_eval(tr, ev)
        row[ev] = v
        tag = "in-domain" if tr == ev else "zero-shot"
        print("  train=%s eval=%s  VRMSE=%.3f  (%s)" % (tr, ev, v, tag), flush=True)
    results[tr] = row

out = os.path.join(BASE, "b1_results.json")
json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
print("\nSaved ->", out, flush=True)

# summary: does any cross-system zero-shot go below the mean baseline (VRMSE<1)?
below = [(tr, ev) for tr in results for ev in results[tr] if tr != ev and results[tr][ev] < 1.0]
print("zero-shot below mean baseline (VRMSE<1):", below, flush=True)
print("gray zero-shot from shear/turb:", results['shear']['gray'], results['turb']['gray'], flush=True)
