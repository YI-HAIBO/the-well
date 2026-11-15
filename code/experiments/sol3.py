# -*- coding: utf-8 -*-
"""sol3: conservation PROJECTION on raw turb (robust). Train on normalized (4D), project in raw."""
import numpy as np, torch, torch.nn as nn, os, json
torch.set_num_threads(4)
BASE = os.path.expanduser("~/well_b3")
R = {}
def mk_xy(A, n_in=4):
    n, st, H, W = A.shape; X, Y = [], []
    for t in range(st - n_in):
        X.append(A[:, t:t+n_in]); Y.append(A[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)
def pstd3(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s
class MoE_lite(nn.Module):
    def __init__(self, ch=16, in_ch=4):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1); self.e1 = nn.Conv2d(ch, ch, 1); self.e2 = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, 1, 1); self.gate = nn.Linear(4, 2); self.desc = nn.Parameter(torch.randn(4))
    def forward(self, x):
        h = self.in_conv(x); g = torch.softmax(self.gate(self.desc[None]), -1)
        return self.out(g[0, 0] * self.e1(h) + g[0, 1] * self.e2(h))
S = np.load(os.path.join(BASE, "data64.npz")); U = S["turb"].astype(np.float32)
Xr, Yr = mk_xy(U)          # Xr (n,4,64,64), Yr (n,1,64,64)
Xrn = np.stack([pstd3(Xr[:, i]) for i in range(4)], 1)  # per-channel standardize
Yrn = pstd3(Yr[:, 0])[:, None]
n = len(Xrn); pr = np.random.RandomState(3).permutation(n); ntr = int(0.8 * n)
torch.manual_seed(0)
m = MoE_lite(ch=16); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
Xt_ = torch.tensor(Xrn[pr[:ntr]]); Yt_ = torch.tensor(Yrn[pr[:ntr]])
for s in range(40): opt.zero_grad(); lf(m(Xt_), Yt_).backward(); opt.step()
m.eval()
with torch.no_grad(): pred = m(torch.tensor(Xrn[pr[ntr:]])).numpy()
pred = pred[:, 0]          # (m,64,64)
yr = Yr[pr[ntr:], 0]       # (m,64,64) raw
xlast = Xr[pr[ntr:], -1]   # (m,64,64) raw
sd = Yrn[pr[ntr:], 0].std(axis=(1, 2)) + 1e-6; mu = yr.mean(axis=(1, 2), keepdims=True)
pred_raw = pred * sd[:, None, None] + mu
def cerr(p): return float(np.abs(p.sum(axis=(1, 2)) - yr.sum(axis=(1, 2))).mean() / (np.abs(yr.sum(axis=(1, 2))).mean() + 1e-9))
def relL2(p): return float(np.sqrt(((p - yr) ** 2).mean() / (yr.var() + 1e-12)))
ratio = (xlast.sum(axis=(1, 2)) + 1e-9) / (pred_raw.sum(axis=(1, 2)) + 1e-9)
pred_proj = pred_raw * ratio[:, None, None]
drift = float(np.abs(yr.sum(axis=(1, 2)) - xlast.sum(axis=(1, 2))).mean() / (np.abs(xlast.sum(axis=(1, 2))).mean() + 1e-9))
R["conservation_projection"] = {
    "cons_err_raw": cerr(pred_raw), "cons_err_projected": cerr(pred_proj),
    "relL2_raw": relL2(pred_raw), "relL2_projected": relL2(pred_proj), "data_drift": drift}
print("cons_err raw %.4e -> projected %.4e (data drift %.4e)" % (cerr(pred_raw), cerr(pred_proj), drift), flush=True)
print("relL2 raw %.3f -> projected %.3f" % (relL2(pred_raw), relL2(pred_proj)), flush=True)
json.dump(R, open(os.path.join(BASE, "sol3_results.json"), "w"), indent=2, ensure_ascii=False)
print("Saved", flush=True)
