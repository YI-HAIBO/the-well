# -*- coding: utf-8 -*-
"""sol2: rework D (conservation on RAW turb) and H (D_op via MMD of one-step increments)."""
import numpy as np, torch, torch.nn as nn, os, json
torch.set_num_threads(4)
BASE = os.path.expanduser("~/well_b3")
R = {}

def mk_xy(A, n_in=4):
    n, st, H, W = A.shape
    X, Y = [], []
    for t in range(st - n_in):
        X.append(A[:, t:t+n_in]); Y.append(A[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

def pstd(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s

def relL2(p, y):
    return float(np.sqrt(((p - y) ** 2).mean() / (y.var() + 1e-12)))

class MoE_lite(nn.Module):
    def __init__(self, ch=16, in_ch=4):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1); self.e1 = nn.Conv2d(ch, ch, 1); self.e2 = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, 1, 1); self.gate = nn.Linear(4, 2); self.desc = nn.Parameter(torch.randn(4))
    def forward(self, x):
        h = self.in_conv(x); g = torch.softmax(self.gate(self.desc[None]), -1)
        return self.out(g[0, 0] * self.e1(h) + g[0, 1] * self.e2(h))

S = np.load(os.path.join(BASE, "data64.npz"))
T = S["shear"].astype(np.float32); U = S["turb"].astype(np.float32)
def load(f): d = np.load(os.path.join(BASE, f + ".npz")); return d[list(d.keys())[0]].astype(np.float32)
G = load("gray64"); E = load("euler64"); V = load("visco64")
Xs, Ys = (pstd(a) for a in mk_xy(T)); Xt, Yt = (pstd(a) for a in mk_xy(U))
Xg, Yg = (pstd(a) for a in mk_xy(G)); Xe, Ye = (pstd(a) for a in mk_xy(E)); Xv, Yv = (pstd(a) for a in mk_xy(V))

# ---------- D2: conservation on RAW turb ----------
print("=== D2 conservation (RAW turb) ===", flush=True)
Xr, Yr = mk_xy(U)
n = len(Xr); pr = np.random.RandomState(2).permutation(n); ntr = int(0.8 * n)
# data drift (raw one-step integral drift)
drift = [abs(float(Yr[i,0].sum() - Xr[i,-1].sum())) / (abs(float(Xr[i,-1].sum())) + 1e-9) for i in range(n)]
print("  data one-step integral drift: mean %.3e max %.3e" % (np.mean(drift), np.max(drift)), flush=True)
cons = {}
for beta in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
    torch.manual_seed(0)
    m = MoE_lite(ch=16); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
    Xt_ = torch.tensor(Xr[pr[:ntr]]); Yt_ = torch.tensor(Yr[pr[:ntr]])
    for s in range(40):
        opt.zero_grad(); pred = m(Xt_); loss = lf(pred, Yt_)
        if beta > 0:
            sp = pred.sum(dim=(2, 3)); sy = Yt_.sum(dim=(2, 3))
            loss = loss + beta * ((sp - sy) ** 2 / (sy ** 2 + 1e-9)).mean()
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(torch.tensor(Xr[pr[ntr:]])).numpy(); yr = Yr[pr[ntr:]]
    c_err = float(np.abs(pred.sum(axis=(1, 2)) - yr.sum(axis=(1, 2))).mean() / (np.abs(yr.sum(axis=(1, 2))).mean() + 1e-9))
    rl2 = relL2(pred, yr)
    cons["beta%s" % beta] = {"cons_rel_err": c_err, "raw_relL2": rl2}
    print("  beta=%.4f: cons_err %.4e  raw_relL2 %.3f" % (beta, c_err, rl2), flush=True)
R["D2_conservation_raw"] = {"data_drift_mean": float(np.mean(drift)), "data_drift_max": float(np.max(drift)), "sweep": cons}

# ---------- H2: D_op via MMD of one-step increments ----------
print("=== H2 D_op (MMD of one-step increments) ===", flush=True)
def inc(X, Y):
    d = Y[:, 0] - X[:, -1]
    m = d.mean(axis=(1, 2), keepdims=True); s = d.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((d - m) / s).reshape(len(d), -1)
Ds = inc(Xs, Ys); Dt = inc(Xt, Yt); Dg = inc(Xg, Yg); De = inc(Xe, Ye); Dv = inc(Xv, Yv)
allD = np.concatenate([Ds, Dt, Dg, De, Dv], 0)
mu = allD.mean(0); C = allD - mu
U_, Sv, Vt = np.linalg.svd(C, full_matrices=False)
K = 8; proj = lambda D: (D - mu) @ Vt[:K].T
ps_, pt_, pg_, pe_, pv_ = proj(Ds), proj(Dt), proj(Dg), proj(De), proj(Dv)
def mmd2(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    # median heuristic bandwidth
    from numpy.linalg import norm
    ab = np.vstack([a, b]); n = len(ab)
    idx = np.random.RandomState(0).choice(n, min(300, n), replace=False)
    sub = ab[idx]
    dm = np.sqrt(((sub[:, None] - sub[None, :]) ** 2).sum(-1))
    h2 = np.median(dm[dm > 0]) ** 2 if (dm > 0).any() else 1.0
    if h2 <= 0: h2 = 1.0
    g = lambda X, Y: np.exp(-((X[:, None] - Y[None, :]) ** 2).sum(-1) / h2)
    return float(g(a, a).mean() + g(b, b).mean() - 2 * g(a, b).mean())
R["H2_Dop_mmd"] = {
    "shear_turb": mmd2(ps_, pt_), "shear_gray": mmd2(ps_, pg_),
    "turb_gray": mmd2(pt_, pg_), "shear_euler": mmd2(ps_, pe_),
    "shear_visco": mmd2(ps_, pv_),
}
print("  MMD^2: s-t %.4f s-g %.4f t-g %.4f s-e %.4f s-v %.4f" % (
    R["H2_Dop_mmd"]["shear_turb"], R["H2_Dop_mmd"]["shear_gray"], R["H2_Dop_mmd"]["turb_gray"],
    R["H2_Dop_mmd"]["shear_euler"], R["H2_Dop_mmd"]["shear_visco"]), flush=True)

out = os.path.join(BASE, "sol2_results.json")
json.dump(R, open(out, "w"), indent=2, ensure_ascii=False)
print("Saved ->", out, flush=True)
