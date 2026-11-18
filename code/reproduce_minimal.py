# -*- coding: utf-8 -*-
"""One-command minimal reproduction of the paper's core results.

Usage:  python reproduce_minimal.py
Needs:  data/slice_train.npz, data/slice_test.npz (official The-Well shear_flow
        slices, Reynolds 1e5 / Schmidt 2e0), optionally turb_rad_layer.hdf5 for
        the cross-system check.  CPU-only, fixed seeds, ~3 minutes.

Outputs headline VRMSE numbers (mean baseline = 1.0):
  1. in-domain (80/20 random windows)
  2. leave-one-trajectory-out (LOO), per-sample-normalized protocol
  3. cross-system zero-shot shear->turb (if turb file present)
  4. cross-system few-shot adaptation (16 target windows)
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(min(8, os.cpu_count() or 4))
BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]

# ---------------- data ----------------
def downsample(T, H=64, W=64):
    oH, oW = T.shape[-2], T.shape[-1]
    fH, fW = oH // H, oW // W
    if T.ndim == 5:
        T = T.reshape(T.shape[0], T.shape[1], fH, H, fW, W, T.shape[-1]).mean(axis=(2, 4))
    else:
        T = T.reshape(T.shape[0], T.shape[1], fH, H, fW, W).mean(axis=(2, 4))
    return T.astype(np.float32)

def std_ps(A):
    m = A.mean(axis=(-2, -1), keepdims=True); s = A.std(axis=(-2, -1), keepdims=True) + 1e-6
    return (A - m) / s

def mk_xy(T, n_in=4):
    X, Y = [], []
    for t in range(T.shape[0] - n_in):
        X.append(T[t:t+n_in]); Y.append(T[t+n_in:t+n_in+1])
    return np.stack(X), np.stack(Y)

def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float("inf")

print("[1/4] loading data ...", flush=True)
ztr = np.load(os.path.join(BASE, "data", "slice_train.npz"))
zte = np.load(os.path.join(BASE, "data", "slice_test.npz"))
Ptr = downsample(ztr["pressure"])          # (4,20,64,64)
Pte = downsample(zte["pressure"])          # (4,20,64,64)

# ---------------- model ----------------
class MoE_lite(nn.Module):
    def __init__(self, ch=16, in_ch=4, n_experts=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(n_experts)])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(4, n_experts)
        self.desc = nn.Parameter(torch.randn(4))
        self.n_experts = n_experts
    def forward(self, x):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[None]), -1)
        h = sum(g[0, i] * self.experts[i](h) for i in range(self.n_experts))
        return self.out(h)

def train(X, Y, steps=60, lr=1e-3, seed=0, ch=16):
    torch.manual_seed(seed)
    m = MoE_lite(ch=ch)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    lf = nn.MSELoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(steps):
        opt.zero_grad(); lf(m(Xt), Yt).backward(); opt.step()
    return m

def finetune(model, X, Y, steps=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(steps):
        opt.zero_grad(); lf(model(Xt), Yt).backward(); opt.step()
    return model

def ev(m, X, Y):
    m.eval()
    with torch.no_grad():
        return vrmse(m(torch.tensor(X)).numpy(), Y)

out = {}

# ---------------- 1) in-domain ----------------
print("[2/4] in-domain (80/20) ...", flush=True)
Xa = np.concatenate([mk_xy(Ptr[i])[0] for i in range(4)] + [mk_xy(Pte[i])[0] for i in range(4)])
Ya = np.concatenate([mk_xy(Ptr[i])[1] for i in range(4)] + [mk_xy(Pte[i])[1] for i in range(4)])
perm = np.random.RandomState(0).permutation(len(Xa)); ntr = int(0.8 * len(Xa))
Xtr, Ytr = std_ps(Xa[perm[:ntr]]), std_ps(Ya[perm[:ntr]])
Xev, Yev = std_ps(Xa[perm[ntr:]]), std_ps(Ya[perm[ntr:]])
vals = [round(ev(train(Xtr, Ytr, seed=s), Xev, Yev), 4) for s in SEEDS]
out["in_domain_80_20_vrmse"] = {"vals": vals, "mean": round(float(np.mean(vals)), 4)}
print("   in-domain VRMSE:", vals, flush=True)

# ---------------- 2) LOO ----------------
print("[3/4] leave-one-trajectory-out ...", flush=True)
P8 = np.concatenate([Ptr, Pte], 0)
loo = []
for held in range(P8.shape[0]):
    tr = [i for i in range(P8.shape[0]) if i != held]
    Xl = np.concatenate([mk_xy(P8[i])[0] for i in tr]); Yl = np.concatenate([mk_xy(P8[i])[1] for i in tr])
    Xh, Yh = mk_xy(P8[held])
    v = ev(train(std_ps(Xl), std_ps(Yl), seed=0), std_ps(Xh), std_ps(Yh))
    loo.append(round(v, 4))
out["loo_vrmse"] = {"per_fold": loo, "mean": round(float(np.mean(loo)), 4)}
print("   LOO per-fold:", loo, flush=True)

# ---------------- 3)+4) cross-system (optional) ----------------
turb_path = os.path.join(BASE, "turb_rad_layer.hdf5")
if os.path.exists(turb_path):
    import h5py
    print("[4/4] cross-system shear->turb (few-shot) ...", flush=True)
    with h5py.File(turb_path, "r") as f:
        Dt = downsample(f["t0_fields/density"][:])
    Xs, Ys = std_ps(np.concatenate([mk_xy(Ptr[i])[0] for i in range(4)])), \
             std_ps(np.concatenate([mk_xy(Ptr[i])[1] for i in range(4)]))
    Xf, Yf = std_ps(np.concatenate([mk_xy(Dt[i])[0] for i in (0, 1)])[:16]), \
             std_ps(np.concatenate([mk_xy(Dt[i])[1] for i in (0, 1)])[:16])
    Xe, Ye = std_ps(np.concatenate([mk_xy(Dt[i])[0] for i in (2, 3)])[:32]), \
             std_ps(np.concatenate([mk_xy(Dt[i])[1] for i in (2, 3)])[:32])
    zs, fs = [], []
    for s in SEEDS:
        m = train(Xs, Ys, steps=100, seed=s)
        zs.append(round(ev(m, Xe, Ye), 4))
        m2 = train(Xs, Ys, steps=100, seed=s)
        finetune(m2, Xf, Yf, steps=60, seed=s)   # full-model few-shot adaptation
        fs.append(round(ev(m2, Xe, Ye), 4))
    out["turb_zero_shot_vrmse"] = {"vals": zs, "mean": round(float(np.mean(zs)), 4)}
    out["turb_fewshot_adapted_vrmse"] = {"vals": fs, "mean": round(float(np.mean(fs)), 4)}
    print("   zero-shot:", zs, " few-shot adapted:", fs, flush=True)
else:
    print("[4/4] turb_rad_layer.hdf5 not found -- skipping cross-system check.", flush=True)

# ---------------- save ----------------
dst = os.path.join(BASE, "reproduce_minimal_results.json")
json.dump(out, open(dst, "w"), indent=2)
print("\n=== REPRODUCTION SUMMARY ===", flush=True)
print(json.dumps(out, indent=2), flush=True)
print("saved ->", dst, flush=True)
