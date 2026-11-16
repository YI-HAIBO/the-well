# -*- coding: utf-8 -*-
"""Targeted experiments attacking E1 / E2 / E3 of The Well JCP manuscript.
Strictly reuses the paper protocol: MoE_lite, per-sample std normalization,
VRMSE = sqrt(MSE / y.var) (mean baseline = 1.0).
Runs on the real The Well 64x64 data in the CPU sandbox.

E1: few-shot GRADIENT adaptation (shear pretrain -> turb 5/10/20/50 steps).
    NOTE: prior 'adapted zero-shot' only set desc to neutral (no gradient).
E2: large capacity + long training sweep on turb (ch=16/64/128 x steps=60/300/1000)
    to test whether turb failure is structural or capacity/compute limited.
E3: trajectory-level augmentation (time shift + noise injection + init jitter)
    to test whether LOO extrapolation gap can be reduced.
"""
import numpy as np, torch, torch.nn as nn, os, json, copy

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
BASE = "/data/home/well_b3"

z = np.load(os.path.join(BASE, "data64.npz"))
DATA = {"shear": z["shear"].astype(np.float32), "turb": z["turb"].astype(np.float32)}

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

XY = {k: tuple(a.astype(np.float32) for a in (std(x) for x in mk_xy(T))) for k, T in DATA.items()}

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

def train_full(model, X, Y, steps, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt_ = torch.tensor(X); Yt_ = torch.tensor(Y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr); lf = nn.MSELoss()
    for s in range(steps):
        opt.zero_grad(); lf(model(Xt_), Yt_).backward(); opt.step()
    return model

def eval_v(model, X, Y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy()
    return vrmse(pred, Y)

results = {"protocol": "MoE_lite, per-sample std, VRMSE=sqrt(MSE/y.var), mean baseline=1.0",
           "data": "The Well shear(4x20) turb(8x20) 64x64"}

# ---------------- E1: few-shot gradient adaptation ----------------
print("=== E1: few-shot gradient adaptation (shear pretrain -> turb) ===", flush=True)
torch.manual_seed(0); m0 = MoE_lite()
m_pre = train_full(m0, *XY["shear"], steps=60, lr=1e-3, seed=0)
zs = eval_v(m_pre, *XY["turb"])
print("  zero-shot shear->turb VRMSE = %.4f" % zs, flush=True)
e1 = {"zero_shot": round(zs, 4), "few_shot": {}}
for k in [1, 5, 10, 20, 50]:
    m = copy.deepcopy(m_pre)
    m = train_full(m, *XY["turb"], steps=k, lr=1e-3, seed=7)
    v = eval_v(m, *XY["turb"])
    e1["few_shot"]["steps_%d" % k] = round(v, 4)
    print("    +%2d grad steps on turb -> VRMSE %.4f" % (k, v), flush=True)
results["E1"] = e1

# ---------------- E2: capacity x training-length on turb ----------------
print("=== E2: capacity x training length on turb (in-domain) ===", flush=True)
e2 = {}
for ch in [16, 64, 128]:
    for steps in [60, 300, 1000]:
        m = MoE_lite(ch=ch)
        m = train_full(m, *XY["turb"], steps=steps, lr=1e-3, seed=0)
        v = eval_v(m, *XY["turb"])
        e2["ch%d_s%d" % (ch, steps)] = round(v, 4)
        print("    ch=%3d steps=%4d -> turb VRMSE %.4f" % (ch, steps, v), flush=True)
results["E2"] = e2

# ---------------- E3: trajectory augmentation + LOO extrapolation ----------------
print("=== E3: trajectory augmentation for LOO extrapolation ===", flush=True)
T = DATA["shear"]
def make_loo(T, t_idx, augment=False, seed=0):
    traj_train = np.delete(T, t_idx, axis=0)
    traj_test = T[t_idx:t_idx+1]
    if augment:
        rng = np.random.RandomState(seed)
        ex = []
        for a in traj_train:
            ex.append(a)
            # time shift (cyclic roll along time axis)
            for sh in [1, 2, 3]:
                ex.append(np.roll(a, sh, axis=0))
            # noise injection
            for _ in range(3):
                ex.append(a + rng.normal(0, 0.02, a.shape).astype(np.float32))
        traj_train = np.concatenate(ex, 0)
    Xtr, Ytr = (std(x) for x in mk_xy(traj_train))
    Xte, Yte = (std(x) for x in mk_xy(traj_test))
    return Xtr.astype(np.float32), Ytr.astype(np.float32), Xte.astype(np.float32), Yte.astype(np.float32)

e3 = {}
for aug in [False, True]:
    vals = []
    for i in range(T.shape[0]):
        Xtr, Ytr, Xte, Yte = make_loo(T, i, augment=aug, seed=0)
        m = train_full(MoE_lite(), Xtr, Ytr, steps=60, lr=1e-3, seed=0)
        vals.append(eval_v(m, Xte, Yte))
    tag = "aug" if aug else "base"
    e3[tag] = {"per_traj": [round(v, 4) for v in vals], "mean": round(float(np.mean(vals)), 4)}
    print("    LOO %s mean VRMSE = %.4f  per=%s" % (tag, np.mean(vals), [round(v,3) for v in vals]), flush=True)
results["E3"] = e3

out = os.path.join(BASE, "attack_e1e2e3_results.json")
json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
print("\nSaved ->", out, flush=True)
print("RESULT=" + json.dumps(results, separators=(",", ":")), flush=True)
