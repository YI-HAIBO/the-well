# -*- coding: utf-8 -*-
"""E10 FULL-DATA re-test on the complete official shear train split
(32 trajectories, 13.44 GB), replacing the 8-trajectory slice used in
attack2a_e3_e10.py.  This closes the "full official data not used" gap.

Protocol identical to attack2a_e3_e10.py (MoE_lite ch16, 64x64 block-averaged
pressure, per-sample-std VRMSE, mean baseline 1.0), scaled up:

  - trajectory-count scaling: train on k in {1,2,4,8,16,28} of trajectories
    0..27 (3 random subsets each), fixed eval on trajectories 28..31
  - temporal-coverage scaling at k=28: f in {0.25,0.5,1.0}

Reads the full train HDF5 trajectory-by-trajectory (each ~420 MB raw), so
peak memory stays ~1-2 GB.  Results -> attack2a_full_results.json
"""
import numpy as np, torch, torch.nn as nn, os, json, time, h5py

torch.set_num_threads(8)
BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
FULL = os.path.join(BASE, "data", "datasets", "shear_flow", "data", "train",
                    "shear_flow_Reynolds_1e5_Schmidt_2e0.hdf5")
RES = os.path.join(BASE, "attack2a_full_results.json")

# ---------------- data ----------------
def downsample(T, H=64, W=64):
    oH, oW = T.shape[-2], T.shape[-1]
    fH, fW = oH // H, oW // W
    T = T.reshape(T.shape[0], T.shape[1], fH, H, fW, W)
    return T.mean(axis=(2, 4)).astype(np.float32)

t0 = time.time()
print("loading full train hdf5 ...", flush=True)
with h5py.File(FULL, "r") as f:
    print("top keys:", list(f.keys()), flush=True)
    # The Well layout: f['pressure'] etc. each (n_traj, n_steps, H, W)
    press = f["pressure"]
    N_TRAJ = press.shape[0]
    print("pressure shape:", press.shape, press.dtype, flush=True)
    P64 = np.stack([downsample(np.array(press[i], dtype=np.float32)) for i in range(N_TRAJ)])
print("loaded P64:", P64.shape, "(%.0fs)" % (time.time() - t0), flush=True)
N_STEPS = P64.shape[1]

def mk_xy(T, n_in=4):
    X, Y = [], []
    for t in range(T.shape[0] - n_in):
        X.append(T[t:t+n_in]); Y.append(T[t+n_in:t+n_in+1])
    return np.stack(X), np.stack(Y)

def std_ps(A):
    m = A.mean(axis=(-2, -1), keepdims=True); s = A.std(axis=(-2, -1), keepdims=True) + 1e-6
    return (A - m) / s

def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float("inf")

# ---------------- model (identical to attack2a) ----------------
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

def train_run(X, Y, steps=60, lr=1e-3, seed=0, ch=16):
    torch.manual_seed(seed)
    m = MoE_lite(ch=ch)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    lf = nn.MSELoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(steps):
        opt.zero_grad(); lf(m(Xt), Yt).backward(); opt.step()
    return m

def eval_m(m, X, Y):
    m.eval()
    with torch.no_grad():
        return vrmse(m(torch.tensor(X)).numpy(), Y)

results = {"protocol": "MoE_lite ch16 per-sample-std VRMSE, 64x64 shear pressure, FULL 32 official train trajectories",
           "n_traj_total": int(N_TRAJ), "n_steps": int(N_STEPS), "E10_full": {}}
e10 = {}

# ---------------- trajectory-count scaling ----------------
print("\n=== E10-full: trajectory-count scaling (fixed eval trajs 28..31) ===", flush=True)
POOL = list(range(N_TRAJ - 4))       # 0..27 training pool
EVAL = list(range(N_TRAJ - 4, N_TRAJ))  # 28..31 fixed eval
Xev = np.concatenate([mk_xy(P64[i])[0] for i in EVAL])
Yev = np.concatenate([mk_xy(P64[i])[1] for i in EVAL])
Xevs, Yevs = std_ps(Xev), std_ps(Yev)
del Xev, Yev

rng = np.random.RandomState(42)
for k in [1, 2, 4, 8, 16, len(POOL)]:
    runs = []
    for rep in range(3):
        sel = sorted(rng.choice(POOL, size=k, replace=False).tolist())
        Xtr = np.concatenate([mk_xy(P64[i])[0] for i in sel])
        Ytr = np.concatenate([mk_xy(P64[i])[1] for i in sel])
        m = train_run(std_ps(Xtr), std_ps(Ytr), steps=60, seed=100 + rep)
        v = eval_m(m, Xevs, Yevs)
        runs.append({"sel": sel, "vrmse": round(v, 4)})
        print("  k=%d sel=%s -> %.4f" % (k, sel if k <= 8 else sel[:8] + ['...'], v), flush=True)
        del Xtr, Ytr, m
    e10["traj_k%d" % k] = {"runs": runs, "mean": round(float(np.mean([r["vrmse"] for r in runs])), 4)}
    print("  ==> traj_k%d mean=%.4f" % (k, e10["traj_k%d" % k]["mean"]), flush=True)

# ---------------- temporal-coverage scaling at k=28 ----------------
print("\n=== E10-full: temporal-coverage scaling at k=%d ===" % len(POOL), flush=True)
Xtr_all = np.concatenate([mk_xy(P64[i])[0] for i in POOL])
Ytr_all = np.concatenate([mk_xy(P64[i])[1] for i in POOL])
n_tot = len(Xtr_all)
for f in [0.25, 0.5, 1.0]:
    runs = []
    for rep in range(3):
        n_use = max(8, int(n_tot * f))
        idx = np.random.RandomState(200 + rep).choice(n_tot, size=n_use, replace=False)
        m = train_run(std_ps(Xtr_all[idx]), std_ps(Ytr_all[idx]), steps=60, seed=300 + rep)
        v = eval_m(m, Xevs, Yevs)
        runs.append(round(v, 4))
        print("  f=%.2f n=%d -> %.4f" % (f, n_use, v), flush=True)
        del m
    e10["frac_%.2f" % f] = {"runs": runs, "mean": round(float(np.mean(runs)), 4),
                            "n_train": int(n_tot * f)}

print("E10-full summary:", {k: v.get("mean") for k, v in e10.items()}, flush=True)
results["E10_full"] = e10
json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)
print("\n=== FULL-DATA E10 DONE ->", RES, "===", flush=True)
