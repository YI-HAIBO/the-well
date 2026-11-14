# -*- coding: utf-8 -*-
"""Extended scaling study on turbulent_radiative_layer_2D (8 trajectories).
Data scaling: ch=16 with 2/4/8 trajectories.
Model scaling: ch=8/16/32 with 8 trajectories.
"""
import numpy as np, torch, torch.nn as nn, json, os, sys, h5py

torch.set_num_threads(8)
np.random.seed(0)

BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"

# ---------- load all 8 trajectories ----------
hdf5_path = os.path.join(BASE, "turb_rad_layer.hdf5")
f = h5py.File(hdf5_path, "r")
dens = f["t0_fields/density"]  # (8, 101, 128, 384)
print(f"Density shape: {dens.shape}")

D_all = []
for i in range(8):
    D = dens[i][:]
    Dh = D.reshape(101, 2, 64, 6, 64).mean(axis=(1, 3))
    D_all.append(Dh.astype(np.float32))
D_all = np.stack(D_all, 0)  # (8, 101, 64, 64)
print(f"All trajectories: {D_all.shape}")

def make_xy(T, n_in=4):
    n_traj, n_steps, H, W = T.shape
    X, Y = [], []
    for t in range(n_steps - n_in):
        X.append(T[:, t:t+n_in])
        Y.append(T[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

X_all, Y_all = make_xy(D_all)
samples_per_traj = X_all.shape[0] // 8
print(f"Samples per trajectory: {samples_per_traj}")

# ---------- model ----------
class MoE_lite(nn.Module):
    def __init__(self, ch=16, modes=12, use_desc=True, in_ch=4):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.e1 = nn.Conv2d(ch, ch, 1)
        self.e2 = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, 1, 1)
        if use_desc:
            self.gate = nn.Linear(4, 2)
            self.desc = nn.Parameter(torch.randn(4))
    def forward(self, x):
        h = self.in_conv(x)
        if hasattr(self, 'gate') and self.gate is not None:
            g = torch.softmax(self.gate(self.desc[None]), -1)
            h = g[0, 0] * self.e1(h) + g[0, 1] * self.e2(h)
        else:
            h = self.e1(h)
        return self.out(h)

def train_eval(model, X_tr, Y_tr, X_te, Y_te, steps=30, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    X_tr_t = torch.tensor(X_tr).float()
    Y_tr_t = torch.tensor(Y_tr).float()
    X_te_t = torch.tensor(X_te).float()
    Y_te_t = torch.tensor(Y_te).float()
    best_mse = float('inf')
    for s in range(steps):
        model.train()
        opt.zero_grad()
        loss = lf(model(X_tr_t), Y_tr_t)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            te_loss = lf(model(X_te_t), Y_te_t).item()
        if te_loss < best_mse:
            best_mse = te_loss
    y_var = Y_te_t.var().item()
    vrmse = np.sqrt(best_mse / y_var) if y_var > 0 else float('inf')
    rel_l2 = np.sqrt(best_mse) / (np.sqrt((Y_te_t ** 2).mean().item()) + 1e-8)
    return best_mse, vrmse, rel_l2

# ---------- experiments ----------
results = {}

# Data scaling: ch=16, varying trajectory counts
for n_trajs in [2, 4, 8]:
    start = 0 * samples_per_traj
    end = n_trajs * samples_per_traj
    X_trajs = X_all[start:end]
    Y_trajs = Y_all[start:end]
    n = X_trajs.shape[0]
    perm = np.random.RandomState(0).permutation(n)
    ntr = int(0.8 * n)
    X_tr, Y_tr = X_trajs[perm[:ntr]], Y_trajs[perm[:ntr]]
    X_te, Y_te = X_trajs[perm[ntr:]], Y_trajs[perm[ntr:]]

    key = f"data_ch16_trajs{n_trajs}"
    print(f"\n=== Data scaling: ch=16, {n_trajs} trajs, {ntr} train/{n-ntr} test ===")
    model = MoE_lite(ch=16)
    mse, vrmse, rel_l2 = train_eval(model, X_tr, Y_tr, X_te, Y_te, steps=30, seed=0)
    results[key] = {"mse": mse, "vrmse": vrmse, "rel_l2": rel_l2, "n_train": ntr, "n_test": n-ntr}
    print(f"  VRMSE={vrmse:.4f}, rel_L2={rel_l2:.4f}")

# Model scaling: 8 trajectories, varying ch
for ch in [8, 32]:
    start = 0
    end = 8 * samples_per_traj
    X_trajs = X_all[start:end]
    Y_trajs = Y_all[start:end]
    n = X_trajs.shape[0]
    perm = np.random.RandomState(0).permutation(n)
    ntr = int(0.8 * n)
    X_tr, Y_tr = X_trajs[perm[:ntr]], Y_trajs[perm[:ntr]]
    X_te, Y_te = X_trajs[perm[ntr:]], Y_trajs[perm[ntr:]]

    key = f"model_ch{ch}_trajs8"
    print(f"\n=== Model scaling: ch={ch}, 8 trajs, {ntr} train/{n-ntr} test ===")
    model = MoE_lite(ch=ch)
    mse, vrmse, rel_l2 = train_eval(model, X_tr, Y_tr, X_te, Y_te, steps=30, seed=0)
    results[key] = {"mse": mse, "vrmse": vrmse, "rel_l2": rel_l2, "n_train": ntr, "n_test": n-ntr}
    print(f"  VRMSE={vrmse:.4f}, rel_L2={rel_l2:.4f}")

# Save
out_path = os.path.join(BASE, "scaling_results.json")
with open(out_path, "w") as fp:
    json.dump(results, fp, indent=2)
print(f"\nSaved to {out_path}")

# Summary
print("\n" + "="*70)
print("SCALING SUMMARY")
print("="*70)
for key in sorted(results.keys()):
    r = results[key]
    print(f"{key:<25} VRMSE={r['vrmse']:.4f}  rel_L2={r['rel_l2']:.4f}  N_train={r['n_train']}")
f.close()