# -*- coding: utf-8 -*-
"""M3/E5 零样本跨系统泛化实验（Theorem 2 验证）。
在远程 VM 上运行：
  - 训练集：shear_flow（slice_train.npz，tracer 场，64x64）
  - 测试集：turbulent_radiative_layer_2D（density 场，128x384 → 下采样 64x64）
  - 评估：零样本（不微调）+ 从头训练对比
"""
import numpy as np, torch, torch.nn as nn, json, os, sys

torch.set_num_threads(4)
np.random.seed(0)

# ---------- data ----------
# shear_flow: (4,20,64,64) tracer field
sf = np.load(r"/home/cloud/slice_train.npz")
T_sf = sf["tracer"].astype(np.float32)  # (4,20,64,64)
# Build input/output pairs: 4 steps input -> 1 step output
def make_xy(T, n_in=4):
    # T: (traj, steps, H, W)
    n_traj, n_steps, H, W = T.shape
    X, Y = [], []
    for t in range(n_steps - n_in):
        X.append(T[:, t:t+n_in])  # (n_traj, n_in, H, W)
        Y.append(T[:, t+n_in:t+n_in+1])  # (n_traj, 1, H, W)
    X = np.concatenate(X, 0)  # (N, n_in, H, W)
    Y = np.concatenate(Y, 0)  # (N, 1, H, W)
    return X, Y

X_sf, Y_sf = make_xy(T_sf)  # (76,4,64,64), (76,1,64,64)

# turbulent_radiative_layer_2D: density field from HDF5
import h5py
f = h5py.File("/home/cloud/turb_rad_layer.hdf5", "r")
# Navigate the HDF5 structure to find density field
# The Well format: /data/train/.../density, /data/train/.../pressure, etc.
# Let's find the density dataset
def find_density(group):
    for k in group.keys():
        if "density" in k.lower():
            return group[k]
        if isinstance(group[k], h5py.Group):
            r = find_density(group[k])
            if r is not None:
                return r
    return None

dens = find_density(f)
if dens is None:
    # Try common paths
    for path in ["/data/train/density", "/density", "/train/density"]:
        try:
            dens = f[path]
            break
        except:
            pass

if dens is None:
    print("ERROR: density field not found in HDF5")
    print("Keys:", list(f.keys()))
    for k in f.keys():
        print(f"  {k}: {type(f[k])}")
        if isinstance(f[k], h5py.Group):
            for k2 in f[k].keys():
                print(f"    {k2}: {type(f[k][k2])}")
    sys.exit(1)

print(f"Density shape: {dens.shape}, dtype: {dens.dtype}")
# Typical shape: (n_traj, n_steps, H, W) or (n_steps, H, W)
D = dens[:]
if D.ndim == 4:
    # (n_traj, n_steps, H, W) - take first trajectory
    D = D[0]  # (n_steps, H, W)
elif D.ndim == 3:
    pass  # (n_steps, H, W)
else:
    print(f"Unexpected density shape: {D.shape}")
    sys.exit(1)

print(f"Density after squeeze: {D.shape}, range: [{D.min():.4f}, {D.max():.4f}]")

# Downsample to 64x64
H, W = D.shape[-2:]
if H >= 64 and W >= 64:
    # Simple block average
    Dh = D.reshape(D.shape[0], H//64, 64, W//64, 64).mean(axis=(1,3))
else:
    # Upsample if needed
    import torch.nn.functional as F
    Dh = F.interpolate(torch.tensor(D[None]).float(), size=(64,64), mode='bilinear').numpy()[0]

Dh = Dh.astype(np.float32)  # (n_steps, 64, 64)
print(f"Downsampled: {Dh.shape}")

X_tr, Y_tr = make_xy(Dh[None] if Dh.ndim == 3 else Dh)  # (N,4,64,64), (N,1,64,64)
print(f"Turbulent radiative: X={X_tr.shape}, Y={Y_tr.shape}")

# ---------- model ----------
class MoE_lite(nn.Module):
    def __init__(self, ch=16, modes=12, use_desc=True, in_ch=1):
        super().__init__()
        self.use_desc = use_desc
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.e1 = nn.Conv2d(ch, ch, 1)
        self.e2 = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, 1, 1)
        if use_desc:
            self.gate = nn.Linear(4, 2)
            self.desc = nn.Parameter(torch.randn(4))
        else:
            self.gate = None
    def forward(self, x):
        h = self.in_conv(x)
        if self.use_desc:
            g = torch.softmax(self.gate(self.desc[None]), -1)
            h = g[0, 0] * self.e1(h) + g[0, 1] * self.e2(h)
        else:
            h = self.e1(h)
        return self.out(h)

# ---------- training ----------
def train(model, X_tr, Y_tr, X_te, Y_te, steps=40, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    X_tr_t = torch.tensor(X_tr).float()
    Y_tr_t = torch.tensor(Y_tr).float()
    X_te_t = torch.tensor(X_te).float()
    Y_te_t = torch.tensor(Y_te).float()
    best = float('inf')
    for s in range(steps):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss = lf(pred, Y_tr_t)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            te_pred = model(X_te_t)
            te_loss = lf(te_pred, Y_te_t).item()
        if te_loss < best:
            best = te_loss
    return best

# ---------- experiment ----------
results = {}

# 1. Train on shear_flow, evaluate on shear_flow (in-domain)
print("\n=== Train on shear_flow, test on shear_flow ===")
# Split shear_flow into train/test
n = X_sf.shape[0]
perm = np.random.RandomState(0).permutation(n)
ntr = int(0.8 * n)
X_sf_tr, Y_sf_tr = X_sf[perm[:ntr]], Y_sf[perm[:ntr]]
X_sf_te, Y_sf_te = X_sf[perm[ntr:]], Y_sf[perm[ntr:]]

model_sf = MoE_lite(use_desc=True)
sf_err = train(model_sf, X_sf_tr, Y_sf_tr, X_sf_te, Y_sf_te, steps=60, lr=1e-3, seed=0)
results["shear_flow_in_domain"] = sf_err
print(f"  shear_flow in-domain error: {sf_err:.6f}")

# 2. Zero-shot: model trained on shear_flow, evaluated on turbulent_radiative_layer_2D
print("\n=== Zero-shot: shear_flow model on turbulent_radiative ===")
model_sf.eval()
with torch.no_grad():
    # Normalize turbulent data using shear_flow stats (zero-shot: no adaptation)
    # Simple: just run the model as-is
    X_tr_t = torch.tensor(X_tr).float()
    Y_tr_t = torch.tensor(Y_tr).float()
    pred = model_sf(X_tr_t)
    # Relative L2
    mse = ((pred - Y_tr_t) ** 2).mean().item()
    y_mean = Y_tr_t.mean().item()
    # variance-scaled
    y_var = Y_tr_t.var().item()
    vrmse = np.sqrt(mse / y_var) if y_var > 0 else float('inf')
    rel_l2 = np.sqrt(mse) / (np.sqrt(((Y_tr_t) ** 2).mean().item()) + 1e-8)
results["zero_shot"] = {"mse": mse, "vrmse": vrmse, "rel_l2": rel_l2}
print(f"  Zero-shot MSE={mse:.6f}, VRMSE={vrmse:.6f}, rel_L2={rel_l2:.6f}")

# 3. Scratch model trained on turbulent_radiative_layer_2D
print("\n=== Scratch: train on turbulent_radiative ===")
n_tr = X_tr.shape[0]
perm2 = np.random.RandomState(0).permutation(n_tr)
ntr2 = int(0.8 * n_tr)
X_tr_tr, Y_tr_tr = X_tr[perm2[:ntr2]], Y_tr[perm2[:ntr2]]
X_tr_te, Y_tr_te = X_tr[perm2[ntr2:]], Y_tr[perm2[ntr2:]]

model_scratch = MoE_lite(use_desc=True)
scratch_err = train(model_scratch, X_tr_tr, Y_tr_tr, X_tr_te, Y_tr_te, steps=60, lr=1e-3, seed=0)
results["scratch"] = scratch_err
print(f"  Scratch error: {scratch_err:.6f}")

# 4. Zero-shot with field-description adaptation (desc set to density)
print("\n=== Zero-shot with field-desc adaptation ===")
model_adapt = MoE_lite(use_desc=True)
# Copy weights from shear_flow model
model_adapt.load_state_dict(model_sf.state_dict())
# Update field description for density
with torch.no_grad():
    model_adapt.desc.fill_(0.5)  # neutral desc for new field
model_adapt.eval()
with torch.no_grad():
    X_tr_t = torch.tensor(X_tr).float()
    Y_tr_t = torch.tensor(Y_tr).float()
    pred = model_adapt(X_tr_t)
    mse = ((pred - Y_tr_t) ** 2).mean().item()
    y_var = Y_tr_t.var().item()
    vrmse = np.sqrt(mse / y_var) if y_var > 0 else float('inf')
    rel_l2 = np.sqrt(mse) / (np.sqrt(((Y_tr_t) ** 2).mean().item()) + 1e-8)
results["zero_shot_adapted"] = {"mse": mse, "vrmse": vrmse, "rel_l2": rel_l2}
print(f"  Adapted zero-shot MSE={mse:.6f}, VRMSE={vrmse:.6f}, rel_L2={rel_l2:.6f}")

# Save results
out_path = "/home/cloud/zero_shot_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(json.dumps(results, indent=2))
f.close()