# -*- coding: utf-8 -*-
"""M3/E5 零样本跨系统泛化实验 v2 — 用 fsspec HTTP 补丁读取 SDSC 数据。
"""
import io, os, sys, yaml, traceback, json
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(4)
np.random.seed(0)

# ---------- patch HTTPFileSystem for WellDataset ----------
import fsspec
from fsspec.implementations.http import HTTPFileSystem
from the_well.utils.download import WELL_REGISTRY

reg = yaml.safe_load(io.open(WELL_REGISTRY, encoding="utf-8"))
known = []
for ds in ("turbulent_radiative_layer_2D",):
    for split in ("train", "valid", "test"):
        known.extend(reg[ds][split])
known_norm = [u.replace("\\", "/") for u in known]

orig_glob = HTTPFileSystem.glob

def _norm(p):
    return str(p).replace("\\", "/")

def patched_glob(self, path, *a, **k):
    p = _norm(path)
    if p.endswith("/*.h5") or p.endswith("/*.hdf5") or p.endswith("*.h5") or p.endswith("*.hdf5"):
        prefix = p.rsplit("/", 1)[0]
        pat = p.rsplit("/", 1)[1].lstrip("*").lstrip(".")
        res = [u for u in known_norm if u.startswith(prefix) and u.endswith("." + pat)]
        return res
    return orig_glob(self, p, *a, **k)

HTTPFileSystem.glob = patched_glob

# ---------- data: shear_flow ----------
sf = np.load(r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\data\slice_train.npz")
T_sf = sf["tracer"].astype(np.float32)  # (4,20,64,64)

def make_xy(T, n_in=4):
    n_traj, n_steps, H, W = T.shape
    X, Y = [], []
    for t in range(n_steps - n_in):
        X.append(T[:, t:t+n_in])
        Y.append(T[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

X_sf, Y_sf = make_xy(T_sf)
print(f"shear_flow: X={X_sf.shape}, Y={Y_sf.shape}")

# ---------- data: turbulent_radiative_layer_2D ----------
print("\nLoading turbulent_radiative_layer_2D from SDSC...")
from the_well.data import WellDataset
base = "https://sdsc-users.flatironinstitute.org/~polymathic/data/the_well/datasets"
ds = WellDataset(
    well_base_path=base,
    well_dataset_name="turbulent_radiative_layer_2D",
    well_split_name="train",
    n_steps_input=4,
    n_steps_output=1,
    cache_small=False,
)
print(f"  n_files={ds.n_files}, n_samples={len(ds)}")

# Extract density field from all samples
densities = []
for i in range(len(ds)):
    s = ds[i]
    d = s.get('density')
    if d is not None:
        densities.append(np.array(d))
D = np.array(densities)  # (N, n_in+1, H, W)
print(f"  Density raw shape: {D.shape}")

# Build input/output pairs: (N, n_in+1, H, W) -> (N, n_in, H, W), (N, 1, H, W)
X_tr = D[:, :-1]  # (N, n_in, H, W)
Y_tr = D[:, -1:]  # (N, 1, H, W)
print(f"turbulent_radiative: X={X_tr.shape}, Y={Y_tr.shape}")

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

def train_model(model, X_tr, Y_tr, X_te, Y_te, steps=40, lr=1e-3, seed=0):
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

# 1. shear_flow in-domain
print("\n=== 1. shear_flow in-domain ===")
n = X_sf.shape[0]
perm = np.random.RandomState(0).permutation(n)
ntr = int(0.8 * n)
model_sf = MoE_lite(use_desc=True)
sf_err = train_model(model_sf, X_sf[perm[:ntr]], Y_sf[perm[:ntr]],
                     X_sf[perm[ntr:]], Y_sf[perm[ntr:]], steps=60, lr=1e-3, seed=0)
results["shear_flow_in_domain"] = sf_err
print(f"  error={sf_err:.6f}")

# 2. Zero-shot
print("\n=== 2. Zero-shot (shear_flow -> turbulent_radiative) ===")
model_sf.eval()
with torch.no_grad():
    X_t = torch.tensor(X_tr).float()
    Y_t = torch.tensor(Y_tr).float()
    pred = model_sf(X_t)
    mse = ((pred - Y_t) ** 2).mean().item()
    y_var = Y_t.var().item()
    vrmse = np.sqrt(mse / y_var) if y_var > 0 else float('inf')
    rel_l2 = np.sqrt(mse) / (np.sqrt(((Y_t) ** 2).mean().item()) + 1e-8)
results["zero_shot"] = {"mse": mse, "vrmse": vrmse, "rel_l2": rel_l2}
print(f"  MSE={mse:.6f}, VRMSE={vrmse:.6f}, rel_L2={rel_l2:.6f}")

# 3. Scratch
print("\n=== 3. Scratch ===")
n_tr = X_tr.shape[0]
perm2 = np.random.RandomState(0).permutation(n_tr)
ntr2 = int(0.8 * n_tr)
model_scratch = MoE_lite(use_desc=True)
scratch_err = train_model(model_scratch, X_tr[perm2[:ntr2]], Y_tr[perm2[:ntr2]],
                          X_tr[perm2[ntr2:]], Y_tr[perm2[ntr2:]], steps=60, lr=1e-3, seed=0)
results["scratch"] = scratch_err
print(f"  error={scratch_err:.6f}")

# 4. Adapted zero-shot
print("\n=== 4. Adapted zero-shot ===")
model_adapt = MoE_lite(use_desc=True)
model_adapt.load_state_dict(model_sf.state_dict())
with torch.no_grad():
    model_adapt.desc.fill_(0.5)
model_adapt.eval()
with torch.no_grad():
    pred = model_adapt(X_t)
    mse = ((pred - Y_t) ** 2).mean().item()
    y_var = Y_t.var().item()
    vrmse = np.sqrt(mse / y_var) if y_var > 0 else float('inf')
    rel_l2 = np.sqrt(mse) / (np.sqrt(((Y_t) ** 2).mean().item()) + 1e-8)
results["zero_shot_adapted"] = {"mse": mse, "vrmse": vrmse, "rel_l2": rel_l2}
print(f"  MSE={mse:.6f}, VRMSE={vrmse:.6f}, rel_L2={rel_l2:.6f}")

# Save
out = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\zero_shot_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out}")
print(json.dumps(results, indent=2))