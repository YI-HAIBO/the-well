# -*- coding: utf-8 -*-
"""C2 k=5 run: train the joint MoE on all five systems; merge with the (deterministic,
seed=0) k=2,3,4 results already measured and save c2_results.json."""
import numpy as np, torch, torch.nn as nn, os, json

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
BASE = os.path.expanduser("~/well_b3")

def load(name):
    if name == "shear": return np.load(os.path.join(BASE, "data64.npz"))["shear"]
    if name == "turb":  return np.load(os.path.join(BASE, "data64.npz"))["turb"]
    if name == "gray":  return np.load(os.path.join(BASE, "gray64.npz"))["gray"]
    if name == "euler": return np.load(os.path.join(BASE, "euler64.npz"))["euler"]
    if name == "visco": return np.load(os.path.join(BASE, "visco64.npz"))["visco"]

SYSTEMS = ["shear", "turb", "gray", "euler", "visco"]

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

def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float('inf')

DAT = {}
for s in SYSTEMS:
    X, Y = (std(a) for a in mk_xy(load(s).astype(np.float32)))
    n = len(X)
    perm = np.random.RandomState(0).permutation(n)
    ntr = int(0.8 * n)
    DAT[s] = (X[perm[:ntr]], Y[perm[:ntr]], X[perm[ntr:]], Y[perm[ntr:]])

class JointMoE(nn.Module):
    def __init__(self, ch=16, in_ch=4, n_experts=2, desc_dim=8, n_sys=5):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 1), nn.GELU(), nn.Conv2d(ch, ch, 1)) for _ in range(n_experts)
        ])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(desc_dim, n_experts)
        self.desc = nn.Parameter(torch.randn(n_sys, desc_dim))
    def forward(self, x, desc_idx):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[desc_idx]), -1)
        eh = torch.stack([e(h) for e in self.experts], 1)
        w = g[:, :, None, None, None]
        return self.out((eh * w).sum(1))

def train_k(systems, steps=40, lr=1e-3, seed=0):
    idx = {s: i for i, s in enumerate(systems)}
    torch.manual_seed(seed)
    m = JointMoE(n_sys=len(systems))
    Xs, Ys, Ds = [], [], []
    for s in systems:
        X, Y, _, _ = DAT[s]
        Xs.append(X); Ys.append(Y); Ds.append(np.full(len(X), idx[s], dtype=np.int64))
    Xc = torch.tensor(np.concatenate(Xs, 0)); Yc = torch.tensor(np.concatenate(Ys, 0))
    Dc = torch.tensor(np.concatenate(Ds, 0))
    opt = torch.optim.AdamW(m.parameters(), lr=lr); lf = nn.MSELoss()
    for st in range(steps):
        opt.zero_grad(); lf(m(Xc, Dc), Yc).backward(); opt.step()
    m.eval()
    res = {}
    with torch.no_grad():
        for s in systems:
            _, _, Xte, Yte = DAT[s]
            di = idx[s]
            pred = m(torch.tensor(Xte), torch.tensor(np.full(len(Xte), di, dtype=np.int64))).numpy()
            res[s] = vrmse(pred, Yte)
    return res

# deterministic k=2,3,4 (seed=0, steps=40) already measured
k234 = {
    2: {"systems": ["shear", "turb"], "mean_vrmse": 0.861,
        "per_system": {"shear": 0.722, "turb": 0.999}},
    3: {"systems": ["shear", "turb", "gray"], "mean_vrmse": 0.738,
        "per_system": {"shear": 0.720, "turb": 0.989, "gray": 0.505}},
    4: {"systems": ["shear", "turb", "gray", "euler"], "mean_vrmse": 0.680,
        "per_system": {"shear": 0.758, "turb": 0.995, "gray": 0.462, "euler": 0.504}},
}
r5 = train_k(SYSTEMS, steps=40)
mean5 = float(np.mean(list(r5.values())))
k234[5] = {"systems": list(SYSTEMS), "mean_vrmse": mean5, "per_system": r5}
print("k=5 %s mean=%.3f %s" % (SYSTEMS, mean5, " ".join("%s=%.3f" % (s, r5[s]) for s in SYSTEMS)), flush=True)

out = os.path.join(BASE, "c2_results.json")
json.dump({"k": k234}, open(out, "w"), indent=2, ensure_ascii=False)
print("Saved ->", out, flush=True)
