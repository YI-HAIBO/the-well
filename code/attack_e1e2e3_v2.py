# -*- coding: utf-8 -*-
"""V2: incremental-save version. Reuses paper protocol exactly."""
import numpy as np, torch, torch.nn as nn, os, json, copy, time

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
BASE = "/data/home/well_b3"
RES = os.path.join(BASE, "attack_e1e2e3_results.json")

z = np.load(os.path.join(BASE, "data64.npz"))
DATA = {"shear": z["shear"].astype(np.float32), "turb": z["turb"].astype(np.float32)}

def mk_xy(T, n_in=4):
    X, Y = [], []
    for t in range(T.shape[1] - n_in):
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

results = {"protocol": "MoE_lite per-sample-std VRMSE=sqrt(MSE/y.var) mean-baseline=1.0",
           "data": "shear(4x20) turb(8x20) 64x64"}
def save():
    json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)

# E1 (confirmed from v1 run)
print("=== E1 few-shot shear->turb ===", flush=True)
torch.manual_seed(0); m_pre = train_full(MoE_lite(), *XY["shear"], steps=60, lr=1e-3, seed=0)
zs = eval_v(m_pre, *XY["turb"])
e1 = {"zero_shot": round(zs, 4), "few_shot": {}}
print("  zero-shot = %.4f" % zs, flush=True)
for k in [1, 5, 10, 20, 50]:
    m = train_full(copy.deepcopy(m_pre), *XY["turb"], steps=k, lr=1e-3, seed=7)
    v = eval_v(m, *XY["turb"]); e1["few_shot"]["steps_%d"%k] = round(v, 4)
    print("    +%d -> %.4f" % (k, v), flush=True)
results["E1"] = e1; save()

# E2: ch x steps (60/300), incremental save
print("=== E2 capacity x length turb ===", flush=True)
e2 = {}; results["E2"] = e2
for ch in [16, 64, 128]:
    for steps in [60, 300]:
        t0=time.time()
        m = train_full(MoE_lite(ch=ch), *XY["turb"], steps=steps, lr=1e-3, seed=0)
        v = eval_v(m, *XY["turb"])
        e2["ch%d_s%d"%(ch,steps)] = round(v, 4); save()
        print("    ch=%d s=%d -> %.4f (%.1fs)" % (ch, steps, v, time.time()-t0), flush=True)

# E3: LOO augmentation
print("=== E3 LOO augmentation ===", flush=True)
T = DATA["shear"]
def make_loo(T, idx, augment=False):
    tr = np.delete(T, idx, 0); te = T[idx:idx+1]
    if augment:
        rng = np.random.RandomState(0); ex=[]
        for a in tr:
            ex.append(a)
            for sh in [1,2,3]: ex.append(np.roll(a, sh, axis=0))
            for _ in range(3): ex.append(a + rng.normal(0,0.02,a.shape).astype(np.float32))
        tr = np.concatenate(ex, 0)
    Xtr,Ytr = (std(x) for x in mk_xy(tr)); Xte,Yte = (std(x) for x in mk_xy(te))
    return Xtr.astype(np.float32),Ytr.astype(np.float32),Xte.astype(np.float32),Yte.astype(np.float32)
e3 = {}; results["E3"] = e3
for aug in [False, True]:
    vals=[]
    for i in range(T.shape[0]):
        a = make_loo(T, i, augment=aug)
        m = train_full(MoE_lite(), a[0], a[1], steps=60, lr=1e-3, seed=0)
        vals.append(eval_v(m, a[2], a[3]))
    tag = "aug" if aug else "base"
    e3[tag] = {"per_traj":[round(v,4) for v in vals], "mean":round(float(np.mean(vals)),4)}; save()
    print("    LOO %s mean=%.4f per=%s" % (tag, np.mean(vals), [round(v,3) for v in vals]), flush=True)

save()
print("RESULT=" + json.dumps(results, separators=(",", ":")), flush=True)
