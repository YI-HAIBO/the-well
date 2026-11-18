# -*- coding: utf-8 -*-
"""Attack round 2 (part A): E3 + E10 on local official The-Well shear data.

E3  Leave-one-out normalization decomposition:
    same LOO protocol, four conditions
      A  per-sample norm, consistent train/eval   (paper protocol, clean)
      B  train-stat norm, consistent train/eval   (alternative clean)
      C  per-sample train, RAW-space eval         (reproduces the artifact mechanism)
      D  in-domain 80/20 per-sample norm          (reference)
    -> decomposes the LOO degradation into normalization artifact vs genuine
       trajectory extrapolation.

E10 Data scaling on official slices:
    fixed held-out eval trajectories; train on k in {1,2,4,6} of the remaining
    six shear trajectories (3 random subsets each), plus temporal-coverage
    scaling f in {0.25, 0.5, 1.0} at k=6.

Protocol: MoE_lite, 64x64 block-averaged pressure, per-sample-std VRMSE,
mean baseline = 1.0.  torch CPU, 8 threads.  Results -> attack2a_results.json
"""
import numpy as np, torch, torch.nn as nn, os, json, time

torch.set_num_threads(8)
BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
RES = os.path.join(BASE, "attack2a_results.json")

# ---------------- data ----------------
def downsample(T, H=64, W=64):
    oH, oW = T.shape[-2], T.shape[-1]
    fH, fW = oH // H, oW // W
    T = T.reshape(T.shape[0], T.shape[1], fH, H, fW, W)
    return T.mean(axis=(2, 4)).astype(np.float32)

ztr = np.load(os.path.join(BASE, "data", "slice_train.npz"))
zte = np.load(os.path.join(BASE, "data", "slice_test.npz"))
P8 = np.concatenate([downsample(ztr["pressure"]), downsample(zte["pressure"])], 0)  # (8,20,64,64)
N_TRAJ, N_STEPS = P8.shape[0], P8.shape[1]
print("P8:", P8.shape, flush=True)

def mk_xy(T, n_in=4):
    X, Y = [], []
    for t in range(T.shape[0] - n_in):
        X.append(T[t:t+n_in]); Y.append(T[t+n_in:t+n_in+1])
    return np.stack(X), np.stack(Y)  # (n,4,H,W),(n,1,H,W)

def std_ps(A):  # per-sample standardization over (H,W)
    m = A.mean(axis=(-2, -1), keepdims=True); s = A.std(axis=(-2, -1), keepdims=True) + 1e-6
    return (A - m) / s

def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float("inf")

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

results = {"protocol": "MoE_lite ch16 per-sample-std VRMSE, 64x64 shear pressure, 8 official trajectories",
           "E3": {}, "E10": {}}

# ============================================================
# E3: LOO x normalization decomposition
# ============================================================
print("\n=== E3: LOO normalization decomposition ===", flush=True)
t0 = time.time()
N_SEED = 3
conds = ["A_persample_consistent", "B_trainstat_consistent", "C_mixed_raw_eval"]
e3 = {c: [] for c in conds}

for held in range(N_TRAJ):
    train_trajs = [i for i in range(N_TRAJ) if i != held]
    Xtr_raw = np.concatenate([mk_xy(P8[i])[0] for i in train_trajs])
    Ytr_raw = np.concatenate([mk_xy(P8[i])[1] for i in train_trajs])
    Xev_raw, Yev_raw = mk_xy(P8[held])

    # condition A: per-sample norm everywhere
    XtrA, YtrA, XevA, YevA = std_ps(Xtr_raw), std_ps(Ytr_raw), std_ps(Xev_raw), std_ps(Yev_raw)
    # condition B: train-stat norm everywhere
    gm, gs = float(Xtr_raw.mean()), float(Xtr_raw.std() + 1e-6)
    ym, ys = float(Ytr_raw.mean()), float(Ytr_raw.std() + 1e-6)
    XtrB = (Xtr_raw - gm) / gs; YtrB = (Ytr_raw - ym) / ys
    XevB = (Xev_raw - gm) / gs; YevB = (Yev_raw - ym) / ys
    # condition C: per-sample train, RAW eval (artifact mechanism)
    for cond, (Xtr, Ytr, Xev, Yev) in [
        ("A_persample_consistent", (XtrA, YtrA, XevA, YevA)),
        ("B_trainstat_consistent", (XtrB, YtrB, XevB, YevB)),
        ("C_mixed_raw_eval",       (XtrA, YtrA, XevA, Yev_raw)),
    ]:
        vals = []
        for seed in range(N_SEED):
            m = train_run(Xtr, Ytr, steps=60, seed=seed)
            vals.append(round(eval_m(m, Xev, Yev), 4))
        e3[cond].append({"held": held, "vals": vals, "mean": round(float(np.mean(vals)), 4)})
        print("  held=%d %s mean=%.4f %s" % (held, cond, np.mean(vals), vals), flush=True)

# condition D: in-domain 80/20 (all windows, random split)
Xall = np.concatenate([mk_xy(P8[i])[0] for i in range(N_TRAJ)])
Yall = np.concatenate([mk_xy(P8[i])[1] for i in range(N_TRAJ)])
perm = np.random.RandomState(0).permutation(len(Xall))
ntr = int(0.8 * len(Xall))
XtrD, YtrD = std_ps(Xall[perm[:ntr]]), std_ps(Yall[perm[:ntr]])
XevD, YevD = std_ps(Xall[perm[ntr:]]), std_ps(Yall[perm[ntr:]])
valsD = []
for seed in range(N_SEED):
    m = train_run(XtrD, YtrD, steps=60, seed=seed)
    valsD.append(round(eval_m(m, XevD, YevD), 4))
e3["D_indomain_80_20"] = {"vals": valsD, "mean": round(float(np.mean(valsD)), 4)}
print("  in-domain 80/20: mean=%.4f %s" % (np.mean(valsD), valsD), flush=True)

for c in conds:
    means = [r["mean"] for r in e3[c]]
    e3[c + "_avg"] = round(float(np.mean(means)), 4)
print("E3 summary:", {c: e3.get(c + "_avg") for c in conds}, "D=", e3["D_indomain_80_20"]["mean"],
      " (%.0fs)" % (time.time() - t0), flush=True)
results["E3"] = e3
json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)

# ============================================================
# E10: data scaling on official trajectories
# ============================================================
print("\n=== E10: trajectory-count scaling (fixed eval trajs 6,7) ===", flush=True)
t0 = time.time()
POOL = list(range(6))          # training pool: trajectories 0..5
EVAL = [6, 7]                  # fixed eval
Xev10 = np.concatenate([mk_xy(P8[i])[0] for i in EVAL])
Yev10 = np.concatenate([mk_xy(P8[i])[1] for i in EVAL])
Xev10s, Yev10s = std_ps(Xev10), std_ps(Yev10)

rng = np.random.RandomState(42)
e10 = {}
for k in [1, 2, 4, 6]:
    runs = []
    for rep in range(3):
        sel = sorted(rng.choice(POOL, size=k, replace=False).tolist())
        Xtr = np.concatenate([mk_xy(P8[i])[0] for i in sel])
        Ytr = np.concatenate([mk_xy(P8[i])[1] for i in sel])
        m = train_run(std_ps(Xtr), std_ps(Ytr), steps=60, seed=100 + rep)
        v = eval_m(m, Xev10s, Yev10s)
        runs.append({"sel": sel, "vrmse": round(v, 4)})
        print("  k=%d sel=%s -> %.4f" % (k, sel, v), flush=True)
    e10["traj_k%d" % k] = {"runs": runs, "mean": round(float(np.mean([r["vrmse"] for r in runs])), 4)}

print("\n=== E10: temporal-coverage scaling at k=6 ===", flush=True)
Xtr6 = np.concatenate([mk_xy(P8[i])[0] for i in POOL])
Ytr6 = np.concatenate([mk_xy(P8[i])[1] for i in POOL])
n_tot = len(Xtr6)
for f in [0.25, 0.5, 1.0]:
    runs = []
    for rep in range(3):
        n_use = max(8, int(n_tot * f))
        idx = np.random.RandomState(200 + rep).choice(n_tot, size=n_use, replace=False)
        m = train_run(std_ps(Xtr6[idx]), std_ps(Ytr6[idx]), steps=60, seed=300 + rep)
        v = eval_m(m, Xev10s, Yev10s)
        runs.append(round(v, 4))
        print("  f=%.2f n=%d -> %.4f" % (f, n_use, v), flush=True)
    e10["frac_%.2f" % f] = {"runs": runs, "mean": round(float(np.mean(runs)), 4),
                            "n_train": int(n_tot * f)}

print("E10 summary:", {k: v.get("mean") for k, v in e10.items()}, " (%.0fs)" % (time.time() - t0), flush=True)
results["E10"] = e10
json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)
print("\n=== PART A DONE ->", RES, "===", flush=True)
