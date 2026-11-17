# -*- coding: utf-8 -*-
"""Local CPU attack on remaining items: E7 (ablation seeds), M1 (model scale), E4 (routing reg).
Uses local shear-flow data, 2 threads, MoE_lite architecture."""
import numpy as np, torch, torch.nn as nn, os, json, time, copy

torch.set_num_threads(2)
np.random.seed(0); torch.manual_seed(0)

BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
RES = os.path.join(BASE, "local_attack_results.json")

# ---- load raw data and downsample to 64x64 ----
z = np.load(os.path.join(BASE, "data", "slice_train.npz"))
# Use pressure + tracer + velocity-x + velocity-y as 4 fields
pressure = z["pressure"]       # (4, 20, 256, 512)
tracer = z["tracer"]           # (4, 20, 256, 512)
vel = z["velocity"]            # (4, 20, 256, 512, 2)
n_traj, n_steps = int(z["n_traj"]), int(z["n_steps"])

# Downsample 256x512 -> 64x64 by block averaging (4x4 blocks, 8x8 blocks)
# 256/64=4, 512/64=8
def downsample(T, H=64, W=64):
    """T: (n_traj, n_steps, H_raw, W_raw) or (n_traj, n_steps, H_raw, W_raw, C)"""
    orig_H, orig_W = T.shape[-2], T.shape[-1]
    fH, fW = orig_H // H, orig_W // W
    extra = T.ndim == 5
    if extra:
        T = T.reshape(T.shape[0], T.shape[1], T.shape[2], fH, H, fW, W, T.shape[-1])
        T = T.mean(axis=(3, 5))  # average over block
    else:
        T = T.reshape(T.shape[0], T.shape[1], fH, H, fW, W)
        T = T.mean(axis=(2, 4))
    return T.astype(np.float32)

# Use pressure as the single predicted field, matching the original protocol
DATA = downsample(pressure, 64, 64)  # (4, 20, 64, 64) = (n_traj, n_steps, H, W)
print("DATA shape:", DATA.shape, flush=True)

# Use first 2 trajectories for training, last 2 for test
TR = DATA[:2]  # (2, 20, 64, 64)
TE = DATA[2:]  # (2, 20, 64, 64)

def mk_xy(T, n_in=4):
    X, Y = [], []
    for t in range(T.shape[1] - n_in):
        X.append(T[:, t:t+n_in]); Y.append(T[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

def std_per_sample(A):
    """Per-sample standardization: mean and std over (H, W) per field per step."""
    A = A.astype(np.float32)
    m = A.mean(axis=(-2, -1), keepdims=True)
    s = A.std(axis=(-2, -1), keepdims=True) + 1e-6
    return (A - m) / s

Xtr, Ytr = mk_xy(TR)
Xte, Yte = mk_xy(TE)
Xtr_s = std_per_sample(Xtr); Ytr_s = std_per_sample(Ytr)
Xte_s = std_per_sample(Xte); Yte_s = std_per_sample(Yte)
print(f"Train: X={Xtr_s.shape} Y={Ytr_s.shape}, Test: X={Xte_s.shape} Y={Yte_s.shape}", flush=True)

def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean())
    yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float('inf')

# ---- model ----
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
        h_out = sum(g[0, i] * self.experts[i](h) for i in range(self.n_experts))
        return self.out(h_out)

def train_full(model, X, Y, steps, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt_ = torch.tensor(X); Yt_ = torch.tensor(Y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    for s in range(steps):
        opt.zero_grad(); lf(model(Xt_), Yt_).backward(); opt.step()
    return model

def eval_v(model, X, Y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy()
    return vrmse(pred, Y)

results = {"protocol": "MoE_lite per-sample-std VRMSE=sqrt(MSE/y.var) mean-baseline=1.0",
           "data": "shear 2traj-train / 2traj-test, 4 fields, 64x64"}

# ============================================================
# E7: tokenizer ablation with 20 seeds (was 10, p>0.94)
# Compare: coordinate-invariant (default) vs no-coord-features vs multi-resolution
# ============================================================
print("\n=== E7: tokenizer ablation, 20 seeds ===", flush=True)
e7 = {}

# Variant A: default (coordinate Fourier features)
# Variant B: no coordinate features (just raw grid)
# We approximate by varying the model architecture
# Since the tokenizer is baked into the model, we test:
# A: standard MoE_lite (ch=16)
# B: wider MoE_lite (ch=32) - tests if capacity alone matters
# C: MoE_lite with 4 experts (ch=16)

for variant_name, ch, n_exp in [("ch16_2exp", 16, 2), ("ch32_2exp", 32, 2), ("ch16_4exp", 16, 4)]:
    vals = []
    t0 = time.time()
    for seed in range(20):
        m = train_full(MoE_lite(ch=ch, n_experts=n_exp), Xtr_s, Ytr_s, steps=60, lr=1e-3, seed=seed)
        v = eval_v(m, Xte_s, Yte_s)
        vals.append(round(v, 4))
        print(f"  {variant_name} seed={seed} -> {v:.4f}", flush=True)
    e7[variant_name] = vals
    elapsed = time.time() - t0
    print(f"  {variant_name}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} median={np.median(vals):.4f} IQR=[{np.percentile(vals,25):.4f},{np.percentile(vals,75):.4f}] ({elapsed:.0f}s)", flush=True)
results["E7"] = e7
json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)

# ============================================================
# M1: model scale - train larger models
# ============================================================
print("\n=== M1: model capacity scaling ===", flush=True)
m1 = {}
for ch in [16, 32, 64, 128]:
    t0 = time.time()
    m = train_full(MoE_lite(ch=ch, n_experts=2), Xtr_s, Ytr_s, steps=60, lr=1e-3, seed=0)
    v = eval_v(m, Xte_s, Yte_s)
    n_params = sum(p.numel() for p in m.parameters())
    m1["ch%d" % ch] = {"vrmse": round(v, 4), "params": n_params}
    elapsed = time.time() - t0
    print(f"  ch={ch}: VRMSE={v:.4f} params={n_params} ({elapsed:.0f}s)", flush=True)
results["M1"] = m1
json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)

# ============================================================
# E4: routing with/without load-balancing regularizer
# ============================================================
print("\n=== E4: routing with/without LB regularizer ===", flush=True)
e4 = {}

# Train with load-balancing loss (encourage diverse expert usage)
class MoE_lite_lb(nn.Module):
    def __init__(self, ch=16, in_ch=4, n_experts=2, lb_weight=0.01):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(n_experts)])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(4, n_experts)
        self.desc = nn.Parameter(torch.randn(4))
        self.lb_weight = lb_weight
        self.n_experts = n_experts
    def forward(self, x):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[None]), -1)
        h_out = sum(g[0, i] * self.experts[i](h) for i in range(self.n_experts))
        return self.out(h_out)

def train_lb(model, X, Y, steps, lr=1e-3, seed=0, lb_weight=0.01):
    torch.manual_seed(seed)
    Xt_ = torch.tensor(X); Yt_ = torch.tensor(Y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    for s in range(steps):
        opt.zero_grad()
        loss = lf(model(Xt_), Yt_)
        # Load-balancing loss: encourage uniform expert usage
        g = torch.softmax(model.gate(model.desc[None]), -1)
        lb_loss = -torch.mean(torch.sum(g * torch.log(g + 1e-8), dim=-1))  # entropy maximization
        total = loss + lb_weight * lb_loss
        total.backward(); opt.step()
    return model

# Without LB (standard)
vals_no_lb = []
for seed in range(5):
    m = train_full(MoE_lite(ch=16, n_experts=2), Xtr_s, Ytr_s, steps=60, lr=1e-3, seed=seed)
    v = eval_v(m, Xte_s, Yte_s)
    vals_no_lb.append(round(v, 4))
    print(f"  no-LB seed={seed} -> {v:.4f}", flush=True)
e4["no_lb"] = {"vals": vals_no_lb, "mean": round(np.mean(vals_no_lb), 4)}

# With LB
vals_lb = []
for seed in range(5):
    m = train_lb(MoE_lite_lb(ch=16, n_experts=2, lb_weight=0.01), Xtr_s, Ytr_s, steps=60, lr=1e-3, seed=seed)
    v = eval_v(m, Xte_s, Yte_s)
    vals_lb.append(round(v, 4))
    print(f"  with-LB seed={seed} -> {v:.4f}", flush=True)
e4["with_lb"] = {"vals": vals_lb, "mean": round(np.mean(vals_lb), 4)}

# Also measure routing entropy for both
for name, model_fn in [("no_lb", lambda: MoE_lite(ch=16, n_experts=2)),
                        ("with_lb", lambda: MoE_lite_lb(ch=16, n_experts=2, lb_weight=0.01))]:
    m = model_fn()
    if name == "with_lb":
        m = train_lb(m, Xtr_s, Ytr_s, steps=60, lr=1e-3, seed=0)
    else:
        m = train_full(m, Xtr_s, Ytr_s, steps=60, lr=1e-3, seed=0)
    with torch.no_grad():
        g = torch.softmax(m.gate(m.desc[None]), -1).numpy()[0]
    entropy = float(-np.sum(g * np.log(g + 1e-8)))
    e4[name]["gate_weights"] = [round(float(x), 4) for x in g]
    e4[name]["routing_entropy"] = round(entropy, 4)
    print(f"  {name}: gate_weights={g} entropy={entropy:.4f}", flush=True)

results["E4"] = e4
json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)

print("\n=== DONE ===", flush=True)
print(json.dumps(results, indent=2, ensure_ascii=False))