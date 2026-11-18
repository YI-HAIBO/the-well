# -*- coding: utf-8 -*-
"""Attack round 2 (part B): E6 + E8.

E6  Cross-system gate adaptation (shear -> turb), testing whether gating-side
    adaptation can compensate operator drift:
      zero_shot     source model, no adaptation
      gate_only     refit fixed-description gate (14 params)
      stat_gate     NEW: learnable input-statistics -> gate mapping (routing
                    becomes input-dependent); adapt only the mapping
      full_refit    all parameters
      scratch_*     fresh model, matched (60) / full (200) budget
    Few-shot: 16 target windows; eval on 32 held-out target windows. 5 seeds.

E8  Masked-reconstruction / contrastive pretraining re-test at ch16 vs ch64:
      pretrain (100 steps, shear pressure, 64 windows) -> few-shot fine-tune
      vs scratch with identical fine-tune budget. 5 seeds, Welch t-test.

Results -> attack2b_results.json
"""
import numpy as np, torch, torch.nn as nn, os, json, time, h5py

torch.set_num_threads(8)
BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
RES = os.path.join(BASE, "attack2b_results.json")

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

def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float("inf")

def mk_xy(T, n_in=4):
    X, Y = [], []
    for t in range(T.shape[0] - n_in):
        X.append(T[t:t+n_in]); Y.append(T[t+n_in:t+n_in+1])
    return np.stack(X), np.stack(Y)

# ---- source: shear pressure (4 traj) ----
ztr = np.load(os.path.join(BASE, "data", "slice_train.npz"))
P_src = downsample(ztr["pressure"])                       # (4,20,64,64)
Xsrc = np.concatenate([mk_xy(P_src[i])[0] for i in range(4)])
Ysrc = np.concatenate([mk_xy(P_src[i])[1] for i in range(4)])
Xsrc_s, Ysrc_s = std_ps(Xsrc), std_ps(Ysrc)

# ---- target: turb density (8 traj, 101 steps) ----
with h5py.File(os.path.join(BASE, "turb_rad_layer.hdf5"), "r") as f:
    D_turb = f["t0_fields/density"][:]
D_turb = downsample(D_turb)                                # (8,101,64,64)
Xt_few = np.concatenate([mk_xy(D_turb[i])[0] for i in (0, 1)])[:16]
Yt_few = np.concatenate([mk_xy(D_turb[i])[1] for i in (0, 1)])[:16]
Xt_ev = np.concatenate([mk_xy(D_turb[i])[0] for i in (2, 3)])[:32]
Yt_ev = np.concatenate([mk_xy(D_turb[i])[1] for i in (2, 3)])[:32]
Xt_few_s, Yt_few_s = std_ps(Xt_few), std_ps(Yt_few)
Xt_ev_s, Yt_ev_s = std_ps(Xt_ev), std_ps(Yt_ev)
print("src:", Xsrc_s.shape, "turb few:", Xt_few_s.shape, "turb eval:", Xt_ev_s.shape, flush=True)

# ---------------- models ----------------
class MoE_lite(nn.Module):
    """Fixed-description gate (paper architecture)."""
    def __init__(self, ch=16, in_ch=4, n_experts=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(n_experts)])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(4, n_experts)
        self.desc = nn.Parameter(torch.randn(4))
        self.n_experts = n_experts
    def features(self, x):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[None]), -1)
        h = sum(g[0, i] * self.experts[i](h) for i in range(self.n_experts))
        return h
    def forward(self, x):
        return self.out(self.features(x))

class MoE_statgate(nn.Module):
    """Learnable input-statistics -> gate mapping (routing input-dependent)."""
    def __init__(self, ch=16, in_ch=4, n_experts=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(n_experts)])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(4 * in_ch, n_experts)  # per-frame mean/std/max/min
        self.n_experts = n_experts
        self.in_ch = in_ch
    def forward(self, x):
        h = self.in_conv(x)
        B = x.shape[0]
        feats = torch.cat([x.mean(dim=(-2, -1)), x.std(dim=(-2, -1)),
                           x.amax(dim=(-2, -1)), x.amin(dim=(-2, -1))], dim=1)
        g = torch.softmax(self.gate(feats), -1)      # (B, n_experts)
        h = sum(g[:, i].view(B, 1, 1, 1) * self.experts[i](h) for i in range(self.n_experts))
        return self.out(h)
    def gate_entropy(self, x):
        with torch.no_grad():
            feats = torch.cat([x.mean(dim=(-2, -1)), x.std(dim=(-2, -1)),
                               x.amax(dim=(-2, -1)), x.amin(dim=(-2, -1))], dim=1)
            g = torch.softmax(self.gate(feats), -1).numpy()
        return float(np.mean([-np.sum(row * np.log(row + 1e-8)) for row in g]))

def train(model, X, Y, steps, lr=1e-3, seed=0, params=None):
    torch.manual_seed(seed)
    ps = list(model.parameters()) if params is None else list(params)
    opt = torch.optim.AdamW(ps, lr=lr)
    lf = nn.MSELoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(steps):
        opt.zero_grad(); lf(model(Xt), Yt).backward(); opt.step()
    return model

def eval_m(model, X, Y):
    model.eval()
    with torch.no_grad():
        return vrmse(model(torch.tensor(X)).numpy(), Y)

results = {"protocol": "cross-system shear->turb few-shot adaptation; VRMSE per-sample-std",
           "E6": {}, "E8": {}}

# ============================================================
# E6
# ============================================================
print("\n=== E6: cross-system gate adaptation (shear->turb) ===", flush=True)
t0 = time.time()
N_SEED = 5
methods = ["zero_shot", "gate_only", "stat_gate", "full_refit", "scratch_60", "scratch_200"]
e6 = {m: [] for m in methods}
for seed in range(N_SEED):
    # shared source models for this seed
    src_fix = train(MoE_lite(ch=16), Xsrc_s, Ysrc_s, steps=100, seed=seed)
    src_stat = train(MoE_statgate(ch=16), Xsrc_s, Ysrc_s, steps=100, seed=seed)

    # zero-shot (fixed-desc source)
    e6["zero_shot"].append(round(eval_m(src_fix, Xt_ev_s, Yt_ev_s), 4))

    # gate-only: train gate.weight/bias + desc
    m = MoE_lite(ch=16); m.load_state_dict(src_fix.state_dict())
    gp = [m.gate.weight, m.gate.bias, m.desc]
    train(m, Xt_few_s, Yt_few_s, steps=60, seed=seed, params=gp)
    e6["gate_only"].append(round(eval_m(m, Xt_ev_s, Yt_ev_s), 4))

    # stat_gate: adapt only the mapping
    m = MoE_statgate(ch=16); m.load_state_dict(src_stat.state_dict())
    train(m, Xt_few_s, Yt_few_s, steps=60, seed=seed, params=list(m.gate.parameters()))
    e6["stat_gate"].append(round(eval_m(m, Xt_ev_s, Yt_ev_s), 4))

    # full refit
    m = MoE_lite(ch=16); m.load_state_dict(src_fix.state_dict())
    train(m, Xt_few_s, Yt_few_s, steps=60, seed=seed)
    e6["full_refit"].append(round(eval_m(m, Xt_ev_s, Yt_ev_s), 4))

    # scratch matched / full
    m = train(MoE_lite(ch=16), Xt_few_s, Yt_few_s, steps=60, seed=seed)
    e6["scratch_60"].append(round(eval_m(m, Xt_ev_s, Yt_ev_s), 4))
    m = train(MoE_lite(ch=16), Xt_few_s, Yt_few_s, steps=200, seed=seed)
    e6["scratch_200"].append(round(eval_m(m, Xt_ev_s, Yt_ev_s), 4))
    print("  seed %d: %s" % (seed, {k: v[-1] for k, v in e6.items()}), flush=True)

# stat-gate routing entropy on target eval (input-dependence evidence)
m = MoE_statgate(ch=16); m.load_state_dict(src_stat.state_dict())
train(m, Xt_few_s, Yt_few_s, steps=60, seed=0, params=list(m.gate.parameters()))
e6["stat_gate_entropy_target"] = round(m.gate_entropy(torch.tensor(Xt_ev_s)), 4)
m0 = MoE_statgate(ch=16); m0.load_state_dict(src_stat.state_dict())
e6["stat_gate_entropy_zeroshot"] = round(m0.gate_entropy(torch.tensor(Xt_ev_s)), 4)

for mname in methods:
    v = e6[mname]
    e6[mname] = {"vals": v, "mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
print("E6 summary:", {k: e6[k].get("mean") for k in methods},
      "entropy zz/es:", e6["stat_gate_entropy_zeroshot"], e6["stat_gate_entropy_target"],
      " (%.0fs)" % (time.time() - t0), flush=True)
results["E6"] = e6
json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)

# ============================================================
# E8: mask / contrast pretraining at ch16 vs ch64
# ============================================================
print("\n=== E8: mask/contrast pretraining, ch16 vs ch64 ===", flush=True)
t0 = time.time()

def pretrain_mask(model, X, steps=100, lr=1e-3, seed=0, mask_frac=0.25, blk=4):
    """Zero-imputation block masking; loss on masked pixels of the target frame."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    B, C, H, W = Xt.shape
    rng = np.random.RandomState(seed)
    for _ in range(steps):
        idx = rng.choice(B, size=min(32, B), replace=False)
        xb, yb = Xt[idx], Yt[idx]
        mask = np.zeros((len(idx), H, W), dtype=bool)
        for b in range(len(idx)):
            cells = rng.choice(H * W, size=int(H * W * mask_frac), replace=False)
            mask[b].ravel()[cells] = True
        mt = torch.tensor(mask).float().unsqueeze(1)
        xb_m = xb.clone(); xb_m[:, -1:] = xb_m[:, -1:] * (1 - mt)
        pred = model(xb_m)
        loss = ((pred - yb) ** 2 * mt).sum() / mt.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def pretrain_contrast(model, X, steps=100, lr=1e-3, seed=0, temp=0.1, feat_dim=16):
    torch.manual_seed(seed)
    proj = nn.Linear(feat_dim, 16)
    opt = torch.optim.AdamW(list(model.parameters()) + list(proj.parameters()), lr=lr)
    Xt = torch.tensor(X)
    B = Xt.shape[0]
    rng = np.random.RandomState(seed)
    for _ in range(steps):
        idx = rng.choice(B, size=min(32, B), replace=False)
        xb = Xt[idx]
        v1 = torch.flip(xb, dims=[3])
        shifts = rng.randint(0, 16, size=2)
        v2 = torch.roll(torch.flip(xb, dims=[2]), shifts=int(shifts[0]), dims=3)
        v1 = torch.roll(v1, shifts=int(shifts[1]), dims=2)
        h1 = model.features(v1); h2 = model.features(v2)
        z1 = torch.nn.functional.normalize(proj(h1.mean(-1).mean(-1)), dim=1)
        z2 = torch.nn.functional.normalize(proj(h2.mean(-1).mean(-1)), dim=1)
        logits = z1 @ z2.t() / temp
        labels = torch.arange(len(idx))
        loss = 0.5 * (nn.functional.cross_entropy(logits, labels) +
                      nn.functional.cross_entropy(logits.t(), labels))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# fine-tune data: few-shot from shear test-slice trajs 4,5; eval 6,7 (from P8 in part A order:
# 0-3 = train slice, 4-7 = test slice)
P_te = downsample(zte["pressure"])
Xft = np.concatenate([mk_xy(P_te[i])[0] for i in (0, 1)])[:16]
Yft = np.concatenate([mk_xy(P_te[i])[1] for i in (0, 1)])[:16]
Xev8 = np.concatenate([mk_xy(P_te[i])[0] for i in (2, 3)])[:32]
Yev8 = np.concatenate([mk_xy(P_te[i])[1] for i in (2, 3)])[:32]
Xft_s, Yft_s = std_ps(Xft), std_ps(Yft)
Xev8_s, Yev8_s = std_ps(Xev8), std_ps(Yev8)

def welch(a, b):
    a, b = np.array(a), np.array(b)
    t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    df = (a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))**2 / (
        (a.var(ddof=1)/len(a))**2/(len(a)-1) + (b.var(ddof=1)/len(b))**2/(len(b)-1))
    from math import erf, sqrt
    # two-sided p via normal approx (df large enough for report)
    p = 2 * (1 - 0.5 * (1 + erf(abs(t)/sqrt(2))))
    return float(t), float(p), float(df)

e8 = {}
for ch in [16, 64]:
    for task in ["mask", "contrast", "both"]:
        pre_vals, scr_vals = [], []
        for seed in range(5):
            # pretrain init
            m = MoE_lite(ch=ch)
            if task in ("mask", "both"):
                pretrain_mask(m, Xsrc_s, Ysrc_s, steps=100, seed=seed)
            if task in ("contrast", "both"):
                pretrain_contrast(m, Xsrc_s, steps=100, seed=seed + 50, feat_dim=ch)
            train(m, Xft_s, Yft_s, steps=60, seed=seed)
            pre_vals.append(round(eval_m(m, Xev8_s, Yev8_s), 4))
            # scratch
            m2 = train(MoE_lite(ch=ch), Xft_s, Yft_s, steps=60, seed=seed)
            scr_vals.append(round(eval_m(m2, Xev8_s, Yev8_s), 4))
        t, p, df = welch(pre_vals, scr_vals)
        key = "ch%d_%s" % (ch, task)
        e8[key] = {"pretrained": {"vals": pre_vals, "mean": round(float(np.mean(pre_vals)), 4)},
                   "scratch": {"vals": scr_vals, "mean": round(float(np.mean(scr_vals)), 4)},
                   "welch_t": round(t, 3), "p": float("%.3g" % p)}
        print("  %s: pre=%.4f scratch=%.4f t=%.2f p=%.3g" % (
            key, np.mean(pre_vals), np.mean(scr_vals), t, p), flush=True)

results["E8"] = e8
json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)
print("\n=== PART B DONE ->", RES, " (%.0fs) ===" % (time.time() - t0), flush=True)
