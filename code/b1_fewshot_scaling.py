# -*- coding: utf-8 -*-
"""B1 (E1 residual): few-shot adaptation budget sweep, shear->turb.

Question: does the 0.995-vs-0.71 recovery gap close with adaptation budget,
and how does label count affect it?
  zero-shot        : shear-pretrained model, no adaptation
  scratch_200/1800 : fresh model trained on the same few-shot windows
  adapted_{S}      : shear-pretrained -> full-param fine-tune, S steps
  windows sweep    : 4 vs 16 label windows at 600 steps
Protocol: normalized 64x64, ch16, eval on 32 held-out turb windows, 3 seeds.
Output: b1_fewshot_scaling.json
"""
import numpy as np, torch, torch.nn as nn, os, json, h5py

torch.set_num_threads(8)
BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"

def downsample(T, H=64, W=64):
    oH, oW = T.shape[-2], T.shape[-1]
    fH, fW = oH // H, oW // W
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

class MoE_lite(nn.Module):
    def __init__(self, ch=16, in_ch=4, n_experts=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(n_experts)])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(4, n_experts)
        self.desc = nn.Parameter(torch.randn(4))
    def forward(self, x):
        g = torch.softmax(self.gate(self.desc[None]), -1)
        h = self.in_conv(x)
        h = sum(g[0, i] * self.experts[i](h) for i in range(2))
        return self.out(h)

def train(model, X, Y, steps, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(Y, dtype=torch.float32)
    for _ in range(steps):
        idx = torch.randint(0, xt.shape[0], (16,))
        opt.zero_grad()
        loss = ((model(xt[idx]) - yt[idx]) ** 2).mean()
        loss.backward(); opt.step()
    return model

def evaluate(model, Xe, Ye):
    with torch.no_grad():
        p = model(torch.tensor(Xe, dtype=torch.float32)).numpy()
    return vrmse(p, Ye)

# ---- data ----
ztr = np.load(os.path.join(BASE, "data", "slice_train.npz"))
P_src = downsample(ztr["pressure"])
Xsrc = np.concatenate([mk_xy(P_src[i])[0] for i in range(4)])
Ysrc = np.concatenate([mk_xy(P_src[i])[1] for i in range(4)])
Xsrc_s, Ysrc_s = std_ps(Xsrc), std_ps(Ysrc)

with h5py.File(os.path.join(BASE, "turb_rad_layer.hdf5"), "r") as f:
    D_turb = downsample(f["t0_fields/density"][:])
Xf16, Yf16 = np.concatenate([mk_xy(D_turb[i])[0] for i in (0, 1)])[:16], np.concatenate([mk_xy(D_turb[i])[1] for i in (0, 1)])[:16]
Xf4, Yf4 = Xf16[:4], Yf16[:4]
Xev, Yev = np.concatenate([mk_xy(D_turb[i])[0] for i in (2, 3)])[:32], np.concatenate([mk_xy(D_turb[i])[1] for i in (2, 3)])[:32]
Xf16_s, Yf16_s, Xf4_s, Yf4_s, Xev_s, Yev_s = map(std_ps, (Xf16, Yf16, Xf4, Yf4, Xev, Yev))
print("data ready | few16:", Xf16_s.shape, "few4:", Xf4_s.shape, "eval:", Xev_s.shape, flush=True)

SEEDS = [0, 1, 2]
res = {"zero_shot": [], "scratch_200": [], "scratch_1800": [],
       "adapted_50": [], "adapted_200": [], "adapted_600": [], "adapted_1800": [],
       "adapted_600_w4": []}

for seed in SEEDS:
    torch.manual_seed(seed)
    src_model = MoE_lite()
    train(src_model, Xsrc_s, Ysrc_s, 100, seed=seed)
    res["zero_shot"].append(evaluate(src_model, Xev_s, Yev_s))

    m = MoE_lite(); train(m, Xf16_s, Yf16_s, 200, seed=seed)
    res["scratch_200"].append(evaluate(m, Xev_s, Yev_s))
    m = MoE_lite(); train(m, Xf16_s, Yf16_s, 1800, seed=seed)
    res["scratch_1800"].append(evaluate(m, Xev_s, Yev_s))

    for steps in (50, 200, 600, 1800):
        m = MoE_lite(); m.load_state_dict(src_model.state_dict())
        train(m, Xf16_s, Yf16_s, steps, seed=seed)
        res[f"adapted_{steps}"].append(evaluate(m, Xev_s, Yev_s))

    m = MoE_lite(); m.load_state_dict(src_model.state_dict())
    train(m, Xf4_s, Yf4_s, 600, seed=seed)
    res["adapted_600_w4"].append(evaluate(m, Xev_s, Yev_s))
    print("seed", seed, "done", flush=True)

def summarize(vals):
    v = np.array(vals)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)), 4), "seeds": [round(x, 4) for x in vals]}

out = {k: summarize(v) for k, v in res.items()}

def welch(a, b):
    a, b = np.array(a), np.array(b)
    t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    df = (a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))**2 / ((a.var(ddof=1)/len(a))**2/(len(a)-1) + (b.var(ddof=1)/len(b))**2/(len(b)-1))
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t)/sqrt(2))))  # normal approx
    return round(float(t), 3), int(df), round(float(p), 4)

out["welch_adapted1800_vs_scratch1800"] = welch(res["adapted_1800"], res["scratch_1800"])
out["welch_adapted600_vs_scratch200"] = welch(res["adapted_600"], res["scratch_200"])

with open(os.path.join(BASE, "b1_fewshot_scaling.json"), "w") as f:
    json.dump(out, f, indent=2)
for k, v in out.items():
    print(k, v)
