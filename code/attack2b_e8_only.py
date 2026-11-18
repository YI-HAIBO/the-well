# -*- coding: utf-8 -*-
"""E8-only runner (E6 already completed and stored in attack2b_results.json).
Masked-reconstruction / contrastive pretraining re-test at ch16 vs ch64.
Fixes vs attack2b_e6_e8.py: loads slice_test.npz locally; pretrain_mask takes Y.
Results merged into attack2b_results.json.
"""
import numpy as np, torch, torch.nn as nn, os, json, time

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

def mk_xy(T, n_in=4):
    X, Y = [], []
    for t in range(T.shape[0] - n_in):
        X.append(T[t:t+n_in]); Y.append(T[t+n_in:t+n_in+1])
    return np.stack(X), np.stack(Y)

# ---- data ----
ztr = np.load(os.path.join(BASE, "data", "slice_train.npz"))
zte = np.load(os.path.join(BASE, "data", "slice_test.npz"))
P_src = downsample(ztr["pressure"])
Xsrc = np.concatenate([mk_xy(P_src[i])[0] for i in range(4)])
Ysrc = np.concatenate([mk_xy(P_src[i])[1] for i in range(4)])
Xsrc_s, Ysrc_s = std_ps(Xsrc), std_ps(Ysrc)
P_te = downsample(zte["pressure"])
Xft = np.concatenate([mk_xy(P_te[i])[0] for i in (0, 1)])[:16]
Yft = np.concatenate([mk_xy(P_te[i])[1] for i in (0, 1)])[:16]
Xev8 = np.concatenate([mk_xy(P_te[i])[0] for i in (2, 3)])[:32]
Yev8 = np.concatenate([mk_xy(P_te[i])[1] for i in (2, 3)])[:32]
Xft_s, Yft_s = std_ps(Xft), std_ps(Yft)
Xev8_s, Yev8_s = std_ps(Xev8), std_ps(Yev8)
print("pretrain:", Xsrc_s.shape, "few-shot:", Xft_s.shape, "eval:", Xev8_s.shape, flush=True)

class MoE_lite(nn.Module):
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

def train(model, X, Y, steps=60, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(steps):
        opt.zero_grad(); lf(model(Xt), Yt).backward(); opt.step()
    return model

def eval_m(model, X, Y):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy()
    mse = float(((pred - Y) ** 2).mean()); yv = float(Y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float("inf")

def pretrain_mask(model, X, Y, steps=100, lr=1e-3, seed=0, mask_frac=0.25):
    """Zero-imputation block masking; loss on masked pixels of the target frame."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    B, _, H, W = Xt.shape
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

def welch(a, b):
    a, b = np.array(a), np.array(b)
    t = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t)/sqrt(2))))
    return float(t), float(p)

# ---- run E8 ----
results = json.load(open(RES, encoding="utf-8")) if os.path.exists(RES) else {}
print("\n=== E8: mask/contrast pretraining, ch16 vs ch64 ===", flush=True)
t0 = time.time()
e8 = {}
for ch in [16, 64]:
    for task in ["mask", "contrast", "both"]:
        pre_vals, scr_vals = [], []
        for seed in range(5):
            m = MoE_lite(ch=ch)
            if task in ("mask", "both"):
                pretrain_mask(m, Xsrc_s, Ysrc_s, steps=100, seed=seed)
            if task in ("contrast", "both"):
                pretrain_contrast(m, Xsrc_s, steps=100, seed=seed + 50, feat_dim=ch)
            train(m, Xft_s, Yft_s, steps=60, seed=seed)
            pre_vals.append(round(eval_m(m, Xev8_s, Yev8_s), 4))
            m2 = train(MoE_lite(ch=ch), Xft_s, Yft_s, steps=60, seed=seed)
            scr_vals.append(round(eval_m(m2, Xev8_s, Yev8_s), 4))
        t, p = welch(pre_vals, scr_vals)
        key = "ch%d_%s" % (ch, task)
        e8[key] = {"pretrained": {"vals": pre_vals, "mean": round(float(np.mean(pre_vals)), 4)},
                   "scratch": {"vals": scr_vals, "mean": round(float(np.mean(scr_vals)), 4)},
                   "welch_t": round(t, 3), "p": float("%.3g" % p)}
        print("  %s: pre=%.4f scratch=%.4f t=%.2f p=%.3g" % (
            key, np.mean(pre_vals), np.mean(scr_vals), t, p), flush=True)

results["E8"] = e8
json.dump(results, open(RES, "w"), indent=2, ensure_ascii=False)
print("\n=== E8 DONE, merged into", RES, " (%.0fs) ===" % (time.time() - t0), flush=True)
