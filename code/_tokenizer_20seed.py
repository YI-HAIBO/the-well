# -*- coding: utf-8 -*-
"""#3 (Round 38): 20-seed tokenizer ablation with per-seed values.

Variants follow the published M4 protocol:
  coord   : 64x64 field + two normalized coordinate channels
  fixed   : 64x64 field only (fixed-grid ablation)
  multires: train on 64x64 and 32x32 pooled copies, evaluate at 64x64
  single  : 64x64 field only, same FNO budget as multires
Model: small FNO-style spectral network, 40 AdamW steps, seed-dependent 80/20 split.
Metric: relative L2 on the held-out tracer field.  Saves raw per-seed values.
"""
import os, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "c3_tokenizer_ablation_20seed.json")
torch.set_num_threads(4)
N_SEED = 20
STEPS = 40
LR = 1e-3

z = np.load(os.path.join(BASE, "data", "slice_train.npz"))
T = z["tracer"].astype(np.float32)
Td = T.reshape(4, 20, 64, 4, 64, 8).mean(axis=(3, 5))
X = np.concatenate([Td[:, t] for t in range(19)], 0)[:, None]
Y = np.concatenate([Td[:, t + 1] for t in range(19)], 0)[:, None]


class FNO2d(nn.Module):
    def __init__(self, modes=12, ch=16, layers=2, in_ch=1):
        super().__init__()
        self.modes, self.layers = modes, layers
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.spectral = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(layers)])
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        h = self.in_conv(x)
        for c in self.spectral:
            hf = torch.fft.rfft2(h)
            hf = hf[:, :, :self.modes, :self.modes]
            h = torch.fft.irfft2(hf, s=h.shape[-2:]) + c(h)
        return self.out(h)


def split(seed):
    n = X.shape[0]
    perm = np.random.RandomState(seed).permutation(n)
    ntr = int(0.8 * n)
    return (torch.tensor(X[perm[:ntr]]), torch.tensor(Y[perm[:ntr]]),
            torch.tensor(X[perm[ntr:]]), torch.tensor(Y[perm[ntr:]]))


def coord_channels(x):
    B, _, H, W = x.shape
    gx = torch.linspace(0, 1, H)[None, None, :, None].expand(B, 1, H, W)
    gy = torch.linspace(0, 1, W)[None, None, None, :].expand(B, 1, H, W)
    return torch.cat([x, gx, gy], 1)


def rel_l2(pred, y):
    return float(torch.sqrt(((pred - y) ** 2).sum()) / torch.sqrt((y ** 2).sum()))


def train_variant(kind, seed):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = split(seed)
    in_ch = 3 if kind == "coord" else 1
    m = FNO2d(modes=12 if kind in ("coord", "fixed") else 8, ch=16, layers=2, in_ch=in_ch)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    lf = nn.MSELoss()
    xtr_in = coord_channels(xtr) if kind == "coord" else xtr
    xtr2 = F.avg_pool2d(xtr, 2)
    ytr2 = F.avg_pool2d(ytr, 2)
    for _ in range(STEPS):
        opt.zero_grad()
        loss = lf(m(xtr_in), ytr)
        if kind == "multires":
            loss = loss + lf(m(xtr2), ytr2)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        xte_in = coord_channels(xte) if kind == "coord" else xte
        return rel_l2(m(xte_in), yte)


def welch(a, b):
    a, b = np.array(a), np.array(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    mde = (1.959964 + 0.841621) * sp * np.sqrt(1 / len(a) + 1 / len(b))
    return float(t), float(p), float(mde)


def summ(vals):
    v = np.array(vals)
    return {"vals": [round(float(x), 5) for x in v], "mean": round(float(v.mean()), 5),
            "std": round(float(v.std(ddof=1)), 5), "median": round(float(np.median(v)), 5),
            "iqr": [round(float(np.percentile(v, 25)), 5), round(float(np.percentile(v, 75)), 5)]}


def main():
    t0 = time.time()
    res = {}
    for kind in ["coord", "fixed", "multires", "single"]:
        vals = []
        for seed in range(N_SEED):
            v = train_variant(kind, seed)
            vals.append(v)
            print(f"{kind} seed={seed}: {v:.5f}", flush=True)
        res[kind] = summ(vals)
        print(kind, res[kind], flush=True)
    comparisons = {}
    for a, b in [("coord", "fixed"), ("multires", "single")]:
        t, p, mde = welch(res[a]["vals"], res[b]["vals"])
        comparisons[f"{a}_vs_{b}"] = {"welch_t": round(t, 3), "p": float("%.4g" % p), "mde_80": round(mde, 5)}
        print(a, "vs", b, comparisons[f"{a}_vs_{b}"], flush=True)
    out = {"protocol": "20-seed tokenizer ablation; shear tracer; 40 AdamW steps; 80/20 split; relative L2",
           "n_seeds": N_SEED, "results": res, "comparisons": comparisons,
           "runtime_sec": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("Saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
