# -*- coding: utf-8 -*-
"""M2 基线对比（修正版 DeepONet）。同协议：4 轨迹 shear_flow tracer 64x64，单步，3 seeds。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, json

torch.set_num_threads(4)
np.random.seed(0)
DATA = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\data\slice_train.npz"
OUT = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\supp_m2.json"

z = np.load(DATA)
T = z["tracer"].astype(np.float32)
Td = T.reshape(4, 20, 64, 4, 64, 8).mean(axis=(3, 5))
X = np.concatenate([Td[:, t] for t in range(19)], 0)[:, None]
Y = np.concatenate([Td[:, t + 1] for t in range(19)], 0)[:, None]

def split(seed):
    n = X.shape[0]
    perm = np.random.RandomState(seed).permutation(n)
    ntr = int(0.8 * n)
    return (torch.tensor(X[perm[:ntr]]), torch.tensor(Y[perm[:ntr]]),
            torch.tensor(X[perm[ntr:]]), torch.tensor(Y[perm[ntr:]]))

class FNO2d(nn.Module):
    def __init__(self, modes=12, ch=16, layers=2, in_ch=1):
        super().__init__()
        self.modes, self.layers = modes, layers
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.spectral = nn.ModuleList()
        for _ in range(layers):
            self.spectral.append(nn.Conv2d(ch, ch, 1))
        self.out = nn.Conv2d(ch, 1, 1)
    def forward(self, x):
        h = self.in_conv(x)
        for c in self.spectral:
            hf = torch.fft.rfft2(h)
            hf = hf[:, :, :self.modes, :self.modes]
            h = torch.fft.irfft2(hf, s=h.shape[-2:]) + c(h)
        return self.out(h)

class DeepONet(nn.Module):
    """标准 DeepONet: u(x)=sum_k branch_k(field)*trunk_k(x)，整场输出 64x64。"""
    def __init__(self, p=32, w=64):
        super().__init__()
        self.p = p
        self.branch = nn.Sequential(nn.Linear(64 * 64, 128), nn.ReLU(), nn.Linear(128, p))
        self.trunk = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, p))
        H2 = W2 = 64
        gx = torch.linspace(0, 1, H2).view(1, H2, 1).expand(1, H2, W2).reshape(-1)
        gy = torch.linspace(0, 1, W2).view(1, 1, W2).expand(1, H2, W2).reshape(-1)
        self.register_buffer("coords", torch.stack([gx, gy], -1))  # (H2*W2,2)
    def forward(self, x):
        xs = x.reshape(x.shape[0], -1)                       # (B,64*64)
        b = self.branch(xs)                                  # (B,p)
        t = self.trunk(self.coords[None].expand(x.shape[0], -1, -1))  # (B,HW,p)
        out = (b[:, None, :] * t).sum(-1)                    # (B,HW)
        return out.view(x.shape[0], 1, 64, 64)

class MoE_lite(nn.Module):
    def __init__(self, ch=16, use_desc=True, in_ch=1):
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

def train(model, x_tr, y_tr, x_te, y_te, steps=40, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    for _ in range(steps):
        model.train(); opt.zero_grad()
        lf(model(x_tr), y_tr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(x_te)
    rl2 = float(torch.sqrt(((out - y_te) ** 2).sum()) / torch.sqrt((y_te ** 2).sum()))
    return rl2

print("== M2 baselines (corrected DeepONet) ==", flush=True)
res = {}
for name, mk in [("FNO", lambda: FNO2d(modes=12, ch=16, layers=2)),
                 ("DeepONet", lambda: DeepONet()),
                 ("MoE (this paper, field-desc gate)", lambda: MoE_lite(use_desc=True)),
                 ("MoE (no field-desc gate)", lambda: MoE_lite(use_desc=False))]:
    rl2s = []
    for s in range(3):
        x_tr, y_tr, x_te, y_te = split(s)
        rl2s.append(train(mk(), x_tr, y_tr, x_te, y_te, steps=40, seed=s))
    res[name] = {"mean": float(np.mean(rl2s)), "std": float(np.std(rl2s)), "runs": rl2s}
    print(f"  {name}: {res[name]['mean']:.4f} +/- {res[name]['std']:.4f}", flush=True)

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("DONE ->", OUT, flush=True)
