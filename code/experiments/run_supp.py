# -*- coding: utf-8 -*-
"""JCP 审稿意见补充实验：M2 基线对比、M4 tokenizer 消融、M6 复杂度、M7 敏感性、D1 路由、D3 联合验证。
数据：slice_train.npz tracer 场（4 轨迹 x 20 步 x 256 x 512）下采样 64x64，单步预测 t->t+1。CPU。"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, json, os, sys

torch.set_num_threads(4)
np.random.seed(0)
DATA = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\data\slice_train.npz"
OUT = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\supp_results.json"

# ---------- data ----------
z = np.load(DATA)
T = z["tracer"].astype(np.float32)                      # (4,20,256,512)
Td = T.reshape(4, 20, 64, 4, 64, 8).mean(axis=(3, 5))   # (4,20,64,64)
X = np.concatenate([Td[:, t] for t in range(19)], 0)[:, None]  # (76,1,64,64)
Y = np.concatenate([Td[:, t + 1] for t in range(19)], 0)[:, None]

def split(seed):
    n = X.shape[0]
    perm = np.random.RandomState(seed).permutation(n)
    ntr = int(0.8 * n)
    return (torch.tensor(X[perm[:ntr]]), torch.tensor(Y[perm[:ntr]]),
            torch.tensor(X[perm[ntr:]]), torch.tensor(Y[perm[ntr:]]))

# ---------- models ----------
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
    def __init__(self, p=32, w=64):
        super().__init__()
        self.branch = nn.Sequential(nn.Linear(32 * 32, 128), nn.ReLU(), nn.Linear(128, p))
        self.trunk = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, p))
    def forward(self, x, coords):
        # x: (B,1,64,64) -> downsample to 32x32, flatten; coords: (B,2)
        xs = F.avg_pool2d(x, 2).reshape(x.shape[0], -1)
        return (self.branch(xs) * self.trunk(coords)).sum(-1, keepdim=True).reshape(x.shape[0], 1, 32, 32)

class MoE_lite(nn.Module):
    """简化版本文模型：2 个专家卷积 + 场描述门控（desc 开关）。"""
    def __init__(self, ch=16, modes=12, use_desc=True, in_ch=1):
        super().__init__()
        self.use_desc = use_desc
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.e1 = nn.Conv2d(ch, ch, 1)
        self.e2 = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, 1, 1)
        if use_desc:
            self.gate = nn.Linear(4, 2)   # field-desc vector (dim 4) -> expert logits
            self.desc = nn.Parameter(torch.randn(4))
        else:
            self.gate = None
    def forward(self, x):
        h = self.in_conv(x)
        if self.use_desc:
            g = torch.softmax(self.gate(self.desc[None]), -1)   # (1,2)
            h = g[0, 0] * self.e1(h) + g[0, 1] * self.e2(h)
        else:
            h = self.e1(h)  # no desc -> single fixed expert
        return self.out(h)

# ---------- training ----------
def train(model, x_tr, y_tr, x_te, y_te, steps=40, lr=1e-3, seed=0, coords_tr=None, coords_te=None, use_deeponet=False):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    t0 = time.time()
    for _ in range(steps):
        model.train(); opt.zero_grad()
        if use_deeponet:
            pred = model(x_tr, coords_tr)
        else:
            pred = model(x_tr)
        loss = lf(pred, y_tr)
        loss.backward(); opt.step()
    train_time = time.time() - t0
    model.eval()
    with torch.no_grad():
        if use_deeponet:
            out = model(x_te, coords_te)
        else:
            out = model(x_te)
    rl2 = float(torch.sqrt(((out - y_te) ** 2).sum()) / torch.sqrt((y_te ** 2).sum()))
    nparam = sum(p.numel() for p in model.parameters())
    return rl2, nparam, train_time

def grid(shape):
    B, _, H, W = shape
    gx = torch.linspace(0, 1, H)[None, :, None].expand(B, H, W)
    gy = torch.linspace(0, 1, W)[None, None, :].expand(B, H, W)
    return torch.cat([gx.reshape(B, -1).mean(1, keepdim=True),
                      gy.reshape(B, -1).mean(1, keepdim=True)], 1)  # (B,2) coords

R = {"M2_baselines": {}, "M4_ablation": {}, "M6_complexity": {}, "M7_sensitivity": {},
     "D1_routing": {}, "D3_joint": {}}

def run_m2():
    print("== M2 baselines ==", flush=True)
    res = {}
    for name, mk in [("FNO", lambda: FNO2d(modes=12, ch=16, layers=2)),
                     ("DeepONet", lambda: DeepONet()),
                     ("MoE(this paper, desc)", lambda: MoE_lite(use_desc=True)),
                     ("MoE(no desc)", lambda: MoE_lite(use_desc=False))]:
        rl2s = []
        for s in range(3):
            x_tr, y_tr, x_te, y_te = split(s)
            use_deep = name == "DeepONet"
            if use_deep:
                # use 32x32 for DeepONet (branch flatten) — coords dummy
                model = mk()
                coords_tr = grid(x_tr.shape); coords_te = grid(x_te.shape)
                rl2, np_, tt = train(model, x_tr, y_tr, x_te, y_te, steps=40, seed=s,
                                     coords_tr=coords_tr, coords_te=coords_te, use_deeponet=True)
            else:
                model = mk()
                rl2, np_, tt = train(model, x_tr, y_tr, x_te, y_te, steps=40, seed=s)
            rl2s.append(rl2)
        res[name] = {"mean": float(np.mean(rl2s)), "std": float(np.std(rl2s)), "runs": rl2s}
        print(f"  {name}: {res[name]['mean']:.4f} +/- {res[name]['std']:.4f}", flush=True)
    R["M2_baselines"] = res
    return res

def run_m4():
    print("== M4 ablation ==", flush=True)
    res = {}
    # A: coords vs no-coords (use FNO, extra coord channel)
    for label, add_coord in [("coord-invariant (with coord ch)", True), ("fixed-grid (no coord ch)", False)]:
        rl2s = []
        for s in range(3):
            x_tr, y_tr, x_te, y_te = split(s)
            if add_coord:
                def make_xy(batch):
                    B, _, H, W = batch.shape
                    gx = torch.linspace(0, 1, H)[None, None, :, None].expand(B, 1, H, W)
                    gy = torch.linspace(0, 1, W)[None, None, None, :].expand(B, 1, H, W)
                    return torch.cat([batch, gx, gy], 1)
                m = FNO2d(modes=12, ch=16, layers=2, in_ch=3)
                rl2, _, _ = train(m, make_xy(x_tr), y_tr, make_xy(x_te), y_te, steps=40, seed=s)
            else:
                m = FNO2d(modes=12, ch=16, layers=2)
                rl2, _, _ = train(m, x_tr, y_tr, x_te, y_te, steps=40, seed=s)
            rl2s.append(rl2)
        res[label] = {"mean": float(np.mean(rl2s)), "std": float(np.std(rl2s))}
        print(f"  {label}: {res[label]['mean']:.4f}+/-{res[label]['std']:.4f}", flush=True)
    # B: multi-res vs single-res
    for label, multi in [("multi-resolution (64+32)", True), ("single-resolution (64)", False)]:
        rl2s = []
        for s in range(3):
            x_tr, y_tr, x_te, y_te = split(s)
            if multi:
                # augment: also train on 32x32 (coarse) version
                m = FNO2d(modes=8, ch=16, layers=2)
                x_tr2 = F.avg_pool2d(x_tr, 2); y_tr2 = F.avg_pool2d(y_tr, 2)
                x_te2 = F.avg_pool2d(x_te, 2); y_te2 = F.avg_pool2d(y_te, 2)
                # interleave coarse into training (train on both)
                torch.manual_seed(s); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
                for _ in range(40):
                    opt.zero_grad()
                    lf(m(x_tr), y_tr).backward()
                    opt.step()
                    opt.zero_grad()
                    lf(m(x_tr2), y_tr2).backward()
                    opt.step()
                m.eval()
                with torch.no_grad(): out = m(x_te)
                rl2 = float(torch.sqrt(((out - y_te) ** 2).sum()) / torch.sqrt((y_te ** 2).sum()))
            else:
                m = FNO2d(modes=8, ch=16, layers=2)
                rl2, _, _ = train(m, x_tr, y_tr, x_te, y_te, steps=40, seed=s)
            rl2s.append(rl2)
        res[label] = {"mean": float(np.mean(rl2s)), "std": float(np.std(rl2s))}
        print(f"  {label}: {res[label]['mean']:.4f}+/-{res[label]['std']:.4f}", flush=True)
    R["M4_ablation"] = res
    return res

def run_m6():
    print("== M6 complexity ==", flush=True)
    res = {}
    for name, mk in [("FNO", lambda: FNO2d(modes=12, ch=16, layers=2)),
                     ("MoE", lambda: MoE_lite(use_desc=True))]:
        x_tr, y_tr, x_te, y_te = split(0)
        m = mk()
        nparam = sum(p.numel() for p in m.parameters())
        # time per epoch + inference
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
        t0 = time.time()
        for _ in range(10):
            opt.zero_grad(); lf(m(x_tr), y_tr).backward(); opt.step()
        te = (time.time() - t0) / 10
        m.eval()
        with torch.no_grad():
            t0 = time.time(); m(x_te); ti = (time.time() - t0) / x_te.shape[0]
        res[name] = {"params": nparam, "train_s_per_epoch": te, "infer_ms_per_sample": ti * 1000}
        print(f"  {name}: params={nparam}, {te:.3f}s/epoch, {ti*1000:.2f}ms/sample", flush=True)
    R["M6_complexity"] = res
    return res

def run_m7():
    print("== M7 sensitivity ==", flush=True)
    res = {}
    x_tr, y_tr, x_te, y_te = split(0)
    for lr in [1e-3, 3e-3]:
        for modes in [8, 12]:
            for ch in [8, 16]:
                key = f"lr={lr},modes={modes},ch={ch}"
                m = FNO2d(modes=modes, ch=ch, layers=2)
                rl2, _, _ = train(m, x_tr, y_tr, x_te, y_te, steps=40, lr=lr, seed=0)
                res[key] = rl2
                print(f"  {key}: {rl2:.4f}", flush=True)
    R["M7_sensitivity"] = res
    return res

def run_d1():
    print("== D1 routing ==", flush=True)
    x_tr, y_tr, x_te, y_te = split(0)
    m = MoE_lite(use_desc=True)
    train(m, x_tr, y_tr, x_te, y_te, steps=40, seed=0)
    with torch.no_grad():
        h = m.in_conv(x_te)
        g = torch.softmax(m.gate(m.desc[None]), -1)
    R["D1_routing"] = {"gate": g[0].tolist(), "desc_norm": float(m.desc.norm().item())}
    print(f"  gate weights (expert1,expert2) = {g[0].tolist()}", flush=True)

def run_d3():
    print("== D3 joint ==", flush=True)
    x_tr, y_tr, x_te, y_te = split(0)
    res = {}
    # vary N (modes) with fixed L=2; measure error -> C1*N^-a fit
    for modes in [4, 8, 12, 16]:
        m = FNO2d(modes=modes, ch=16, layers=2)
        rl2, _, _ = train(m, x_tr, y_tr, x_te, y_te, steps=40, seed=0)
        res[f"modes={modes}"] = rl2
        print(f"  modes={modes}: {rl2:.4f}", flush=True)
    # vary L (layers) with fixed modes=12
    for layers in [1, 2, 3]:
        m = FNO2d(modes=12, ch=16, layers=layers)
        rl2, _, _ = train(m, x_tr, y_tr, x_te, y_te, steps=40, seed=0)
        res[f"layers={layers}"] = rl2
        print(f"  layers={layers}: {rl2:.4f}", flush=True)
    R["D3_joint"] = res
    return res

if __name__ == "__main__":
    for fn in [run_m2, run_m4, run_m6, run_m7, run_d1, run_d3]:
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {fn.__name__} failed: {e}", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print("\nDONE ->", OUT, flush=True)
