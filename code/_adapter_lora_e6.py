# -*- coding: utf-8 -*-
"""#5 (Round 38): lightweight cross-family adaptation beyond gate-only.

Protocol follows the published E6 setup: pretrain on shear pressure, adapt to
turbulent-radiative-layer density with 16 few-shot windows, evaluate on 32 held-out
target windows, 5 seeds.  New arms:
  adapter_only: freeze source backbone/gate; train a zero-initialized residual
                1x1 adapter before the output head.
  lora_only:    freeze source backbone/gate; train rank-4 low-rank residuals on
                the 1x1 convolution inside each expert.
Existing arms: zero_shot, gate_only, full_refit.  Metric: per-sample-std VRMSE.
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "c5_adapter_lora_shear_to_turb.json")
torch.set_num_threads(4)
N_SEED = 5


def downsample(T, H=64, W=64):
    oH, oW = T.shape[-2], T.shape[-1]
    fH, fW = oH // H, oW // W
    if T.ndim == 5:
        T = T.reshape(T.shape[0], T.shape[1], fH, H, fW, W, T.shape[-1]).mean(axis=(2, 4))
    else:
        T = T.reshape(T.shape[0], T.shape[1], fH, H, fW, W).mean(axis=(2, 4))
    return T.astype(np.float32)


def std_ps(A):
    m = A.mean(axis=(-2, -1), keepdims=True)
    s = A.std(axis=(-2, -1), keepdims=True) + 1e-6
    return (A - m) / s


def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean())
    yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float("inf")


def mk_xy(T, n_in=4):
    X, Y = [], []
    for t in range(T.shape[0] - n_in):
        X.append(T[t:t + n_in])
        Y.append(T[t + n_in:t + n_in + 1])
    return np.stack(X), np.stack(Y)


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
        return sum(g[0, i] * self.experts[i](h) for i in range(self.n_experts))

    def forward(self, x):
        return self.out(self.features(x))


class MoE_adapter(MoE_lite):
    def __init__(self, ch=16, in_ch=4, n_experts=2):
        super().__init__(ch, in_ch, n_experts)
        self.adapter = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.GELU(), nn.Conv2d(ch, ch, 1))
        nn.init.zeros_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

    def forward(self, x):
        return self.out(self.adapter(self.features(x)))


class MoE_lora(nn.Module):
    """Two-expert MoE with rank-r residuals on each expert conv; source weights frozen."""
    def __init__(self, src: MoE_lite, rank=4):
        super().__init__()
        ch = src.in_conv.out_channels
        self.in_conv = src.in_conv
        self.out = src.out
        self.gate = src.gate
        self.desc = src.desc
        self.n_experts = src.n_experts
        self.exp_conv = nn.ModuleList()
        self.lora = nn.ModuleList()
        for e in src.experts:
            self.exp_conv.append(e)
            self.lora.append(nn.ParameterDict({
                "A": nn.Parameter(torch.randn(rank, ch, 1, 1) * 0.01),
                "B": nn.Parameter(torch.zeros(ch, rank, 1, 1))}))
        for p in self.parameters():
            p.requires_grad = False
        for d in self.lora:
            d["A"].requires_grad = True
            d["B"].requires_grad = True

    def forward(self, x):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[None]), -1)
        hs = []
        for i in range(self.n_experts):
            dw = self.lora[i]["B"].squeeze(-1).squeeze(-1) @ self.lora[i]["A"].squeeze(-1).squeeze(-1)
            z = self.exp_conv[i](h) + F.conv2d(h, dw.unsqueeze(-1).unsqueeze(-1))
            hs.append(z)
        hmix = sum(g[0, i] * hs[i] for i in range(self.n_experts))
        return self.out(hmix)


def train(model, X, Y, steps, lr=1e-3, seed=0, params=None):
    torch.manual_seed(seed)
    ps = list(model.parameters()) if params is None else list(params)
    ps = [p for p in ps if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr)
    lf = nn.MSELoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(steps):
        opt.zero_grad()
        lf(model(Xt), Yt).backward()
        opt.step()
    return model


def eval_m(model, X, Y):
    model.eval()
    with torch.no_grad():
        return vrmse(model(torch.tensor(X)).numpy(), Y)


def count_trainable(model, params):
    names = {id(p) for p in params}
    return sum(p.numel() for p in model.parameters() if id(p) in names and p.requires_grad)


def main():
    ztr = np.load(os.path.join(BASE, "data", "slice_train.npz"))
    P_src = downsample(ztr["pressure"])
    Xsrc = np.concatenate([mk_xy(P_src[i])[0] for i in range(4)])
    Ysrc = np.concatenate([mk_xy(P_src[i])[1] for i in range(4)])
    Xsrc_s, Ysrc_s = std_ps(Xsrc), std_ps(Ysrc)

    with h5py.File(os.path.join(BASE, "turb_rad_layer.hdf5"), "r") as f:
        D_turb = f["t0_fields/density"][:]
    D_turb = downsample(D_turb)
    Xt_few = np.concatenate([mk_xy(D_turb[i])[0] for i in (0, 1)])[:16]
    Yt_few = np.concatenate([mk_xy(D_turb[i])[1] for i in (0, 1)])[:16]
    Xt_ev = np.concatenate([mk_xy(D_turb[i])[0] for i in (2, 3)])[:32]
    Yt_ev = np.concatenate([mk_xy(D_turb[i])[1] for i in (2, 3)])[:32]
    Xt_few_s, Yt_few_s = std_ps(Xt_few), std_ps(Yt_few)
    Xt_ev_s, Yt_ev_s = std_ps(Xt_ev), std_ps(Yt_ev)
    print("src", Xsrc_s.shape, "few", Xt_few_s.shape, "eval", Xt_ev_s.shape, flush=True)

    methods = ["zero_shot", "gate_only", "adapter_only", "lora_only", "full_refit"]
    res = {m: [] for m in methods}
    params = {}
    t0 = time.time()
    for seed in range(N_SEED):
        src = train(MoE_lite(ch=16), Xsrc_s, Ysrc_s, steps=100, seed=seed)
        res["zero_shot"].append(round(eval_m(src, Xt_ev_s, Yt_ev_s), 4))

        m = MoE_lite(ch=16)
        m.load_state_dict(src.state_dict())
        gp = [m.gate.weight, m.gate.bias, m.desc]
        for p in m.parameters():
            p.requires_grad = False
        for p in gp:
            p.requires_grad = True
        train(m, Xt_few_s, Yt_few_s, steps=60, seed=seed, params=gp)
        res["gate_only"].append(round(eval_m(m, Xt_ev_s, Yt_ev_s), 4))
        if seed == 0:
            params["gate_only"] = count_trainable(m, gp)

        m = MoE_adapter(ch=16)
        m.load_state_dict(src.state_dict(), strict=False)
        ap = list(m.adapter.parameters())
        for p in m.parameters():
            p.requires_grad = False
        for p in ap:
            p.requires_grad = True
        train(m, Xt_few_s, Yt_few_s, steps=60, seed=seed, params=ap)
        res["adapter_only"].append(round(eval_m(m, Xt_ev_s, Yt_ev_s), 4))
        if seed == 0:
            params["adapter_only"] = count_trainable(m, ap)

        m = MoE_lora(src, rank=4)
        lp = [p for p in m.parameters() if p.requires_grad]
        train(m, Xt_few_s, Yt_few_s, steps=60, seed=seed, params=lp)
        res["lora_only"].append(round(eval_m(m, Xt_ev_s, Yt_ev_s), 4))
        if seed == 0:
            params["lora_only"] = count_trainable(m, lp)

        m = MoE_lite(ch=16)
        m.load_state_dict(src.state_dict())
        train(m, Xt_few_s, Yt_few_s, steps=60, seed=seed)
        res["full_refit"].append(round(eval_m(m, Xt_ev_s, Yt_ev_s), 4))
        if seed == 0:
            params["full_refit"] = sum(p.numel() for p in m.parameters())
        print("seed", seed, {k: v[-1] for k, v in res.items()}, flush=True)

    summary = {}
    for m in methods:
        vals = res[m]
        summary[m] = {"vals": vals, "mean": round(float(np.mean(vals)), 4),
                      "std": round(float(np.std(vals)), 4)}
    out = {"protocol": "shear pressure -> turbulent radiative density; 16 few-shot windows; 32 eval; 5 seeds; 60 adaptation steps",
           "trainable_params": params, "results": summary,
           "runtime_sec": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("summary", summary, "params", params, flush=True)
    print("Saved ->", OUT, flush=True)


if __name__ == "__main__":
    main()
