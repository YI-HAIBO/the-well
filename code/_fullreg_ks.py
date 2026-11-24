# -*- coding: utf-8 -*-
"""#7 (Round 38, resumed): paper's FULL routing regularizer vs none, k=2..8,
with INCREMENTAL JSON saves after each k (survives interruption).

Regularizer (sol.py section C / paper default):
    loss + 0.1 * sum(frac**2) - 0.05 * mean(g**2),  frac = g.mean(0)
Protocol: JointMoE, learned desc, k systems -> k experts, 60 steps AdamW lr=1e-3,
80/20 split seed 0, MSE base loss. Same as Round 37 entropy run for comparability.
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routing_regularizer_by_family import SYSTEMS, IN_CH, STEPS, LR, mk_xy, std, load_system, measure

torch.set_num_threads(4)
BASE = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.expanduser("~/well_b3")
OUT = os.path.join(BASE, "sol4_routing_fullreg_byfamily.json")
W_BAL, W_SHARP = 0.1, 0.05


class JointMoE(nn.Module):
    def __init__(self, ch=16, in_ch=IN_CH, n_experts=8, desc_dim=8, n_sys=8):
        super().__init__()
        self.n_experts = n_experts
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, ch, 1), nn.GELU(), nn.Conv2d(ch, ch, 1))
            for _ in range(n_experts)])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(desc_dim, n_experts)
        self.desc = nn.Parameter(torch.randn(n_sys, desc_dim))

    def forward(self, x, desc_idx):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[desc_idx]), -1)
        eh = torch.stack([e(h) for e in self.experts], 1)
        w = g[:, :, None, None, None]
        return self.out((eh * w).sum(1)), g


def train(dat, systems, use_reg, seed=0):
    idx = {s: i for i, s in enumerate(systems)}
    k = len(systems)
    torch.manual_seed(seed); np.random.seed(seed)
    m = JointMoE(n_experts=k, n_sys=k)
    Xs, Ys, Ds = [], [], []
    for s in systems:
        X, Y, _, _ = dat[s]
        Xs.append(X); Ys.append(Y); Ds.append(np.full(len(X), idx[s], dtype=np.int64))
    Xc = torch.tensor(np.concatenate(Xs, 0)); Yc = torch.tensor(np.concatenate(Ys, 0))
    Dc = torch.tensor(np.concatenate(Ds, 0))
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        pred, g = m(Xc, Dc)
        loss = nn.functional.mse_loss(pred, Yc)
        if use_reg:
            frac = g.mean(0)
            loss = loss + W_BAL * (frac ** 2).sum() - W_SHARP * (g ** 2).mean()
        loss.backward(); opt.step()
    m.eval()
    return m


def main():
    dat = {}
    for s in SYSTEMS:
        X, Y = (std(a) for a in mk_xy(load_system(DATADIR, s).astype(np.float32)))
        n = len(X); perm = np.random.RandomState(0).permutation(n); ntr = int(0.8 * n)
        dat[s] = (X[perm[:ntr]], Y[perm[:ntr]], X[perm[ntr:]], Y[perm[ntr:]])

    results = {}
    if os.path.exists(OUT):
        try:
            results = json.load(open(OUT, encoding="utf-8")).get("by_family", {})
        except Exception:
            results = {}

    print(f"{'k':>2} | {'delta_none':>10} | {'delta_fullreg':>12} | {'ded_none':>8} | {'ded_fullreg':>11} | {'used_none':>9} | {'used_fullreg':>12}", flush=True)
    print("-" * 84, flush=True)
    for k in range(2, 9):
        key = f"k{k}"
        if key in results and "none" in results[key] and "full_regularzier" in results[key]:
            r0, r1 = results[key]["none"], results[key]["full_regularzier"]
            print(f"{k:>2} | (cached) {r0['delta_sample']:.4f} | {r1['delta_sample']:.4f}", flush=True)
            continue
        systems = SYSTEMS[:k]
        m0 = train(dat, systems, use_reg=False, seed=0)
        r0 = measure(m0, dat, systems)
        del m0
        m1 = train(dat, systems, use_reg=True, seed=0)
        r1 = measure(m1, dat, systems)
        del m1
        results[key] = {"systems": systems, "none": r0, "full_regularzier": r1,
                        "w_bal": W_BAL, "w_sharp": W_SHARP, "steps": STEPS, "lr": LR}
        json.dump({"regularizer": "0.1*sum(frac^2) - 0.05*mean(g^2)  (sol.py C / paper default)",
                   "by_family": results}, open(OUT, "w"), indent=2, ensure_ascii=False)
        print(f"{k:>2} | {r0['delta_sample']:>10.4f} | {r1['delta_sample']:>12.4f} "
              f"| {r0['systems_dedicated']:>4}/{k:<3} | {r1['systems_dedicated']:>5}/{k:<3} "
              f"| {r0['experts_used']:>5}/{k:<3} | {r1['experts_used']:>6}/{k:<3}  [saved]", flush=True)
    print("-" * 84, flush=True)
    print("Done. Saved ->", OUT, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("total %.1f min" % ((time.time() - t0) / 60), flush=True)
