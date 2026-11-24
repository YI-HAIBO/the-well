# -*- coding: utf-8 -*-
"""#2 (Round 38): multi-system (k=8) capacity sweep into the >1e5-parameter regime.

Protocol: all 8 systems jointly, JointMoE (learned desc, k=8 experts), the paper's
full routing regularizer (multi-system default), AdamW lr=1e-3, 60 steps, minibatch
SGD (batch 128, seed 0) — uniform across capacities so the ch-trend is internally
consistent. Metric: per-system held-out VRMSE = sqrt(MSE/var(y_test)), mean over 8.
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routing_regularizer_by_family import SYSTEMS, IN_CH, mk_xy, std, load_system

torch.set_num_threads(min(8, max(4, (os.cpu_count() or 8) - 2)))
DATADIR = os.path.expanduser("~/well_b3")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c2_capacity_sweep_k8.json")
W_BAL, W_SHARP = 0.1, 0.05
STEPS, LR, BATCH = 60, 1e-3, 128
CHS = [16, 32, 64, 128]


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


def main():
    dat = {}
    for s in SYSTEMS:
        X, Y = (std(a) for a in mk_xy(load_system(DATADIR, s).astype(np.float32)))
        n = len(X); perm = np.random.RandomState(0).permutation(n); ntr = int(0.8 * n)
        dat[s] = (X[perm[:ntr]], Y[perm[:ntr]], X[perm[ntr:]], Y[perm[ntr:]])
    idx = {s: i for i, s in enumerate(SYSTEMS)}
    Xs, Ys, Ds = [], [], []
    for s in SYSTEMS:
        X, Y, _, _ = dat[s]
        Xs.append(X); Ys.append(Y); Ds.append(np.full(len(X), idx[s], dtype=np.int64))
    Xall = np.concatenate(Xs, 0); Yall = np.concatenate(Ys, 0); Dall = np.concatenate(Ds, 0)
    n = len(Xall)
    print("joint train samples:", n, "threads:", torch.get_num_threads(), flush=True)

    results = {}
    for ch in CHS:
        torch.manual_seed(0); np.random.seed(0)
        m = JointMoE(ch=ch, n_experts=8, n_sys=8)
        n_params = sum(p.numel() for p in m.parameters())
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
        rng = np.random.RandomState(0)
        t0 = time.time()
        for step in range(STEPS):
            mb = rng.choice(n, size=min(BATCH, n), replace=False)
            Xb = torch.tensor(Xall[mb]); Yb = torch.tensor(Yall[mb]); Db = torch.tensor(Dall[mb])
            opt.zero_grad()
            pred, g = m(Xb, Db)
            loss = nn.functional.mse_loss(pred, Yb)
            frac = g.mean(0)
            loss = loss + W_BAL * (frac ** 2).sum() - W_SHARP * (g ** 2).mean()
            loss.backward(); opt.step()
        m.eval()
        # per-system held-out VRMSE
        per = {}
        with torch.no_grad():
            for s in SYSTEMS:
                Xte, Yte = dat[s][2], dat[s][3]
                pred, _ = m(torch.tensor(Xte), torch.full((len(Xte),), idx[s], dtype=torch.long))
                pred = pred.numpy()
                mse = float(((pred - Yte) ** 2).mean())
                var = float(Yte.var())
                per[s] = float(np.sqrt(mse / var)) if var > 0 else float("inf")
            # routing concentration at eval
            gs = []
            for s in SYSTEMS:
                g = torch.softmax(m.gate(m.desc[idx[s]]), -1)
                gs.append(g.numpy())
            conf = np.array(gs); major = conf.argmax(1)
            experts_used = len(set(major.tolist()))
        mean_v = float(np.mean(list(per.values())))
        results[f"ch{ch}"] = {"params": n_params, "mean_vrmse": mean_v, "per_system": per,
                              "experts_used_majority": experts_used,
                              "s_per_step": round((time.time() - t0) / STEPS, 2)}
        print(f"ch={ch:>3}: params={n_params:>7,} meanVRMSE={mean_v:.4f} "
              f"experts_used={experts_used}/8 ({results[f'ch{ch}']['s_per_step']}s/step)", flush=True)
        json.dump(results, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("Saved ->", OUT, flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("total %.1f min" % ((time.time() - t0) / 60), flush=True)
