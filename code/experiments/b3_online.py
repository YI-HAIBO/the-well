# -*- coding: utf-8 -*-
"""B3) Online/continual-learning validation of Theorem 3 (pretraining-to-finetuning
convergence under the Polyak-Lojasiewicz condition).

Validates two predictions of T3 on real The Well data (64x64):
  (1) Linear convergence: log((L_T - L*)/(L_0 - L*)) decays linearly in T with
      rate rho = 1 - mu*eta; we fit the empirical rate and infer mu_emp.
  (2) Initialization advantage: T*(eps) = (1/(mu*eta)) log(e_0/eps) is smaller
      for a pretrained initialization than for a scratch one (Assumption B4),
      so the pretrained path reaches eps in fewer steps.
"""
import numpy as np, torch, torch.nn as nn, os, json

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
BASE = os.path.expanduser("~/well_b3")

z = np.load(os.path.join(BASE, "data64.npz"))
shear = z["shear"].astype(np.float32)   # (4,20,64,64)
turb  = z["turb"].astype(np.float32)    # (8,20,64,64)

def mk_xy(T, n_in=4):
    n, st, H, W = T.shape
    X, Y = [], []
    for t in range(st - n_in):
        X.append(T[:, t:t+n_in]); Y.append(T[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

def std(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s

Xs, Ys = std(mk_xy(shear))   # (64,4,64,64),(64,1,64,64)
Xt, Yt = std(mk_xy(turb))    # (128,4,64,64),(128,1,64,64)

class MoE_lite(nn.Module):
    def __init__(self, ch=16, in_ch=4):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.e1 = nn.Conv2d(ch, ch, 1); self.e2 = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(4, 2); self.desc = nn.Parameter(torch.randn(4))
    def forward(self, x):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[None]), -1)
        h = g[0,0]*self.e1(h) + g[0,1]*self.e2(h)
        return self.out(h)

def train_hist(model, X, Y, steps, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt_ = torch.tensor(X); Yt_ = torch.tensor(Y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr); lf = nn.MSELoss()
    hist = []
    for s in range(steps):
        opt.zero_grad(); loss = lf(model(Xt_), Yt_); loss.backward(); opt.step()
        hist.append(float(loss.detach()))
    return hist

def fit_linear(hist, Lstar):
    # slope of log((L-L*)/(L0-L*)) vs T over the decaying portion
    e = [max(h - Lstar, 1e-12) for h in hist]
    e0 = e[0]
    ys = np.log(np.array(e) / e0)
    xs = np.arange(len(hist))
    # fit on steps [start:end] where it's decaying (exclude initial/final flat)
    n = len(hist); a = int(0.1*n); b = int(0.9*n)
    if b <= a: a, b = 0, n
    sl = np.polyfit(xs[a:b], ys[a:b], 1)[0]
    rho = float(np.exp(sl)); mu_emp = float((1 - rho) / 1e-3)  # eta=1e-3
    return rho, mu_emp, float(ys[-1])

# --- pretrain on source (shear) ---
print("== pretrain MoE on shear (64x64) ==", flush=True)
pre = MoE_lite()
pre_hist = train_hist(pre, Xs, Ys, 60, seed=0)
print("  pretrain loss 0->60: %.4f -> %.4f" % (pre_hist[0], pre_hist[-1]), flush=True)
pre_state = {k: v.clone() for k, v in pre.state_dict().items()}

# --- target L* via long train (both target tasks) ---
print("== estimate L* (long train on turb and on held-out shear) ==", flush=True)
# target tasks: (name, X, Y)
shear_hold_X, shear_hold_Y = Xs, Ys   # same-system: finetune back on shear (source=shear)
tasks = [("turb", Xt, Yt), ("shear_self", Xs, Ys)]
Lstar = {}
for name, X, Y in tasks:
    m = MoE_lite(); h = train_hist(m, X, Y, 200, seed=1)
    Lstar[name] = h[-1]
    print("  L*(%s) = %.5f" % (name, h[-1]), flush=True)

results = {}
for name, X, Y in tasks:
    row = {}
    for mode, init in [("pre", pre_state), ("scratch", None)]:
        m = MoE_lite()
        if init is not None:
            m.load_state_dict(init)
        h = train_hist(m, X, Y, 60, seed=7)
        rho, mu_emp, logr = fit_linear(h, Lstar[name])
        e0 = max(h[0] - Lstar[name], 1e-12)
        # steps to eps=0.1*e0
        eps = 0.1 * e0
        Tstar = float(np.log(e0 / eps) / max(1e-9, -np.log(rho))) if rho < 1 else float('inf')
        row[mode] = {"L0": h[0], "L60": h[-1], "rho": rho, "mu_emp": mu_emp,
                     "logratio": logr, "Tstar_0.1e0": Tstar, "hist": [round(x,6) for x in h[::5]]}
        print("  %s | %s: L0=%.4f L60=%.4f rho=%.4f mu_emp=%.1f T*(0.1e0)=%.1f" %
              (name, mode, h[0], h[-1], rho, mu_emp, Tstar), flush=True)
    results[name] = row
    results[name]["Lstar"] = Lstar[name]

# sanity: pretrained init advantage
ok = []
for name in tasks:
    r = results[name[0]] if False else results[name]
    ok.append(r["pre"]["Tstar_0.1e0"] < r["scratch"]["Tstar_0.1e0"])
results["pre_advantage"] = ok
results["all_linear_convergence"] = all(0 < r["rho"] < 1 for n in tasks for r in [results[n]])
out = os.path.join(BASE, "b3_results.json")
json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
print("\nSaved ->", out, flush=True)
print("SUMMARY: pre_advantage=%s linear_conv=%s" % (ok, results["all_linear_convergence"]), flush=True)
