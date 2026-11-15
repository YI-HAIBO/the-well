# -*- coding: utf-8 -*-
"""#20 tokenizer aggregation error vs token count; #10 turb learnability via log transform."""
import numpy as np, torch, torch.nn as nn, os, json, time
torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
BASE = os.path.expanduser("~/well_b3")

# ---------- native shear 256x256 ----------
raw = open(os.path.join(BASE, "_sh.raw"), "rb").read()
T = np.frombuffer(raw, dtype=np.float32).reshape(4, 20, 256, 512)
S = T.reshape(4, 20, 256, 1, 512 // 2, 2).mean(axis=(3, 5))  # (4,20,256,256)
print("native 256x256 shear", S.shape, flush=True)

def block_mean(A, R):
    """downsample (4,20,256,256) -> (4,20,R,R) by block mean, then upsample back to 256."""
    f = 256 // R
    d = A.reshape(4, 20, R, f, R, f).mean(axis=(3, 5))
    up = np.repeat(np.repeat(d, f, axis=2), f, axis=3)
    return d, up

print("\n===== #20: tokenizer (block-mean) aggregation error vs token count =====", flush=True)
rows = []
for R in [32, 64, 128, 256]:
    d, up = block_mean(S, R)
    err = np.sqrt(((up - S) ** 2).mean()) / np.sqrt((S ** 2).mean())
    rows.append((R, R * R, float(err)))
    print("  R=%3d  tokens=%6d  relL2 reconstruction error=%.4f" % (R, R*R, err), flush=True)
fit = [(t, e) for (R, t, e) in rows if e > 0]
beta = None
if len(fit) >= 2:
    lx = np.log([t for t, _ in fit]); ly = np.log([e for _, e in fit])
    beta = -np.polyfit(lx, ly, 1)[0]
    print("  fitted decay exponent beta = %.3f  (err ~ tokens^-%.3f)" % (beta, beta), flush=True)

# ---------- #10 turb learnability: log transform ----------
def mk_xy(A, n_in=4):
    n, st, H, W = A.shape; X, Y = [], []
    for t in range(st - n_in):
        X.append(A[:, t:t+n_in]); Y.append(A[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)
def std(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s
def vrmse(pred, y):
    mse = float(((pred - y) ** 2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float('inf')
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
turb = np.load(os.path.join(BASE, "data64.npz"))["turb"].astype(np.float32)
print("\nturb raw: mean=%.4f std=%.4f max=%.4f" % (turb.mean(), turb.max(), turb.std()), flush=True)
print("===== #10: turb 64^2 learnability (std-normalized vs log-transform) =====", flush=True)
Xs, Ys = (std(a) for a in mk_xy(turb))  # baseline per-sample std normalization
Xs = torch.tensor(Xs); Ys = torch.tensor(Ys)
perm = np.random.RandomState(0).permutation(len(Xs)); ntr = int(0.8*len(Xs))
res = {}
for name, Xt, Yt in [("std", Xs, Ys)]:
    torch.manual_seed(0)
    m = MoE_lite(ch=16); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
    for s in range(40): opt.zero_grad(); lf(m(Xt[perm[:ntr]]), Yt[perm[:ntr]]).backward(); opt.step()
    m.eval()
    with torch.no_grad(): pred = m(Xt[perm[ntr:]]).numpy()
    res[name] = vrmse(pred, Yt[perm[ntr:]].numpy())
    print("  %-8s VRMSE=%.3f" % (name, res[name]), flush=True)
# log transform: log(1+|x|) then per-sample std
Lt = np.log1p(np.abs(turb))
Xl, Yl = (std(a) for a in mk_xy(Lt))
Xl = torch.tensor(Xl); Yl = torch.tensor(Yl)
torch.manual_seed(0)
m = MoE_lite(ch=16); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
for s in range(40): opt.zero_grad(); lf(m(Xl[perm[:ntr]]), Yl[perm[:ntr]]).backward(); opt.step()
m.eval()
with torch.no_grad(): pred = m(Xl[perm[ntr:]]).numpy()
res["log"] = vrmse(pred, Yl[perm[ntr:]].numpy())
print("  %-8s VRMSE=%.3f" % ("log", res["log"]), flush=True)
print("  => log-transform VRMSE %.3f vs std %.3f" % (res["log"], res["std"]), flush=True)

out = {"20_tokenizer": {"rows": [(int(R), int(t), float(e)) for (R, t, e) in rows], "beta": float(beta) if beta is not None else None},
       "10_turb_log": res}
json.dump(out, open(os.path.join(BASE, "sol5_results.json"), "w"), indent=2, ensure_ascii=False)
print("\nSaved ->", os.path.join(BASE, "sol5_results.json"), flush=True)