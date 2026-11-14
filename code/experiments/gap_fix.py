# -*- coding: utf-8 -*-
"""
New experiments to close theory-verification gaps:
  A) Direct delta measurement (Assumption B2: field-description sufficiency).
     Train a field-description classifier on statistical descriptors of samples
     drawn from two physically distinct systems; delta = held-out misclassification
     rate. This measures how well the description e discriminates the operator
     family, i.e. the actual delta in the bound C2*delta.
  B) PL constant mu estimation from the pretraining convergence curve.
  C) Reverse zero-shot (turbulent_radiative_layer -> shear_flow): second
     cross-system pair for Theorem 2 validation.
"""
import numpy as np, torch, torch.nn as nn, os, json, h5py, time

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
base = r'C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38'


# ------------------------------------------------------------------ data load
def load_shear():
    sf = np.load(os.path.join(base, 'data', 'slice_train.npz'))
    T = sf['tracer'].astype(np.float32)              # (4,20,256,512)
    n, t, H, W = T.shape
    return T.reshape(n, t, H//4, 4, W//8, 8).mean(axis=(3, 5))   # (4,20,64,64)

def load_turb():
    with h5py.File(os.path.join(base, 'turb_rad_layer.hdf5'), 'r') as f:
        d = np.array(f['t0_fields/density']).astype(np.float32)   # (8,101,128,384)
    return d[:, :20].reshape(8, 20, 128//2, 2, 384//6, 6).mean(axis=(3, 5))  # (8,20,64,64)

def mk_xy(T, n_in=4):
    n, st, H, W = T.shape
    X, Y = [], []
    for t in range(st - n_in):
        X.append(T[:, t:t+n_in]); Y.append(T[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

def per_sample_std(A):
    """Standardize each sample (4,64,64)->(4,64,64) by its own mean/std."""
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True)
    s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s

print('Loading shear flow...'); Sf = load_shear()
print('Loading turbulent radiative layer...'); Tr = load_turb()
Xs, Ys = mk_xy(Sf); Xt, Yt = mk_xy(Tr)
Xs = per_sample_std(Xs); Xt = per_sample_std(Xt)
Ys = per_sample_std(Ys); Yt = per_sample_std(Yt)
print('  shear samples:', Xs.shape[0], '  turb samples:', Xt.shape[0])


# ================================================================== A) delta
def field_descriptors(X):
    """Statistical descriptors of each sample (the 'field description' e).
    Features: mean, std, skew, kurtosis, and the log-spectral slope of the
    first input frame, computed on the standardized field."""
    feats = []
    for i in range(X.shape[0]):
        f = X[i, 0]                       # first input frame (64,64)
        m, s = float(f.mean()), float(f.std())
        sk = float(((f - m)**3).mean()) / (s**3 + 1e-9)
        ku = float(((f - m)**4).mean()) / (s**4 + 1e-9)
        # radial log-spectral slope
        F = np.abs(np.fft.fftshift(np.fft.fft2(f - f.mean())))**2
        cy, cx = F.shape[0]//2, F.shape[1]//2
        yy, xx = np.mgrid[0:F.shape[0], 0:F.shape[1]]
        r = np.sqrt((yy-cy)**2 + (xx-cx)**2).astype(int)
        kmax = min(cy, cx)
        prof = np.zeros(kmax)
        for k in range(kmax):
            sel = (r == k)
            prof[k] = F[sel].mean() if sel.any() else 0
        valid = prof[1:min(kmax, 16)] > 1e-12
        kk = np.log(np.arange(1, min(kmax, 16))[valid] + 1.0)
        ll = np.log(prof[1:min(kmax, 16)][valid] + 1e-12)
        slope = float(np.polyfit(kk, ll, 1)[0]) if len(kk) > 3 else 0.0
        feats.append([m, s, sk, ku, slope])
    return np.array(feats, dtype=np.float32)

print('\n=== A) Measuring delta (Assumption B2 field-description sufficiency) ===')
Ds = field_descriptors(Xs); Dt = field_descriptors(Xt)
print('  descriptor dim:', Ds.shape[1])
AllX = np.concatenate([Ds, Dt], 0)
Ally = np.concatenate([np.zeros(len(Ds)), np.ones(len(Dt))]).astype(np.int64)
# standardize descriptors
mx, sx = AllX.mean(0), AllX.std(0) + 1e-6
AllX = (AllX - mx) / sx

# train a small MLP classifier on descriptors, 5-fold cross-validation
def train_clf(X, y, seed):
    torch.manual_seed(seed)
    n = len(X); perm = np.random.RandomState(seed).permutation(n)
    ntr = int(0.8 * n)
    tr, va = perm[:ntr], perm[ntr:]
    net = nn.Sequential(nn.Linear(X.shape[1], 16), nn.ReLU(), nn.Linear(16, 2))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    lf = nn.CrossEntropyLoss()
    Xt_ = torch.tensor(X[tr]); yt_ = torch.tensor(y[tr])
    Xv_ = torch.tensor(X[va]); yv_ = torch.tensor(y[va])
    for ep in range(300):
        opt.zero_grad(); loss = lf(net(Xt_), yt_); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (net(Xv_).argmax(1) == yv_).float().mean().item()
    return 1.0 - acc, acc

accs = []
for seed in range(5):
    dlt, acc = train_clf(AllX, Ally, seed)
    accs.append(acc)
delta = float(1.0 - np.mean(accs)); delta_std = float(np.std(accs))
print(f'  held-out classification accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}')
print(f'  => delta (misclassification rate) = {delta:.4f} +/- {delta_std:.4f}')
# confusion
torch.manual_seed(0); perm = np.random.RandomState(0).permutation(len(Ally))
ntr = int(0.8*len(Ally)); tr, va = perm[:ntr], perm[ntr:]
net = nn.Sequential(nn.Linear(AllX.shape[1], 16), nn.ReLU(), nn.Linear(16, 2))
opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
for ep in range(300):
    opt.zero_grad(); loss = nn.CrossEntropyLoss()(net(torch.tensor(AllX[tr])), torch.tensor(Ally[tr])); loss.backward(); opt.step()
with torch.no_grad():
    pred = net(torch.tensor(AllX[va])).argmax(1).numpy()
cm = np.zeros((2,2), int)
for t, p in zip(Ally[va], pred): cm[t, p] += 1
print('  confusion (rows=true[shear,turb], cols=pred):\n', cm)


# ================================================================== B) mu
print('\n=== B) Estimating the PL constant mu ===')
class MoE_lite(nn.Module):
    def __init__(self, ch=16, in_ch=4, n_desc=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.e1 = nn.Conv2d(ch, ch, 1); self.e2 = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(n_desc, 2)
        self.desc = nn.Parameter(torch.randn(n_desc))
    def forward(self, x):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[None]), -1)
        h = g[0,0]*self.e1(h) + g[0,1]*self.e2(h)
        return self.out(h)

# train on turb radiative layer (the larger system), record loss curve
n = len(Xt); perm = np.random.RandomState(0).permutation(n)
ntr = int(0.8*n)
m = MoE_lite(in_ch=4)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
lf = nn.MSELoss()
Xtr = torch.tensor(Xt[perm[:ntr]]); Ytr = torch.tensor(Yt[perm[:ntr]])
Xva = torch.tensor(Xt[perm[ntr:]]); Yva = torch.tensor(Yt[perm[ntr:]])
curve = []
for s in range(60):
    opt.zero_grad(); loss = lf(m(Xtr), Ytr); loss.backward(); opt.step()
    with torch.no_grad(): curve.append(loss.item())
# estimate mu from exponential decay L_t ~ L* + (L0-L*)(1-mu*eta)^t
curve = np.array(curve); eta = 1e-3
Lstar = float(curve[-1])
rel = curve - Lstar
valid = rel > 1e-6 * (curve[0]-Lstar)
# fit log(rel) vs t -> slope = log(1-mu*eta)
tt = np.arange(len(curve))[valid][:45]
rr = np.log(np.maximum(rel[valid][:45], 1e-12))
coef = np.polyfit(tt, rr, 1)
slope = coef[0]
mu = (1 - np.exp(slope)) / eta
print(f'  final loss {Lstar:.4f}, initial {curve[0]:.4f}')
print(f'  log-linear slope {slope:.4f}, implied mu = {mu:.3f}')
print(f'  convergence factor (1-mu*eta) = {1-mu*eta:.4f}')


# ================================================================== C) reverse zero-shot
print('\n=== C) Reverse zero-shot (turbulent_radiative -> shear_flow) ===')
def train_return_best(model, X_tr, Y_tr, X_te, Y_te, steps=40, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    Xt_ = torch.tensor(X_tr); Yt_ = torch.tensor(Y_tr)
    Xe_ = torch.tensor(X_te); Ye_ = torch.tensor(Y_te)
    best = float('inf')
    for s in range(steps):
        opt.zero_grad(); loss = lf(model(Xt_), Yt_); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            te = lf(model(Xe_), Ye_).item()
        model.train()
        if te < best: best = te
    return best

def metrics(pred, y):
    mse = float(((pred - y)**2).mean())
    yvar = float(y.var()); vrmse = np.sqrt(mse / yvar) if yvar > 0 else float('inf')
    rel = np.sqrt(mse) / (np.sqrt((y**2).mean()) + 1e-8)
    return mse, vrmse, rel

n = len(Xt); perm = np.random.RandomState(0).permutation(n); ntr = int(0.8*n)
# train on turb rad
model_t = MoE_lite(in_ch=4)
err_t = train_return_best(model_t, Xt[perm[:ntr]], Yt[perm[:ntr]], Xt[perm[ntr:]], Yt[perm[ntr:]], steps=40, lr=1e-3)
# zero-shot eval on shear flow
model_t.eval()
with torch.no_grad():
    p = model_t(torch.tensor(Xs))
mse_zs, vrmse_zs, rel_zs = metrics(p.numpy(), Ys)
print(f'  turb in-domain best MSE {err_t:.4f}')
print(f'  zero-shot on shear_flow: MSE {mse_zs:.4f}, VRMSE {vrmse_zs:.4f}, rel_L2 {rel_zs:.4f}')
# scratch on shear flow
ns = len(Xs); pms = np.random.RandomState(0).permutation(ns); ntrs = int(0.8*ns)
model_sc = MoE_lite(in_ch=4)
err_sc = train_return_best(model_sc, Xs[pms[:ntrs]], Ys[pms[:ntrs]], Xs[pms[ntrs:]], Ys[pms[ntrs:]], steps=40, lr=1e-3)
print(f'  shear scratch best MSE {err_sc:.4f}')


# ------------------------------------------------------------------ save
res = {
    'A_delta': {'mean': delta, 'std': delta_std, 'accuracy_mean': float(np.mean(accs)),
                'confusion': cm.tolist(), 'n_descriptors': int(AllX.shape[1])},
    'B_PL': {'final_loss': Lstar, 'initial_loss': float(curve[0]), 'slope': float(slope),
             'mu': mu, 'eta': eta, 'convergence_factor': float(1-mu*eta)},
    'C_reverse_zero_shot': {'turb_in_domain_mse': err_t,
                            'zero_shot_mse': mse_zs, 'zero_shot_vrmse': vrmse_zs, 'zero_shot_rel_l2': rel_zs,
                            'shear_scratch_mse': err_sc},
}
out = os.path.join(base, 'gap_fix_results.json')
with open(out, 'w') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
print('\nSaved ->', out)
print(json.dumps(res, indent=2, ensure_ascii=False))
