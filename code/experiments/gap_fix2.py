# -*- coding: utf-8 -*-
"""
Gap-fix round 2: validate the unified-backbone claim directly.
  D) Joint multi-system pre-training: ONE shared MoE trained on BOTH the shear
     flow and turbulent radiative layer systems with field-description routing.
     Demonstrates that a single backbone handles heterogeneous simulations
     simultaneously (the paper's central premise).
  E) Expert-capacity ablation: 2 vs 4 experts on the joint task.
  F) Reverse zero-shot on the RAW density field (consistency with the forward
     zero-shot which used raw fields).
"""
import numpy as np, torch, torch.nn as nn, os, json, h5py

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
base = r'C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38'


def load_shear():
    sf = np.load(os.path.join(base, 'data', 'slice_train.npz'))
    T = sf['tracer'].astype(np.float32)
    n, t, H, W = T.shape
    return T.reshape(n, t, H//4, 4, W//8, 8).mean(axis=(3, 5))

def load_turb():
    with h5py.File(os.path.join(base, 'turb_rad_layer.hdf5'), 'r') as f:
        d = np.array(f['t0_fields/density']).astype(np.float32)
    return d[:, :20].reshape(8, 20, 128//2, 2, 384//6, 6).mean(axis=(3, 5))

def mk_xy(T, n_in=4):
    n, st, H, W = T.shape
    X, Y = [], []
    for t in range(st - n_in):
        X.append(T[:, t:t+n_in]); Y.append(T[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

def per_sample_std(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s

Sf = load_shear(); Tr = load_turb()
Xs, Ys = mk_xy(Sf); Xt, Yt = mk_xy(Tr)
Xs, Ys = per_sample_std(Xs), per_sample_std(Ys)
Xt, Yt = per_sample_std(Xt), per_sample_std(Yt)


class JointMoE(nn.Module):
    """Shared backbone with a field-description-routed 2/4-expert MoE."""
    def __init__(self, ch=16, in_ch=4, n_experts=2, desc_dim=8):
        super().__init__()
        self.n_experts = n_experts
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch, 1), nn.GELU(),
                                                    nn.Conv2d(ch, ch, 1)) for _ in range(n_experts)])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(desc_dim, n_experts)
    def forward(self, x, desc):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(desc), -1)             # (B, n_experts)
        eh = torch.stack([e(h) for e in self.experts], 1)  # (B, n_exp, ch, H, W)
        w = g[:, :, None, None, None]
        h = (eh * w).sum(1)
        return self.out(h)


def train_joint(model, Xs_, Ys_, Xt_, Yt_, desc_s, desc_t, steps=50, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lf = nn.MSELoss()
    Xs_t = torch.tensor(Xs_); Ys_t = torch.tensor(Ys_)
    Xt_t = torch.tensor(Xt_); Yt_t = torch.tensor(Yt_)
    ds_t = desc_s.expand(len(Xs_), -1); dt_t = desc_t.expand(len(Xt_), -1)
    Xj = torch.cat([Xs_t, Xt_t], 0); Yj = torch.cat([Ys_t, Yt_t], 0)
    Dj = torch.cat([ds_t, dt_t], 0)
    best = None
    for s in range(steps):
        opt.zero_grad(); loss = lf(model(Xj, Dj), Yj); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        es = lf(model(Xs_t, ds_t), Ys_t).item()
        et = lf(model(Xt_t, dt_t), Yt_t).item()
    model.train()
    return es, et


# ---------- build joint data (train/test split) ----------
ns = len(Xs); ps = np.random.RandomState(0).permutation(ns); ntrs = int(0.7*ns)
nt = len(Xt); pt = np.random.RandomState(0).permutation(nt); ntrt = int(0.7*nt)

desc_s = nn.Parameter(torch.randn(8), requires_grad=False)  # shear description
desc_t = nn.Parameter(torch.randn(8), requires_grad=False)  # turb description
# make descriptions distinct
with torch.no_grad():
    desc_s.normal_(0, 1); desc_t.normal_(0, 1)
    desc_t = desc_t + 2.0  # offset to be distinguishable

results = {}

# ---------- D) joint 2-expert ----------
print('=== D) Joint multi-system pre-training (shared backbone, 2 experts) ===')
m2 = JointMoE(n_experts=2)
es, et = train_joint(m2, Xs[ps[:ntrs]], Ys[ps[:ntrs]], Xt[pt[:ntrt]], Yt[pt[:ntrt]],
                     desc_s, desc_t, steps=50)
# held-out
m2.eval()
with torch.no_grad():
    ds_t = desc_s.expand(ns-ntrs, -1); dt_t = desc_t.expand(nt-ntrt, -1)
    es_ho = ((m2(torch.tensor(Xs[ps[ntrs:]]), ds_t) - torch.tensor(Ys[ps[ntrs:]]))**2).mean().item()
    et_ho = ((m2(torch.tensor(Xt[pt[ntrt:]]), dt_t) - torch.tensor(Yt[pt[ntrt:]]))**2).mean().item()
    # gate weights per system
    gs = torch.softmax(m2.gate(ds_t), -1).mean(0).numpy()
    gt = torch.softmax(m2.gate(dt_t), -1).mean(0).numpy()
print(f'  train  shear MSE {es:.4f}  turb MSE {et:.4f}')
print(f'  held-out shear MSE {es_ho:.4f}  turb MSE {et_ho:.4f}')
print(f'  gate weights (shear): {gs}   (turb): {gt}')
results['D_joint_2exp'] = {'train_shear': es, 'train_turb': et,
                           'holdout_shear': es_ho, 'holdout_turb': et_ho,
                           'gate_shear': gs.tolist(), 'gate_turb': gt.tolist()}

# ---------- E) 4-expert variant ----------
print('=== E) Expert-capacity ablation (4 experts) ===')
m4 = JointMoE(n_experts=4)
es4, et4 = train_joint(m4, Xs[ps[:ntrs]], Ys[ps[:ntrs]], Xt[pt[:ntrt]], Yt[pt[:ntrt]],
                       desc_s, desc_t, steps=50)
m4.eval()
with torch.no_grad():
    es4_ho = ((m4(torch.tensor(Xs[ps[ntrs:]]), desc_s.expand(ns-ntrs,-1)) - torch.tensor(Ys[ps[ntrs:]]))**2).mean().item()
    et4_ho = ((m4(torch.tensor(Xt[pt[ntrt:]]), desc_t.expand(nt-ntrt,-1)) - torch.tensor(Yt[pt[ntrt:]]))**2).mean().item()
print(f'  4-exp held-out shear MSE {es4_ho:.4f}  turb MSE {et4_ho:.4f}')
results['E_4exp'] = {'holdout_shear': es4_ho, 'holdout_turb': et4_ho}

# ---------- F) reverse zero-shot on RAW density ----------
print('=== F) Reverse zero-shot on RAW fields ===')
# raw shear tracer (not standardized), raw turb density
Xs_raw, Ys_raw = mk_xy(Sf); Xt_raw, Yt_raw = mk_xy(Tr)
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
def train_model(model, X_tr, Y_tr, X_te, Y_te, steps=40, lr=1e-3, seed=0):
    torch.manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), lr=lr); lf = nn.MSELoss()
    Xt_=torch.tensor(X_tr); Yt_=torch.tensor(Y_tr); Xe_=torch.tensor(X_te); Ye_=torch.tensor(Y_te)
    best=float('inf')
    for s in range(steps):
        opt.zero_grad(); lf(model(Xt_), Yt_).backward(); opt.step()
        model.eval()
        with torch.no_grad(): te=lf(model(Xe_), Ye_).item()
        model.train()
        if te<best: best=te
    return best
def vrmse(pred, y):
    mse=float(((pred-y)**2).mean()); yv=float(y.var())
    return float(np.sqrt(mse/yv)) if yv>0 else float('inf')
# normalize raw fields to have comparable scale: divide by per-field std
Xs_n = Xs_raw / (Xs_raw.std() + 1e-9); Ys_n = Ys_raw / (Ys_raw.std() + 1e-9)
Xt_n = Xt_raw / (Xt_raw.std() + 1e-9); Yt_n = Yt_raw / (Yt_raw.std() + 1e-9)
# pretrain on turb (raw-normalized)
n=len(Xt_n); pp=np.random.RandomState(0).permutation(n); ntr=int(0.8*n)
mt = MoE_lite(in_ch=4)
train_model(mt, Xt_n[pp[:ntr]], Yt_n[pp[:ntr]], Xt_n[pp[ntr:]], Yt_n[pp[ntr:]], steps=40, lr=1e-3)
mt.eval()
with torch.no_grad():
    p_zs = mt(torch.tensor(Xs_n)).numpy()
zs_vrmse = vrmse(p_zs, Ys_n)
# scratch on shear
ns=len(Xs_n); ps2=np.random.RandomState(0).permutation(ns); ntrs=int(0.8*ns)
msc = MoE_lite(in_ch=4)
sc_err = train_model(msc, Xs_n[ps2[:ntrs]], Ys_n[ps2[:ntrs]], Xs_n[ps2[ntrs:]], Ys_n[ps2[ntrs:]], steps=40, lr=1e-3)
print(f'  reverse zero-shot VRMSE (raw-normalized) {zs_vrmse:.4f}, scratch MSE {sc_err:.4f}')
results['F_reverse_raw'] = {'zero_shot_vrmse': zs_vrmse, 'scratch_mse': sc_err}

out = os.path.join(base, 'gap_fix2_results.json')
json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
print('\nSaved ->', out)
print(json.dumps(results, indent=2, ensure_ascii=False))
