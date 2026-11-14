# -*- coding: utf-8 -*-
"""Gap-fix round 3: run key experiments at 128x128 resolution.
  - In-domain shear flow at 128x128 (compare to 64x64).
  - In-domain turbulent radiative layer at 128x128.
  - Joint multi-system pre-training at 128x128.
Reports VRMSE (paper's main metric) for comparability.
"""
import numpy as np, torch, torch.nn as nn, os, json, h5py, time

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
base = r'C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38'
RES = 128


def load_shear():
    sf = np.load(os.path.join(base, 'data', 'slice_train.npz'))
    T = sf['tracer'].astype(np.float32)                    # (4,20,256,512)
    n, t, H, W = T.shape
    return T.reshape(n, t, H//2, 2, W//4, 4).mean(axis=(3, 5))   # -> (4,20,128,128)

def load_turb():
    with h5py.File(os.path.join(base, 'turb_rad_layer.hdf5'), 'r') as f:
        d = np.array(f['t0_fields/density']).astype(np.float32)  # (8,101,128,384)
    return d[:, :20].reshape(8, 20, 128, 1, 384//3, 3).mean(axis=(3, 5))  # -> (8,20,128,128)

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

def vrmse(pred, y):
    mse = float(((pred - y)**2).mean()); yv = float(y.var())
    return float(np.sqrt(mse / yv)) if yv > 0 else float('inf'), float(np.sqrt(mse/(y**2).mean()+1e-12))


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

class JointMoE(nn.Module):
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
        g = torch.softmax(self.gate(desc), -1)
        eh = torch.stack([e(h) for e in self.experts], 1)
        w = g[:, :, None, None, None]
        return self.out((eh * w).sum(1))

def train_in(model, X_tr, Y_tr, X_te, Y_te, steps=50, lr=1e-3, seed=0):
    torch.manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), lr=lr); lf = nn.MSELoss()
    Xt_=torch.tensor(X_tr); Yt_=torch.tensor(Y_tr); Xe_=torch.tensor(X_te); Ye_=torch.tensor(Y_te)
    for s in range(steps):
        opt.zero_grad(); lf(model(Xt_), Yt_).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p = model(Xe_)
    return vrmse(p.numpy(), Y_te), p.numpy()

Sf = load_shear(); Tr = load_turb()
print('shear 128:', Sf.shape, ' turb 128:', Tr.shape)
Xs, Ys = mk_xy(Sf); Xt, Yt = mk_xy(Tr)
Xs, Ys = per_sample_std(Xs), per_sample_std(Ys)
Xt, Yt = per_sample_std(Xt), per_sample_std(Yt)
results = {'resolution': RES, 'n_shear': len(Xs), 'n_turb': len(Xt)}

# ---------- in-domain shear 128 ----------
print('\n=== In-domain shear flow @128 ===')
ns = len(Xs); ps = np.random.RandomState(0).permutation(ns); ntrs = int(0.8*ns)
m = MoE_lite(in_ch=4)
t0=time.time()
(rv, rl), _ = train_in(m, Xs[ps[:ntrs]], Ys[ps[:ntrs]], Xs[ps[ntrs:]], Ys[ps[ntrs:]], steps=50)
print(f'  shear VRMSE {rv:.4f} rel_L2 {rl:.4f}  ({time.time()-t0:.0f}s)')
results['in_domain_shear_128'] = {'vrmse': rv, 'rel_l2': rl}

# ---------- in-domain turb 128 ----------
print('\n=== In-domain turbulent radiative @128 ===')
nt = len(Xt); pt = np.random.RandomState(0).permutation(nt); ntrt = int(0.8*nt)
m2 = MoE_lite(in_ch=4)
t0=time.time()
(rv2, rl2), _ = train_in(m2, Xt[pt[:ntrt]], Yt[pt[:ntrt]], Xt[pt[ntrt:]], Yt[pt[ntrt:]], steps=50)
print(f'  turb VRMSE {rv2:.4f} rel_L2 {rl2:.4f}  ({time.time()-t0:.0f}s)')
results['in_domain_turb_128'] = {'vrmse': rv2, 'rel_l2': rl2}

# ---------- joint 128 ----------
print('\n=== Joint multi-system @128 ===')
nsts = int(0.7*ns); nstt = int(0.7*nt)
desc_s = nn.Parameter(torch.randn(8), requires_grad=False)
desc_t = nn.Parameter(torch.randn(8), requires_grad=False)
with torch.no_grad():
    desc_s.normal_(0, 1); desc_t.normal_(0, 1); desc_t += 2.0
mj = JointMoE(n_experts=2)
Xs_t=torch.tensor(Xs[ps[:nsts]]); Ys_t=torch.tensor(Ys[ps[:nsts]])
Xt_t=torch.tensor(Xt[pt[:nstt]]); Yt_t=torch.tensor(Yt[pt[:nstt]])
Xj=torch.cat([Xs_t, Xt_t],0); Yj=torch.cat([Ys_t, Yt_t],0)
Dj=torch.cat([desc_s.expand(nsts,-1), desc_t.expand(nstt,-1)],0)
opt=torch.optim.AdamW(mj.parameters(), lr=1e-3); lf=nn.MSELoss()
t0=time.time()
for s in range(50):
    opt.zero_grad(); lf(mj(Xj, Dj), Yj).backward(); opt.step()
mj.eval()
with torch.no_grad():
    psh = mj(torch.tensor(Xs[ps[nsts:]]), desc_s.expand(ns-nsts,-1)).numpy()
    ptb = mj(torch.tensor(Xt[pt[nstt:]]), desc_t.expand(nt-nstt,-1)).numpy()
    gs = torch.softmax(mj.gate(desc_s.expand(ns-nsts,-1)), -1).mean(0).numpy()
    gt = torch.softmax(mj.gate(desc_t.expand(nt-nstt,-1)), -1).mean(0).numpy()
vs, ls = vrmse(psh, Ys[ps[nsts:]]); vt, lt = vrmse(ptb, Yt[pt[nstt:]])
print(f'  joint shear VRMSE {vs:.4f} rel_L2 {ls:.4f}  gate {gs}')
print(f'  joint turb  VRMSE {vt:.4f} rel_L2 {lt:.4f}  gate {gt}')
print(f'  ({time.time()-t0:.0f}s)')
results['joint_128'] = {'shear_vrmse': vs, 'shear_rel_l2': ls, 'turb_vrmse': vt, 'turb_rel_l2': lt,
                        'gate_shear': gs.tolist(), 'gate_turb': gt.tolist()}

out = os.path.join(base, 'gap_fix3_results.json')
json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
print('\nSaved ->', out)
print(json.dumps(results, indent=2, ensure_ascii=False))
