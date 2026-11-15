# -*- coding: utf-8 -*-
"""Gap-fix round 4: capacity scaling for the hardest system (turbulent radiative
layer) at 128x128 and in joint mode, plus a stronger local baseline (UNet).
Goal: show that increasing capacity brings turb below the mean baseline, and
that the model beats a standard UNet baseline.
"""
import numpy as np, torch, torch.nn as nn, os, json, h5py, time

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
base = r'C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38'


def load_shear():
    sf = np.load(os.path.join(base, 'data', 'slice_train.npz'))
    T = sf['tracer'].astype(np.float32)
    n, t, H, W = T.shape
    return T.reshape(n, t, H//2, 2, W//4, 4).mean(axis=(3, 5))

def load_turb():
    with h5py.File(os.path.join(base, 'turb_rad_layer.hdf5'), 'r') as f:
        d = np.array(f['t0_fields/density']).astype(np.float32)
    return d[:, :20].reshape(8, 20, 128, 1, 384//3, 3).mean(axis=(3, 5))

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
    return float(np.sqrt(mse / yv)) if yv > 0 else float('inf')

Sf = load_shear(); Tr = load_turb()
Xs, Ys = mk_xy(Sf); Xt, Yt = mk_xy(Tr)
Xs, Ys = per_sample_std(Xs), per_sample_std(Ys)
Xt, Yt = per_sample_std(Xt), per_sample_std(Yt)
nt = len(Xt); pt = np.random.RandomState(0).permutation(nt); ntrt = int(0.8*nt)
Xt_tr, Yt_tr = Xt[pt[:ntrt]], Yt[pt[:ntrt]]
Xt_te, Yt_te = Xt[pt[ntrt:]], Yt[pt[ntrt:]]
results = {}

# ---------------- MoE_lite with capacity (ch) ----------------
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

def train_in(model, X_tr, Y_tr, X_te, Y_te, steps=50, lr=1e-3, seed=0):
    torch.manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), lr=lr); lf = nn.MSELoss()
    Xt_=torch.tensor(X_tr); Yt_=torch.tensor(Y_tr); Xe_=torch.tensor(X_te); Ye_=torch.tensor(Y_te)
    for s in range(steps):
        opt.zero_grad(); lf(model(Xt_), Yt_).backward(); opt.step()
    model.eval()
    with torch.no_grad(): p = model(Xe_)
    return vrmse(p.numpy(), Y_te)

# ---------------- UNet baseline ----------------
class UNet(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.GELU(), nn.Conv2d(base, base, 3, padding=1), nn.GELU())
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(base, base*2, 3, padding=1), nn.GELU(), nn.Conv2d(base*2, base*2, 3, padding=1), nn.GELU())
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec = nn.Sequential(nn.Conv2d(base*3, base, 3, padding=1), nn.GELU(), nn.Conv2d(base, 1, 3, padding=1))
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e2u = self.up(e2)
        if e2u.shape[-2:] != e1.shape[-2:]:
            e2u = nn.functional.interpolate(e2u, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        return self.dec(torch.cat([e1, e2u], 1))

print('=== Capacity scaling for turbulent radiative layer @128x128 ===')
for ch in [8, 16, 32]:
    t0=time.time()
    rv = train_in(MoE_lite(ch=ch, in_ch=4), Xt_tr, Yt_tr, Xt_te, Yt_te, steps=50)
    print(f'  ch={ch:2d}  turb VRMSE {rv:.4f}  ({time.time()-t0:.0f}s)')
    results[f'turb128_ch{ch}'] = rv

print('\n=== UNet baseline on turb @128x128 ===')
t0=time.time()
un = UNet(in_ch=4, base=32)
un_rv = train_in(un, Xt_tr, Yt_tr, Xt_te, Yt_te, steps=50)
print(f'  UNet turb VRMSE {un_rv:.4f}  ({time.time()-t0:.0f}s)  params={sum(p.numel() for p in un.parameters())}')
results['turb128_unet'] = un_rv

# MoE params for reference
mo = MoE_lite(ch=32, in_ch=4)
print(f'  MoE ch=32 params={sum(p.numel() for p in mo.parameters())}')

# ---------------- joint 128 with ch=32 ----------------
class JointMoE(nn.Module):
    def __init__(self, ch=16, in_ch=4, n_experts=2, desc_dim=8):
        super().__init__()
        self.n_experts=n_experts; self.in_conv=nn.Conv2d(in_ch,ch,1)
        self.experts=nn.ModuleList([nn.Sequential(nn.Conv2d(ch,ch,1),nn.GELU(),nn.Conv2d(ch,ch,1)) for _ in range(n_experts)])
        self.out=nn.Conv2d(ch,1,1); self.gate=nn.Linear(desc_dim,n_experts)
    def forward(self,x,desc):
        h=self.in_conv(x); g=torch.softmax(self.gate(desc),-1)
        eh=torch.stack([e(h) for e in self.experts],1); w=g[:,:,None,None,None]
        return self.out((eh*w).sum(1))

print('\n=== Joint multi-system @128x128 with ch=32 ===')
ns=len(Xs); ps=np.random.RandomState(0).permutation(ns); nsts=int(0.7*ns); nstt=int(0.7*nt)
desc_s=nn.Parameter(torch.randn(8),requires_grad=False); desc_t=nn.Parameter(torch.randn(8),requires_grad=False)
with torch.no_grad(): desc_s.normal_(0,1); desc_t.normal_(0,1); desc_t+=2.0
mj=JointMoE(ch=32,n_experts=2)
Xs_t=torch.tensor(Xs[ps[:nsts]]); Ys_t=torch.tensor(Ys[ps[:nsts]])
Xt_t=torch.tensor(Xt[pt[:nstt]]); Yt_t=torch.tensor(Yt[pt[:nstt]])
Xj=torch.cat([Xs_t,Xt_t],0); Yj=torch.cat([Ys_t,Yt_t],0)
Dj=torch.cat([desc_s.expand(nsts,-1),desc_t.expand(nstt,-1)],0)
opt=torch.optim.AdamW(mj.parameters(),lr=1e-3); lf=nn.MSELoss()
t0=time.time()
for s in range(50): opt.zero_grad(); lf(mj(Xj,Dj),Yj).backward(); opt.step()
mj.eval()
with torch.no_grad():
    psh=mj(torch.tensor(Xs[ps[nsts:]]),desc_s.expand(ns-nsts,-1)).numpy()
    ptb=mj(torch.tensor(Xt[pt[nstt:]]),desc_t.expand(nt-nstt,-1)).numpy()
    gs=torch.softmax(mj.gate(desc_s.expand(ns-nsts,-1)),-1).mean(0).numpy()
    gt=torch.softmax(mj.gate(desc_t.expand(nt-nstt,-1)),-1).mean(0).numpy()
jvs=vrmse(psh,Ys[ps[nsts:]]); jvt=vrmse(ptb,Yt[pt[nstt:]])
print(f'  joint ch32 shear VRMSE {jvs:.4f}  turb VRMSE {jvt:.4f}  ({time.time()-t0:.0f}s)')
print(f'  gates shear {gs}  turb {gt}')
results['joint128_ch32'] = {'shear_vrmse': jvs, 'turb_vrmse': jvt,
                            'gate_shear': gs.tolist(), 'gate_turb': gt.tolist()}

out = os.path.join(base, 'gap_fix4_results.json')
json.dump(results, open(out, 'w'), indent=2, ensure_ascii=False)
print('\nSaved ->', out)
print(json.dumps(results, indent=2))
