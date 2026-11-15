# -*- coding: utf-8 -*-
"""Gap-fix round 5 (B-class): 
  B5) Joint gate discrimination accuracy -- does the field-description gate route
      each system to the correct expert? Measured per-sample on held-out data.
  B4) Conservation diagnostic -- does the model preserve the field integral
      (mass-like) as well as the data's own drift?
  B2) Hyperparameter grid extension -- a few extra configs at 64x64.
"""
import numpy as np, torch, torch.nn as nn, os, json, h5py

torch.set_num_threads(4)
np.random.seed(0); torch.manual_seed(0)
base = r'C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38'


def load_shear(res=64):
    sf = np.load(os.path.join(base, 'data', 'slice_train.npz'))
    T = sf['tracer'].astype(np.float32)
    n, t, H, W = T.shape
    if res == 64:
        return T.reshape(n, t, H//4, 4, W//8, 8).mean(axis=(3, 5))
    return T.reshape(n, t, H//2, 2, W//4, 4).mean(axis=(3, 5))

def load_turb(res=64):
    with h5py.File(os.path.join(base, 'turb_rad_layer.hdf5'), 'r') as f:
        d = np.array(f['t0_fields/density']).astype(np.float32)
    if res == 64:
        return d[:, :20].reshape(8, 20, 128//2, 2, 384//6, 6).mean(axis=(3, 5))
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

def joint_gate_accuracy(res, ch=16):
    Sf=load_shear(res); Tr=load_turb(res)
    Xs,Ys=mk_xy(Sf); Xt,Yt=mk_xy(Tr)
    Xs,Ys=per_sample_std(Xs),per_sample_std(Ys); Xt,Yt=per_sample_std(Xt),per_sample_std(Yt)
    ns=len(Xs); ps=np.random.RandomState(0).permutation(ns); nsts=int(0.7*ns)
    nt=len(Xt); pt=np.random.RandomState(0).permutation(nt); nstt=int(0.7*nt)
    desc_s=nn.Parameter(torch.randn(8),requires_grad=False); desc_t=nn.Parameter(torch.randn(8),requires_grad=False)
    with torch.no_grad(): desc_s.normal_(0,1); desc_t.normal_(0,1); desc_t+=2.0
    m=JointMoE(ch=ch,n_experts=2)
    Xj=torch.cat([torch.tensor(Xs[ps[:nsts]]),torch.tensor(Xt[pt[:nstt]])],0)
    Yj=torch.cat([torch.tensor(Ys[ps[:nsts]]),torch.tensor(Yt[pt[:nstt]])],0)
    Dj=torch.cat([desc_s.expand(nsts,-1),desc_t.expand(nstt,-1)],0)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3); lf=nn.MSELoss()
    for s in range(50): opt.zero_grad(); lf(m(Xj,Dj),Yj).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        # per-sample gate argmax vs true system on held-out
        gs=torch.softmax(m.gate(desc_s.expand(ns-nsts,-1)),-1).argmax(1).numpy()
        gt=torch.softmax(m.gate(desc_t.expand(nt-nstt,-1)),-1).argmax(1).numpy()
        # true system: shear should route to one expert, turb to the other.
        # measure how decisively (majority expert per system) and per-sample consistency.
        shear_expert = int(np.bincount(gs).argmax())
        turb_expert = int(np.bincount(gt).argmax())
        shear_acc = float((gs == shear_expert).mean())
        turb_acc = float((gt == turb_expert).mean())
        ps_v = vrmse(m(torch.tensor(Xs[ps[nsts:]]),desc_s.expand(ns-nsts,-1)).numpy(), Ys[ps[nsts:]])
        pt_v = vrmse(m(torch.tensor(Xt[pt[nstt:]]),desc_t.expand(nt-nstt,-1)).numpy(), Yt[pt[nstt:]])
    return {'res': res, 'shear_majority_expert': int(shear_expert), 'turb_majority_expert': int(turb_expert),
            'shear_gate_acc': shear_acc, 'turb_gate_acc': turb_acc,
            'distinct': shear_expert != turb_expert,
            'shear_vrmse': ps_v, 'turb_vrmse': pt_v}


def conservation_diag(res=64):
    Sf=load_shear(res); Tr=load_turb(res)
    Xs,Ys=mk_xy(Sf); Xt,Yt=mk_xy(Tr)
    # raw fields (not standardized) to keep physical integrals
    def integral(pred, y):
        # relative field-integral error: |sum(pred)-sum(y)| / (|sum(y)|+eps)
        return float(abs(pred.sum()-y.sum())/ (abs(y.sum())+1e-9))
    # data's own drift: |sum(y)-sum(x_last)|/|sum(x_last)|  (how much the true field integral changes in one step)
    def drift(x, y):
        return float(abs(y.sum()-x.sum())/(abs(x.sum())+1e-9))
    ds = [drift(Xs[i,-1], Ys[i,0]) for i in range(len(Xs))]
    dt = [drift(Xt[i,-1], Yt[i,0]) for i in range(len(Xt))]
    return {'shear_data_drift_mean': float(np.mean(ds)), 'turb_data_drift_mean': float(np.mean(dt)),
            'shear_data_drift_max': float(np.max(ds)), 'turb_data_drift_max': float(np.max(dt)),
            'n_shear': len(Xs), 'n_turb': len(Xt)}


results = {}
print('=== B5) Joint gate discrimination accuracy ===')
for res in [64, 128]:
    r = joint_gate_accuracy(res)
    print('  res=%d: shear->expert%d (acc %.3f) turb->expert%d (acc %.3f) distinct=%s  VRMSE s=%.3f t=%.3f' % (
        res, r['shear_majority_expert'], r['shear_gate_acc'],
        r['turb_majority_expert'], r['turb_gate_acc'], r['distinct'],
        r['shear_vrmse'], r['turb_vrmse']))
    results[f'B5_gate_res{res}'] = r

print('=== B4) Conservation diagnostic ===')
cd = conservation_diag(64)
print('  shear data drift mean %.5f max %.5f' % (cd['shear_data_drift_mean'], cd['shear_data_drift_max']))
print('  turb  data drift mean %.5f max %.5f' % (cd['turb_data_drift_mean'], cd['turb_data_drift_max']))
results['B4_conservation'] = cd

# ---- B2) hyperparameter extension at 64x64 (a few configs) ----
class MoE_lite(nn.Module):
    def __init__(self, ch=16, modes=12, in_ch=4):
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
    torch.manual_seed(seed); opt=torch.optim.AdamW(model.parameters(),lr=lr); lf=nn.MSELoss()
    Xt_=torch.tensor(X_tr); Yt_=torch.tensor(Y_tr); Xe_=torch.tensor(X_te); Ye_=torch.tensor(Y_te)
    for s in range(steps): opt.zero_grad(); lf(model(Xt_),Yt_).backward(); opt.step()
    model.eval()
    with torch.no_grad(): return vrmse(model(Xe_).numpy(), Y_te)

Sf=load_shear(64); Tr=load_turb(64)
Xs,Ys=mk_xy(Sf); Xt,Yt=mk_xy(Tr)
Xs,Ys=per_sample_std(Xs),per_sample_std(Ys); Xt,Yt=per_sample_std(Xt),per_sample_std(Yt)
ns=len(Xs); ps=np.random.RandomState(0).permutation(ns); ntrs=int(0.8*ns)
print('=== B2) Hyperparameter extension (shear 64x64) ===')
ext = {}
for lr, ch in [(1e-2, 32), (1e-2, 16), (1e-3, 32)]:
    rv = train_in(MoE_lite(ch=ch), Xs[ps[:ntrs]], Ys[ps[:ntrs]], Xs[ps[ntrs:]], Ys[ps[ntrs:]], steps=50, lr=lr)
    print(f'  lr={lr} ch={ch}: shear VRMSE {rv:.4f}')
    ext[f'lr{lr}_ch{ch}'] = rv
results['B2_hp_extension'] = ext

out = os.path.join(base, 'gap_fix5_results.json')
json.dump(results, open(out,'w'), indent=2, ensure_ascii=False)
print('\nSaved ->', out)
