# -*- coding: utf-8 -*-
"""B4 completion: measure the MODEL's field-integral (mass-like) conservation
error on predictions vs the data's own drift. If the model's integral error is
comparable to the data drift, it respects conservation to the data's accuracy.
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
    torch.manual_seed(seed); opt=torch.optim.AdamW(model.parameters(),lr=lr); lf=nn.MSELoss()
    Xt_=torch.tensor(X_tr); Yt_=torch.tensor(Y_tr); Xe_=torch.tensor(X_te); Ye_=torch.tensor(Y_te)
    for s in range(steps): opt.zero_grad(); lf(model(Xt_),Yt_).backward(); opt.step()
    model.eval()
    with torch.no_grad(): return model(Xe_).numpy()

def integral_err(pred, y):
    return float(abs(pred.sum()-y.sum())/(abs(y.sum())+1e-9))

Sf=load_shear(); Tr=load_turb()
Xs,Ys=mk_xy(Sf); Xt,Yt=mk_xy(Tr)
# raw fields for physical integrals
Xsr,Ysr=Xs,Ys; Xtr,Ytr=Xt,Yt
results={}
# train on raw-normalized fields (scale to unit std for training stability)
Xs_n = Xs/(Xs.std()+1e-9); Ys_n = Ys/(Ys.std()+1e-9)
Xt_n = Xt/(Xt.std()+1e-9); Yt_n = Yt/(Yt.std()+1e-9)
for name, X, Y, Xn, Yn in [('shear',Xsr,Ysr,Xs_n,Ys_n), ('turb',Xtr,Ytr,Xt_n,Yt_n)]:
    n=len(X); perm=np.random.RandomState(0).permutation(n); ntr=int(0.8*n)
    m=MoE_lite(in_ch=4)
    pred_n = train_in(m, Xn[perm[:ntr]], Yn[perm[:ntr]], Xn[perm[ntr:]], Yn[perm[ntr:]])
    pred = pred_n * (Y.std()+1e-9)  # back to physical scale
    y = Y[perm[ntr:]]
    # model prediction integral error
    model_ie = float(np.mean([integral_err(pred[i], y[i]) for i in range(len(y))]))
    # data's own drift (input last frame -> true next)
    x = X[perm[ntr:]]
    data_drift = float(np.mean([integral_err(x[i,-1], y[i]) for i in range(len(y))]))
    # also a mean-baseline predictor integral error
    mean_ie = float(np.mean([integral_err(np.full_like(y[i], y[i].mean()), y[i]) for i in range(len(y))]))
    print(f'{name}: model_integral_err {model_ie:.2e}  data_drift {data_drift:.2e}  mean_baseline {mean_ie:.2e}  n={len(y)}')
    results[name]={'model_integral_err':model_ie,'data_drift':data_drift,'mean_baseline_integral_err':mean_ie}

out=os.path.join(base,'gap_fix5b_results.json')
json.dump(results,open(out,'w'),indent=2,ensure_ascii=False)
print('\nSaved ->',out)
