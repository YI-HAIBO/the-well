# -*- coding: utf-8 -*-
"""Solutions batch: execute P0/P1 fixes from the limitations roadmap on the sandbox.
mode a: A(#26 lr) B(#19/5 zs-normalized) C(#8 route) D(#1 conservation) E(#27 LO-trajectory)
mode b: F(#28 seeds) G(#33 alpha) H(#21/35 D_op) I(#3/23 gate delta)
"""
import numpy as np, torch, torch.nn as nn, os, json, sys, time
torch.set_num_threads(4)
BASE = os.path.expanduser("~/well_b3")
mode = sys.argv[1] if len(sys.argv) > 1 else "a"
R = {}

def load(f):
    d = np.load(os.path.join(BASE, f + ".npz"))
    return d[d.keys()[0]].astype(np.float32)

def mk_xy(A, n_in=4):
    n, st, H, W = A.shape
    X, Y = [], []
    for t in range(st - n_in):
        X.append(A[:, t:t+n_in]); Y.append(A[:, t+n_in:t+n_in+1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)

def pstd(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s

def vrmse(p, y):
    mse = float(((p - y) ** 2).mean()); yv = float(y.var())
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
        h = g[0, 0] * self.e1(h) + g[0, 1] * self.e2(h)
        return self.out(h)

def train_in(Xtr, Ytr, Xte, Yte, ch=16, lr=1e-3, steps=40, seed=0, cons_beta=0.0):
    torch.manual_seed(seed)
    m = MoE_lite(ch=ch); opt = torch.optim.AdamW(m.parameters(), lr=lr)
    Xt_ = torch.tensor(Xtr); Yt_ = torch.tensor(Ytr)
    lf = nn.MSELoss()
    for s in range(steps):
        opt.zero_grad()
        pred = m(Xt_)
        loss = lf(pred, Yt_)
        if cons_beta > 0:
            sp = pred.sum(dim=(2, 3)); sy = Yt_.sum(dim=(2, 3))
            loss = loss + cons_beta * ((sp - sy) ** 2 / (sy ** 2 + 1e-9)).mean()
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return vrmse(m(torch.tensor(Xte)).numpy(), Yte), m

# ---------------- load once ----------------
shear = load("data64")["shear"] if False else None
S = np.load(os.path.join(BASE, "data64.npz")); T = S["shear"].astype(np.float32); U = S["turb"].astype(np.float32)
G = load("gray64")["gray"]; E = load("euler64")["euler"]; V = load("visco64")["visco"]
Xs, Ys = (pstd(a) for a in mk_xy(T))
Xt, Yt = (pstd(a) for a in mk_xy(U))
Xg, Yg = (pstd(a) for a in mk_xy(G))
Xe, Ye = (pstd(a) for a in mk_xy(E))
Xv, Yv = (pstd(a) for a in mk_xy(V))
ns = len(Xs)
ps = np.random.RandomState(0).permutation(ns); ntr = int(0.8 * ns)
nt = len(Xt); pt = np.random.RandomState(0).permutation(nt)
nstt = int(0.7 * nt); nss = int(0.7 * ns)

if mode == "a":
    # ---- A. #26 lr/ch sweep (shear 64^2) ----
    print("=== A #26 lr/ch sweep (shear 64x64) ===", flush=True)
    cfg = {}
    for lr, ch in [(1e-3, 16), (1e-2, 8), (1e-2, 16), (1e-2, 32), (3e-3, 16)]:
        vs = [train_in(Xs[ps[:ntr]], Ys[ps[:ntr]], Xs[ps[ntr:]], Ys[ps[ntr:]], ch=ch, lr=lr, seed=s) [0] for s in range(3)]
        cfg["lr%s_ch%s" % (lr, ch)] = {"mean": float(np.mean(vs)), "std": float(np.std(vs)), "per_seed": [round(v, 4) for v in vs]}
        print("  lr=%s ch=%d: %.4f +/- %.4f" % (lr, ch, np.mean(vs), np.std(vs)), flush=True)
    R["A_lr_sweep"] = cfg

    # ---- B. #19/5 zero-shot matrix per-sample normalized (shear/turb/gray) ----
    print("=== B #19/5 zero-shot matrix (normalized) ===", flush=True)
    sets = {"shear": (Xs, Ys), "turb": (Xt, Yt), "gray": (Xg, Yg)}
    inn = {}; zs = {}
    for nm, (X, Y) in sets.items():
        inn[nm] = train_in(X[:ntr], Y[:ntr], X[ntr:], Y[ntr:], ch=16, lr=1e-3, steps=40, seed=0)[0]
        print("  in-domain %s: %.4f" % (nm, inn[nm]), flush=True)
    for src, (Xs_, Ys_) in sets.items():
        for tgt, (Xt_, Yt_) in sets.items():
            if src == tgt: continue
            v = train_in(Xs_[:ntr], Ys_[:ntr], Xt_[ntr:], Yt_[ntr:], ch=16, lr=1e-3, steps=40, seed=0)[0]
            zs["%s->%s" % (src, tgt)] = v
            print("  zero-shot %s->%s: %.4f" % (src, tgt, v), flush=True)
    R["B_zs_normalized"] = {"in_domain": inn, "zero_shot": zs}

    # ---- C. #8 routing: JointMoE 4 experts + load-balance/diversity ----
    print("=== C #8 routing (4 experts + load-balance) ===", flush=True)
    class JMoE(nn.Module):
        def __init__(self, ch=16, ne=4):
            super().__init__()
            self.ne = ne; self.in_conv = nn.Conv2d(4, ch, 1)
            self.experts = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(ne)])
            self.out = nn.Conv2d(ch, 1, 1); self.gate = nn.Linear(8, ne)
        def forward(self, x, d):
            h = self.in_conv(x); g = torch.softmax(self.gate(d), -1)
            eh = torch.stack([e(h) for e in self.experts], 1)
            return self.out((eh * g[:, :, None, None, None]).sum(1)), g
    nt = len(Xt); pt = np.random.RandomState(0).permutation(nt); nstt = int(0.7 * nt)
    nss = int(0.7 * ns)
    ds = nn.Parameter(torch.randn(8), requires_grad=False); dt_ = nn.Parameter(torch.randn(8), requires_grad=False)
    with torch.no_grad(): ds.normal_(0, 1); dt_.normal_(0, 1); dt_ += 2.0
    Xj = torch.cat([torch.tensor(Xs[ps[:nss]]), torch.tensor(Xt[pt[:nstt]])], 0)
    Yj = torch.cat([torch.tensor(Ys[ps[:nss]]), torch.tensor(Yt[pt[:nstt]])], 0)
    Dj = torch.cat([ds.expand(nss, -1), dt_.expand(nstt, -1)], 0)
    m = JMoE(ne=4); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
    for s in range(50):
        opt.zero_grad(); pred, g = m(Xj, Dj)
        loss = lf(pred, Yj)
        # load-balance + route-diversity regularizer
        frac = g.mean(0); loss = loss + 0.1 * (frac ** 2).sum() - 0.05 * (g ** 2).mean()
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        _, gs_ = m(torch.tensor(Xs[ps[nss:]]), ds.expand(ns - nss, -1))
        _, gt_ = m(torch.tensor(Xt[pt[nstt:]]), dt_.expand(nt - nstt, -1))
        occ_s = gs_.mean(0).numpy(); occ_t = gt_.mean(0).numpy()
        arg_s = gs_.argmax(1).numpy(); arg_t = gt_.argmax(1).numpy()
    maj_s = int(np.bincount(arg_s).argmax()); maj_t = int(np.bincount(arg_t).argmax())
    R["C_routing"] = {"expert_occupancy_shear": [round(float(x), 3) for x in occ_s],
                      "expert_occupancy_turb": [round(float(x), 3) for x in occ_t],
                      "majority_expert_shear": maj_s, "majority_expert_turb": maj_t,
                      "distinct": maj_s != maj_t,
                      "shear_frac_majority": float((arg_s == maj_s).mean()),
                      "turb_frac_majority": float((arg_t == maj_t).mean())}
    print("  occ shear %s turb %s distinct=%s" % (np.round(occ_s, 2), np.round(occ_t, 2), maj_s != maj_t), flush=True)

    # ---- D. #1 conservation regularizer on turb (raw fields) ----
    print("=== D #1 conservation reg (turb raw) ===", flush=True)
    Xr, Yr = mk_xy(U)  # raw, keep integrals
    Xrn, Yrn = pstd(Xr), pstd(Yr)
    ntr2 = int(0.8 * len(Xrn)); pr = np.random.RandomState(1).permutation(len(Xrn))
    cons = {}
    for beta in [0.0, 1e-2, 1e-1, 1.0]:
        v, m = train_in(Xrn[pr[:ntr2]], Yrn[pr[:ntr2]], Xrn[pr[ntr2:]], Yrn[pr[ntr2:]], ch=16, lr=1e-3, steps=40, seed=0, cons_beta=beta)
        # measure conservation rel error on raw held-out (unnormalize)
        with torch.no_grad():
            pred = m(torch.tensor(Xrn[pr[ntr2:]])).numpy()
        yr = Yr[pr[ntr2:]]
        # reconstruct pred in raw scale: pred is standardized; multiply by y-std (per-sample) then add y-mean
        pstd_y = Yrn[pr[ntr2:]].std(axis=(1, 2)) + 1e-6; mean_y = Yr[pr[ntr2:]].mean(axis=(1, 2), keepdims=True)
        pred_raw = pred * pstd_y[:, None, None] + mean_y
        c_err = float(np.abs(pred_raw.sum(axis=(1, 2)) - yr.sum(axis=(1, 2))).mean() / (np.abs(yr.sum(axis=(1, 2))).mean() + 1e-9))
        cons["beta%s" % beta] = {"vrmse": v, "conservation_rel_err": c_err}
        print("  beta=%.2f: VRMSE %.4f cons_err %.5f" % (beta, v, c_err), flush=True)
    R["D_conservation"] = cons

    # ---- E. #27 leave-one-trajectory-out (shear 4 traj) ----
    print("=== E #27 LO-trajectory (shear) ===", flush=True)
    # samples per trajectory = st - n_in = 16
    per = 16
    lo = {}
    for k in range(4):
        idx = np.arange(ns)
        te = np.arange(k * per, (k + 1) * per); tr = np.setdiff1d(idx, te)
        v = train_in(Xs[tr], Ys[tr], Xs[te], Ys[te], ch=16, lr=1e-3, steps=40, seed=0)[0]
        lo["traj%d" % k] = v
        print("  LO-trajectory %d: %.4f" % (k, v), flush=True)
    R["E_LO_trajectory"] = {"per_fold": lo, "mean": float(np.mean(list(lo.values()))), "max": float(np.max(list(lo.values())))}

    out = os.path.join(BASE, "sol_a_results.json")
    json.dump(R, open(out, "w"), indent=2, ensure_ascii=False)
    print("Saved ->", out, flush=True)

else:
    # ---- F. #28 seeds: main single-set + gated vs ungated, Welch t ----
    print("=== F #28 seeds (main + gate) ===", flush=True)
    ntr = int(0.8 * ns)
    main = [train_in(Xs[ps[:ntr]], Ys[ps[:ntr]], Xs[ps[ntr:]], Ys[ps[ntr:]], ch=16, lr=1e-3, steps=40, seed=s)[0] for s in range(10)]
    R["F_seeds_main"] = {"mean": float(np.mean(main)), "std": float(np.std(main)), "per_seed": [round(v, 4) for v in main]}
    print("  main seeds=10: %.4f +/- %.4f" % (np.mean(main), np.std(main)), flush=True)
    # gated vs ungated (MoE_lite always gated; ungated = average of two experts fixed)
    ung = []
    for s in range(10):
        torch.manual_seed(s)
        m = MoE_lite(ch=16)
        # freeze gate to 0.5/0.5
        for p in m.gate.parameters(): p.requires_grad = False
        m.desc.requires_grad = False
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
        lf = nn.MSELoss(); Xt_ = torch.tensor(Xs[ps[:ntr]]); Yt_ = torch.tensor(Ys[ps[:ntr]])
        for _ in range(40): opt.zero_grad(); lf(m(Xt_), Yt_).backward(); opt.step()
        m.eval()
        with torch.no_grad(): ung.append(vrmse(m(torch.tensor(Xs[ps[ntr:]])).numpy(), Ys[ps[ntr:]]))
    R["F_gate_ungated"] = {"mean": float(np.mean(ung)), "std": float(np.std(ung)), "per_seed": [round(v, 4) for v in ung]}
    # Welch t-test
    a, b = np.array(main), np.array(ung)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    t = (a.mean() - b.mean()) / np.sqrt(va / len(a) + vb / len(b))
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    R["F_welch"] = {"t": float(t), "df": float(df)}
    print("  ungated seeds=10: %.4f +/- %.4f ; Welch t=%.3f" % (np.mean(ung), np.std(ung), t), flush=True)

    # ---- G. #33 spectral decay alpha per system ----
    print("=== G #33 spectral alpha per system ===", flush=True)
    def spec_alpha(field):
        # radial-mean 2D power spectrum slope of one mean-normalized sample
        f = field[0].astype(np.float64); f = f - f.mean()
        F = np.fft.fftshift(np.abs(np.fft.fft2(f)) ** 2)
        H, W = F.shape; cy, cx = H // 2, W // 2
        yy, xx = np.mgrid[0:H, 0:W]; k = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        kb = np.arange(1, min(H, W) // 2); Pb = []
        for km in kb:
            mask = (k >= km - 0.5) & (k < km + 0.5)
            if mask.sum() > 0: Pb.append(F[mask].mean())
            else: Pb.append(np.nan)
        Pb = np.array(Pb); ok = ~np.isnan(Pb) & (Pb > 0)
        kk = np.log(kb[ok][1:]); PP = np.log(Pb[ok][1:])
        if len(kk) < 5: return None
        sl = np.polyfit(kk, PP, 1)[0]
        return -float(sl)  # P ~ k^-gamma
    alphas = {}
    for nm, (X, _) in [("shear", (Xs, Ys)), ("turb", (Xt, Yt)), ("gray", (Xg, Yg)), ("euler", (Xe, Ye)), ("visco", (Xv, Yv))]:
        g_ = spec_alpha(X); alphas[nm] = g_
        print("  %s: gamma=%.3f" % (nm, g_), flush=True)
    R["G_spectral_alpha"] = alphas

    # ---- H. #21/35 D_op via PCA linear one-step maps ----
    print("=== H #21/35 D_op (PCA linear one-step) ===", flush=True)
    from numpy.linalg import lstsq
    K = 16
    def pca_fit(Xs, Ys):
        # joint PCA on x
        allx = np.concatenate([Xs, Xt, Xg], 0)
        meanx = allx.mean(0); Xc = allx.reshape(len(allx), -1) - meanx.reshape(1, -1)
        U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        basis = Vt[:K]
        return meanx, basis
    meanx, basis = pca_fit(Xs, Xt)
    def lin_A(X, Y):
        xp = (X.reshape(len(X), -1) - meanx.reshape(1, -1)) @ basis.T
        yp = (Y.reshape(len(Y), -1) - meanx.reshape(1, -1)) @ basis.T
        A, _, _, _ = lstsq(xp, yp, rcond=None)
        return A
    As = lin_A(Xs, Ys); At = lin_A(Xt, Yt); Ag = lin_A(Xg, Yg)
    def dop(A, B): return float(np.linalg.norm(A - B, 'fro') / np.sqrt(np.linalg.norm(A, 'fro') ** 2 + np.linalg.norm(B, 'fro') ** 2 + 1e-12))
    R["H_Dop"] = {"shear_turb": dop(As, At), "shear_gray": dop(As, Ag), "turb_gray": dop(At, Ag)}
    print("  D_op shear-turb %.4f shear-gray %.4f turb-gray %.4f" % (R["H_Dop"]["shear_turb"], R["H_Dop"]["shear_gray"], R["H_Dop"]["turb_gray"]), flush=True)

    # ---- I. #3/23 real-gate delta + CI ----
    print("=== I #3/23 real-gate delta ===", flush=True)
    class JM2(nn.Module):
        def __init__(self, ch=16):
            super().__init__()
            self.in_conv = nn.Conv2d(4, ch, 1); self.e1 = nn.Conv2d(ch, ch, 1); self.e2 = nn.Conv2d(ch, ch, 1)
            self.out = nn.Conv2d(ch, 1, 1); self.gate = nn.Linear(8, 2)
        def forward(self, x, d):
            h = self.in_conv(x); g = torch.softmax(self.gate(d), -1)
            return self.out(g[0, 0] * self.e1(h) + g[0, 1] * self.e2(h)), g
    nss = int(0.7 * ns); nstt = int(0.7 * nt)
    ds = nn.Parameter(torch.randn(8), requires_grad=False); dtt = nn.Parameter(torch.randn(8), requires_grad=False)
    with torch.no_grad(): ds.normal_(0, 1); dtt.normal_(0, 1); dtt += 2.0
    Xj = torch.cat([torch.tensor(Xs[ps[:nss]]), torch.tensor(Xt[pt[:nstt]])], 0)
    Yj = torch.cat([torch.tensor(Ys[ps[:nss]]), torch.tensor(Yt[pt[:nstt]])], 0)
    Dj = torch.cat([ds.expand(nss, -1), dtt.expand(nstt, -1)], 0)
    misr = []
    for rep in range(20):
        torch.manual_seed(rep)
        m = JM2(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); lf = nn.MSELoss()
        for s in range(40): opt.zero_grad(); lf(m(Xj, Dj)[0], Yj).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            _, gs_ = m(torch.tensor(Xs[ps[nss:]]), ds.expand(ns - nss, -1))
            _, gt_ = m(torch.tensor(Xt[pt[nstt:]]), dtt.expand(nt - nstt, -1))
        # true label: shear=0, turb=1; misroute = fraction where gate argmax disagrees with a fixed optimal assignment
        arg_s = gs_.argmax(1).numpy(); arg_t = gt_.argmax(1).numpy()
        # choose assignment minimizing misroute
        m00 = (arg_s == 0).mean(); m01 = (arg_s == 1).mean()
        m10 = (arg_t == 0).mean(); m11 = (arg_t == 1).mean()
        err_a = 1 - 0.5 * (m00 + m11); err_b = 1 - 0.5 * (m01 + m10)
        misr.append(min(err_a, err_b))
    m_arr = np.array(misr)
    R["I_gate_delta"] = {"mean": float(m_arr.mean()), "std": float(m_arr.std()),
                         "ci95_low": float(m_arr.mean() - 1.96 * m_arr.std() / np.sqrt(len(m_arr))),
                         "ci95_high": float(m_arr.mean() + 1.96 * m_arr.std() / np.sqrt(len(m_arr)))}
    print("  real-gate delta=%.4f (95%%CI %.4f-%.4f)" % (m_arr.mean(), R["I_gate_delta"]["ci95_low"], R["I_gate_delta"]["ci95_high"]), flush=True)

    out = os.path.join(BASE, "sol_b_results.json")
    json.dump(R, open(out, "w"), indent=2, ensure_ascii=False)
    print("Saved ->", out, flush=True)
