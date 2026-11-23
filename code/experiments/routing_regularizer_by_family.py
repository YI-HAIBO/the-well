# -*- coding: utf-8 -*-
"""
#7 路由正则化器按族数对照（per-family-count with/without load-balancing regularizer）

用途：补齐论文 R 轮披露的“路由在无负载均衡正则化器下退化（0.30->0.540）”在
各系统族数 k=2..8 下的逐点对照，隔离负载均衡（LB）正则化器的独立贡献。

方法学（忠实沿用论文已有协议）：
  - 架构: JointMoE（joint gate, desc 行 = 系统描述向量, n_experts = k）,
         与 sol4_routing.py 相同。
  - 数据: 每个系统一个 64x64 切片 (n_traj, n_time, 64, 64) float32，
         由 mk_xy 切 4 帧 -> 预测下一帧，std 归一化（与 sol4_routing.py 相同）。
  - 训练: 60 步 AdamW lr=1e-3，MSE only（without） vs MSE + 0.01*lb_loss（with）。
         lb_loss = -mean(sum(g*log(g+1e-8),-1)) 熵最大化（与 local_attack.py E4 相同）。
  - 指标: delta_sample（样本级误分配份额：被分配到共享 majority expert 的样本占比，
         按 300 次 bootstrap 求 95%CI），delta_system（系统级共享 majority 比例）。
         与 sol4_results.json 的 3_23_routing.delta_sample=0.54 同一口径。

数据清单（缺失即报错并列出）:
  <datadir>/data64.npz : {"shear": (n,st,64,64), "turb": (n,st,64,64)}
  <datadir>/gray64.npz  : {"gray": (n,st,64,64)}
  <datadir>/euler64.npz : {"euler": (n,st,64,64)}
  <datadir>/visco64.npz : {"visco": (n,st,64,64)}
  <datadir>/rb64.npz    : {"rb": (n,st,64,64)}
  <datadir>/ac64.npz    : {"ac": (n,st,64,64)}
  <datadir>/ps64.npz    : {"ps": (n,st,64,64)}

用法:
  python routing_regularizer_by_family.py [datadir] [out_json]

示例:
  python routing_regularizer_by_family.py ~/well_b3 sol4_routing_byfamily.json

输出: 控制台逐 k 表格 + JSON（含 delta_with / delta_without / CI / entropy）。
运行时: 全流程 k=2..8 双条件约 3-5 分钟 CPU（8 专家 x 64x64 卷积 x 60 步 x 14 次训练）。
"""
import numpy as np, torch, torch.nn as nn, os, sys, json
from collections import Counter

torch.set_num_threads(4)

SYSTEMS = ["shear", "turb", "gray", "euler", "visco", "rb", "ac", "ps"]
LB_WEIGHT = 0.01          # 与 local_attack.py E4 一致
STEPS = 60                # 与 local_attack.py E4 一致
LR = 1e-3
N_BOOT = 300
IN_CH = 4


def mk_xy(A, n_in=IN_CH):
    n, st, H, W = A.shape
    X, Y = [], []
    for t in range(st - n_in):
        X.append(A[:, t:t + n_in]); Y.append(A[:, t + n_in:t + n_in + 1])
    return np.concatenate(X, 0), np.concatenate(Y, 0)


def std(A):
    A = A.astype(np.float32)
    m = A.mean(axis=(1, 2), keepdims=True); s = A.std(axis=(1, 2), keepdims=True) + 1e-6
    return (A - m) / s


def load_system(datadir, name):
    if name in ("shear", "turb"):
        p = os.path.join(datadir, "data64.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        if name not in d.files:
            return None
        return d[name]
    fn = {"gray": "gray64.npz", "euler": "euler64.npz", "visco": "visco64.npz",
          "rb": "rb64.npz", "ac": "ac64.npz", "ps": "ps64.npz"}[name]
    p = os.path.join(datadir, fn)
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    if name not in d.files:
        return None
    return d[name]


class JointMoE(nn.Module):
    def __init__(self, ch=16, in_ch=IN_CH, n_experts=8, desc_dim=8, n_sys=8, lb_weight=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.lb_weight = lb_weight
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

    def loss(self, x, y, desc_idx, use_lb):
        pred, g = self.forward(x, desc_idx)
        mse = nn.functional.mse_loss(pred, y)
        if use_lb and self.lb_weight > 0:
            # local_attack E4 intent: encourage diverse expert usage by maximizing
            # gate entropy.  The entropy term is -mean(sum g log g); minimizing
            # mse - weight*entropy is equivalent to mse + weight*(-entropy).
            lb = -torch.mean(torch.sum(g * torch.log(g + 1e-8), dim=-1))
            return mse + self.lb_weight * lb
        return mse


def train_joint(dat, systems, use_lb, seed=0, verbose=False):
    idx = {s: i for i, s in enumerate(systems)}
    k = len(systems)
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = JointMoE(n_experts=k, n_sys=k, lb_weight=LB_WEIGHT if use_lb else 0.0)
    Xs, Ys, Ds = [], [], []
    for s in systems:
        X, Y, _, _ = dat[s]
        Xs.append(X); Ys.append(Y); Ds.append(np.full(len(X), idx[s], dtype=np.int64))
    Xc = torch.tensor(np.concatenate(Xs, 0)); Yc = torch.tensor(np.concatenate(Ys, 0))
    Dc = torch.tensor(np.concatenate(Ds, 0))
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for step in range(STEPS):
        opt.zero_grad()
        loss = m.loss(Xc, Yc, Dc, use_lb)
        loss.backward()
        opt.step()
        if verbose and step in (0, 10, 20, 30, 40, 50, 59):
            with torch.no_grad():
                g = torch.softmax(m.gate(m.desc), -1)
                ent = -(g * g.log().sum(-1)).mean().item()
            print(f"    step={step} loss={loss.item():.4f} ent={ent:.4f}")
    m.eval()
    return m


def measure(m, dat, systems):
    idx = {s: i for i, s in enumerate(systems)}
    k = len(systems)
    conf = np.zeros((k, k)); n_hold = np.zeros(k)
    with torch.no_grad():
        for s in systems:
            di = idx[s]
            g = torch.softmax(m.gate(m.desc[di]), -1).numpy()
            conf[di] = g; n_hold[di] = len(dat[s][2])
    major = conf.argmax(1)
    cnt = Counter(major.tolist())
    dedicated = sum(1 for e in major if cnt[e] == 1)
    shared_experts = {e for e in major if cnt[e] > 1}
    sample_share = sum(n_hold[idx[s]] for s in systems if major[idx[s]] in shared_experts) / n_hold.sum()
    delta_sys = (k - dedicated) / float(k)
    rng = np.random.RandomState(1)
    bs = []
    for _ in range(N_BOOT):
        draws = [int(np.argmax(np.random.multinomial(1, conf[idx[s]]))) for s in systems]
        cc = Counter(draws)
        sh = sum(n_hold[idx[s]] for s in systems if cc[draws[idx[s]]] > 1)
        bs.append(sh / n_hold.sum())
    bs = np.array(bs)
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    return {"confusion": conf.tolist(), "majority": major.tolist(),
            "experts_used": len(cnt), "systems_dedicated": int(dedicated),
            "delta_system": float(delta_sys), "delta_sample": float(sample_share),
            "delta_ci": ci, "n_experts": k, "n_systems": k}


def main():
    datadir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/well_b3")
    outpath = sys.argv[2] if len(sys.argv) > 2 else "sol4_routing_byfamily.json"
    datadir = os.path.expanduser(datadir)

    # 数据可用性检查
    missing = []
    avail = []
    for s in SYSTEMS:
        a = load_system(datadir, s)
        if a is None:
            missing.append(s)
        else:
            avail.append(s)
    if missing:
        print("MISSING SYSTEM SLICES (cannot run per-family-count experiment):")
        for s in SYSTEMS:
            if s in missing:
                print("  -", s)
        print()
        print("Available:", avail)
        print("Place the following in", datadir, "then re-run:")
        print("  data64.npz{'shear','turb'} + gray64/euler64/visco64/rb64/ac64/ps64.npz")
        sys.exit(2)

    # 构建 80/20 划分（与 sol4_routing.py 相同：seed 0 打乱，80% train）
    dat = {}
    for s in SYSTEMS:
        X, Y = (std(a) for a in mk_xy(load_system(datadir, s).astype(np.float32)))
        n = len(X); perm = np.random.RandomState(0).permutation(n); ntr = int(0.8 * n)
        dat[s] = (X[perm[:ntr]], Y[perm[:ntr]], X[perm[ntr:]], Y[perm[ntr:]])

    results = {}
    print(f"{'k':>2} | {'delta_without':>12} | {'delta_with':>12} | {'delta_sys_w/o':>12} | {'delta_sys_w':>12}")
    print("-" * 62)
    for k in range(2, 9):
        systems = SYSTEMS[:k]
        print(f"k={k} {systems}", flush=True)
        m_wo = train_joint(dat, systems, use_lb=False, seed=0, verbose=True)
        m_w = train_joint(dat, systems, use_lb=True, seed=0, verbose=True)
        r_wo = measure(m_wo, dat, systems)
        r_w = measure(m_w, dat, systems)
        results[f"k{k}"] = {
            "systems": systems,
            "without_LB": r_wo,
            "with_LB": r_w,
            "lb_weight": LB_WEIGHT,
            "steps": STEPS,
            "lr": LR,
        }
        print(f"{k:>2} | {r_wo['delta_sample']:>12.4f} | {r_w['delta_sample']:>12.4f} "
              f"| {r_wo['delta_system']:>12.4f} | {r_w['delta_system']:>12.4f}")
    print("-" * 62)
    print("CI (95%) for delta_sample, without_LB / with_LB:")
    for k in range(2, 9):
        r_wo = results[f"k{k}"]["without_LB"]; r_w = results[f"k{k}"]["with_LB"]
        print(f"  k={k}: without {r_wo['delta_ci'][0]:.3f}-{r_wo['delta_ci'][1]:.3f} | "
              f"with {r_w['delta_ci'][0]:.3f}-{r_w['delta_ci'][1]:.3f}")
    with open(outpath, "w") as f:
        json.dump({"lb_weight": LB_WEIGHT, "steps": STEPS, "lr": LR,
                   "by_family": results}, f, indent=2, ensure_ascii=False)
    print("\nSaved ->", outpath)


if __name__ == "__main__":
    main()
