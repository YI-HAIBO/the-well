# -*- coding: utf-8 -*-
"""
S1 reference implementation for the The-Well physics world-model proposal.
Companion to: 03_TheWell_物理世界模型基础模型.md, Section 16 (manual) & Section 17 (plan).

Design constraints (must match the proposal's audited positioning):
  * The Well is ALL uniform grids -> NO point clouds, NO unstructured meshes.
  * Keep the FFT spectral branch (that is the dataset's core advantage).
  * "Coordinate-system agnostic" := normalized parametric coordinates fed through
    a SHARED Fourier-feature encoder (Cartesian / Spherical / Angular / Log-Spherical
    all map to (u,v) in [-1,1]^2 and use the SAME encoder), NOT geometry conversion.
  * Field description embedding + RevIN + field-aware TopK Fourier MoE.
  * Metric: VRMSE (The Well official main metric; 1.0 == predicting the mean).

Modes:
  --mode synthetic : run WITHOUT downloading any dataset; verifies the whole
                     pipeline (tokenizer -> Fourier MoE -> RevIN -> train -> VRMSE)
                     and executes built-in self-tests. Use this for CI / smoke test.
  --mode well      : load official the_well.data.WellDataset (requires
                     `pip install the_well[benchmark]` + a downloaded subset).

Examples:
  python s1_well_pytorch.py --mode synthetic --steps 30
  python s1_well_pytorch.py --mode synthetic --self-test
  python s1_well_pytorch.py --mode well --dataset shear_flow --data-root ./data

NOTE: This is a deliberately compact *research scaffold*, not the full S2/S3
system. It implements E1 (single-subset closed loop) of the experiment plan.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class Cfg:
    in_steps: int = 4           # n_in, matches proposal 16.4
    out_steps: int = 1          # n_out
    height: int = 32
    width: int = 32
    n_fields: int = 3          # channels (e.g. velocity-x, velocity-y, tracer)
    hidden: int = 32
    n_modes: int = 8           # Fourier modes kept per axis (N in Theorem 1)
    n_layers: int = 2          # L in Theorem 1
    n_experts: int = 4         # K experts
    top_k: int = 2
    n_coord_freqs: int = 6
    n_desc_ids: int = 16
    desc_dim: int = 32
    lr: float = 1e-3
    revin_eps: float = 1e-6
    seed: int = 0


# --------------------------------------------------------------------------------------
# 1) Coordinate Fourier features -- the SHARED encoder for all coordinate systems
# --------------------------------------------------------------------------------------
class CoordFourierFeatures(nn.Module):
    """Normalized parametric coordinates -> multi-frequency sin/cos channels.

    Whatever the physical coordinate system is (Cartesian / Spherical / Angular /
    Log-Spherical), The Well stores fields on a uniform grid; we take the grid's
    normalized index coordinates (u,v) in [-1,1] and encode them identically.
    This is what 'coordinate-system agnostic' means in this proposal. It does NOT
    mean converting grids to point clouds.
    """
    def __init__(self, n_freqs: int):
        super().__init__()
        freqs = (2.0 ** torch.arange(n_freqs, dtype=torch.float32)) * math.pi
        self.register_buffer("freqs", freqs)          # (n_freqs,)

    def forward(self, h: int, w: int, device=None, dtype=None):
        u = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype or torch.float32)
        v = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype or torch.float32)
        grid_u, grid_v = torch.meshgrid(u, v, indexing="ij")  # (h,w)
        feats = []
        for axis in (grid_u, grid_v):
            a = axis.unsqueeze(-1) * self.freqs.view(1, 1, -1)  # (h,w,n_freqs)
            feats += [torch.sin(a), torch.cos(a)]
        out = torch.cat(feats, dim=-1)                  # (h,w,4*n_freqs)
        return out.permute(2, 0, 1).contiguous()        # (4*n_freqs,h,w)


# --------------------------------------------------------------------------------------
# 2) RevIN -- per-field instance normalization + denormalization (handles scale gaps)
# --------------------------------------------------------------------------------------
class RevIN(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self._mean = None
        self._std = None

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W); per-sample, per-channel statistics over spatial dims
        self._mean = x.mean(dim=(-2, -1), keepdim=True).detach()
        var = x.var(dim=(-2, -1), keepdim=True, unbiased=False).detach()
        self._std = torch.sqrt(var + self.eps)
        return (x - self._mean) / self._std

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._std is None:
            raise RuntimeError("RevIN.denorm called before RevIN.norm")
        return x * self._std + self._mean


# --------------------------------------------------------------------------------------
# 3) Field description embedding -- variable-order agnostic (mean over field tokens)
# --------------------------------------------------------------------------------------
class FieldDescriptorEmbedding(nn.Module):
    def __init__(self, n_ids: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_ids, dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, field_ids: torch.Tensor) -> torch.Tensor:
        # field_ids: (B, n_fields); mean-pooling makes it invariant to field ordering
        z = self.emb(field_ids).mean(dim=1)
        return self.proj(z)                              # (B, dim)


# --------------------------------------------------------------------------------------
# 4) Spectral convolution (keeps the uniform-grid FFT advantage)
# --------------------------------------------------------------------------------------
class SpectralConv2d(nn.Module):
    """2D Fourier layer with a FIXED number of modes -> resolution-agnostic weights."""
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        # store real/imag separately, view_as_complex at runtime
        self.w_lo = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, 2))
        self.w_hi = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        m1 = min(self.modes, H // 2)
        m2 = min(self.modes, W // 2 + 1)
        x_ft = torch.fft.rfft2(x)                        # (B,C,H,W//2+1)
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        w_lo = torch.view_as_complex(self.w_lo.contiguous())[:, :, :m1, :m2]
        w_hi = torch.view_as_complex(self.w_hi.contiguous())[:, :, :m1, :m2]
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bcij,coij->boij", x_ft[:, :, :m1, :m2], w_lo)
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bcij,coij->boij", x_ft[:, :, -m1:, :m2], w_hi)
        return torch.fft.irfft2(out_ft, s=(H, W))


class FourierExpert(nn.Module):
    def __init__(self, ch: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv2d(ch, ch, modes)
        self.pointwise = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(4, ch), ch)

    def forward(self, x):
        return F.gelu(self.norm(x + self.spectral(x) + self.pointwise(x)))


# --------------------------------------------------------------------------------------
# 5) Field-aware TopK Fourier MoE
# --------------------------------------------------------------------------------------
class FourierMoELayer(nn.Module):
    """Router is conditioned on the FIELD DESCRIPTION embedding (not on pixels)."""
    def __init__(self, ch: int, modes: int, n_experts: int, top_k: int, desc_dim: int):
        super().__init__()
        self.experts = nn.ModuleList([FourierExpert(ch, modes) for _ in range(n_experts)])
        self.router = nn.Sequential(nn.Linear(desc_dim, desc_dim), nn.GELU(),
                                    nn.Linear(desc_dim, n_experts))
        self.top_k = min(top_k, n_experts)
        self.last_balance_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, desc: torch.Tensor):
        B = x.shape[0]
        logits = self.router(desc)                        # (B,K)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(x)
        # hard TopK dispatch (only selected experts receive gradient)
        for k in range(self.top_k):
            sel = topi[:, k]                             # (B,)
            w = topv[:, k].view(B, 1, 1, 1)
            # group samples by chosen expert to keep it batched & simple
            for e in range(len(self.experts)):
                mask = (sel == e)
                if mask.any():
                    out[mask] = out[mask] + w[mask] * self.experts[e](x[mask])

        # load-balancing auxiliary loss (importance == uniform)
        importance = probs.mean(dim=0)
        self.last_balance_loss = ((importance - 1.0 / len(self.experts)) ** 2).sum()
        return out


# --------------------------------------------------------------------------------------
# 6) Unified tokenizer + full model
# --------------------------------------------------------------------------------------
class UnifiedTokenizer(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.coord = CoordFourierFeatures(cfg.n_coord_freqs)
        in_ch = cfg.in_steps * cfg.n_fields + 4 * cfg.n_coord_freqs
        self.in_proj = nn.Conv2d(in_ch, cfg.hidden, 1)
        self.desc = FieldDescriptorEmbedding(cfg.n_desc_ids, cfg.desc_dim)
        # FiLM modulation from description embedding
        self.film = nn.Linear(cfg.desc_dim, 2 * cfg.hidden)

    def forward(self, x: torch.Tensor, field_ids: torch.Tensor):
        # x: (B, T_in, H, W, C) -> (B, T_in*C, H, W)
        B, T, H, W, C = x.shape
        xf = x.permute(0, 1, 4, 2, 3).contiguous().view(B, T * C, H, W)
        cf = self.coord(H, W, x.device, x.dtype).unsqueeze(0).expand(B, -1, -1, -1)
        h = self.in_proj(torch.cat([xf, cf], dim=1))
        e = self.desc(field_ids)
        gamma, beta = self.film(e).chunk(2, dim=-1)
        h = h * (1.0 + gamma.view(B, -1, 1, 1)) + beta.view(B, -1, 1, 1)
        return h, e


class WellS1Model(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.revin = RevIN(cfg.revin_eps)
        self.tokenizer = UnifiedTokenizer(cfg)
        self.layers = nn.ModuleList([
            FourierMoELayer(cfg.hidden, cfg.n_modes, cfg.n_experts,
                            cfg.top_k, cfg.desc_dim)
            for _ in range(cfg.n_layers)
        ])
        self.head = nn.Sequential(nn.Conv2d(cfg.hidden, cfg.hidden, 1), nn.GELU(),
                                  nn.Conv2d(cfg.hidden, cfg.n_fields, 1))

    def forward(self, x: torch.Tensor, field_ids: torch.Tensor) -> torch.Tensor:
        # x: (B,T_in,H,W,C) ; output: (B,H,W,C_out) -- one-step prediction
        B, T, H, W, C = x.shape
        # RevIN on the LAST input frame only (per-channel statistics for denorm)
        last = x[:, -1].permute(0, 3, 1, 2).contiguous()   # (B,C,H,W)
        std_last = torch.sqrt(last.var(dim=(-2, -1), keepdim=True,
                                       unbiased=False) + self.cfg.revin_eps)
        mean_last = last.mean(dim=(-2, -1), keepdim=True)
        # (B,C,1,1) -> (B,1,1,1,C) so it broadcasts over (B,T,H,W,C)
        m5 = mean_last.permute(0, 2, 3, 1).unsqueeze(1)
        s5 = std_last.permute(0, 2, 3, 1).unsqueeze(1)
        x_norm = (x - m5) / s5

        h, e = self.tokenizer(x_norm, field_ids)
        bal = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            h = layer(h, e)
            bal = bal + layer.last_balance_loss
        y_norm = self.head(h)                              # (B,C,H,W)
        y = y_norm * std_last + mean_last
        self.last_balance_loss = bal
        return y.permute(0, 2, 3, 1).contiguous()          # (B,H,W,C_out)


# --------------------------------------------------------------------------------------
# 7) Metrics (VRMSE is the official The Well main metric)
# --------------------------------------------------------------------------------------
def vrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Variance-scaled RMSE; 1.0 == predicting the per-sample mean field."""
    var = target.var(dim=(1, 2), unbiased=False).mean(dim=-1)  # (B,)
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)        # (B,)
    return torch.sqrt((mse / (var + eps)).mean()).item()


def rel_l2(pred, target, eps=1e-8):
    return (torch.linalg.vector_norm(pred - target) /
            (torch.linalg.vector_norm(target) + eps)).item()


def spectral_rel_error(pred, target):
    """Relative error in low / mid / high frequency bands (dissects the FFT branch)."""
    p = torch.fft.rfft2(pred.permute(0, 3, 1, 2)).abs()
    t = torch.fft.rfft2(target.permute(0, 3, 1, 2)).abs()
    Fmax = p.shape[-1]
    cuts = [0, max(1, Fmax // 8), max(2, Fmax // 2), Fmax]
    out = {}
    for name, a, b in zip(("low", "mid", "high"), cuts[:-1], cuts[1:]):
        out[name] = ((p[..., a:b] - t[..., a:b]).pow(2).sum().sqrt() /
                     (t[..., a:b].pow(2).sum().sqrt() + 1e-8)).item()
    return out


# --------------------------------------------------------------------------------------
# 8) Synthetic data (no download needed): smooth 2D fields with a simple evolution
# --------------------------------------------------------------------------------------
def make_synthetic(n_traj: int, n_steps: int, h: int, w: int, c: int,
                   device, seed: int = 0):
    """Deterministic smooth random fields evolving by phase drift + diffusion.
    Mimics a 2D shear_flow / active_matter-like field purely for pipeline testing."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    yy = torch.linspace(-1, 1, h).view(1, h, 1)
    xx = torch.linspace(-1, 1, w).view(1, 1, w)
    data = []
    for tr in range(n_traj):
        amps = torch.rand(c, 3, 3, generator=g) * 0.8 + 0.2
        kx = torch.randint(1, 4, (c, 3, 3), generator=g).float()
        ky = torch.randint(1, 4, (c, 3, 3), generator=g).float()
        ph = torch.rand(c, 3, 3, generator=g) * 2 * math.pi
        traj = []
        for t in range(n_steps):
            drift = 0.06 * t
            decay = math.exp(-0.004 * t)
            field = torch.zeros(c, h, w)
            for ci in range(c):
                for i in range(3):
                    for j in range(3):
                        wave = torch.sin(kx[ci, i, j] * math.pi * xx +
                                         ky[ci, i, j] * math.pi * yy +
                                         ph[ci, i, j] + drift)
                        field[ci] += (amps[ci, i, j] * decay * wave).squeeze(0)
            traj.append(field)
        data.append(torch.stack(traj))
    return torch.stack(data).permute(0, 1, 3, 4, 2).contiguous().to(device)  # (N,T,H,W,C)


def make_windows(seq, t_in, t_out, batch, device, field_ids=None):
    """seq: (N,T,H,W,C) -> random sliding windows; NO frame-level random split
    across trajectories in real use (official split is trajectory-level)."""
    N, T, H, W, C = seq.shape
    idx_tr = torch.randint(0, N, (batch,), device=device)
    idx_t = torch.randint(0, T - t_in - t_out + 1, (batch,), device=device)
    xs = torch.stack([seq[i, s:s + t_in] for i, s in zip(idx_tr, idx_t)])
    ys = torch.stack([seq[i, s + t_in:s + t_in + t_out] for i, s in zip(idx_tr, idx_t)])
    y = ys[:, 0]                                          # one-step target (B,H,W,C)
    if field_ids is None:
        ids = torch.arange(C, device=device).unsqueeze(0).expand(batch, -1)
    else:
        ids = field_ids
    return xs, y, ids


# --------------------------------------------------------------------------------------
# 9) Self-tests
# --------------------------------------------------------------------------------------
def run_self_tests(cfg: Cfg, device) -> bool:
    print("\n[SELF-TEST] running built-in assertions")
    torch.manual_seed(cfg.seed)
    model = WellS1Model(cfg).to(device)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))
        print("  %-46s %s %s" % (name, "PASS" if cond else "FAIL", detail))

    # (a) forward shape
    B, T, H, W, C = 2, cfg.in_steps, cfg.height, cfg.width, cfg.n_fields
    x = torch.randn(B, T, H, W, C, device=device)
    ids = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)
    y = model(x, ids)
    check("forward output shape (B,H,W,C_out)", y.shape == (B, H, W, C), str(tuple(y.shape)))

    # (b) backward finite, no NaN
    loss = F.mse_loss(y, torch.randn_like(y))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(torch.isfinite(g).all().item() for g in grads)
    check("backward gradients finite / non-NaN", finite and len(grads) > 0,
          "%d tensors with grad" % len(grads))
    check("load-balancing aux loss >= 0",
          float(model.last_balance_loss.item()) >= 0.0,
          "%.6f" % float(model.last_balance_loss.item()))

    # (c) RevIN round-trip
    rev = RevIN().to(device)
    a = torch.randn(2, C, H, W, device=device) * 5.0 + 3.0
    z = rev.norm(a)
    rec = rev.denorm(z)
    err = (rec - a).abs().max().item()
    check("RevIN norm/denorm round-trip", err < 1e-4, "max err %.2e" % err)

    # (d) resolution-agnostic: same weights forward on a different grid size
    model.eval()
    with torch.no_grad():
        x2 = torch.randn(1, T, 48, 48, C, device=device)
        y2 = model(x2, torch.arange(C, device=device).unsqueeze(0))
    check("resolution-agnostic (32x32 -> 48x48 weights)",
          y2.shape == (1, 48, 48, C) and torch.isfinite(y2).all().item(),
          str(tuple(y2.shape)))

    # (e) coordinate-system agnostic: same encoder for 4 coordinate tags
    with torch.no_grad():
        cf = model.tokenizer.coord(H, W, device, torch.float32)
    check("shared coord encoder for all 4 systems",
          cf.shape == (4 * cfg.n_coord_freqs, H, W) and torch.isfinite(cf).all().item(),
          str(tuple(cf.shape)))

    # (f) VRMSE sanity: predicting the mean should give ~1.0
    tgt = torch.randn(2, H, W, C, device=device)
    mean_pred = tgt.mean(dim=(1, 2), keepdim=True).expand_as(tgt)
    v = vrmse(mean_pred, tgt)
    check("VRMSE(mean predictor) ~ 1.0", abs(v - 1.0) < 0.05, "%.4f" % v)

    # (g) variable-order agnostic descriptor
    with torch.no_grad():
        e1 = model.tokenizer.desc(ids)
        e2 = model.tokenizer.desc(ids.flip(1))
    check("field-description embedding order-invariant",
          torch.allclose(e1, e2, atol=1e-6),
          "max diff %.2e" % (e1 - e2).abs().max().item())

    ok = all(r[1] for r in results)
    print("[SELF-TEST] %d/%d passed -> %s\n" %
          (sum(r[1] for r in results), len(results), "ALL PASS" if ok else "HAS FAILURES"))
    return ok


# --------------------------------------------------------------------------------------
# 10) Training / evaluation
# --------------------------------------------------------------------------------------
def train_synthetic(cfg: Cfg, steps: int, device, batch: int = 4):
    print("[TRAIN] mode=synthetic  device=%s  steps=%d" % (device, steps))
    torch.manual_seed(cfg.seed)
    seq = make_synthetic(n_traj=24, n_steps=cfg.in_steps + 12,
                         h=cfg.height, w=cfg.width, c=cfg.n_fields,
                         device=device, seed=cfg.seed + 1)
    # trajectory-level split: last 4 trajectories = val (never trained on)
    train_seq, val_seq = seq[:20], seq[20:]
    model = WellS1Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ids = torch.arange(cfg.n_fields, device=device).unsqueeze(0).expand(batch, -1)
    t0 = time.time()
    for step in range(steps):
        model.train()
        x, y, fid = make_windows(train_seq, cfg.in_steps, cfg.out_steps, batch, device, ids)
        opt.zero_grad()
        pred = model(x, fid)
        loss = F.mse_loss(pred, y) + 0.01 * model.last_balance_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, steps // 5) == 0 or step == 0:
            print("  step %3d  train MSE %.6f" % (step + 1, loss.item()))

    model.eval()
    with torch.no_grad():
        vx, vy, vfid = make_windows(val_seq, cfg.in_steps, cfg.out_steps,
                                    min(batch, 8), device)
        vp = model(vx, vfid)
        # baseline: predict per-sample mean field (VRMSE should be ~1.0)
        baseline = vy.mean(dim=(1, 2), keepdim=True).expand_as(vy)
    v_model = vrmse(vp, vy)
    v_base = vrmse(baseline, vy)
    rl2 = rel_l2(vp, vy)
    spec = spectral_rel_error(vp, vy)
    n_params = sum(p.numel() for p in model.parameters())
    print("\n[EVAL] held-out trajectories (never trained on)")
    print("  VRMSE  model = %.4f   |  mean-baseline = %.4f" % (v_model, v_base))
    print("  rel L2 model = %.4f" % rl2)
    print("  spectral rel-err  low %.4f / mid %.4f / high %.4f" %
          (spec["low"], spec["mid"], spec["high"]))
    print("  params = %d   elapsed = %.1fs" % (n_params, time.time() - t0))
    improved = v_model < v_base
    print("  E1 sanity (model beats mean baseline): %s" % ("PASS" if improved else "FAIL"))
    print("  NOTE: short smoke run on synthetic data; NOT a scientific result.")
    return {"vrmse_model": v_model, "vrmse_baseline": v_base,
            "rel_l2": rl2, "params": n_params, "e1_pass": improved,
            "spec": spec}


def train_well(cfg: Cfg, args, device):
    """Official WellDataset path. Requires `pip install the_well[benchmark]`."""
    try:
        from the_well.data import WellDataset
        from torch.utils.data import DataLoader
    except ImportError as exc:
        print("[ERROR] --mode well requires the_well (+ h5py). %s" % exc)
        print("        Run: pip install the_well[benchmark]")
        print("        Then: the-well-download --base-path ./data --dataset %s --split train"
              % args.dataset)
        print("        Falling back is NOT done silently; use --mode synthetic to smoke test.")
        return None
    ds = WellDataset(
        well_base_path=args.data_root,       # or "hf://datasets/polymathic-ai/"
        well_dataset_name=args.dataset,
        well_split_name="train",
        n_steps_input=cfg.in_steps,
        n_steps_output=cfg.out_steps,
    )
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True)
    print("[TRAIN] mode=well  dataset=%s  batches=%d" % (args.dataset, len(loader)))
    model = WellS1Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        x = batch["input_fields"].to(device)      # (B,n_in,*spatial,C)
        y = batch["output_fields"].to(device)[:, 0]
        B = x.shape[0]
        C = x.shape[-1]
        ids = torch.arange(C, device=device).unsqueeze(0).expand(B, -1) % cfg.n_desc_ids
        opt.zero_grad()
        pred = model(x, ids)
        loss = F.mse_loss(pred, y) + 0.01 * model.last_balance_loss
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print("  step %4d  MSE %.6f  VRMSE %.4f" %
                  (step, loss.item(), vrmse(pred.detach(), y)))
    return {"well_steps": min(args.steps, len(loader))}


def main():
    ap = argparse.ArgumentParser(description="S1 reference implementation (The Well world model)")
    ap.add_argument("--mode", choices=["synthetic", "well"], default="synthetic")
    ap.add_argument("--dataset", default="shear_flow")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    cfg = Cfg(height=args.height, width=args.width, seed=args.seed)
    print("=" * 78)
    print("The Well world-model / S1 reference implementation")
    print("mode=%s  device=%s  HxW=%dx%d  in_steps=%d  modes=%d  layers=%d  experts=%d"
          % (args.mode, device, cfg.height, cfg.width, cfg.in_steps,
             cfg.n_modes, cfg.n_layers, cfg.n_experts))
    print("=" * 78)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    st_ok = True
    if args.self_test or args.mode == "synthetic":
        st_ok = run_self_tests(cfg, device)

    if args.mode == "synthetic":
        res = train_synthetic(cfg, args.steps, device, args.batch)
        print("\n[SUMMARY] self_test=%s  e1_sanity=%s  vrmse=%.4f (baseline 1.0)"
              % ("PASS" if st_ok else "FAIL",
                 "PASS" if res["e1_pass"] else "FAIL", res["vrmse_model"]))
        return 0 if (st_ok and res["e1_pass"]) else 1
    else:
        out = train_well(cfg, args, device)
        return 0 if out else 2


if __name__ == "__main__":
    sys.exit(main())
