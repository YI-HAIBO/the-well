# -*- coding: utf-8 -*-
"""Closed-loop (autoregressive rollout) end-to-end latency benchmark — M2 residual.

Measures per-step latency for batch-1, 64x64, ch in {16,32,64,128}:
  open-loop   : single forward on a fixed input
  closed-loop : R-step autoregressive rollout, output re-fed as next input
Devices: CPU (4 threads) and AMD DirectML GPU when torch-directml is available.
"""
import json, time, statistics, sys, os

import torch
import torch.nn as nn

torch.set_num_threads(4)
BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"


class MoE_lite(nn.Module):
    def __init__(self, ch=16, in_ch=4, n_experts=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.experts = nn.ModuleList([nn.Conv2d(ch, ch, 1) for _ in range(n_experts)])
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(4, n_experts)
        self.desc = nn.Parameter(torch.randn(4))

    def forward(self, x):
        g = torch.softmax(self.gate(self.desc), -1)
        h = self.in_conv(x)
        y = sum(w * e(h) for w, e in zip(g, self.experts))
        return self.out(y)


def bench_device(dev_name, to_dev):
    rows = []
    for ch in (16, 32, 64, 128):
        m = MoE_lite(ch).to(to_dev)
        x = torch.randn(1, 4, 64, 64, device=to_dev)

        def step():
            with torch.no_grad():
                return m(x)

        for _ in range(10):
            step()
        if hasattr(torch, "cpu") and to_dev.type == "privateuseone":
            torch.cpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            step()
        if to_dev.type == "privateuseone":
            torch.cpu.synchronize()
        open_ms = (time.perf_counter() - t0) / 50 * 1e3

        R, warm = 100, 10
        buf = x
        for _ in range(warm):
            with torch.no_grad():
                buf = buf.clone()
                o = m(buf)
                buf = torch.cat([o, buf[:, 1:]], dim=1)

        per_step = []
        buf = x
        t_roll = time.perf_counter()
        for _ in range(R):
            t1 = time.perf_counter()
            with torch.no_grad():
                o = m(buf)
                buf = torch.cat([o, buf[:, 1:]], dim=1)
            if to_dev.type == "privateuseone":
                torch.cpu.synchronize()
            per_step.append((time.perf_counter() - t1) * 1e3)
        roll_s = time.perf_counter() - t_roll
        rows.append({
            "ch": ch,
            "open_ms": round(open_ms, 4),
            "closed_ms_median": round(statistics.median(per_step), 4),
            "closed_ms_mean": round(sum(per_step) / len(per_step), 4),
            "rollout_100s_total_ms": round(roll_s * 1e3, 1),
        })
        print(dev_name, rows[-1])
    return rows


out = {}
if "gpu" not in sys.argv:
    out["cpu"] = bench_device("cpu", torch.device("cpu"))

try:
    import torch_directml
    dev = torch_directml.device()
    out["amd_dml_gpu"] = bench_device("amd_dml_gpu", dev)
except Exception as e:
    out["amd_dml_gpu"] = f"unavailable: {e}"
    print("DirectML unavailable:", e)

path = os.path.join(BASE, "latency_closedloop_results.json")
if "gpu" in sys.argv and os.path.exists(path):
    with open(path) as f:
        prev = json.load(f)
    prev.update(out)
    out = prev
with open(path, "w") as f:
    json.dump(out, f, indent=2)
print("saved latency_closedloop_results.json")
