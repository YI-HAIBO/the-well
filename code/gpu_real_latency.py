# -*- coding: utf-8 -*-
"""M2: REAL GPU inference latency benchmark on AMD RX 7650 GRE via torch-directml.
Replicates the exact MoE_lite used in the JCP manuscript. Replaces the previous
'CPU x15 estimated' GPU number with a hardware measurement.
"""
import time, json, torch, torch.nn as nn
import torch_directml as tdml

torch.set_num_threads(4)
dml = tdml.device()
print("GPU device:", tdml.device_name(0), "| torch", torch.__version__, flush=True)

# ---- exact MoE_lite from the manuscript ----
class MoE_lite(nn.Module):
    def __init__(self, ch=16, in_ch=4):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, ch, 1)
        self.e1 = nn.Conv2d(ch, ch, 1)
        self.e2 = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, 1, 1)
        self.gate = nn.Linear(4, 2)
        self.desc = nn.Parameter(torch.randn(4))

    def forward(self, x):
        h = self.in_conv(x)
        g = torch.softmax(self.gate(self.desc[None]), -1)
        h = g[0, 0] * self.e1(h) + g[0, 1] * self.e2(h)
        return self.out(h)

def sync(dev):
    # DirectML has no cuda.synchronize; copying a scalar to CPU forces queue drain.
    if str(dev).startswith("privateuseone"):
        _ = torch.zeros(1, device=dev).cpu().item()
    else:
        _ = torch.zeros(1, device=dev).item()

def bench(model, x, dev, iters=200, warm=30):
    model.to(dev).eval()
    xs = x.to(dev)
    with torch.no_grad():
        for _ in range(warm):
            _ = model(xs)
        sync(dev)
        # per-call latency (sync each call -> online rolling step cost)
        t0 = time.perf_counter()
        for _ in range(iters):
            y = model(xs)
            sync(dev)
        t1 = time.perf_counter()
        lat_call_ms = (t1 - t0) / iters * 1000.0
        # steady-state throughput (pipeline, sync once at end)
        t0 = time.perf_counter()
        for _ in range(iters):
            y = model(xs)
        sync(dev)
        t1 = time.perf_counter()
        thr = iters / (t1 - t0)  # calls/s
    return lat_call_ms, thr

out = {"device": tdml.device_name(0), "torch": torch.__version__, "results": {}}
torch.manual_seed(0)
for ch in [16, 32, 64, 128]:
    m = MoE_lite(ch=ch)
    npar = sum(p.numel() for p in m.parameters())
    for batch in [1, 8]:
        x = torch.randn(batch, 4, 64, 64)
        # CPU
        torch.set_num_threads(4)
        cpu_lat, cpu_thr = bench(MoE_lite(ch=ch).eval(), x, torch.device("cpu"))
        # GPU
        gpu_lat, gpu_thr = bench(MoE_lite(ch=ch), x, dml)
        key = "ch%d_b%d" % (ch, batch)
        rec = {
            "ch": ch, "batch": batch, "n_params": npar,
            "cpu_latency_ms": round(cpu_lat, 4),
            "gpu_latency_ms": round(gpu_lat, 4),
            "cpu_throughput_s": round(cpu_thr, 1),
            "gpu_throughput_s": round(gpu_thr, 1),
            "measured_speedup": round(cpu_lat / gpu_lat, 2),
        }
        out["results"][key] = rec
        print("  ch=%-3d b=%d | CPU %7.3f ms | GPU %7.3f ms | speedup %5.2fx | %d params" % (
            ch, batch, cpu_lat, gpu_lat, cpu_lat/gpu_lat, npar), flush=True)

path = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\gpu_real_latency_results.json"
json.dump(out, open(path, "w"), indent=2)
print("saved ->", path, flush=True)
