import json, time, torch
import torch.nn as n

class M(n.Module):
    def __init__(s, c=16):
        super().__init__()
        s.i = n.Conv2d(4, c, 1); s.e1 = n.Conv2d(c, c, 1); s.e2 = n.Conv2d(c, c, 1)
        s.o = n.Conv2d(c, 1, 1); s.g = n.Linear(4, 2)
        s.d = n.Parameter(torch.randn(4))
    def forward(s, x):
        h = s.i(x); g = torch.softmax(s.g(s.d[None]), -1)
        return s.o(g[0, 0] * s.e1(h) + g[0, 1] * s.e2(h))

def bench(m, x, N=200, W=30):
    for _ in range(W): m(x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(N): m(x); torch.cuda.synchronize()
    a = (time.perf_counter() - t) / N * 1000
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(N): m(x)
    torch.cuda.synchronize(); b = (time.perf_counter() - t) / N * 1000
    return {"sync_ms": round(a, 3), "async_ms": round(b, 3)}

assert torch.cuda.is_available(), "NO_CUDA"
p = torch.cuda.get_device_properties(0)
o = {"device": torch.cuda.get_device_name(0), "vram_gb": round(p.total_memory / 1e9, 1),
     "torch": torch.__version__, "cuda": torch.version.cuda, "results": {}}
for c in [16, 32, 64, 128]:
    for b in [1, 8]:
        torch.manual_seed(0)
        m = M(c).cuda().eval(); x = torch.randn(b, 4, 64, 64, device="cuda")
        o["results"]["ch%d_b%d" % (c, b)] = bench(m, x)
        print("ch%d_b%d" % (c, b), o["results"]["ch%d_b%d" % (c, b)], flush=True)
json.dump(o, open("/mnt/workspace/a10_latency_results.json", "w"), indent=2)
print("RESULT=" + json.dumps(o, separators=(",", ":")))
