# -*- coding: utf-8 -*-
"""Environment self-check for the one-click reproduction. Run: python check_env.py"""
import importlib, os, sys

# Compatible-version expectations (minimum leading numeric), matching
# requirements.txt. Version may be dotted (torch 2.4.1) or year-based
# (fsspec 2024.10.0) -> compare the leading integer.
EXPECT = {
    "torch": (2, "torch>=2.0,<3.0"),
    "numpy": (2, "numpy>=1.24,<3.0"),
    "h5py": (3, "h5py>=3.8"),
    "fsspec": (2023, "fsspec>=2023.1"),
}
ok = True
for mod, (want_min, spec) in EXPECT.items():
    try:
        m = importlib.import_module(mod)
        got = getattr(m, "__version__", "?")
        lead = "".join(ch for ch in got.split(".")[0] if ch.isdigit())
        good = int(lead) >= want_min
        ok &= good
        print(f"[{'OK' if good else 'WARN'}] {mod} {got} (need {spec})")
    except Exception as e:
        ok = False
        print(f"[FAIL] {mod}: {e}")

try:
    import torch
    x = torch.randn(2, 4, 8, 8)
    y = torch.nn.Conv2d(4, 4, 1)(x)
    print("[OK] torch CPU forward pass:", tuple(y.shape), "| threads:", torch.get_num_threads())
except Exception as e:
    ok = False
    print("[FAIL] torch op:", e)

here = os.path.dirname(os.path.abspath(__file__))
for f in ("data/slice_train.npz", "data/slice_test.npz"):
    p = os.path.join(here, f)
    print(("[OK] " if os.path.exists(p) else "[WARN] missing (reproduce_minimal will fetch/skip): ") + f)

print("ENV CHECK:", "PASS" if ok else "ISSUES FOUND (see above)")
sys.exit(0 if ok else 1)
