# -*- coding: utf-8 -*-
"""Environment self-check for the one-click reproduction. Run: python check_env.py"""
import importlib, os, sys

EXPECT = {"torch": "2.14.0", "numpy": "2.5.3", "h5py": "3.16.0", "fsspec": "2024.10.0"}
ok = True
for mod, ver in EXPECT.items():
    try:
        m = importlib.import_module(mod)
        got = getattr(m, "__version__", "?")
        good = got.split("+")[0] == ver
        ok &= good
        print(f"[{'OK' if good else 'WARN'}] {mod} {got} (expected {ver})")
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
