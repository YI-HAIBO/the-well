# -*- coding: utf-8 -*-
"""Compile manuscript_en_jcp.tex with MiKTeX pdflatex (PATH filtered per known env issue)."""
import os, subprocess, sys

DIR = r"C:\Users\Yi\WorkBuddy\2026-11-09-18-02-06\outputs\TheWell_论文"
EXE = r"C:\CTEX\MiKTeX\miktex\bin\x64\pdflatex.exe"

env = os.environ.copy()
parts = env.get("PATH", "").split(os.pathsep)
clean = [p for p in parts if p.strip() and not os.path.isfile(p)]
env["PATH"] = os.pathsep.join(clean)

for i in range(2):
    r = subprocess.run([EXE, "-interaction=nonstopmode", "manuscript_en_jcp.tex"],
                       cwd=DIR, env=env, capture_output=True, text=True, errors="replace")
    log = open(os.path.join(DIR, "manuscript_en_jcp.log"), encoding="utf-8", errors="replace").read()
    errs = [l for l in log.split("\n") if l.startswith("!")]
    undef = [l for l in log.split("\n") if "undefined" in l.lower() and "There were" not in l]
    print(f"pass {i+1}: returncode={r.returncode} errors={len(errs)} undefined_lines={len(undef)}")
    if errs:
        print("first errors:", errs[:5])
print("pages line:", [l for l in log.split("\n") if "Output written" in l])
