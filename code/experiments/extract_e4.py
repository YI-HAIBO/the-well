# -*- coding: utf-8 -*-
import io, sys, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
t = io.open(r"C:\Users\Yi\WorkBuddy\2026-11-09-18-02-06\outputs\选题方案\03_TheWell_物理世界模型基础模型.md", encoding="utf-8").read()
lines = t.split("\n")

def show(pat, ctx=1, maxhits=8):
    print("=" * 70)
    print("PATTERN:", pat)
    n = 0
    for i, l in enumerate(lines):
        if re.search(pat, l):
            print("--- line %d ---" % (i + 1))
            for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
                print("  %d| %s" % (j + 1, lines[j][:200]))
            n += 1
            if n >= maxhits:
                break

show(r"E4\b")
show(r"定理 1")
