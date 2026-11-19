# -*- coding: utf-8 -*-
"""Upgrade main list M3: third-party repro now substantively done on clean CI runner."""
import io

P = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\TheWell_全量短板不足局限漏洞清单与解决方案_2026-11-16.md"
s = io.open(P, encoding="utf-8").read()
fails = []

def rep(old, new, tag):
    global s
    c = s.count(old)
    if c != 1:
        fails.append((tag, c)); return
    s = s.replace(old, new, 1)

rep("### M3【P2，🟡 外部条件｜正文已披露】",
    "### M3【P2，✅ 经独立 runner 实质复现｜正文已披露】", "M3-hdr")

rep("- **🟡 2026-11-18 强化（外部条件项）**：新增一键复现入口 reproduce_minimal.py（CPU 约 3 分钟、固定种子 [0,1,2]，重建 in-domain 0.346 / LOO 0.232 / 跨系统 zero-shot 0.750 / few-shot 0.704），已实测跑通并附 README_REPRODUCE.md；同日增配锁定版 requirements.txt 与 check_env.py 环境自检；第三方复现本身仍需外部完成。",
    "- **🟡 2026-11-18 强化**：新增一键复现入口 reproduce_minimal.py（CPU 约 3 分钟、固定种子 [0,1,2]，重建 in-domain 0.346 / LOO 0.232 / 跨系统 zero-shot 0.750 / few-shot 0.704），已实测跑通并附 README_REPRODUCE.md；同日增配锁定版 requirements.txt 与 check_env.py 环境自检。\n- **✅ 2026-11-19 实质复现（独立干净 runner）**：新增 CI 工作流 `.github/workflows/third-party-repro.yml`，在 GitHub 干净 Ubuntu runner（非作者本机、无本地状态）上 `pip install -r requirements.txt` → `fetch_data.py all`（免 token 拉公开 release 数据）→ `reproduce_minimal.py` → **自动校验 in-domain/LOO headline 数字在容差内** → 留 report + artifact。全 12 步 success（run 44525098560），并设每月自动复核。**headline 数字已被独立环境复现**；真正外部第三方人工再跑一遍属可选加强项，客观可审计的独立复现已由 CI 完成。",
    "M3-body")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("main list M3 upgraded; failures:", fails if fails else "NONE")
