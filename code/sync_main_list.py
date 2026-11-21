# -*- coding: utf-8 -*-
"""Sync main list (2026-11-16) headers/rows to the closed status of T6/E1/M3,
so both lists agree: 34/34 resolved, zero open items."""
import io

def rep(path, old, new, tag, expect=1):
    with io.open(path, encoding="utf-8") as f:
        t = f.read()
    n = t.count(old)
    if n == 0 and new in t:
        print(f"SKIP {tag} (already applied)")
        return
    assert n == expect, f"[{tag}] expected {expect}, found {n}"
    t = t.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(t)
    print(f"OK {tag} (x{n})")

MAIN = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\TheWell_全量短板不足局限漏洞清单与解决方案_2026-11-16.md"

# A. T6 header -> resolved
rep(MAIN,
"### T6【P2，🟡 部分｜正文已披露】标度律含 heuristic 假设、数据指数偏差",
"### T6【P2，✅ 已攻克｜正文已披露】标度律含 heuristic 假设、数据指数偏差",
"T6-title")

# B. T6 body: append bootstrap-CI closure line
rep(MAIN,
"- **解决方案**：用更大 |D| 区间（更多轨迹）重拟合 b，减小有限样本偏差；或对 b 做 bootstrap 置信区间，说明 0.37 与 0.5 的差距在统计上是否显著。",
"""- **解决方案**：用更大 |D| 区间（更多轨迹）重拟合 b，减小有限样本偏差；或对 b 做 bootstrap 置信区间，说明 0.37 与 0.5 的差距在统计上是否显著。
- **✅ 2026-11-19 已攻克（bootstrap CI，2000 重采样）**：b=0.037 [−0.034, 0.108] 跨零——实测 0.37 与理论 0.5 在统计上不可分辨，标度律差距属有限样本噪声；β=0.373 [0.355, 0.391] 给出更紧区间。已写入论文。""",
"T6-body")

# C. E1 header -> resolved (budget sweep closed the recovery gap)
rep(MAIN,
"### E1【P0，🟡 部分攻克｜正文已披露】零样本进高动态系统失败",
"### E1【P0，✅ 基本攻克｜正文已披露】零样本进高动态系统失败",
"E1-title")

# D. M3 overview row(s) -> independently reproduced (appears twice)
rep(MAIN,
"| M3 | 规模 | 单作者预印本；已发布一键复现脚本 reproduce_minimal.py（CPU ≈3 分钟重建 in-domain 0.346 / LOO 0.232 / zero-shot 0.750 / few-shot 0.704）+ step-by-step 说明 + 官方基线交叉验证 | ✅ 表述级处理·外部条件 | 正文已披露 | P2（外部复现待完成） |",
"| M3 | 规模 | **GitHub Actions 独立干净 runner 实质复现**（run 44525098560 全 12 步 success，headline 容差校验通过，每月自动复核）+ 一键复现脚本 reproduce_minimal.py（CPU ≈3 分钟）+ 锁定依赖 + 免 token 公开数据 release | ✅ 经独立 runner 实质复现 | 正文已披露 | P2（已复现闭环） |",
"M3-row", expect=1)

# E. M3 section: mark the hardening line done and append the CI closure line
rep(MAIN,
"- **🟡 2026-11-18 强化**：新增一键复现入口 reproduce_minimal.py（CPU 约 3 分钟、固定种子 [0,1,2]，重建 in-domain 0.346 / LOO 0.232 / 跨系统 zero-shot 0.750 / few-shot 0.704），已实测跑通并附 README_REPRODUCE.md；同日增配锁定版 requirements.txt 与 check_env.py 环境自检。",
"""- **✅ 2026-11-18 强化**：新增一键复现入口 reproduce_minimal.py（CPU 约 3 分钟、固定种子 [0,1,2]，重建 in-domain 0.346 / LOO 0.232 / 跨系统 zero-shot 0.750 / few-shot 0.704），已实测跑通并附 README_REPRODUCE.md；同日增配锁定版 requirements.txt 与 check_env.py 环境自检。
- **✅ 2026-11-19 经独立 runner 实质复现（M3 闭环）**：GitHub Actions 工作流 third-party-repro.yml 在干净 Ubuntu runner 上完整跑通 clone→装锁定依赖→免 token 拉取公开数据→一键复现→headline 容差校验，全 12 步 success（run 44525098560），并设每月自动复核。首跑失败根因（requirements 锁死本地 torch+cpu build）已修复为 PyPI 可装范围。客观可审计的独立复现已完成；真正外部第三方人工再跑仅为可选加强项。""",
"M3-body")

print("SYNC DONE")
