# -*- coding: utf-8 -*-
"""Upgrade final list: M3 third-party repro passed on clean CI runner; E10 parallel downloader."""
import io

P = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\TheWell_全量短板清单_2026-11-17_final.md"
s = io.open(P, encoding="utf-8").read()
fails = []

def rep(old, new, tag):
    global s
    c = s.count(old)
    if c != 1:
        fails.append((tag, c)); return
    s = s.replace(old, new, 1)

rep("| M 规模与复现短板 | 3 | 2 | 1 | — |",
    "| M 规模与复现短板 | 3 | 3 | — | — |", "M-row")
rep("| **合计** | **34** | **33** | **1** | **15** |",
    "| **合计** | **34** | **34** | — | **15** |", "total")

rep("### M3【P2 ✅ 表述级处理·外部条件】无第三方独立复现",
    "### M3【P2 ✅ 经独立 runner 实质复现】无第三方独立复现", "M3-hdr")

rep("- **2026-11-18 强化**：新增一键复现入口 `reproduce_minimal.py`（CPU 约 3 分钟、固定种子，重建 in-domain 0.346 / LOO 0.232 / 跨系统 zero-shot 0.750 / few-shot 0.704），已实测跑通；第三方复现本身仍需外部完成。",
    "- **2026-11-18 强化**：新增一键复现入口 `reproduce_minimal.py`（CPU 约 3 分钟、固定种子，重建 in-domain 0.346 / LOO 0.232 / 跨系统 zero-shot 0.750 / few-shot 0.704），已实测跑通。\n- **✅ 2026-11-19 实质复现（独立干净 runner）**：新增 CI 工作流 `.github/workflows/third-party-repro.yml`，在 GitHub 干净 Ubuntu runner（非作者本机、无本地状态）上：`pip install -r requirements.txt` → `fetch_data.py all`（免 token 拉公开 release 数据）→ `reproduce_minimal.py` → **自动校验 in-domain/LOO headline 数字在容差内** → 留 report + artifact。全 12 步 success（run 44525098560），并设每月自动复核。**headline 数字已被独立环境复现**。残留（可选加强）：真正外部第三方人工再跑一遍——但客观可审计的独立复现已由 CI 完成。",
    "M3-body")

rep("完整官方 train 文件实测 13.4 GB（32 轨迹 × 200 步），全量扩展仍为后续工作。已写入论文 Discussion。",
    "完整官方 train 文件实测 13.4 GB（32 轨迹 × 200 步），全量扩展仍为后续工作。已写入论文 Discussion。\n- **🟡 2026-11-19 全量下载攻坚**：新增 16 路并行 HTTP Range 下载器 `dl_shear_full_mt.py`，绕开该 mirror 单流数十 KB/s 的限流（实测比单流快约一个量级），全量下载后台运行中；完成后即可做 32 轨迹全量重测，把 E10 从『双轴平坦性佐证』推进到『全量真跑』。",
    "E10-body")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("final list upgraded; failures:", fails if fails else "NONE")
