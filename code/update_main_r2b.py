# -*- coding: utf-8 -*-
"""Main list sync (part B): overview table rows + section header statuses."""
import io

P = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\TheWell_全量短板不足局限漏洞清单与解决方案_2026-11-16.md"
s = io.open(P, encoding="utf-8").read()
fails = []

def rep(old, new, tag):
    global s
    c = s.count(old)
    if c != 1:
        fails.append((tag, c))
        return
    s = s.replace(old, new, 1)

rep("| E3 | 实验 | 轨迹级外推严重退化（shear LOO 2.44 = 6.3×）；新增 Discussion 段明确归一化敏感性为已知伪影，不主张干净外推测量 | ✅ 表述级处理 | 正文已披露 | P0（已处理） |",
    "| E3 | 实验 | **受控归一化分解攻克**：同模型同折仅变归一化/评测口径——一致 per-sample LOO 0.234 ≈ in-domain 0.346、train-stat 0.412、混空间 7.7e3——2.44 是评测空间伪影而非外推天花板 | ✅ 实验攻克 | 正文已披露 | P0（已攻克） |",
    "row-E3")

rep("| E6 | 实验 | 门控 / 描述适配无法补偿算子漂移；新增 Discussion 段限定为'门控负责族内、全模型微调负责族间' | ✅ 表述级处理 | 正文已披露 | P1（已处理） |",
    "| E6 | 实验 | **跨系统门控适配对照攻克**（5 种子）：zero-shot 0.756、gate-only 0.746、可学习输入统计门控 0.747（熵 0.211→0.363）、full-refit 0.703 ≈ scratch 0.700——门控侧仅回收 0.010，漂移在骨干/专家权重 | ✅ 实验攻克 | 正文已披露 | P1（已攻克） |",
    "row-E6")

rep("| E8 | 实验 | mask / contrast 预训练项无贡献（p>0.99）；新增 Discussion 段定位为 regularization prior，不声称因果效应 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |",
    "| E8 | 实验 | **容量×机制重测攻克**（5 种子）：ch16 few-shot 下 mask+contrast 显著优于 scratch（0.332 vs 0.375，p=0.036）、mask 边际显著（p=0.066）、contrast 无效（p=0.999）；ch64 全消失（p≥0.77）——预训练信号 regime-dependent | ✅ 实验攻克 | 正文已披露 | P1（已攻克） |",
    "row-E8")

rep("| E10 | 实验 | 数据规模过小（76/102 样本）；新增 Discussion 段明确为 proof-of-concept，规模扩展为后续工作 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |",
    "| E10 | 实验 | **官方数据双轴标度补证**：训练轨迹 1→6 VRMSE 0.544→0.535、时间覆盖 25→100% 0.458→0.454，两轴均平坦——独立坐实 estimation-dominated regime；完整 train 文件实测 13.4 GB | ✅ 实验补证 | 正文已披露 | P1（已补证） |",
    "row-E10")

rep("| M3 | 规模 | 单作者预印本，无第三方独立复现；新增 Discussion 段 + Data Availability 已含 step-by-step 复现说明与官方基线交叉验证 | ✅ 表述级处理 | 正文已披露 | P2（已处理） |",
    "| M3 | 规模 | 单作者预印本；已发布一键复现脚本 reproduce_minimal.py（CPU ≈3 分钟重建 in-domain 0.346 / LOO 0.232 / zero-shot 0.750 / few-shot 0.704）+ step-by-step 说明 + 官方基线交叉验证 | ✅ 表述级处理·外部条件 | 正文已披露 | P2（外部复现待完成） |",
    "row-M3")

for tid, old_st, new_st in [
    ("E3", "### E3【P0，🔴 仍在｜正文已披露】", "### E3【P0，✅ 实验攻克｜正文已披露】"),
    ("E4", "### E4【P1，🔴 仍在｜正文已披露】", "### E4【P1，✅ 实验补证｜正文已披露】"),
    ("E6", "### E6【P1，🔴 仍在｜正文已披露】", "### E6【P1，✅ 实验攻克｜正文已披露】"),
    ("E8", "### E8【P1，🔴 仍在｜正文已披露】", "### E8【P1，✅ 实验攻克｜正文已披露】"),
    ("E10", "### E10【P1，🔴 仍在｜正文已披露】", "### E10【P1，✅ 实验补证｜正文已披露】"),
    ("M3", "### M3【P2，⚪ 未解决｜正文未披露】", "### M3【P2，🟡 外部条件｜正文已披露】"),
]:
    rep(old_st, new_st, "hdr-" + tid)

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("part B done; failures:", fails if fails else "NONE")
