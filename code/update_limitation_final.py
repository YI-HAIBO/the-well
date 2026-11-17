# -*- coding: utf-8 -*-
"""Update limitation-list status after Discussion and Limitations section added.
Marks E3/E4/E6/E7/E8/E10/M1/M3 as 'disclosed and framed' (表述级处理完成)."""
import io

P = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\TheWell_全量短板不足局限漏洞清单与解决方案_2026-11-16.md"
src = io.open(P, encoding="utf-8").read()

ROWS = [
    # E3: LOO protocol - now explicitly discussed in new section
    ("| E3 | 实验 | 轨迹级外推严重退化（shear LOO 2.44 = 6.3×） | 🔴 仍在 | 正文已披露 | P0 |",
     "| E3 | 实验 | 轨迹级外推严重退化（shear LOO 2.44 = 6.3×）；新增 Discussion 段明确归一化敏感性为已知伪影，不主张干净外推测量 | ✅ 表述级处理 | 正文已披露 | P0（已处理） |"),

    # E4: routing regex - now discussed in new section
    ("| E4 | 实验 | 真实门控误路由高且随 K 增长（K=8 达 0.54），自然训练坍缩 | 🔴 仍在 | 正文已披露 | P1 |",
     "| E4 | 实验 | 真实门控误路由高且随 K 增长（K=8 达 0.54）；新增 Discussion 段说明正则为默认配置、结果以正则为条件 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |"),

    # E6: gate-only adaptation - now discussed
    ("| E6 | 实验 | 门控 / 描述适配无法补偿算子漂移，需全模型重训 | 🔴 仍在 | 正文已披露 | P1 |",
     "| E6 | 实验 | 门控 / 描述适配无法补偿算子漂移；新增 Discussion 段限定为'门控负责族内、全模型微调负责族间' | ✅ 表述级处理 | 正文已披露 | P1（已处理） |"),

    # E7: tokenizer ablation stats - now reframed
    ("| E7 | 实验 | 分词器消融各变体无显著差异（p>0.94），卖点缺统计支撑 | 🔴 仍在 | 正文已披露 | P1 |",
     "| E7 | 实验 | 分词器消融各变体无显著差异（p>0.94）；新增 Discussion 段改用'consistent with'措辞+中位数/IQR，弱化因果声称 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |"),

    # E8: mask/contrast pretraining - now reframed
    ("| E8 | 实验 | mask / contrast 预训练项无贡献（p>0.99） | 🔴 仍在 | 正文已披露 | P1 |",
     "| E8 | 实验 | mask / contrast 预训练项无贡献（p>0.99）；新增 Discussion 段定位为 regularization prior，不声称因果效应 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |"),

    # E10: data scale - now discussed
    ("| E10 | 实验 | 数据规模过小（76/102 样本），完整 15TB v2.0 未用 | 🔴 仍在 | 正文已披露 | P1 |",
     "| E10 | 实验 | 数据规模过小（76/102 样本）；新增 Discussion 段明确为 proof-of-concept，规模扩展为后续工作 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |"),

    # M1: model scale - now discussed
    ("| M1 | 规模 | 模型仅 2319 参数，概念验证非真基础模型 | 🔴 仍在 | 正文已披露 | P1 |",
     "| M1 | 规模 | 模型仅 2319 参数；新增 Discussion 段明确为 proof-of-concept，非 foundation-model 性能声明 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |"),

    # M3: third-party reproduction - now discussed
    ("| M3 | 规模 | 单作者预印本，无第三方独立复现 | ⚪ 未解决 | 正文未披露 | P2 |",
     "| M3 | 规模 | 单作者预印本，无第三方独立复现；新增 Discussion 段 + Data Availability 已含 step-by-step 复现说明与官方基线交叉验证 | ✅ 表述级处理 | 正文已披露 | P2（已处理） |"),
]

fails = []
for old, new in ROWS:
    c = src.count(old)
    if c != 1:
        fails.append((old[:60], c))
        continue
    src = src.replace(old, new, 1)

if fails:
    print("FAILED ROWS:")
    for s, c in fails:
        print(" ", s, "matches=", c)
    raise SystemExit(1)

# Update summary table
src = src.replace(
    "| **合计** | **34** | **26** | **0** | **8** | 其中 P0 共 9 条（已全部攻克） |",
    "| **合计** | **34** | **26** | **0** | **8** | 其中 P0 共 9 条（已全部攻克）；8 条残留已完成表述级处理 |")

# Add banner for the limitations section
src = src.replace(
    "> **【2026-11-17 全量攻关 · 9 条攻克】**",
    "> **【2026-11-17 表述级收尾 · 8 条残留全部处理】** 新增 Discussion and Limitations 段（5 个子节），将 E3/E4/E6/E7/E8/E10/M1/M3 逐条写入论文正文做诚实框定：LOO 归一化敏感性、正则条件依赖、gate-only 适用边界、消融统计口径（consistent with + 中位数/IQR）、数据与模型规模定位为 proof-of-concept、复现说明与官方基线交叉验证。PDF 36 页（原 34 页增 2 页）、0 errors、0 undefined。**9 条攻克 + 8 条表述级处理 = 全量 34 条全部闭环。**\n\n> **【2026-11-17 全量攻关 · 9 条攻克】**")

io.open(P, "w", encoding="utf-8").write(src)
print("Updated limitation list: 8 rows + summary + banner")