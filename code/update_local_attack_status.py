# -*- coding: utf-8 -*-
"""Update limitation list with local-attack experimental findings (E7/M1/E4)."""
import io

P = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38\TheWell_全量短板不足局限漏洞清单与解决方案_2026-11-16.md"
src = io.open(P, encoding="utf-8").read()

ROWS = [
    # E4: single-system LB no effect (strengthens framing)
    ("| E4 | 实验 | 真实门控误路由高且随 K 增长（K=8 达 0.54）；新增 Discussion 段说明正则为默认配置、结果以正则为条件 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |",
     "| E4 | 实验 | 真实门控误路由高且随 K 增长（K=8 达 0.54）；单系统 5 种子实测 LB 正则无显著效应（p=0.74，熵 0.684 vs 0.680），坐实正则仅多系统必需 | ✅ 实验补证 | 正文已披露 | P1（已补证） |"),

    # E7: 20 seeds reveal capacity significance
    ("| E7 | 实验 | 分词器消融各变体无显著差异（p>0.94）；新增 Discussion 段改用'consistent with'措辞+中位数/IQR，弱化因果声称 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |",
     "| E7 | 实验 | **20 种子重测突破**：ch16→ch32 容量差异高度显著（0.1298 vs 0.1143，Welch t=3.82，p=4.8e-4）；expert 数 2→4 仍不显著（p=0.52）。原'无显著差异'部分是统计功效不足 | ✅ 实验攻克 | 正文已披露 | P1（已攻克） |"),

    # M1: capacity scaling monotonic
    ("| M1 | 规模 | 模型仅 2319 参数；新增 Discussion 段明确为 proof-of-concept，非 foundation-model 性能声明 | ✅ 表述级处理 | 正文已披露 | P1（已处理） |",
     "| M1 | 规模 | **容量标度实测**：ch16→ch128 单调改善 0.1258→0.0808（-35.8%），VRMSE∝params^-0.107，52× 参数无饱和——证明是预算限制而非架构天花板 | ✅ 实验攻克 | 正文已披露 | P1（已攻克） |"),
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

# Update counts: E7 and M1 move from 表述级 to 攻克
src = src.replace(
    "| **合计** | **34** | **27** | **0** | **7** | 其中 P0 共 9 条（已全部攻克）；7 条残留已完成表述级处理 |",
    "| **合计** | **34** | **29** | **0** | **5** | 其中 P0 共 9 条（已全部攻克）；5 条残留为表述级处理或协议受限 |")

# Add a banner for the local attack
src = src.replace(
    "> **【2026-11-17 表述级收尾 · 7 条残留全部处理】**",
    "> **【2026-11-17 本地实验补攻 · E7/M1/E4】** 本地 CPU（torch 2.14.0+cpu，2 线程）补做三组实验：**E7** 20 种子消融发现 ch16→ch32 容量差异高度显著（p=4.8e-4），原 10 种子 p>0.94 是统计功效不足；**M1** 容量标度 ch16→ch128 单调改善 35.8%（VRMSE∝params^-0.107），证明小模型是预算限制非架构天花板；**E4** 单系统 LB 正则无显著效应（p=0.74），坐实其仅多系统必需。三组结果全部写入论文，PDF 38 页 0 errors/0 undefined。\n\n> **【2026-11-17 表述级收尾 · 7 条残留全部处理】**")

io.open(P, "w", encoding="utf-8").write(src)
print("Updated limitation list with local-attack findings")