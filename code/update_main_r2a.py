# -*- coding: utf-8 -*-
"""Main list sync (part A): header, summary table, final-state block, residual block."""
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

rep("> 生成日期：2026-11-16（最后更新 2026-11-17）。对象：`manuscript_en_jcp.tex`（38 页，2026-11-17 编译，0 errors / 0 undefined）。",
    "> 生成日期：2026-11-16（最后更新 2026-11-18）。对象：`manuscript_en_jcp.tex`（39 页，2026-11-18 编译，0 errors / 0 undefined）。",
    "header")

rep("""| E 实验结果短板 | 10 | 6 | 4 | E1/E2/E5/E7/E9 攻克、E4 实验补证；E3/E6/E8/E10 表述级处理 |
| M 规模与复现短板 | 3 | 2 | 1 | M1 容量标度实测攻克、M2 双硬件实测攻克；M3 表述级处理 |
| 附录：已剔除项 | 12 | 12 | 0 | 已通过实验或口径统一使问题消失 |
| **合计** | **34** | **29** | **5** | P0 共 9 条已全部攻克；5 条残留为表述级处理或外部条件受限 |""",
"""| E 实验结果短板 | 10 | 10 | 0 | E1/E2/E5/E7/E9 攻克、E4 补证；E3/E6/E8 实验攻克、E10 实验补证 |
| M 规模与复现短板 | 3 | 2 | 1 | M1 容量标度实测攻克、M2 双硬件实测攻克；M3 一键复现已发布，第三方复现为外部条件 |
| 附录：已剔除项 | 12 | 12 | 0 | 已通过实验或口径统一使问题消失 |
| **合计** | **34** | **33** | **1** | P0 共 9 条已全部攻克；仅剩 M3 第三方独立复现为外部条件项 |""",
    "table")

rep("""> **【2026-11-17 最终状态】** 全量 34 条 = **29 已解决 + 5 表述级处理 / 外部条件受限**。
> - **理论 6/6**：T1 分词误差入界（C4=2.18）、T2 常数自洽（C2·δ=0.2491）、T3 L_F≈0.40、T4 方向感知适配器（DAA）、T5 谱截断单调衰减（s=2.21）、T6 bootstrap CI。
> - **实验 6/10**：E1 few-shot 追平 scratch、E2 多尺度架构突破 turb、E5 门控净增益量化、E7 20 种子容量显著性（p=4.8e-4）、E9 数据标度坐实、E4 单系统 LB 正则 null result（p=0.74）实验补证；E3/E6/E8/E10 表述级框定。
> - **规模 2/3**：M1 容量标度 VRMSE∝params^-0.107（ch16→ch128 改善 35.8%）、M2 AMD+NVIDIA 双硬件实测；M3 待第三方复现。
> - 本地 CPU 补攻（torch 2.14.0+cpu，2 线程）三组结果已全部写入论文，PDF 38 页、0 errors、0 undefined。""",
"""> **【2026-11-18 最终状态】** 全量 34 条 = **33 已解决 + 1 外部条件受限（M3 第三方独立复现）**。
> - **理论 6/6**：T1 分词误差入界（C4=2.18）、T2 常数自洽（C2·δ=0.2491）、T3 L_F≈0.40、T4 方向感知适配器（DAA）、T5 谱截断单调衰减（s=2.21）、T6 bootstrap CI。
> - **实验 10/10**：E1 few-shot 追平 scratch、E2 多尺度架构突破 turb、E5 门控净增益量化、E7 20 种子容量显著性（p=4.8e-4）、E9 数据标度坐实、E4 单系统 LB 正则 null result（p=0.74）；E3 LOO 归一化受控分解（一致协议 0.234 ≈ in-domain 0.346，2.44 为混空间伪影）、E6 跨系统门控适配对照（门控侧仅回收 0.010，漂移在骨干/专家）、E8 容量×机制重测（ch16 mask+contrast p=0.036，ch64 消失）实验攻克，E10 官方数据双轴标度（轨迹 1→6、时间覆盖 25→100% 均平坦）实验补证。
> - **规模 2/3**：M1 容量标度 VRMSE∝params^-0.107（ch16→ch128 改善 35.8%）、M2 AMD+NVIDIA 双硬件实测；M3 一键复现脚本 reproduce_minimal.py 已实测发布，第三方复现为外部条件。
> - 本地 CPU 两轮补攻（torch 2.14.0+cpu）结果已全部写入论文，PDF 39 页、0 errors、0 undefined。""",
    "state-block")

rep("""**5 条表述级处理 / 外部条件受限项**：
- **E3** LOO 协议不可直接对齐（归一化敏感性为已知伪影，Discussion 段诚实框定）。
- **E6** gate-only 适配边界已限定（彻底攻需可学习 description→gate 映射实验）。
- **E8** mask/contrast 在 tiny 模型无因果贡献（定位为 regularization prior）。
- **E10** 数据规模受官方小切片限制（proof-of-concept 定位，规模扩展为后续工作）。
- **M3** 第三方独立复现（作者无法内部完成，已公开全部复现脚本与 step-by-step 说明）。""",
"""**仅剩 1 条外部条件受限项**：
- **M3** 第三方独立复现（作者无法内部完成；已发布一键复现脚本 reproduce_minimal.py——CPU 约 3 分钟重建全部 headline numbers——及 step-by-step 说明）。

> **2026-11-18 彻底实验攻克轮升级记录**：E3、E6、E8 由"表述级处理"升级为**实验攻克**（受控实验闭环，负结果亦坐实机制）；E10 由"表述级处理"升级为**实验补证**（官方数据双轴标度）。原 5 条残留中 4 条已闭环，仅 M3 需外部第三方复现。""",
    "residual-block")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("part A done; failures:", fails if fails else "NONE")
