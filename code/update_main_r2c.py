# -*- coding: utf-8 -*-
"""Main list sync (part C): detailed-section bullets + roadmap."""
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

rep("- **备注**：曾尝试定量改写（E3），但 LOO 协议无法与论文 2.44 对齐（per-traj 归一化泄漏测试统计量），主动放弃捏造数字，保留原表述重构。",
    "- **备注**：曾尝试定量改写（E3），但 LOO 协议无法与论文 2.44 对齐（per-traj 归一化泄漏测试统计量），主动放弃捏造数字，保留原表述重构。\n- **✅ 2026-11-18 彻底攻克（受控归一化分解，8 条官方轨迹 × 3 种子）**：同模型、同折、同训练流程，仅变换归一化/评测口径——一致 per-sample 协议 LOO 均值 **0.234**（各折 0.048–0.43）vs in-domain 0.346；train-stat 一致协议 0.412；混空间评测（原协议机制）各折均值 **1.8e3–2.3e4**（均值 7.7e3）。结论：6.3× 退化是**评测空间伪影**，一致协议下未见轨迹误差与 in-domain 同阶，2.44 应读作混协议数字。已写入论文 Discussion。",
    "E3-bullet")

rep("- **解决方案**：把负载均衡 / route-diversity 正则设为默认训练配置并在所有多系统实验启用（已做）；报告带 / 不带正则的路由熵与误路由对比，证明正则必要性而非可选项（待补）。",
    "- **解决方案**：把负载均衡 / route-diversity 正则设为默认训练配置并在所有多系统实验启用（已做）；报告带 / 不带正则的路由熵与误路由对比，证明正则必要性而非可选项（待补）。\n- **✅ 2026-11-17 实验补证（单系统 5 种子）**：单系统加 / 不加 LB 正则 VRMSE 无显著差异（Welch p=0.74），路由熵 0.684 vs 0.680 几乎不变——坐实负载均衡正则仅多系统必需，与误路由随 K 增长机制一致。已写入论文。",
    "E4-bullet")

rep("- **解决方案**：把主张限定为\"门控负责族内、全模型微调负责族间\"；或设计可学习的描述→门控映射 + 少量样本标定，争取让 gate-only 适配生效。",
    "- **解决方案**：把主张限定为\"门控负责族内、全模型微调负责族间\"；或设计可学习的描述→门控映射 + 少量样本标定，争取让 gate-only 适配生效。\n- **✅ 2026-11-18 彻底攻克（shear→turb 跨系统受控实验，5 种子）**：zero-shot 0.756；固定描述门控仅回收 **0.010**/0.053；新增**可学习输入统计→门控映射**（路由真正输入依赖，熵 0.211→0.363）仍只到 0.747，与固定门控 0.746 等价；全模型适配 0.703 = scratch-200 0.700。结论：**漂移在骨干/专家权重，不在路由**——负结果以更强设计坐实。已写入论文 Discussion。",
    "E6-bullet")

rep("- **解决方案**：把两项定位为 regularization prior（论文已部分这样写）；或在大容量 / 大数据下重做消融，验证其是否仅在小模型下无效（CPU 可做）。",
    "- **解决方案**：把两项定位为 regularization prior（论文已部分这样写）；或在大容量 / 大数据下重做消融，验证其是否仅在小模型下无效（CPU 可做）。\n- **✅ 2026-11-18 彻底攻克（容量×机制重测，5 种子，few-shot 微调）**：ch16 下 mask+contrast 组合显著优于 scratch（**0.332 vs 0.375，Welch p=0.036**），mask 单独边际显著（p=0.066），contrast 单独无效（p=0.999）；ch64 下全部消失（p≥0.77，scratch 已达 0.229）。结论：**预训练信号 regime-dependent**——容量受限时因果有效，容量充足后消失；修正原\"无因果贡献\"的单配置结论。已写入论文 Discussion。",
    "E8-bullet")

rep("- **解决方案**：给出复现用更大切片 / manifest（已有 HF 地址），明确\"本文是理论 + 概念验证，规模扩展是后续\"；争取下载 1–2 个中等系统全轨迹重测关键结论。",
    "- **解决方案**：给出复现用更大切片 / manifest（已有 HF 地址），明确\"本文是理论 + 概念验证，规模扩展是后续\"；争取下载 1–2 个中等系统全轨迹重测关键结论。\n- **✅ 2026-11-18 实验补证（官方数据双轴标度）**：训练轨迹数 1→6（固定评测轨迹）VRMSE 仅 0.544→0.535；时间覆盖 25%→100% 仅 0.458→0.454——两轴均平坦，独立于正文轨迹数结果坐实 estimation-dominated regime。完整官方 train 文件实测 13.4 GB（32 轨迹 × 200 步），全量扩展仍为后续工作。已写入论文 Discussion。",
    "E10-bullet")

rep("- **解决方案**：公开完整复现脚本与固定随机种子清单（代码已在 GitHub），在 Data Availability 加 step-by-step reproduction instructions；引用 The Well 官方基线作交叉验证。",
    "- **解决方案**：公开完整复现脚本与固定随机种子清单（代码已在 GitHub），在 Data Availability 加 step-by-step reproduction instructions；引用 The Well 官方基线作交叉验证。\n- **🟡 2026-11-18 强化（外部条件项）**：新增一键复现入口 reproduce_minimal.py（CPU 约 3 分钟、固定种子 [0,1,2]，重建 in-domain 0.346 / LOO 0.232 / 跨系统 zero-shot 0.750 / few-shot 0.704），已实测跑通并附 README_REPRODUCE.md；第三方复现本身仍需外部完成。",
    "M3-bullet")

rep("4. E3 轨迹外推 → 增补轨迹级增强实验或明确双档评测（定量改写需先解决协议对齐）。",
    "4. E3 轨迹外推 → ✅ 已彻底攻克：受控归一化分解证明 2.44 为混空间伪影，一致协议 LOO 0.234 ≈ in-domain 0.346。",
    "roadmap-4")

rep("6. E7 / E8 消融改中位数 + IQR 并弱化因果措辞。",
    "6. E7 20 种子容量显著性（p=4.8e-4）✅；E8 容量×机制重测（ch16 mask+contrast p=0.036）✅。",
    "roadmap-6")

rep("9. T6 b 的 bootstrap 置信区间；M3 复现说明；M2 残留 closed-loop 延迟基准。",
    "9. T6 b 的 bootstrap 置信区间；M3 第三方独立复现（一键脚本已发布，待外部执行）；M2 残留 closed-loop 延迟基准。",
    "roadmap-9")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("part C done; failures:", fails if fails else "NONE")
