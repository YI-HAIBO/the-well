# -*- coding: utf-8 -*-
"""Sync both limitation lists with the residual-boundary push round (B1/B2/B3/B4)."""
import io

BASE = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
fails = []

def edit(path, pairs):
    s = io.open(path, encoding="utf-8").read()
    for old, new, tag in pairs:
        c = s.count(old)
        if c != 1:
            fails.append((tag, c))
            continue
        s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)

# ---------- final consolidated list ----------
FIN = BASE + r"\TheWell_全量短板清单_2026-11-17_final.md"
edit(FIN, [
    ("> 对象：`manuscript_en_jcp.tex`（39 页，2026-11-18 编译，0 errors / 0 undefined）",
     "> 对象：`manuscript_en_jcp.tex`（40 页，2026-11-18 编译，0 errors / 0 undefined）",
     "fin-header"),
    ("论文 39 页 0 errors/0 undefined。",
     "论文 39 页 0 errors/0 undefined。\n>\n> **2026-11-18 残留边界推进轮**：E1 预算扫描（full-param 适配 200 步即达 scratch 水平 0.6993 vs 0.6996，p=0.65；4 标签窗口仅升至 0.7134——恢复差距随预算闭合，残留成本=需要标签本身）、M2 闭环基准（100 步 autoregressive rollout，CPU 0.33–1.25 ms/步、AMD GPU 0.22–0.26 ms/步，与开环一致，反馈路径无额外开销）均已写入论文（40 页）；E10 全量 13.4 GB 下载实测带宽受限（单流数十 KB/s，8.5 分钟 0 字节），断点续传器 dl_shear_full.py 入库，维持 proof-of-concept 披露；M3 增配锁定版 requirements.txt 与 check_env.py 环境自检。",
     "fin-banner"),
])

# ---------- main full list ----------
MAIN = BASE + r"\TheWell_全量短板不足局限漏洞清单与解决方案_2026-11-16.md"
edit(MAIN, [
    ("> 生成日期：2026-11-16（最后更新 2026-11-18）。对象：`manuscript_en_jcp.tex`（39 页，2026-11-18 编译，0 errors / 0 undefined）。",
     "> 生成日期：2026-11-16（最后更新 2026-11-18）。对象：`manuscript_en_jcp.tex`（40 页，2026-11-18 编译，0 errors / 0 undefined）。",
     "main-header"),
    ("- **✅ 2026-11-16 部分攻克（方案 B）**：归一化 64×64 协议下真梯度 few-shot，1/5/10/20/50 步 VRMSE=1.364/1.289/1.210/1.099/**0.995**，单调下降且 50 步越过均值基线，两次独立运行完全一致，已写入论文（line 369 后）。**残留边界**：需少量目标标签（非零样本），只恢复到基线附近（0.995）而非 scratch 的 0.71。",
     "- **✅ 2026-11-16 部分攻克（方案 B）**：归一化 64×64 协议下真梯度 few-shot，1/5/10/20/50 步 VRMSE=1.364/1.289/1.210/1.099/**0.995**，单调下降且 50 步越过均值基线，两次独立运行完全一致，已写入论文（line 369 后）。**残留边界**：需少量目标标签（非零样本），只恢复到基线附近（0.995）而非 scratch 的 0.71。\n- **✅ 2026-11-18 残留边界推进（适配预算扫描，3 种子）**：shear→turb full-param 适配 50/200/600/1800 步，held-out（32 个未见轨迹窗口）VRMSE 0.751(零样本)→0.706(50 步)→**0.6993±0.0009(200 步) = scratch 0.6996±0.0011（Welch p=0.65）**，1800 步仍平（p=0.62）；标签窗口 16→4 仅升至 0.7134±0.0005。**恢复差距随适配预算闭合，残留成本=需要标签本身，不存在恢复天花板**。已写入论文。",
     "main-E1"),
    ("- **残留边界（不属原 34 条）**：仍无 closed-loop 端到端延迟基准，在线部署主张止于单步推理延迟——列为后续工作备注。",
     "- **残留边界（不属原 34 条）**：仍无 closed-loop 端到端延迟基准，在线部署主张止于单步推理延迟——列为后续工作备注。\n- **✅ 2026-11-18 残留边界闭合（闭环基准）**：100 步 autoregressive rollout（预测回灌、batch-1、64×64）：CPU 每步 0.33/0.37/0.55/1.25 ms（ch16/32/64/128，100 步共 34–135 ms）、AMD GPU 0.22–0.26 ms/步——与开环单步一致，**反馈路径无可测开销，GPU 100 步 rollout <30 ms**。已写入论文。",
     "main-M2"),
    ("- **🟡 2026-11-18 强化（外部条件项）**：新增一键复现入口 reproduce_minimal.py（CPU 约 3 分钟、固定种子 [0,1,2]，重建 in-domain 0.346 / LOO 0.232 / 跨系统 zero-shot 0.750 / few-shot 0.704），已实测跑通并附 README_REPRODUCE.md；第三方复现本身仍需外部完成。",
     "- **🟡 2026-11-18 强化（外部条件项）**：新增一键复现入口 reproduce_minimal.py（CPU 约 3 分钟、固定种子 [0,1,2]，重建 in-domain 0.346 / LOO 0.232 / 跨系统 zero-shot 0.750 / few-shot 0.704），已实测跑通并附 README_REPRODUCE.md；同日增配锁定版 requirements.txt 与 check_env.py 环境自检；第三方复现本身仍需外部完成。",
     "main-M3"),
    ("- **✅ 2026-11-18 实验补证（官方数据双轴标度）**：训练轨迹数 1→6（固定评测轨迹）VRMSE 仅 0.544→0.535；时间覆盖 25%→100% 仅 0.458→0.454——两轴均平坦，独立于正文轨迹数结果坐实 estimation-dominated regime。完整官方 train 文件实测 13.4 GB（32 轨迹 × 200 步），全量扩展仍为后续工作。已写入论文 Discussion。",
     "- **✅ 2026-11-18 实验补证（官方数据双轴标度）**：训练轨迹数 1→6（固定评测轨迹）VRMSE 仅 0.544→0.535；时间覆盖 25%→100% 仅 0.458→0.454——两轴均平坦，独立于正文轨迹数结果坐实 estimation-dominated regime。完整官方 train 文件实测 13.4 GB（32 轨迹 × 200 步），全量扩展仍为后续工作。已写入论文 Discussion。\n- **🟡 2026-11-18 全量数据尝试（残留边界核实）**：可断点续传分块下载器 dl_shear_full.py 已入库；实测该链路单流仅数十 KB/s（8.5 分钟 0 字节），全量传输带宽受限——维持 proof-of-concept 披露边界，下载器留作后续基础设施。",
     "main-E10"),
])

print("lists updated; failures:", fails if fails else "NONE")
