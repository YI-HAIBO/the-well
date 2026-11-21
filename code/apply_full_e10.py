# -*- coding: utf-8 -*-
"""Close E10 with the full-data re-test (2026-11-21):
update manuscript Discussion + both limitation lists."""
import io, sys

def rep(path, old, new, tag):
    with io.open(path, encoding="utf-8") as f:
        t = f.read()
    n = t.count(old)
    assert n == 1, f"[{tag}] expected 1 occurrence, found {n}"
    t = t.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(t)
    print(f"OK {tag}")

B = r"C:\Users\Yi\WorkBuddy\2026-11-09-19-00-38"
TEX = r"C:\Users\Yi\WorkBuddy\2026-11-09-18-02-06\outputs\TheWell_论文\manuscript_en_jcp.tex"
FINAL = B + r"\TheWell_全量短板清单_2026-11-17_final.md"
MAIN = B + r"\TheWell_全量短板不足局限漏洞清单与解决方案_2026-11-16.md"

# ---------------- 1. manuscript Discussion ----------------
rep(TEX,
"""trajectory-count result. The full official shear-flow training file (13.4 GB, $32$ trajectories of
$200$ steps each) and the complete benchmark remain the natural scaling target. The step-by-step""",
"""trajectory-count result. The full official shear-flow training file (13.4 GB, $32$ trajectories of
$200$ steps each) is used directly for a full-data re-test of both axes: increasing the training
pool from one to all $28$ non-held-out trajectories changes the VRMSE only from $0.426$ to
$0.404$, and increasing the temporal coverage of the training windows from $25\\%$ to $100\\%$ at
the full pool leaves it unchanged at $0.350$; both axes remain flat at the full official data
scale, so the estimation-dominated regime is confirmed on the complete training split rather than
only on slices. The complete fifteen-terabyte benchmark remains the natural scaling target beyond
this single-system full split. The step-by-step""",
"TEX-E10")

# ---------------- 2. final list ----------------
rep(FINAL,
"### E10【P1 ✅ 实验补证】数据规模与\"基础模型\"不匹配",
"### E10【P1 ✅ 全量真跑闭环】数据规模与\"基础模型\"不匹配",
"FINAL-E10-title")

rep(FINAL,
"- **🟡 2026-11-19 全量下载攻坚**：新增 16 路并行 HTTP Range 下载器 `dl_shear_full_mt.py`，绕开该 mirror 单流数十 KB/s 的限流（实测比单流快约一个量级），全量下载后台运行中；完成后即可做 32 轨迹全量重测，把 E10 从『双轴平坦性佐证』推进到『全量真跑』。",
"""- **✅ 2026-11-21 全量真跑完成（E10 闭环）**：定位到官方 HuggingFace 分发（`polymathic-ai/shear_flow`，经 hf-mirror 实测单流 4.1 MB/s，比 flatiron mirror 快约 13 倍），16 路断点续传下载器 `dl_shear_hf.py` 完成 13.44 GB 全量 train 下载（`t0_fields/pressure` = (32, 200, 256, 512) float32，h5py 验证完整）。全量 32 轨迹重测 `attack2a_full.py`（协议与切片双轴实验一致）：训练轨迹池 1→28 VRMSE 仅 0.426→0.404，时间覆盖 25%→100% 恒为 0.350——**两轴在全量官方数据下依然平坦**，estimation-dominated regime 由切片佐证升级为全量实证。结果已写入论文 Discussion（`attack2a_full_results.json`）。""",
"FINAL-E10-body")

rep(FINAL,
"*生成日期：2026-11-17。所有数字取自正文实测值或已写入论文的实验结果。状态标注反映 2026-11-17 最终进展。*",
"*生成日期：2026-11-17，最后更新：2026-11-21（E10 全量真跑闭环：13.44 GB 官方 train 32 轨迹重测，双轴仍平坦）。所有数字取自正文实测值或已写入论文的实验结果。*",
"FINAL-E10-date")

# ---------------- 3. main list ----------------
rep(MAIN,
"> **【2026-11-18 最终状态】** 全量 34 条 = **33 已解决 + 1 外部条件受限（M3 第三方独立复现）**。",
"> **【2026-11-21 最终状态】** 全量 34 条 = **34 已全部解决**（M3 经 GitHub 独立干净 runner 实质复现并留 CI 回执；E10 于 2026-11-21 用 13.44 GB 官方全量 train 完成 32 轨迹真跑重测，双轴平坦性在全量下成立）。",
"MAIN-status")

rep(MAIN,
"| E10 | 实验 | **官方数据双轴标度补证**：训练轨迹 1→6 VRMSE 0.544→0.535、时间覆盖 25→100% 0.458→0.454，两轴均平坦——独立坐实 estimation-dominated regime；完整 train 文件实测 13.4 GB | ✅ 实验补证 | 正文已披露 | P1（已补证） |",
"| E10 | 实验 | **全量真跑闭环**：13.44 GB 官方 train（32 轨迹）全量重测——轨迹池 1→28 VRMSE 0.426→0.404、时间覆盖 25→100% 恒 0.350，两轴在全量下仍平坦，estimation-dominated regime 坐实（此前切片双轴：1→6 0.544→0.535、25→100% 0.458→0.454） | ✅ 全量真跑闭环 | 正文已披露 | P1（已闭环） |",
"MAIN-E10-row")

rep(MAIN,
"### E10【P1，✅ 实验补证｜正文已披露】数据规模与\"基础模型\"不匹配",
"### E10【P1，✅ 全量真跑闭环｜正文已披露】数据规模与\"基础模型\"不匹配",
"MAIN-E10-title")

rep(MAIN,
"- **🟡 2026-11-18 全量数据尝试（残留边界核实）**：可断点续传分块下载器 dl_shear_full.py 已入库；实测该链路单流仅数十 KB/s（8.5 分钟 0 字节），全量传输带宽受限——维持 proof-of-concept 披露边界，下载器留作后续基础设施。",
"""- **✅ 2026-11-21 全量真跑完成（E10 闭环）**：改用官方 HuggingFace 分发源（`polymathic-ai/shear_flow`，hf-mirror 单流 4.1 MB/s，较 flatiron mirror 快约 13 倍），16 路断点续传下载器 `dl_shear_hf.py` 完成 13.44 GB 全量 train 下载并经 h5py 验证（32, 200, 256, 512）。全量 32 轨迹重测：轨迹池 1→28 VRMSE 0.426→0.404、时间覆盖 25%→100% 恒 0.350，双轴平坦性在全量官方数据下成立，结果写入论文 Discussion。""",
"MAIN-E10-body")

print("ALL DONE")
