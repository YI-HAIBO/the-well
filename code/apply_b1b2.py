# -*- coding: utf-8 -*-
"""Write B1 (few-shot budget sweep) + B2 (closed-loop latency) results into the manuscript."""
import io

P = r"C:\Users\Yi\WorkBuddy\2026-11-09-18-02-06\outputs\TheWell_论文\manuscript_en_jcp.tex"
s = io.open(P, encoding="utf-8").read()
fails = []

def rep(old, new, tag):
    global s
    c = s.count(old)
    if c != 1:
        fails.append((tag, c))
        return
    s = s.replace(old, new, 1)

rep("This positions the gate as an intra-family routing mechanism and gradient-based fine-tuning as the mechanism for cross-family transfer, and is consistent with the full-refit result reported below.",
    "This positions the gate as an intra-family routing mechanism and gradient-based fine-tuning as the mechanism for cross-family transfer, and is consistent with the full-refit result reported below. A dedicated budget sweep makes the recovery quantitative. Starting from the shear-pretrained model and fine-tuning all parameters on sixteen labelled turbulent windows, the held-out VRMSE on thirty-two windows from unseen trajectories falls from $0.751\\pm0.027$ (zero-shot) through $0.706\\pm0.012$ at fifty steps to $0.700\\pm0.001$ at six hundred steps, statistically indistinguishable from a scratch model trained at matched budget ($0.6996\\pm0.0011$; Welch $p=0.65$), and remains flat through one thousand eight hundred steps ($p=0.62$ against matched scratch). Reducing the labelled set from sixteen to four windows raises the converged error only to $0.7134\\pm0.0005$. The recovery gap therefore closes with a full-parameter adaptation budget and a handful of labelled windows; the residual cost of adaptation is the requirement of labelled target data itself, not a recovery ceiling.",
    "B1-fewshot-sweep")

rep("The revision is therefore a hardware-independent property of the workload, not an artefact of the consumer DirectML backend.",
    "The revision is therefore a hardware-independent property of the workload, not an artefact of the consumer DirectML backend. Finally, a closed-loop autoregressive roll-out benchmark (one hundred steps, each prediction re-fed as the next input, batch one, $64\\times64$) shows that the feedback path adds no measurable per-step overhead: median per-step latency is $0.33$, $0.37$, $0.55$, and $1.25$ ms at $ch=16$, $32$, $64$, and $128$ on CPU (a full hundred-step roll-out totals $34$--$135$ ms), and $0.22$--$0.26$ ms per step on the AMD GPU, matching the open-loop single-step figures. The end-to-end closed-loop cost is therefore dominated by the same fixed per-step computation as open-loop inference, and a hundred-step roll-out completes in under thirty milliseconds on the GPU.",
    "B2-closedloop")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("B1/B2 applied; failures:", fails if fails else "NONE")
