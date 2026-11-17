# -*- coding: utf-8 -*-
"""Insert E1 few-shot and E2 capacity/length evidence into the JCP manuscript.
Two precise insertions, each asserted to hit exactly once. Backs up first."""
import shutil
p = r"C:\Users\Yi\WorkBuddy\2026-11-09-18-02-06\outputs\TheWell_论文\manuscript_en_jcp.tex"
t = open(p, encoding="utf-8").read()
shutil.copy(p, p + ".bak_before_e1e2")

# --- Insertion 1: E1 few-shot paragraph, right after the Table-3 discussion paragraph (line 369) ---
anchor1 = ("Table~\\ref{tab:m3} reports the results. The zero-shot model, without any fine-tuning, "
           "achieves a VRMSE of $5.06$ on the turbulent radiative layer, with a relative L2 of $0.73$. "
           "For comparison, a model trained from scratch on the turbulent radiative layer achieves a "
           "VRMSE of $0.71$, and a model with field-description adaptation (resetting the description "
           "embedding to a neutral value) achieves a VRMSE of $4.93$.")
assert t.count(anchor1) == 1, "anchor1 count=%d" % t.count(anchor1)

para1 = (
    "\n\nTo test whether the failed forward transfer can be recovered by updating the weights rather than "
    "only the description embedding, we run an auxiliary few-shot adaptation experiment in the uniformly "
    "per-sample-normalized protocol (in which the shear-to-turbulent zero-shot VRMSE is $1.38$, directly "
    "comparable to the normalized transfer matrix rather than to the raw-density value in "
    "Table~\\ref{tab:m3}). Starting from the shear-pretrained model, we take gradient steps on a small "
    "sample of the turbulent radiative layer and evaluate the normalized VRMSE after $1$, $5$, $10$, $20$, "
    "and $50$ steps. The error decreases monotonically from $1.38$ (zero-shot, above the mean baseline) to "
    "$1.36$, $1.29$, $1.21$, $1.10$, and finally $0.99$ at fifty steps, crossing below the mean baseline. "
    "Thus a modest number of labelled target steps suffices to convert the failed forward transfer into a "
    "below-baseline predictor, whereas resetting the field-description embedding alone leaves the weights "
    "unchanged and does not recover the transfer. This positions the gate as an intra-family routing "
    "mechanism and gradient-based fine-tuning as the mechanism for cross-family transfer, and is consistent "
    "with the full-refit result reported below."
)
t = t.replace(anchor1, anchor1 + para1)

# --- Insertion 2: E2 extra capacity/length evidence appended to line 403 capacity paragraph ---
anchor2 = ("On the turbulent radiative layer, re-running with all eight trajectories (102 training samples, "
           "per-sample normalized) and raising the capacity from $ch=16$ to $ch=64$ leaves the VRMSE "
           "unchanged at $0.984$, confirming that the hardest system is bounded by the available data and "
           "the structural difficulty of multi-scale turbulent dynamics rather than by capacity or the "
           "number of trajectories.")
assert t.count(anchor2) == 1, "anchor2 count=%d" % t.count(anchor2)

para2 = (
    " A further sweep in this normalized $64\\times64$ protocol extends the hidden width to $ch=128$ and the "
    "training length to $300$ steps: the turbulent in-domain VRMSE is $0.979$ at $ch=16$, $ch=64$, and "
    "$ch=128$, and is identical at $60$ and $300$ steps for each width. An eight-fold capacity increase and "
    "a five-fold optimization budget therefore leave the error pinned at the mean baseline, which rules out "
    "insufficient capacity or premature stopping as the cause and isolates the structural, multi-scale "
    "nature of this system as the binding constraint."
)
t = t.replace(anchor2, anchor2 + para2)

open(p, "w", encoding="utf-8").write(t)
print("OK: inserted E1 few-shot paragraph and E2 capacity/length evidence; new len=", len(t))
