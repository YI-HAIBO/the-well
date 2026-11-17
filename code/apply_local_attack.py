# -*- coding: utf-8 -*-
"""Apply local-attack findings (E7 20-seed significance, M1 capacity scaling, E4 single-system LB)
to manuscript_en_jcp.tex with unique-match assertions."""
import io

P = r"C:\Users\Yi\WorkBuddy\2026-11-09-18-02-06\outputs\TheWell_论文\manuscript_en_jcp.tex"
src = io.open(P, encoding="utf-8").read()
orig = src

EDITS = []

# ---- E7: 20-seed ablation reveals capacity significance ----
EDITS.append((
 "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\subsection{Computational complexity}",
 "\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
 "To determine whether the absence of significance is a power limitation rather than a true null effect, "
 "we extend the ablation to twenty seeds on a single-field shear-flow configuration and vary the capacity "
 "and the expert count. Widening the hidden dimension from $ch=16$ to $ch=32$ at a fixed two experts lowers "
 "the mean VRMSE from $0.1298\\pm0.0130$ to $0.1143\\pm0.0125$ (median $0.1300$ vs.\\ $0.1144$, IQR "
 "$[0.1201,0.1355]$ vs.\\ $[0.1036,0.1227]$), a difference that is highly significant under a Welch "
 "$t$-test ($t=3.82$, $p=4.8\\times10^{-4}$). Increasing the expert count from two to four at fixed "
 "$ch=16$ leaves the error unchanged ($0.1329\\pm0.0169$, $p=0.52$ against $ch=16$), consistent with the "
 "routing term being small on a single homogeneous system. The twenty-seed experiment therefore shows "
 "that capacity effects are statistically resolvable once the sample size is sufficient, whereas the "
 "expert-count and tokenizer-variant differences at the original scale reflect genuine insensitivity "
 "rather than insufficient power alone.\n\n\\subsection{Computational complexity}"
))

# ---- M1: capacity scaling trend ----
EDITS.append((
 "so the shared backbone does not degrade as the family count grows --- the average is limited by the hardest new system rather than by a collapse of the shared representation.",
 "so the shared backbone does not degrade as the family count grows --- the average is limited by the hardest new system rather than by a collapse of the shared representation. "
 "To establish that the proof-of-concept scale is a limitation of the reported budget rather than of the architecture, we additionally sweep the hidden dimension on the shear flow at $64\\times64$ up to $ch=128$. "
 "The VRMSE decreases monotonically from $0.1258$ at $ch=16$ ($655$ parameters) through $0.1041$ at $ch=32$ ($2{,}319$), $0.0964$ at $ch=64$ ($8{,}719$), and $0.0808$ at $ch=128$ ($33{,}807$), "
 "a $35.8\\%$ reduction over the $52\\times$ parameter increase, fitted as $\\mathrm{VRMSE}\\propto|\\Theta|^{-0.107}$. "
 "The scaling is therefore positive and unbroken across the accessible range, indicating that the reported errors reflect the small parameter budget rather than an architectural ceiling, "
 "and that a larger allocation would continue to improve accuracy at a sub-linear parameter cost."
))

# ---- E4: single-system LB no effect ----
EDITS.append((
 "the load-balancing and route-diversity regularizer is therefore adopted as the default in all multi-system experiments.",
 "the load-balancing and route-diversity regularizer is therefore adopted as the default in all multi-system experiments. "
 "On a single homogeneous system the regularizer is unnecessary: a five-seed comparison with and without the load-balancing term gives a mean VRMSE of $0.1205$ without and $0.1241$ with the regularizer "
 "(Welch $t=-0.34$, $p=0.74$), and the learned gate weights remain near-uniform in both cases "
 "($[0.569,0.431]$ without, $[0.420,0.580]$ with; entropy $0.684$ vs.\\ $0.680$ against a maximum of $\\ln 2=0.693$). "
 "The routing is already balanced on a single family, so the regularizer neither helps nor hurts there; its role is confined to preventing expert collapse as the number of operator families grows, "
 "which is precisely the multi-system setting where it is applied."
))

fails = []
for i, (old, new) in enumerate(EDITS):
    c = src.count(old)
    if c != 1:
        fails.append((i, c, old[:80]))
        continue
    src = src.replace(old, new, 1)

if fails:
    print("FAILED EDITS:")
    for i, c, s in fails:
        print("  #%d matches=%d anchor=%r" % (i, c, s))
    raise SystemExit(1)

io.open(P, "w", encoding="utf-8").write(src)
print("Applied %d edits, all unique." % len(EDITS))
print("delta chars:", len(src) - len(orig))