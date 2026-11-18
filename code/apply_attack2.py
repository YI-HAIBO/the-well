# -*- coding: utf-8 -*-
"""Write attack-round-2 results (E3/E6/E8/E10/M3) back into manuscript_en_jcp.tex.
Every replacement asserts exactly one hit."""
import io, os

TEX = r"C:\Users\Yi\WorkBuddy\2026-11-09-18-02-06\outputs\TheWell_论文\manuscript_en_jcp.tex"
src = io.open(TEX, encoding="utf-8").read()
fails = []

def rep(old, new, tag):
    global src
    c = src.count(old)
    if c != 1:
        fails.append((tag, c))
        return
    src = src.replace(old, new, 1)

# ---------------- E3: LOO normalization decomposition ----------------
rep(r"""\subsection{Trajectory-level extrapolation and the leave-one-out protocol}
The headline in-domain results are interpolated within the training trajectories; the more demanding
setting is extrapolation to unseen initial conditions. Leave-one-trajectory-out testing on the shear
flow gives a mean VRMSE of $2.44$ against $0.39$ in-domain (a $6.3\times$ gap) and $1.26$ against $0.98$
on the turbulent radiative layer, so trajectory-level generalization is substantially harder and the
in-domain numbers should not be read as extrapolation performance. The leave-one-out protocol uses
per-sample standardization on the training trajectories and raw field values on the held-out
trajectory, which introduces a scale mismatch at the boundary; a fully consistent train-stat
standardization gives a much lower error but is not directly comparable with the reported $2.44$,
so we retain the original protocol and flag the normalization sensitivity as a known artifact
rather than claiming a clean trajectory-extrapolation measurement.""",
r"""\subsection{Trajectory-level extrapolation and the leave-one-out protocol}
The headline in-domain results are interpolated within the training trajectories; the more demanding
setting is extrapolation to unseen initial conditions. Under the originally reported protocol,
leave-one-trajectory-out testing on the shear flow gives a mean VRMSE of $2.44$ against $0.39$
in-domain (a $6.3\times$ gap) and $1.26$ against $0.98$ on the turbulent radiative layer, which
suggests that trajectory-level generalization is substantially harder. A controlled normalization
decomposition on the eight official shear-flow trajectories (three seeds per fold, model and folds
held fixed) shows that this gap is dominated by the evaluation-space mismatch rather than by
extrapolation failure: under the consistent per-sample protocol the mean leave-one-out VRMSE is
$0.234$ (per-fold means $0.048$--$0.43$) against an in-domain $80/20$ reference of $0.346$, and a
train-statistics consistent variant gives $0.412$, the same order as in-domain. Evaluating the same
trained models against raw field values---the mixed-space convention of the original protocol, which
standardizes the training trajectories but compares predictions against unstandardized held-out
fields---inflates the per-fold means to $1.8\times10^{3}$--$2.3\times10^{4}$ (average
$7.7\times10^{3}$), three to four orders of magnitude above the consistent-protocol value. The
apparent extrapolation gap is therefore a normalization artifact: under a consistent protocol,
unseen-trajectory error is of the same order as in-domain error, and per-sample standardization
additionally absorbs trajectory-level level and scale shifts. The reported $2.44$ should be read as
a mixed-protocol number, not as an extrapolation ceiling.""",
"E3")

# ---------------- E6: gate-only adaptation, incl. input-dependent gate ----------------
rep(r"""\subsection{Gate-only adaptation}
The field-description embedding alone does not compensate for operator drift between families:
re-initializing only the gate and description parameters while keeping the shared backbone frozen
leaves the held-out VRMSE unchanged, and only a full-model fine-tune recovers meaningful improvement.
The present model therefore does not support one-shot adaptation to a new operator family through
the description mechanism alone; the practical workflow for a new family is cross-system pretraining
followed by full-model fine-tuning, and the description gate should be understood as a routing
mechanism within a fixed family set rather than as a zero-shot adapter.""",
r"""\subsection{Gate-only adaptation}
The field-description embedding alone does not compensate for operator drift between families. A
controlled cross-system experiment---pretraining on the shear flow and adapting with a
sixteen-window few-shot budget on the turbulent radiative layer, five seeds---quantifies the effect
and extends it to input-dependent routing: re-initializing only the fixed-description gate recovers
$0.010$ VRMSE of the $0.053$ zero-shot-to-adapted gap ($0.756$ zero-shot versus $0.703$ full-model
adaptation), and a learnable input-statistics-to-gate mapping that makes the routing input-dependent
performs equivalently ($0.747$ versus $0.746$) even though its routing entropy rises from $0.211$ to
$0.363$ after adaptation, confirming that the routing actually changed. The drift therefore lives in
the shared backbone and expert weights, not in the routing: full-model fine-tuning ($0.703$) closes
the gap and matches a $200$-step scratch reference ($0.700$). The description gate should be
understood as a routing mechanism within a fixed family set rather than as a zero-shot adapter, and
the practical workflow for a new family remains cross-system pretraining followed by full-model
fine-tuning.""",
"E6")

# ---------------- E8: regime- and capacity-resolved pretraining re-test ----------------
rep(r"""\subsection{Statistical reporting of ablations}
The tokenizer ablation and the pretraining-object ablation are reported as mean differences across
seeds, but the observed differences between variants are not statistically significant at the
current data scale (tokenizer variants $p>0.94$; pretraining-object variants $p>0.99$). The
coordinate-invariance and multi-resolution variants are therefore consistent with the data rather
than demonstrated to be causal, and the claims of benefit are phrased accordingly. Median and
interquartile-range summaries are reported alongside the means where available, and the
interpretation is that the ablations confirm the design choices do not hurt performance rather
than that they provide a measurable improvement at the present scale.""",
r"""\subsection{Statistical reporting of ablations}
The tokenizer ablation and the pretraining-object ablation are reported as mean differences across
seeds, but the observed differences between variants are not statistically significant at the
current data scale in the full-data regime (tokenizer variants $p>0.94$; pretraining-object variants
$p>0.99$). The coordinate-invariance and multi-resolution variants are therefore consistent with the
data rather than demonstrated to be causal, and the claims of benefit are phrased accordingly. Median
and interquartile-range summaries are reported alongside the means where available, and the
interpretation is that the ablations confirm the design choices do not hurt performance rather than
that they provide a measurable improvement at the present scale. A capacity- and regime-resolved
re-test of the pretraining objectives (masked reconstruction, contrastive alignment, and their
combination; five seeds; few-shot fine-tuning on sixteen windows) refines this picture: at $ch=16$
the combined objective gives a significant improvement over the scratch baseline ($0.332$ versus
$0.375$, Welch $p=0.036$), masked reconstruction alone is marginal ($0.341$ versus $0.375$,
$p=0.066$), and contrast alone is null ($p=0.999$); at $ch=64$ all three effects vanish
($p\ge0.77$) because the scratch model already reaches $0.229$. The pretraining signal is therefore
regime-dependent: it contributes when capacity is the binding constraint in the few-shot regime and
disappears once capacity suffices, which reconciles the earlier single-configuration null report
with a causal role for the pretraining objectives at low capacity.""",
"E8")

# ---------------- E10 + M3: data/model scale ----------------
rep(r"""flat data-scaling behavior observed when increasing the trajectory count is consistent with an
estimation-dominated regime at small data, and extending to the full benchmark would be the
natural next step. The step-by-step reproduction instructions in the Data Availability section
describe how the reported configuration can be re-run, and the official The Well baselines provide
an independent cross-check of the error scale; we are not aware of an independent third-party
reproduction of the specific model and bounds reported here.""",
r"""flat data-scaling behavior observed when increasing the trajectory count is consistent with an
estimation-dominated regime at small data. Two further scaling axes are measured on the official
slices: increasing the number of training trajectories from one to six at a fixed held-out
evaluation changes the VRMSE only from $0.544$ to $0.535$, and increasing the temporal coverage of
the training windows from $25\%$ to $100\%$ changes it from $0.458$ to $0.454$; both axes are flat
at the present scale, confirming the estimation-dominated regime independently of the
trajectory-count result. The full official shear-flow training file (13.4 GB, $32$ trajectories of
$200$ steps each) and the complete benchmark remain the natural scaling target. The step-by-step
reproduction instructions in the Data Availability section describe how the reported configuration
can be re-run, and the official The Well baselines provide an independent cross-check of the error
scale; we are not aware of an independent third-party reproduction of the specific model and bounds
reported here. To lower the barrier to such a reproduction, the repository ships a one-command
minimal pipeline (\texttt{reproduce\_minimal.py}) that regenerates the in-domain,
leave-one-trajectory-out, and cross-system few-shot headline numbers from the bundled official
slices on CPU in about three minutes with fixed seeds.""",
"E10M3")

# ---------------- Abstract: capacity sweep range ----------------
rep(r"with a capacity sweep up to $8719$ parameters",
    r"with a capacity sweep up to $33807$ parameters",
    "ABSTRACT")

# ---------------- Conclusion: LOO clause ----------------
rep(r"leave-one-trajectory-out testing on unseen trajectories gives a mean VRMSE of $2.44$ on the shear flow and $1.26$ on the turbulent radiative layer, substantially above the in-domain values, so trajectory-level extrapolation is substantially harder than in-domain evaluation, and the operator-divergence term of the zero-shot bound is measured numerically.",
    r"leave-one-trajectory-out testing on unseen trajectories gives a mean VRMSE of $2.44$ on the shear flow under the originally reported mixed-normalization protocol and $1.26$ on the turbulent radiative layer; a controlled normalization decomposition on the official shear-flow slices shows that this gap is an evaluation-space artifact---under a consistent per-sample protocol the leave-one-out error ($0.234$) is of the same order as the in-domain error ($0.346$), whereas evaluating in raw field units inflates the error by three to four orders of magnitude---so trajectory-level extrapolation at this scale is not the bottleneck that the raw numbers suggest, and the operator-divergence term of the zero-shot bound is measured numerically.",
    "CONCLUSION")

# ---------------- Data Availability: one-command repro ----------------
rep(r"(iv) compare against the stored result artifacts.",
    r"(iv) compare against the stored result artifacts; or (v) run \texttt{python reproduce\_minimal.py} for a three-minute CPU-only regeneration of the headline numbers from the bundled slices.",
    "DATAAVAIL")

io.open(TEX, "w", encoding="utf-8", newline="\n").write(src)
print("Applied; failures:", fails if fails else "NONE")
