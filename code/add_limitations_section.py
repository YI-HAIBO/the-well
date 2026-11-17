# -*- coding: utf-8 -*-
"""Add a consolidated Discussion and Limitations section to manuscript_en_jcp.tex.
Inserts between 'Assumption validation and robustness' and 'Conclusion'."""
import io

P = r"C:\Users\Yi\WorkBuddy\2026-11-09-18-02-06\outputs\TheWell_论文\manuscript_en_jcp.tex"
src = io.open(P, encoding="utf-8").read()
orig = src

# The anchor: the line right before \section{Conclusion}
ANCHOR = r"\section{Conclusion}"

LIMITATIONS = r"""\section{Discussion and limitations}
The measurements reported above close several gaps between the theoretical bounds and the empirical
performance, but a number of limitations remain that shape the interpretation of the results.

\subsection{Trajectory-level extrapolation and the leave-one-out protocol}
The headline in-domain results are interpolated within the training trajectories; the more demanding
setting is extrapolation to unseen initial conditions. Leave-one-trajectory-out testing on the shear
flow gives a mean VRMSE of $2.44$ against $0.39$ in-domain (a $6.3\times$ gap) and $1.26$ against $0.98$
on the turbulent radiative layer, so trajectory-level generalization is substantially harder and the
in-domain numbers should not be read as extrapolation performance. The leave-one-out protocol uses
per-sample standardization on the training trajectories and raw field values on the held-out
trajectory, which introduces a scale mismatch at the boundary; a fully consistent train-stat
standardization gives a much lower error but is not directly comparable with the reported $2.44$,
so we retain the original protocol and flag the normalization sensitivity as a known artifact
rather than claiming a clean trajectory-extrapolation measurement.

\subsection{Routing under load balancing}
The field-description gate achieves fully system-specific routing ($100\%$ assignment separation)
only when the load-balancing and route-diversity regularizer is applied; without it the natural
training collapses toward a small number of experts and the misassignment rate rises with the number
of operator families (from $0.30$ at two families to $0.540$ at eight). The regularizer is therefore
adopted as the default in all multi-system experiments, and the reported routing results should be
read as conditional on this regularizer. A direct comparison of learned routing with and without a
regularizer, at each family count, would further isolate the contribution of the regularizer itself,
but the present experiments already establish that routing without it is non-viable in the
multi-family setting.

\subsection{Gate-only adaptation}
The field-description embedding alone does not compensate for operator drift between families:
re-initializing only the gate and description parameters while keeping the shared backbone frozen
leaves the held-out VRMSE unchanged, and only a full-model fine-tune recovers meaningful improvement.
The present model therefore does not support one-shot adaptation to a new operator family through
the description mechanism alone; the practical workflow for a new family is cross-system pretraining
followed by full-model fine-tuning, and the description gate should be understood as a routing
mechanism within a fixed family set rather than as a zero-shot adapter.

\subsection{Statistical reporting of ablations}
The tokenizer ablation and the pretraining-object ablation are reported as mean differences across
seeds, but the observed differences between variants are not statistically significant at the
current data scale (tokenizer variants $p>0.94$; pretraining-object variants $p>0.99$). The
coordinate-invariance and multi-resolution variants are therefore consistent with the data rather
than demonstrated to be causal, and the claims of benefit are phrased accordingly. Median and
interquartile-range summaries are reported alongside the means where available, and the
interpretation is that the ablations confirm the design choices do not hurt performance rather
than that they provide a measurable improvement at the present scale.

\subsection{Data and model scale}
All experiments are conducted on small slices of The Well (on the order of one hundred samples per
system) with a lightweight model of approximately two thousand parameters. The results are
accordingly best interpreted as a proof of concept for the unified representation and the
approximation-bound structure, not as a claim about foundation-model performance at scale. The
flat data-scaling behavior observed when increasing the trajectory count is consistent with an
estimation-dominated regime at small data, and extending to the full benchmark would be the
natural next step. The step-by-step reproduction instructions in the Data Availability section
describe how the reported configuration can be re-run, and the official The Well baselines provide
an independent cross-check of the error scale; we are not aware of an independent third-party
reproduction of the specific model and bounds reported here.

"""

# Insert before \section{Conclusion}
c = src.count(ANCHOR)
if c != 1:
    raise SystemExit(f"Anchor not unique: {c} matches for {ANCHOR!r}")

src = src.replace(ANCHOR, LIMITATIONS + ANCHOR, 1)

io.open(P, "w", encoding="utf-8").write(src)
print("Inserted Discussion and limitations section.")
print("delta chars:", len(src) - len(orig))