# -*- coding: utf-8 -*-
"""Generate paper figures from the audit results (all values are measured)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG, exist_ok=True)

# ----------------------------------------------------------------------------
# Fig 1: E1 single-set precision, VRMSE per seed vs mean baseline
# ----------------------------------------------------------------------------
seeds = ["seed 0", "seed 1", "seed 2"]
vrmse = [0.2697, 0.2734, 0.2700]
mean_v = 0.2710
plt.figure(figsize=(4.4, 3.0))
bars = plt.bar(seeds, vrmse, color=["#3b7ddd", "#3b7ddd", "#3b7ddd"], width=0.5)
plt.axhline(1.0, color="#d43f3a", linestyle="--", linewidth=1.4, label="mean baseline (1.0000)")
plt.axhline(mean_v, color="#2f9e44", linestyle=":", linewidth=1.2, label="ours mean (0.2710)")
plt.ylabel("VRMSE (lower is better)")
plt.ylim(0.0, 1.15)
plt.legend(fontsize=7, frameon=False)
plt.title("Single-set prediction error", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_e1_vrmse.png"), dpi=220)
plt.close()

# ----------------------------------------------------------------------------
# Fig 2: E4b routing-error linearity  mean|err| = 1.985*delta + 0.0004
# ----------------------------------------------------------------------------
delta = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
err = [0.0000, 0.0993, 0.1985, 0.4002, 0.5907, 0.8012, 0.9897]
plt.figure(figsize=(4.4, 3.0))
plt.plot(delta, err, "o", color="#3b7ddd", ms=5, label="measured")
dd = np.linspace(0, 0.5, 100)
plt.plot(dd, 1.985 * dd + 0.0004, "-", color="#d43f3a", lw=1.4, label="fit  1.985 delta + 0.0004")
plt.plot(dd, 2.0 * dd, ":", color="#2f9e44", lw=1.4, label="bound  2M delta")
plt.xlabel("routing misassignment probability  delta")
plt.ylabel("mean |error|")
plt.legend(fontsize=7, frameon=False)
plt.title("Routing-error term C2 delta", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_e4b_routing.png"), dpi=220)
plt.close()

# ----------------------------------------------------------------------------
# Fig 3: E4c spectral-layer term: U-Net multiscale vs flat FNO, L=1->2 ratio
# ----------------------------------------------------------------------------
labels = ["U-Net doubling\nchannels", "U-Net fixed\nchannels", "flat FNO\n(depth=capacity)"]
vals = [0.545, 0.693, 0.960]
colors = ["#3b7ddd", "#3b7ddd", "#d43f3a"]
plt.figure(figsize=(4.4, 3.0))
bars = plt.bar(labels, vals, color=colors, width=0.5)
plt.axhline(0.5, color="#2f9e44", linestyle="--", linewidth=1.2, label="theory 0.5")
plt.ylabel("error ratio, L=1 to 2")
plt.ylim(0.0, 1.1)
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
plt.legend(fontsize=7, frameon=False)
plt.title("Multiscale hierarchy realizes 2^-L", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_e4c_unet.png"), dpi=220)
plt.close()

# ----------------------------------------------------------------------------
# Fig 4: E4a spectral-truncation decay exponent (log-log slopes)
# ----------------------------------------------------------------------------
fields = ["pressure", "tracer", "speed"]
slope = [-2.134, -1.393, -0.884]
r2 = [0.9933, 0.9519, 0.9956]
plt.figure(figsize=(4.4, 3.0))
plt.bar(fields, slope, color=["#3b7ddd", "#3b7ddd", "#3b7ddd"], width=0.5)
plt.axhline(-0.5, color="#2f9e44", linestyle="--", linewidth=1.2, label="alpha = d_max/2")
plt.ylabel("asymptotic decay exponent")
plt.ylim(-2.6, 0.0)
for i, (s, r) in enumerate(zip(slope, r2)):
    plt.text(i, s - 0.12, f"R2={r:.3f}", ha="center", fontsize=7)
plt.legend(fontsize=7, frameon=False)
plt.title("Spectral-truncation decay", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_e4a_decay.png"), dpi=220)
plt.close()

print("figures generated:")
for f in sorted(os.listdir(FIG)):
    print("  ", f)
