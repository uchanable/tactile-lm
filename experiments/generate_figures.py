#!/usr/bin/env python3
"""
TCDS 논문용 Figure 생성 스크립트.

Usage:
    python generate_figures.py                  # 모든 figure 생성
    python generate_figures.py --fig reward_reach reward_selfbody
    python generate_figures.py --fig touch scatter forest ct_curve
    python generate_figures.py --list           # 사용 가능한 figure 목록
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
REACH_DIR = BASE_DIR / "from_macstudio" / "reach"
SELFBODY_DIR = BASE_DIR / "from_macstudio" / "selfbody"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STEPS = [50, 100, 500, 1000]
STEP_LABELS = ["50K", "100K", "500K", "1M"]

# ──────────────────────────────────────────────
# IEEE journal style
# ──────────────────────────────────────────────
IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "pdf.fonttype": 42,   # TrueType — editable in Illustrator
    "ps.fonttype": 42,
}

# Colors
C_OFF = "#4472C4"   # blue
C_ON = "#C44E52"    # red


def apply_style():
    plt.rcParams.update(IEEE_RC)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
def load_results(data_dir: Path, condition: str, steps_k: int):
    """Load all seed results for a given condition and timestep."""
    pattern = f"{condition}_{steps_k}K_seed*.json"
    files = sorted(glob(str(data_dir / pattern)))
    results = []
    for f in files:
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def aggregate(data_dir: Path):
    """Return dict[condition][steps_k] = list of dicts."""
    agg = {}
    for cond in ["CT_OFF", "CT_ON"]:
        agg[cond] = {}
        for sk in STEPS:
            agg[cond][sk] = load_results(data_dir, cond, sk)
    return agg


def reward_stats(records):
    """Mean and SD of mean_reward across seeds."""
    vals = np.array([r["mean_reward"] for r in records])
    return vals.mean(), vals.std(ddof=1)


def touch_stats(records):
    """Mean and SD of mean_touch across seeds."""
    vals = np.array([r["mean_touch"] for r in records])
    return vals.mean(), vals.std(ddof=1)


# ──────────────────────────────────────────────
# Hedges' g helper
# ──────────────────────────────────────────────
def hedges_g(x, y):
    """
    Compute Hedges' g (bias-corrected Cohen's d) and 95 % CI.
    Returns (g, ci_low, ci_high).
    """
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

    # Pooled SD
    sp = np.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    d = (my - mx) / sp  # positive if CT_ON > CT_OFF

    # Bias correction factor J
    df = nx + ny - 2
    j = 1 - 3 / (4 * df - 1)
    g = d * j

    # Variance of g (Borenstein et al. 2009)
    var_g = (nx + ny) / (nx * ny) + g ** 2 / (2 * (nx + ny))
    var_g *= j ** 2
    se_g = np.sqrt(var_g)

    ci_lo = g - 1.96 * se_g
    ci_hi = g + 1.96 * se_g
    return g, ci_lo, ci_hi, se_g


# ──────────────────────────────────────────────
# Figure 1 & 2: Reward comparison (grouped bar)
# ──────────────────────────────────────────────
def fig_reward_comparison(data_dir: Path, task_name: str, out_name: str):
    """Grouped bar chart: CT OFF vs CT ON across 4 timesteps."""
    apply_style()
    agg = aggregate(data_dir)

    means_off, sds_off = [], []
    means_on, sds_on = [], []
    for sk in STEPS:
        m, s = reward_stats(agg["CT_OFF"][sk])
        means_off.append(m)
        sds_off.append(s)
        m, s = reward_stats(agg["CT_ON"][sk])
        means_on.append(m)
        sds_on.append(s)

    x = np.arange(len(STEPS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    bars_off = ax.bar(
        x - width / 2, means_off, width,
        yerr=sds_off, label="CT OFF",
        color=C_OFF, edgecolor="black", linewidth=0.4,
        capsize=2, error_kw={"linewidth": 0.6},
    )
    bars_on = ax.bar(
        x + width / 2, means_on, width,
        yerr=sds_on, label="CT ON",
        color=C_ON, edgecolor="black", linewidth=0.4,
        capsize=2, error_kw={"linewidth": 0.6},
    )

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"PPO Reward Comparison ({task_name})")
    ax.set_xticks(x)
    ax.set_xticklabels(STEP_LABELS)
    ax.legend(frameon=True, edgecolor="gray", fancybox=False)
    ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")

    fig.tight_layout()
    out_path = FIG_DIR / out_name
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_reward_reach():
    fig_reward_comparison(REACH_DIR, "Reach", "fig_ppo_reach_reward.pdf")


def fig_reward_selfbody():
    fig_reward_comparison(SELFBODY_DIR, "Selfbody", "fig_ppo_selfbody_reward.pdf")


# ──────────────────────────────────────────────
# Figure 3: Touch activation comparison
# ──────────────────────────────────────────────
def fig_touch_activation():
    """Bar chart showing touch activation for CT OFF vs CT ON (Reach)."""
    apply_style()
    agg = aggregate(REACH_DIR)

    means_off, sds_off = [], []
    means_on, sds_on = [], []
    for sk in STEPS:
        m, s = touch_stats(agg["CT_OFF"][sk])
        means_off.append(m)
        sds_off.append(s)
        m, s = touch_stats(agg["CT_ON"][sk])
        means_on.append(m)
        sds_on.append(s)

    x = np.arange(len(STEPS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar(
        x - width / 2, means_off, width,
        yerr=sds_off, label="CT OFF",
        color=C_OFF, edgecolor="black", linewidth=0.4,
        capsize=2, error_kw={"linewidth": 0.6},
    )
    ax.bar(
        x + width / 2, means_on, width,
        yerr=sds_on, label="CT ON",
        color=C_ON, edgecolor="black", linewidth=0.4,
        capsize=2, error_kw={"linewidth": 0.6},
    )

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Touch Activation")
    ax.set_title("Touch Activation Comparison (Reach)")
    ax.set_xticks(x)
    ax.set_xticklabels(STEP_LABELS)
    ax.legend(frameon=True, edgecolor="gray", fancybox=False)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_touch_activation.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
# Figure 4: Per-seed scatter (Reach 1M)
# ──────────────────────────────────────────────
def fig_seed_scatter():
    """Scatter plot — each seed's reward for CT OFF vs CT ON at 1M steps."""
    apply_style()

    off_records = load_results(REACH_DIR, "CT_OFF", 1000)
    on_records = load_results(REACH_DIR, "CT_ON", 1000)

    # Build seed -> reward mapping
    off_by_seed = {r["seed"]: r["mean_reward"] for r in off_records}
    on_by_seed = {r["seed"]: r["mean_reward"] for r in on_records}

    common_seeds = sorted(set(off_by_seed) & set(on_by_seed))
    x_vals = np.array([off_by_seed[s] for s in common_seeds])
    y_vals = np.array([on_by_seed[s] for s in common_seeds])

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.scatter(x_vals, y_vals, s=18, c=C_ON, edgecolors="black",
               linewidths=0.3, alpha=0.8, zorder=3)

    # Diagonal — equality line
    all_vals = np.concatenate([x_vals, y_vals])
    lo, hi = all_vals.min() - 10, all_vals.max() + 10
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.6, zorder=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("CT OFF Reward")
    ax.set_ylabel("CT ON Reward")
    ax.set_title("Per-seed Reward at 1M Steps (Reach)")

    # Annotation: fraction above diagonal
    n_above = np.sum(y_vals > x_vals)
    ax.text(
        0.05, 0.95,
        f"CT ON > CT OFF: {n_above}/{len(common_seeds)}",
        transform=ax.transAxes, fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", linewidth=0.4),
    )

    fig.tight_layout()
    out_path = FIG_DIR / "fig_seed_scatter.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
# Figure 5: Effect size forest plot (Reach)
# ──────────────────────────────────────────────
def fig_forest_plot():
    """Forest plot of Hedges' g for each timestep + combined estimate."""
    apply_style()
    agg = aggregate(REACH_DIR)

    gs, ci_los, ci_his, ses, weights = [], [], [], [], []
    for sk in STEPS:
        off_vals = np.array([r["mean_reward"] for r in agg["CT_OFF"][sk]])
        on_vals = np.array([r["mean_reward"] for r in agg["CT_ON"][sk]])
        g, lo, hi, se = hedges_g(off_vals, on_vals)
        gs.append(g)
        ci_los.append(lo)
        ci_his.append(hi)
        ses.append(se)
        weights.append(1.0 / (se ** 2))

    # Fixed-effect combined estimate
    w = np.array(weights)
    g_combined = np.sum(np.array(gs) * w) / np.sum(w)
    se_combined = np.sqrt(1.0 / np.sum(w))
    ci_lo_combined = g_combined - 1.96 * se_combined
    ci_hi_combined = g_combined + 1.96 * se_combined

    # Plotting
    labels = STEP_LABELS + ["Combined"]
    all_g = gs + [g_combined]
    all_lo = ci_los + [ci_lo_combined]
    all_hi = ci_his + [ci_hi_combined]

    n = len(labels)
    y_pos = np.arange(n)[::-1]  # top to bottom

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for i in range(n):
        color = "black" if i < n - 1 else C_ON
        marker = "o" if i < n - 1 else "D"
        ms = 4 if i < n - 1 else 5
        ax.plot(all_g[i], y_pos[i], marker, color=color, markersize=ms, zorder=3)
        ax.plot([all_lo[i], all_hi[i]], [y_pos[i], y_pos[i]],
                "-", color=color, linewidth=1.0, zorder=2)

    # Separation line before "Combined"
    ax.axhline(y_pos[-1] + 0.5, color="gray", linewidth=0.3, linestyle="--")

    # Zero line
    ax.axvline(0, color="gray", linewidth=0.4, linestyle="-")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hedges' $g$ (CT ON $-$ CT OFF)")
    ax.set_title("Effect Size Forest Plot (Reach)")

    # Add numeric values on the right
    x_text = max(all_hi) + 0.15
    for i in range(n):
        ax.text(
            x_text, y_pos[i],
            f"{all_g[i]:.2f} [{all_lo[i]:.2f}, {all_hi[i]:.2f}]",
            va="center", fontsize=6,
        )

    # Extend x-axis to fit text
    ax.set_xlim(min(all_lo) - 0.3, x_text + 1.8)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_forest_plot.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
# Figure 6: CT Afferent Firing Rate Curve
# ──────────────────────────────────────────────
def fig_ct_firing_rate():
    """
    CT afferent firing rate vs velocity curve (Loken et al. 2009).
    Log-Gaussian model: firing rate peaks at ~3 cm/s.
    Model: FR(v) = A * exp(-0.5 * ((ln(v) - ln(v_peak)) / sigma)^2)
    """
    apply_style()

    v = np.linspace(0.1, 30, 500)
    v_peak = 3.0       # cm/s — peak velocity
    sigma = 0.8        # width of the log-Gaussian
    A = 1.0            # normalized peak

    fr = A * np.exp(-0.5 * ((np.log(v) - np.log(v_peak)) / sigma) ** 2)

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(v, fr, "-", color=C_ON, linewidth=1.5)
    ax.fill_between(v, 0, fr, alpha=0.15, color=C_ON)

    # Mark peak
    ax.axvline(v_peak, color="gray", linewidth=0.5, linestyle="--")
    ax.text(
        v_peak + 0.5, 0.95,
        f"Peak: {v_peak} cm/s",
        fontsize=7, color="gray",
    )

    # Pleasant touch range (1–10 cm/s)
    ax.axvspan(1, 10, alpha=0.07, color="green", zorder=0)
    ax.text(
        5.5, 0.05,
        "Pleasant touch\nrange",
        fontsize=6, ha="center", color="green",
        fontstyle="italic",
    )

    ax.set_xlabel("Stroking Velocity (cm/s)")
    ax.set_ylabel("Normalized Firing Rate")
    ax.set_title("CT Afferent Firing Rate Curve")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    out_path = FIG_DIR / "fig_ct_firing_rate.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ──────────────────────────────────────────────
# Registry & main
# ──────────────────────────────────────────────
FIGURES = {
    "reward_reach": fig_reward_reach,
    "reward_selfbody": fig_reward_selfbody,
    "touch": fig_touch_activation,
    "scatter": fig_seed_scatter,
    "forest": fig_forest_plot,
    "ct_curve": fig_ct_firing_rate,
}


def main():
    parser = argparse.ArgumentParser(description="Generate TCDS figures.")
    parser.add_argument(
        "--fig", nargs="+", choices=list(FIGURES.keys()),
        help="Specific figures to generate (default: all).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available figure names.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available figures:")
        for name, fn in FIGURES.items():
            print(f"  {name:20s}  {fn.__doc__.strip().splitlines()[0]}")
        sys.exit(0)

    targets = args.fig if args.fig else list(FIGURES.keys())

    print(f"Generating {len(targets)} figure(s)...")
    for name in targets:
        print(f"  [{name}]")
        FIGURES[name]()

    print("Done.")


if __name__ == "__main__":
    main()
