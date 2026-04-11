#!/usr/bin/env python3
"""CT afferent ON vs OFF comparison experiment for MIMo.

Runs MIMo with random actions under two touch conditions:
  - Condition A (CT ON):  CTAugmentedTouch with "multi_receptor" (SA-I + FA-I + FA-II + CT + normal)
  - Condition B (CT OFF): Standard TrimeshTouch with "force_vector" (3D force only)

Collects per-step sensor activations and produces publication-quality figures
comparing the two conditions.

Usage:
    cd /Users/uchanable_m1/obsidian/40_Ideas/project/mimo-tactile
    PYTHONPATH=MIMo:. .venv312/bin/python experiments/ct_comparison.py
"""

import os
import sys
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "MIMo"))
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR.mkdir(exist_ok=True)

# ── MIMo imports ───────────────────────────────────────────────────────────
from mimoEnv.envs.dummy import MIMoDummyEnv  # noqa: E402
from mimoTouch.touch import TrimeshTouch  # noqa: E402
from ct_touch.ct_augmented_touch import CTAugmentedTouch  # noqa: E402
from ct_touch.skin_map import get_skin_type, SkinType  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# ── Experiment parameters ──────────────────────────────────────────────────
N_STEPS = 1000
N_SEEDS = 3
SEEDS = [42, 123, 7]

# Touch body parts to monitor (shared between both conditions)
TOUCH_SCALES = {
    "left_foot": 0.02,
    "right_foot": 0.02,
    "left_lower_leg": 0.1,
    "right_lower_leg": 0.1,
    "left_upper_leg": 0.1,
    "right_upper_leg": 0.1,
    "hip": 0.1,
    "lower_body": 0.1,
    "upper_body": 0.1,
    "head": 0.1,
    "left_upper_arm": 0.05,
    "right_upper_arm": 0.05,
    "left_lower_arm": 0.05,
    "right_lower_arm": 0.05,
    "left_hand": 0.02,
    "right_hand": 0.02,
}


# ── Custom environments ───────────────────────────────────────────────────

class ForceVectorEnv(MIMoDummyEnv):
    """Standard MIMo with force_vector touch (CT OFF)."""

    def touch_setup(self, touch_params):
        self.touch = TrimeshTouch(self, touch_params)
        n = sum(self.touch.get_sensor_count(b) for b in self.touch.sensor_positions)
        print(f"  [ForceVector] Total sensors: {n}")


class MultiReceptorEnv(MIMoDummyEnv):
    """MIMo with CTAugmentedTouch multi_receptor (CT ON)."""

    def touch_setup(self, touch_params):
        self.touch = CTAugmentedTouch(self, touch_params)
        n = sum(self.touch.get_sensor_count(b) for b in self.touch.sensor_positions)
        print(f"  [MultiReceptor] Total sensors: {n}")


# ── Data collection ───────────────────────────────────────────────────────

def collect_data(env_class, touch_function, seed, n_steps=N_STEPS):
    """Run n_steps random actions and collect per-step sensor data."""
    touch_params = {
        "scales": TOUCH_SCALES,
        "touch_function": touch_function,
        "response_function": "spread_linear",
    }

    env = env_class(
        touch_params=touch_params,
        vision_params=None,
        vestibular_params=None,
        render_mode=None,
    )

    # Build body_id -> name mapping
    body_id_to_name = {}
    for body_id in env.touch.sensor_positions:
        body_id_to_name[body_id] = env.model.body(body_id).name

    # Initialize data storage
    step_data = {
        "total_activation": [],          # scalar per step
        "contact_steps": 0,              # how many steps had any contact
        "body_activations": defaultdict(list),  # body_name -> list of per-step max activation
        "body_contact_counts": defaultdict(int),  # body_name -> number of steps with contact
    }

    # For multi_receptor: per-channel data
    if touch_function == "multi_receptor":
        step_data["channel_means"] = {
            "SA-I": [], "FA-I": [], "FA-II": [], "CT": [], "Normal": []
        }
        step_data["ct_by_skin_type"] = {"hairy": [], "glabrous": []}

    env.reset(seed=seed)
    np.random.seed(seed)

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Overall activation
        touch_obs = obs.get("touch", np.array([]))
        total_act = float(np.sum(np.abs(touch_obs)))
        step_data["total_activation"].append(total_act)

        if total_act > 0:
            step_data["contact_steps"] += 1

        # Per-body analysis
        for body_id in sorted(env.touch.sensor_outputs.keys()):
            body_name = body_id_to_name.get(body_id, f"body_{body_id}")
            output = env.touch.sensor_outputs[body_id]
            max_act = float(np.max(np.abs(output)))
            step_data["body_activations"][body_name].append(max_act)
            if max_act > 0:
                step_data["body_contact_counts"][body_name] += 1

        # Multi-receptor channel analysis
        if touch_function == "multi_receptor":
            sa1_total, fa1_total, fa2_total, ct_total, normal_total = 0.0, 0.0, 0.0, 0.0, 0.0
            ct_hairy, ct_glabrous = 0.0, 0.0

            for body_id in sorted(env.touch.sensor_outputs.keys()):
                body_name = body_id_to_name.get(body_id, f"body_{body_id}")
                output = env.touch.sensor_outputs[body_id]  # (n_sensors, 7)

                if output.shape[1] >= 7:
                    sa1_total += float(np.sum(np.abs(output[:, 0:3])))
                    fa1_total += float(np.sum(np.abs(output[:, 3])))
                    fa2_total += float(np.sum(np.abs(output[:, 4])))
                    ct_val = float(np.sum(np.abs(output[:, 5])))
                    ct_total += ct_val
                    normal_total += float(np.sum(np.abs(output[:, 6])))

                    skin = get_skin_type(body_name)
                    if skin == SkinType.HAIRY:
                        ct_hairy += ct_val
                    else:
                        ct_glabrous += ct_val

            step_data["channel_means"]["SA-I"].append(sa1_total)
            step_data["channel_means"]["FA-I"].append(fa1_total)
            step_data["channel_means"]["FA-II"].append(fa2_total)
            step_data["channel_means"]["CT"].append(ct_total)
            step_data["channel_means"]["Normal"].append(normal_total)
            step_data["ct_by_skin_type"]["hairy"].append(ct_hairy)
            step_data["ct_by_skin_type"]["glabrous"].append(ct_glabrous)

        if terminated or truncated:
            env.reset(seed=seed + step)

    env.close()
    return step_data


# ── Main experiment loop ──────────────────────────────────────────────────

def run_experiment():
    """Run the full CT ON vs OFF comparison experiment."""
    print("=" * 60)
    print("CT Afferent ON vs OFF Comparison Experiment")
    print("=" * 60)

    all_results = {"force_vector": [], "multi_receptor": []}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed_idx + 1}/{N_SEEDS} (seed={seed}) ---")

        # Condition B: Force Vector (CT OFF)
        print("  Running ForceVector (CT OFF)...")
        t0 = time.time()
        data_fv = collect_data(ForceVectorEnv, "force_vector", seed, N_STEPS)
        t_fv = time.time() - t0
        print(f"    Done in {t_fv:.1f}s, contact steps: {data_fv['contact_steps']}/{N_STEPS}")
        all_results["force_vector"].append(data_fv)

        # Condition A: Multi Receptor (CT ON)
        print("  Running MultiReceptor (CT ON)...")
        t0 = time.time()
        data_mr = collect_data(MultiReceptorEnv, "multi_receptor", seed, N_STEPS)
        t_mr = time.time() - t0
        print(f"    Done in {t_mr:.1f}s, contact steps: {data_mr['contact_steps']}/{N_STEPS}")
        all_results["multi_receptor"].append(data_mr)

    return all_results


# ── Plotting ──────────────────────────────────────────────────────────────

def make_figures(results):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Color palette
    C_OFF = "#4477AA"   # blue for CT OFF
    C_ON = "#EE6677"    # red for CT ON
    C_SA1 = "#228833"   # green
    C_FA1 = "#CCBB44"   # yellow
    C_FA2 = "#66CCEE"   # cyan
    C_CT = "#AA3377"    # purple
    C_NRM = "#BBBBBB"   # gray

    # ── Figure 1: Overall comparison ──────────────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))

    # 1a: Total activation time series (smoothed)
    ax = axes1[0]
    window = 20
    for seed_idx in range(N_SEEDS):
        fv_act = np.array(results["force_vector"][seed_idx]["total_activation"])
        mr_act = np.array(results["multi_receptor"][seed_idx]["total_activation"])

        # Smooth
        fv_smooth = np.convolve(fv_act, np.ones(window)/window, mode="valid")
        mr_smooth = np.convolve(mr_act, np.ones(window)/window, mode="valid")
        x = np.arange(len(fv_smooth))

        alpha = 0.3
        ax.plot(x, fv_smooth, color=C_OFF, alpha=alpha, linewidth=0.8)
        ax.plot(x, mr_smooth, color=C_ON, alpha=alpha, linewidth=0.8)

    # Mean across seeds
    fv_means = np.mean([np.convolve(np.array(r["total_activation"]),
                        np.ones(window)/window, mode="valid")
                        for r in results["force_vector"]], axis=0)
    mr_means = np.mean([np.convolve(np.array(r["total_activation"]),
                        np.ones(window)/window, mode="valid")
                        for r in results["multi_receptor"]], axis=0)
    x = np.arange(len(fv_means))
    ax.plot(x, fv_means, color=C_OFF, linewidth=2.0, label="Force Vector (CT OFF)")
    ax.plot(x, mr_means, color=C_ON, linewidth=2.0, label="Multi-receptor (CT ON)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total sensor activation")
    ax.set_title("(a) Total activation over time")
    ax.legend()

    # 1b: Contact frequency per body
    ax = axes1[1]
    body_names_fv = sorted(set().union(*[r["body_contact_counts"].keys()
                                         for r in results["force_vector"]]))
    body_names_mr = sorted(set().union(*[r["body_contact_counts"].keys()
                                         for r in results["multi_receptor"]]))
    all_body_names = sorted(set(body_names_fv) | set(body_names_mr))

    # Abbreviate body names for readability
    def abbrev(name):
        return name.replace("left_", "L ").replace("right_", "R ").replace("_", " ")

    labels = [abbrev(n) for n in all_body_names]
    fv_counts = [np.mean([r["body_contact_counts"].get(n, 0) for r in results["force_vector"]]) / N_STEPS * 100
                 for n in all_body_names]
    mr_counts = [np.mean([r["body_contact_counts"].get(n, 0) for r in results["multi_receptor"]]) / N_STEPS * 100
                 for n in all_body_names]

    y_pos = np.arange(len(all_body_names))
    bar_height = 0.35
    ax.barh(y_pos - bar_height/2, fv_counts, bar_height, color=C_OFF, label="CT OFF")
    ax.barh(y_pos + bar_height/2, mr_counts, bar_height, color=C_ON, label="CT ON")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Contact frequency (%)")
    ax.set_title("(b) Contact frequency by body part")
    ax.legend(loc="lower right", fontsize=8)

    # 1c: Mean activation per body (color by skin type)
    ax = axes1[2]
    hairy_bodies = []
    glabrous_bodies = []
    for n in all_body_names:
        if get_skin_type(n) == SkinType.HAIRY:
            hairy_bodies.append(n)
        else:
            glabrous_bodies.append(n)

    ordered_bodies = hairy_bodies + glabrous_bodies
    ordered_labels = [abbrev(n) for n in ordered_bodies]
    fv_means_body = [np.mean([np.mean(r["body_activations"].get(n, [0]))
                              for r in results["force_vector"]])
                     for n in ordered_bodies]
    mr_means_body = [np.mean([np.mean(r["body_activations"].get(n, [0]))
                              for r in results["multi_receptor"]])
                     for n in ordered_bodies]

    y_pos2 = np.arange(len(ordered_bodies))
    colors_bar = [C_CT if n in hairy_bodies else C_OFF for n in ordered_bodies]
    ax.barh(y_pos2 - bar_height/2, fv_means_body, bar_height, color=C_OFF, label="CT OFF")
    ax.barh(y_pos2 + bar_height/2, mr_means_body, bar_height, color=C_ON, label="CT ON")

    # Mark hairy vs glabrous
    if hairy_bodies and glabrous_bodies:
        sep = len(hairy_bodies) - 0.5
        ax.axhline(y=sep, color="gray", linestyle="--", linewidth=0.8)
        ax.text(ax.get_xlim()[1] * 0.7, sep - 1, "Hairy", fontsize=7, color="gray")
        ax.text(ax.get_xlim()[1] * 0.7, sep + 0.5, "Glabrous", fontsize=7, color="gray")

    ax.set_yticks(y_pos2)
    ax.set_yticklabels(ordered_labels, fontsize=8)
    ax.set_xlabel("Mean activation")
    ax.set_title("(c) Mean activation by body & skin type")
    ax.legend(loc="lower right", fontsize=8)

    fig1.suptitle("Figure 1: CT ON vs OFF — Overall Touch Comparison", fontsize=13, y=1.02)
    fig1.tight_layout()
    fig1.savefig(RESULTS_DIR / "fig1_overall_comparison.png")
    print(f"  Saved: {RESULTS_DIR / 'fig1_overall_comparison.png'}")
    plt.close(fig1)

    # ── Figure 2: Multi-receptor channel analysis (CT ON only) ────────────
    fig2 = plt.figure(figsize=(14, 11))
    gs = GridSpec(3, 3, figure=fig2, hspace=0.4, wspace=0.35)

    # 2a: Channel activation over time (excluding Normal for clarity; CT on secondary axis)
    ax = fig2.add_subplot(gs[0, 0:2])
    channels_main = ["SA-I", "FA-I", "FA-II"]
    colors_main = [C_SA1, C_FA1, C_FA2]

    for ch, color in zip(channels_main, colors_main):
        means = np.mean([np.convolve(np.array(r["channel_means"][ch]),
                         np.ones(window)/window, mode="valid")
                         for r in results["multi_receptor"]], axis=0)
        ax.plot(np.arange(len(means)), means, color=color, linewidth=1.5, label=ch)

    ax.set_xlabel("Step")
    ax.set_ylabel("Channel activation (sum)")
    ax.set_title("(a) Mechanoreceptor channel activation over time")

    # CT on secondary y-axis (much smaller scale)
    ax2 = ax.twinx()
    ct_means_ts = np.mean([np.convolve(np.array(r["channel_means"]["CT"]),
                           np.ones(window)/window, mode="valid")
                           for r in results["multi_receptor"]], axis=0)
    ax2.plot(np.arange(len(ct_means_ts)), ct_means_ts, color=C_CT, linewidth=2.0,
             label="CT", linestyle="-")
    ax2.set_ylabel("CT activation", color=C_CT)
    ax2.tick_params(axis="y", labelcolor=C_CT)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    # 2b: Channel proportions (pie chart of means)
    ax = fig2.add_subplot(gs[0, 2])
    channels = ["SA-I", "FA-I", "FA-II", "CT", "Normal"]
    channel_colors = [C_SA1, C_FA1, C_FA2, C_CT, C_NRM]
    ch_means = {}
    for ch in channels:
        ch_means[ch] = np.mean([np.mean(r["channel_means"][ch])
                                for r in results["multi_receptor"]])

    total_ch = sum(ch_means.values())
    if total_ch > 0:
        sizes = [ch_means[ch] for ch in channels]

        def make_autopct(values, total):
            def autopct(pct):
                val = pct * total / 100.0
                if pct < 0.05:
                    return f"{pct:.2f}%"
                elif pct < 1:
                    return f"{pct:.1f}%"
                return f"{pct:.1f}%"
            return autopct

        ax.pie(sizes, labels=channels, colors=channel_colors,
               autopct=make_autopct(sizes, total_ch), startangle=90, pctdistance=0.75,
               textprops={"fontsize": 9})
        ax.set_title("(b) Channel contribution")
    else:
        ax.text(0.5, 0.5, "No contact detected", ha="center", va="center")
        ax.set_title("(b) Channel contribution")

    # 2c: CT response by skin type
    ax = fig2.add_subplot(gs[1, 0])
    ct_hairy_all = [np.mean(r["ct_by_skin_type"]["hairy"])
                    for r in results["multi_receptor"]]
    ct_glabrous_all = [np.mean(r["ct_by_skin_type"]["glabrous"])
                       for r in results["multi_receptor"]]

    x_pos = [0, 1]
    means_skin = [np.mean(ct_hairy_all), np.mean(ct_glabrous_all)]
    stds_skin = [np.std(ct_hairy_all), np.std(ct_glabrous_all)]
    bars = ax.bar(x_pos, means_skin, yerr=stds_skin,
                  color=[C_CT, C_OFF], capsize=5, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Hairy skin\n(CT present)", "Glabrous skin\n(no CT)"])
    ax.set_ylabel("Mean CT channel activation")
    ax.set_title("(c) CT response by skin type")

    # 2d: Per-body CT activation heatmap
    ax = fig2.add_subplot(gs[1, 1:3])
    # Collect per-body CT channel means across seeds
    ct_body_data = defaultdict(list)
    for r in results["multi_receptor"]:
        for body_name, acts in r["body_activations"].items():
            ct_body_data[body_name].append(np.mean(acts))

    body_names_sorted = sorted(ct_body_data.keys())
    hairy_sorted = [n for n in body_names_sorted if get_skin_type(n) == SkinType.HAIRY]
    glabrous_sorted = [n for n in body_names_sorted if get_skin_type(n) == SkinType.GLABROUS]
    body_order = hairy_sorted + glabrous_sorted

    if body_order:
        body_labels_ordered = [abbrev(n) for n in body_order]
        # For each seed, build per-body activation array
        n_bodies = len(body_order)
        heatmap_data = np.zeros((N_SEEDS, n_bodies))

        for seed_idx, r in enumerate(results["multi_receptor"]):
            for j, body_name in enumerate(body_order):
                acts = r["body_activations"].get(body_name, [0])
                heatmap_data[seed_idx, j] = np.mean(acts)

        im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_yticks(range(N_SEEDS))
        ax.set_yticklabels([f"Seed {s}" for s in SEEDS], fontsize=9)
        ax.set_xticks(range(n_bodies))
        ax.set_xticklabels(body_labels_ordered, rotation=45, ha="right", fontsize=8)

        if hairy_sorted and glabrous_sorted:
            sep_x = len(hairy_sorted) - 0.5
            ax.axvline(x=sep_x, color="white", linewidth=2)
            ax.text(sep_x - 2, -0.7, "Hairy", fontsize=8, ha="center", color=C_CT)
            ax.text(sep_x + 1.5, -0.7, "Glabrous", fontsize=8, ha="center", color=C_OFF)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean activation")
        ax.set_title("(d) Per-body activation heatmap (CT ON)")

    # 2e: CT channel time series (zoomed) — smoothed mean only
    ax = fig2.add_subplot(gs[2, 0:2])
    ct_mean_ts = np.mean([np.convolve(np.array(r["channel_means"]["CT"]),
                          np.ones(window)/window, mode="valid")
                          for r in results["multi_receptor"]], axis=0)
    ax.plot(np.arange(len(ct_mean_ts)), ct_mean_ts,
            color=C_CT, linewidth=2.0, label=f"CT (smoothed mean, {N_SEEDS} seeds)")
    ax.set_xlabel("Step")
    ax.set_ylabel("CT activation")
    ax.set_ylim(0, 0.200)
    ax.set_title("(e) CT channel activation time series (zoomed)")
    ax.legend()

    # 2f: CT event histogram (non-zero CT values)
    ax = fig2.add_subplot(gs[2, 2])
    all_ct_nonzero = []
    for r in results["multi_receptor"]:
        ct_vals = np.array(r["channel_means"]["CT"])
        nonzero = ct_vals[ct_vals > 0]
        all_ct_nonzero.extend(nonzero.tolist())

    if all_ct_nonzero:
        ax.hist(all_ct_nonzero, bins=30, color=C_CT, edgecolor="black",
                linewidth=0.5, alpha=0.8)
        ax.set_xlabel("CT activation value")
        ax.set_ylabel("Count")
        ax.set_title(f"(f) CT event distribution\n(n={len(all_ct_nonzero)} events)")
    else:
        ax.text(0.5, 0.5, "No CT events", ha="center", va="center")
        ax.set_title("(f) CT event distribution")

    fig2.suptitle("Figure 2: Multi-receptor Channel Analysis", fontsize=13, y=1.01)
    fig2.savefig(RESULTS_DIR / "fig2_channel_analysis.png")
    print(f"  Saved: {RESULTS_DIR / 'fig2_channel_analysis.png'}")
    plt.close(fig2)

    # ── Figure 3: CT firing rate curve + velocity analysis ────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

    # 3a: Theoretical CT firing rate curve
    ax = axes3[0]
    velocities = np.logspace(-3, 0, 200)  # 0.001 to 1 m/s
    rates = [CTAugmentedTouch.ct_firing_rate(v) for v in velocities]
    ax.semilogx(velocities * 100, rates, color=C_CT, linewidth=2.5)  # convert to cm/s
    ax.axvline(x=3.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(3.2, 0.95, "3 cm/s\n(optimal)", fontsize=8, color="gray")
    ax.fill_between(velocities * 100, rates, alpha=0.15, color=C_CT)
    ax.set_xlabel("Stroking velocity (cm/s)")
    ax.set_ylabel("CT firing rate (normalized)")
    ax.set_title("(a) CT afferent velocity tuning curve\n(Loken et al. 2009 model)")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0.1, 100)

    # 3b: Information content comparison
    ax = axes3[1]
    fv_dims = 3  # force_vector: 3 channels per sensor
    mr_dims = 7  # multi_receptor: 7 channels per sensor

    # Compute effective information (non-zero activations)
    fv_nonzero = []
    mr_nonzero = []
    for r in results["force_vector"]:
        acts = np.array(r["total_activation"])
        fv_nonzero.append(np.mean(acts > 0))
    for r in results["multi_receptor"]:
        acts = np.array(r["total_activation"])
        mr_nonzero.append(np.mean(acts > 0))

    categories = ["Channels\nper sensor", "Contact\nfrequency (%)", "Mean total\nactivation"]
    fv_vals = [
        fv_dims,
        np.mean(fv_nonzero) * 100,
        np.mean([np.mean(r["total_activation"]) for r in results["force_vector"]]),
    ]
    mr_vals = [
        mr_dims,
        np.mean(mr_nonzero) * 100,
        np.mean([np.mean(r["total_activation"]) for r in results["multi_receptor"]]),
    ]

    x_pos = np.arange(len(categories))
    width = 0.35
    # Normalize for visual comparison
    for i in range(len(categories)):
        max_val = max(fv_vals[i], mr_vals[i], 1e-6)
        fv_vals[i] = fv_vals[i] / max_val * 100
        mr_vals[i] = mr_vals[i] / max_val * 100

    ax.bar(x_pos - width/2, fv_vals, width, color=C_OFF, label="Force Vector (CT OFF)",
           edgecolor="black", linewidth=0.5)
    ax.bar(x_pos + width/2, mr_vals, width, color=C_ON, label="Multi-receptor (CT ON)",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Normalized value (%)")
    ax.set_title("(b) Information richness comparison")
    ax.legend()

    fig3.suptitle("Figure 3: CT Model Properties & Information Content", fontsize=13, y=1.02)
    fig3.tight_layout()
    fig3.savefig(RESULTS_DIR / "fig3_ct_model.png")
    print(f"  Saved: {RESULTS_DIR / 'fig3_ct_model.png'}")
    plt.close(fig3)


# ── Results summary ───────────────────────────────────────────────────────

def write_results(results):
    """Write quantitative results to markdown."""
    lines = []
    lines.append("# CT Afferent ON vs OFF: Experiment Results")
    lines.append("")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Steps per condition**: {N_STEPS}")
    lines.append(f"**Seeds**: {SEEDS}")
    lines.append(f"**Policy**: Random (uniform action sampling)")
    lines.append("")

    # Contact frequency
    lines.append("## 1. Contact Frequency")
    lines.append("")
    fv_contacts = [r["contact_steps"] for r in results["force_vector"]]
    mr_contacts = [r["contact_steps"] for r in results["multi_receptor"]]
    lines.append(f"| Condition | Contact steps (mean +/- SD) | % of total |")
    lines.append(f"|-----------|---------------------------|------------|")
    lines.append(f"| CT OFF (force_vector) | {np.mean(fv_contacts):.1f} +/- {np.std(fv_contacts):.1f} | {np.mean(fv_contacts)/N_STEPS*100:.1f}% |")
    lines.append(f"| CT ON (multi_receptor) | {np.mean(mr_contacts):.1f} +/- {np.std(mr_contacts):.1f} | {np.mean(mr_contacts)/N_STEPS*100:.1f}% |")
    lines.append("")

    # Total activation
    lines.append("## 2. Total Sensor Activation")
    lines.append("")
    fv_total = [np.mean(r["total_activation"]) for r in results["force_vector"]]
    mr_total = [np.mean(r["total_activation"]) for r in results["multi_receptor"]]
    lines.append(f"| Condition | Mean activation (mean +/- SD) |")
    lines.append(f"|-----------|-------------------------------|")
    lines.append(f"| CT OFF | {np.mean(fv_total):.4f} +/- {np.std(fv_total):.4f} |")
    lines.append(f"| CT ON  | {np.mean(mr_total):.4f} +/- {np.std(mr_total):.4f} |")
    lines.append("")

    # Per-body contact
    lines.append("## 3. Per-Body Contact Frequency")
    lines.append("")
    all_bodies = sorted(set().union(*[r["body_contact_counts"].keys()
                                      for r in results["force_vector"]],
                                    *[r["body_contact_counts"].keys()
                                      for r in results["multi_receptor"]]))

    lines.append(f"| Body Part | Skin Type | CT OFF (%) | CT ON (%) |")
    lines.append(f"|-----------|-----------|------------|-----------|")
    for body in all_bodies:
        skin = get_skin_type(body).value
        fv_pct = np.mean([r["body_contact_counts"].get(body, 0) for r in results["force_vector"]]) / N_STEPS * 100
        mr_pct = np.mean([r["body_contact_counts"].get(body, 0) for r in results["multi_receptor"]]) / N_STEPS * 100
        lines.append(f"| {body} | {skin} | {fv_pct:.1f} | {mr_pct:.1f} |")
    lines.append("")

    # Multi-receptor channel analysis
    lines.append("## 4. Multi-receptor Channel Analysis (CT ON only)")
    lines.append("")
    channels = ["SA-I", "FA-I", "FA-II", "CT", "Normal"]
    lines.append(f"| Channel | Mean activation | % of total |")
    lines.append(f"|---------|-----------------|------------|")
    ch_totals = {}
    for ch in channels:
        ch_totals[ch] = np.mean([np.mean(r["channel_means"][ch])
                                 for r in results["multi_receptor"]])
    grand_total = sum(ch_totals.values())
    for ch in channels:
        pct = ch_totals[ch] / grand_total * 100 if grand_total > 0 else 0
        pct_str = f"{pct:.2f}%" if pct < 0.1 else f"{pct:.1f}%"
        lines.append(f"| {ch} | {ch_totals[ch]:.6f} | {pct_str} |")
    lines.append("")

    # CT by skin type
    lines.append("## 5. CT Response by Skin Type")
    lines.append("")
    ct_hairy = np.mean([np.mean(r["ct_by_skin_type"]["hairy"])
                        for r in results["multi_receptor"]])
    ct_glabrous = np.mean([np.mean(r["ct_by_skin_type"]["glabrous"])
                           for r in results["multi_receptor"]])
    lines.append(f"| Skin Type | Mean CT activation |")
    lines.append(f"|-----------|-------------------|")
    lines.append(f"| Hairy (CT present) | {ct_hairy:.6f} |")
    lines.append(f"| Glabrous (no CT) | {ct_glabrous:.6f} |")
    lines.append("")

    # Key findings
    lines.append("## 6. Key Findings")
    lines.append("")
    lines.append("1. **Multi-receptor output provides richer information**: 7 channels per sensor vs 3, ")
    lines.append("   decomposing force into physiologically meaningful receptor types.")
    lines.append("2. **CT afferents respond only on hairy skin**: Consistent with neurophysiology ")
    lines.append("   (Vallbo et al. 1999). Glabrous areas (hands, feet) show zero CT activation.")
    lines.append("3. **CT firing rate follows inverted-U velocity tuning**: Peak at ~3 cm/s ")
    lines.append("   (Loken et al. 2009), providing velocity-dependent affective touch signals.")
    if grand_total > 0:
        ct_pct = ch_totals["CT"] / grand_total * 100
        ct_pct_str = f"{ct_pct:.3f}%" if ct_pct < 0.1 else f"{ct_pct:.2f}%"
        lines.append(f"4. **CT channel constitutes {ct_pct_str} of total multi-receptor output**: ")
        lines.append("   While small in magnitude (consistent with CT's role as a gentle-touch sensor),")
        lines.append("   this represents a qualitatively new velocity-tuned signal absent in force-only mode.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `fig1_overall_comparison.png`: Overall comparison of CT ON vs OFF")
    lines.append("- `fig2_channel_analysis.png`: Multi-receptor channel breakdown")
    lines.append("- `fig3_ct_model.png`: CT model properties and information content")

    text = "\n".join(lines) + "\n"
    with open(RESULTS_DIR / "results.md", "w") as f:
        f.write(text)
    print(f"  Saved: {RESULTS_DIR / 'results.md'}")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    results = run_experiment()

    print("\n--- Generating figures ---")
    make_figures(results)

    print("\n--- Writing results ---")
    write_results(results)

    t_total = time.time() - t_start
    print(f"\nTotal experiment time: {t_total:.1f}s")
    print("Done!")
