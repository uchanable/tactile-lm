#!/usr/bin/env python3
"""
SOM 2x2 Factorial Analysis
===========================

PPO vs SOM × CT OFF vs CT ON の 2×2 要因分析.

4 conditions:
  - PPO_CT_OFF (from_macstudio/reach/)
  - PPO_CT_ON  (from_macstudio/reach/)
  - SOM_CT_OFF (rl_results/som_reach/)
  - SOM_CT_ON  (rl_results/som_reach/)

Usage:
    cd /path/to/mimo-tactile
    python3 experiments/analyze_som_results.py

Output:
    - Console: Markdown tables
    - File: 30_Research/09_TactileLM/02_IEEE_TCDS/som_factorial_analysis.md
"""

import json
import os
import sys
from glob import glob
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats


# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PPO_DIR = SCRIPT_DIR / "from_macstudio" / "reach"
SOM_DIR = SCRIPT_DIR / "rl_results" / "som_reach"
OUTPUT_MD = (
    Path.home()
    / "obsidian"
    / "30_Research"
    / "09_TactileLM"
    / "02_IEEE_TCDS"
    / "som_factorial_analysis.md"
)


# =============================================================================
# Data loading
# =============================================================================

def load_condition(directory: Path, prefix: str, steps_k: int = 1000) -> list[dict]:
    """Load all JSON results for a condition at a given step count.

    Args:
        directory: Path to the results directory.
        prefix: File prefix, e.g. "CT_ON" or "SOM_CT_ON".
        steps_k: Training steps in thousands (1000 = 1M).

    Returns:
        List of result dicts, sorted by seed.
    """
    pattern = f"{prefix}_{steps_k}K_seed*.json"
    files = sorted(glob(str(directory / pattern)))
    results = []
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Failed to load {fpath}: {e}", file=sys.stderr)
    return results


def load_all_conditions() -> dict[str, list[dict]]:
    """Load all 4 conditions.

    Returns:
        Dict mapping condition name to list of result dicts.
    """
    conditions = {}

    # PPO conditions (from_macstudio/reach/)
    if PPO_DIR.exists():
        ppo_off = load_condition(PPO_DIR, "CT_OFF")
        ppo_on = load_condition(PPO_DIR, "CT_ON")
        if ppo_off:
            conditions["PPO_CT_OFF"] = ppo_off
        if ppo_on:
            conditions["PPO_CT_ON"] = ppo_on
    else:
        print(f"  [INFO] PPO directory not found: {PPO_DIR}", file=sys.stderr)

    # SOM conditions (rl_results/som_reach/)
    if SOM_DIR.exists():
        som_off = load_condition(SOM_DIR, "SOM_CT_OFF")
        som_on = load_condition(SOM_DIR, "SOM_CT_ON")
        if som_off:
            conditions["SOM_CT_OFF"] = som_off
        if som_on:
            conditions["SOM_CT_ON"] = som_on
    else:
        print(f"  [INFO] SOM directory not found: {SOM_DIR}", file=sys.stderr)

    return conditions


# =============================================================================
# Statistical helpers
# =============================================================================

def hedges_g(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Compute Hedges' g with 95% CI.

    Args:
        x: Group 1 values.
        y: Group 2 values.

    Returns:
        (g, ci_low, ci_high)
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan"), float("nan"), float("nan")

    mean_diff = np.mean(y) - np.mean(x)
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (
        nx + ny - 2
    )
    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd == 0:
        return float("nan"), float("nan"), float("nan")

    d = mean_diff / pooled_sd
    # Hedges' correction factor
    df = nx + ny - 2
    j = 1 - 3 / (4 * df - 1)
    g = d * j

    # SE of g
    se = np.sqrt((nx + ny) / (nx * ny) + g**2 / (2 * (nx + ny)))
    ci_low = g - 1.96 * se
    ci_high = g + 1.96 * se

    return g, ci_low, ci_high


def mann_whitney(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Mann-Whitney U test.

    Returns:
        (U, p_value, rank_biserial_r)
    """
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan"), float("nan")

    u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
    # Rank-biserial correlation
    n1, n2 = len(x), len(y)
    r = 1 - (2 * u_stat) / (n1 * n2)
    return u_stat, p_val, r


def kruskal_wallis_2x2(data: dict[str, np.ndarray]) -> dict:
    """Perform a non-parametric 2x2 factorial analysis using
    aligned rank transform (ART-like) approach.

    For each factor and interaction, we compute:
    - Main effect via Kruskal-Wallis or Mann-Whitney
    - Effect sizes

    Args:
        data: Dict with keys PPO_CT_OFF, PPO_CT_ON, SOM_CT_OFF, SOM_CT_ON.

    Returns:
        Dict with test results.
    """
    results = {}

    # Check all 4 conditions exist
    required = ["PPO_CT_OFF", "PPO_CT_ON", "SOM_CT_OFF", "SOM_CT_ON"]
    available = [k for k in required if k in data and len(data[k]) > 0]

    if len(available) < 4:
        results["error"] = f"Only {len(available)}/4 conditions available: {available}"
        return results

    ppo_off = data["PPO_CT_OFF"]
    ppo_on = data["PPO_CT_ON"]
    som_off = data["SOM_CT_OFF"]
    som_on = data["SOM_CT_ON"]

    # --- Main effect of Architecture (PPO vs SOM) ---
    ppo_all = np.concatenate([ppo_off, ppo_on])
    som_all = np.concatenate([som_off, som_on])
    u_arch, p_arch, r_arch = mann_whitney(ppo_all, som_all)
    g_arch, g_arch_lo, g_arch_hi = hedges_g(ppo_all, som_all)
    results["architecture"] = {
        "U": u_arch,
        "p": p_arch,
        "r": r_arch,
        "hedges_g": g_arch,
        "g_ci": (g_arch_lo, g_arch_hi),
        "ppo_mean": float(np.mean(ppo_all)),
        "ppo_sd": float(np.std(ppo_all, ddof=1)),
        "som_mean": float(np.mean(som_all)),
        "som_sd": float(np.std(som_all, ddof=1)),
    }

    # --- Main effect of CT (OFF vs ON) ---
    ct_off_all = np.concatenate([ppo_off, som_off])
    ct_on_all = np.concatenate([ppo_on, som_on])
    u_ct, p_ct, r_ct = mann_whitney(ct_off_all, ct_on_all)
    g_ct, g_ct_lo, g_ct_hi = hedges_g(ct_off_all, ct_on_all)
    results["ct"] = {
        "U": u_ct,
        "p": p_ct,
        "r": r_ct,
        "hedges_g": g_ct,
        "g_ci": (g_ct_lo, g_ct_hi),
        "off_mean": float(np.mean(ct_off_all)),
        "off_sd": float(np.std(ct_off_all, ddof=1)),
        "on_mean": float(np.mean(ct_on_all)),
        "on_sd": float(np.std(ct_on_all, ddof=1)),
    }

    # --- Interaction (Architecture x CT) ---
    # CT benefit in PPO vs CT benefit in SOM
    ct_benefit_ppo = ppo_on - ppo_off  # element-wise (assumes matched seeds)
    ct_benefit_som = som_on - som_off

    # If seed counts differ, use mean-based interaction
    if len(ppo_off) == len(ppo_on) and len(som_off) == len(som_on):
        u_int, p_int, r_int = mann_whitney(ct_benefit_ppo, ct_benefit_som)
        g_int, g_int_lo, g_int_hi = hedges_g(ct_benefit_ppo, ct_benefit_som)
    else:
        # Fallback: compare means
        ppo_benefit = np.mean(ppo_on) - np.mean(ppo_off)
        som_benefit = np.mean(som_on) - np.mean(som_off)
        u_int, p_int, r_int = float("nan"), float("nan"), float("nan")
        g_int, g_int_lo, g_int_hi = float("nan"), float("nan"), float("nan")

    results["interaction"] = {
        "U": u_int,
        "p": p_int,
        "r": r_int,
        "hedges_g": g_int,
        "g_ci": (g_int_lo, g_int_hi),
        "ct_benefit_ppo_mean": float(np.mean(ct_benefit_ppo)) if len(ct_benefit_ppo) > 0 else float("nan"),
        "ct_benefit_ppo_sd": float(np.std(ct_benefit_ppo, ddof=1)) if len(ct_benefit_ppo) > 1 else float("nan"),
        "ct_benefit_som_mean": float(np.mean(ct_benefit_som)) if len(ct_benefit_som) > 0 else float("nan"),
        "ct_benefit_som_sd": float(np.std(ct_benefit_som, ddof=1)) if len(ct_benefit_som) > 1 else float("nan"),
    }

    return results


def pairwise_comparisons(data: dict[str, np.ndarray]) -> list[dict]:
    """All 6 pairwise Mann-Whitney U comparisons.

    Returns:
        List of dicts with comparison results.
    """
    conditions = [k for k in ["PPO_CT_OFF", "PPO_CT_ON", "SOM_CT_OFF", "SOM_CT_ON"] if k in data]
    pairs = []
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            a, b = conditions[i], conditions[j]
            x, y = data[a], data[b]
            u, p, r = mann_whitney(x, y)
            g, g_lo, g_hi = hedges_g(x, y)
            pairs.append({
                "comparison": f"{a} vs {b}",
                "a": a,
                "b": b,
                "n_a": len(x),
                "n_b": len(y),
                "mean_a": float(np.mean(x)),
                "mean_b": float(np.mean(y)),
                "U": u,
                "p": p,
                "r": r,
                "hedges_g": g,
                "g_ci_lo": g_lo,
                "g_ci_hi": g_hi,
            })
    return pairs


# =============================================================================
# SOM-specific analysis
# =============================================================================

def analyze_som_metrics(conditions: dict[str, list[dict]]) -> dict:
    """Analyze SOM-specific metrics (Hebbian binding, topographic error, etc.).

    Returns:
        Dict with SOM metric comparisons.
    """
    results = {}

    for cond_name in ["SOM_CT_OFF", "SOM_CT_ON"]:
        if cond_name not in conditions:
            continue

        metrics_data = {}
        for run in conditions[cond_name]:
            som = run.get("som_metrics", {})
            for k, v in som.items():
                if isinstance(v, (int, float)):
                    metrics_data.setdefault(k, []).append(v)

        if metrics_data:
            results[cond_name] = {
                k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
                for k, v in metrics_data.items()
            }

    # Compare SOM_CT_ON vs SOM_CT_OFF on shared metrics
    if "SOM_CT_OFF" in results and "SOM_CT_ON" in results:
        shared_keys = set(results["SOM_CT_OFF"].keys()) & set(results["SOM_CT_ON"].keys())
        comparisons = {}
        for k in sorted(shared_keys):
            off_vals = []
            on_vals = []
            for run in conditions["SOM_CT_OFF"]:
                v = run.get("som_metrics", {}).get(k)
                if v is not None and isinstance(v, (int, float)):
                    off_vals.append(v)
            for run in conditions["SOM_CT_ON"]:
                v = run.get("som_metrics", {}).get(k)
                if v is not None and isinstance(v, (int, float)):
                    on_vals.append(v)

            if len(off_vals) >= 2 and len(on_vals) >= 2:
                off_arr = np.array(off_vals)
                on_arr = np.array(on_vals)
                u, p, r = mann_whitney(off_arr, on_arr)
                g, g_lo, g_hi = hedges_g(off_arr, on_arr)
                comparisons[k] = {
                    "off_mean": float(np.mean(off_arr)),
                    "off_sd": float(np.std(off_arr, ddof=1)),
                    "on_mean": float(np.mean(on_arr)),
                    "on_sd": float(np.std(on_arr, ddof=1)),
                    "U": u,
                    "p": p,
                    "hedges_g": g,
                }

        results["comparisons"] = comparisons

    return results


def analyze_som_history(conditions: dict[str, list[dict]]) -> dict:
    """Analyze SOM metrics over training time (from som_metrics_history).

    Returns summary at key checkpoints.
    """
    results = {}
    for cond_name in ["SOM_CT_OFF", "SOM_CT_ON"]:
        if cond_name not in conditions:
            continue

        # Collect all histories
        histories = [run.get("som_metrics_history", []) for run in conditions[cond_name]]
        histories = [h for h in histories if h]  # filter empty

        if not histories:
            continue

        # Find common steps
        checkpoints = [100_000, 250_000, 500_000, 750_000, 1_000_000]
        cond_results = {}

        for cp in checkpoints:
            metric_vals = {}
            for hist in histories:
                # Find entry closest to checkpoint
                best = None
                best_dist = float("inf")
                for entry in hist:
                    step = entry.get("step", 0)
                    dist = abs(step - cp)
                    if dist < best_dist:
                        best_dist = dist
                        best = entry
                if best and best_dist <= 50_000:  # within 50K tolerance
                    for k, v in best.items():
                        if k == "step":
                            continue
                        if isinstance(v, (int, float)):
                            metric_vals.setdefault(k, []).append(v)

            if metric_vals:
                cond_results[f"{cp // 1000}K"] = {
                    k: {"mean": float(np.mean(v)), "n": len(v)}
                    for k, v in metric_vals.items()
                }

        if cond_results:
            results[cond_name] = cond_results

    return results


# =============================================================================
# Report generation
# =============================================================================

def fmt(val: float, decimals: int = 2) -> str:
    """Format a float, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def fmt_p(p: float) -> str:
    """Format p-value with significance markers."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "N/A"
    if p < 0.001:
        return f"{p:.4f} ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f}"


def effect_label(g: float) -> str:
    """Interpret effect size."""
    if np.isnan(g):
        return "N/A"
    ag = abs(g)
    if ag < 0.2:
        return "negligible"
    elif ag < 0.5:
        return "small"
    elif ag < 0.8:
        return "medium"
    else:
        return "large"


def generate_report(
    conditions: dict[str, list[dict]],
    reward_data: dict[str, np.ndarray],
    factorial: dict,
    pairs: list[dict],
    som_metrics: dict,
    som_history: dict,
) -> str:
    """Generate full Markdown report.

    Returns:
        Markdown string.
    """
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("# SOM × CT 2×2 Factorial Analysis")
    lines.append("")
    lines.append(f"**Generated**: {now}")
    lines.append(f"**Task**: Reach (1M steps)")
    lines.append(f"**Conditions**: {', '.join(sorted(conditions.keys()))}")
    lines.append(f"**Seeds**: " + ", ".join(
        f"{k}: {len(v)}" for k, v in sorted(conditions.items())
    ))
    lines.append("")
    lines.append("---")

    # ── Section 1: Descriptive statistics ──
    lines.append("")
    lines.append("## 1. Descriptive Statistics")
    lines.append("")
    lines.append("| Condition | N | Reward (M +/- SD) | Touch (M) | CT (M) |")
    lines.append("|-----------|---|-------------------|-----------|--------|")

    for cname in ["PPO_CT_OFF", "PPO_CT_ON", "SOM_CT_OFF", "SOM_CT_ON"]:
        if cname not in conditions:
            continue
        runs = conditions[cname]
        n = len(runs)
        rewards = [r["mean_reward"] for r in runs]
        touches = [r.get("mean_touch", float("nan")) for r in runs]
        cts = [r.get("mean_ct", float("nan")) for r in runs]

        r_mean = np.mean(rewards)
        r_sd = np.std(rewards, ddof=1) if n > 1 else 0.0
        t_mean = np.nanmean(touches) if any(not np.isnan(t) for t in touches) else float("nan")
        c_mean = np.nanmean(cts) if any(not np.isnan(c) for c in cts) else float("nan")

        touch_str = fmt(t_mean, 0) if not np.isnan(t_mean) else "-"
        ct_str = fmt(c_mean, 2) if not np.isnan(c_mean) else "-"

        lines.append(
            f"| {cname} | {n} | {fmt(r_mean, 1)} +/- {fmt(r_sd, 1)} | {touch_str} | {ct_str} |"
        )

    # ── Section 2: 2x2 Factorial ──
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 2×2 Factorial Analysis (Non-parametric)")
    lines.append("")

    if "error" in factorial:
        lines.append(f"> **Note**: {factorial['error']}")
        lines.append("")
    else:
        lines.append("| Effect | U | p | rank-biserial r | Hedges' g | 95% CI | Size |")
        lines.append("|--------|---|---|----------------|-----------|--------|------|")

        for eff_name, eff_label_str in [
            ("architecture", "Architecture (PPO vs SOM)"),
            ("ct", "CT (OFF vs ON)"),
            ("interaction", "Architecture × CT"),
        ]:
            eff = factorial[eff_name]
            g = eff["hedges_g"]
            ci = eff["g_ci"]
            lines.append(
                f"| {eff_label_str} | {fmt(eff['U'], 0)} | {fmt_p(eff['p'])} | "
                f"{fmt(eff['r'], 3)} | {fmt(g, 2)} | [{fmt(ci[0], 2)}, {fmt(ci[1], 2)}] | "
                f"{effect_label(g)} |"
            )

        lines.append("")
        lines.append("### Interaction Detail")
        lines.append("")
        inter = factorial["interaction"]
        lines.append(f"- CT benefit in PPO: {fmt(inter['ct_benefit_ppo_mean'], 1)} +/- {fmt(inter['ct_benefit_ppo_sd'], 1)}")
        lines.append(f"- CT benefit in SOM: {fmt(inter['ct_benefit_som_mean'], 1)} +/- {fmt(inter['ct_benefit_som_sd'], 1)}")
        lines.append("")

        # Interpretation
        if not np.isnan(inter.get("p", float("nan"))):
            if inter["p"] < 0.05:
                lines.append("> **Significant interaction**: CT의 효과가 Architecture에 따라 달라진다.")
            else:
                lines.append("> Interaction 비유의: CT의 효과가 PPO와 SOM에서 유사하다.")
        lines.append("")

    # ── Section 3: Pairwise comparisons ──
    lines.append("---")
    lines.append("")
    lines.append("## 3. Pairwise Comparisons (Mann-Whitney U)")
    lines.append("")
    lines.append("| Comparison | Mean A | Mean B | Diff% | U | p | Hedges' g | 95% CI | Size |")
    lines.append("|------------|--------|--------|-------|---|---|-----------|--------|------|")

    for pair in pairs:
        mean_a = pair["mean_a"]
        mean_b = pair["mean_b"]

        # Compute percentage difference (relative to worse = more negative baseline)
        if mean_a != 0:
            # For negative rewards: positive diff% means B is better (less negative)
            diff_pct = (mean_b - mean_a) / abs(mean_a) * 100
        else:
            diff_pct = float("nan")

        g = pair["hedges_g"]
        lines.append(
            f"| {pair['comparison']} | {fmt(mean_a, 1)} | {fmt(mean_b, 1)} | "
            f"{fmt(diff_pct, 1)}% | {fmt(pair['U'], 0)} | {fmt_p(pair['p'])} | "
            f"{fmt(g, 2)} | [{fmt(pair['g_ci_lo'], 2)}, {fmt(pair['g_ci_hi'], 2)}] | "
            f"{effect_label(g)} |"
        )

    # ── Section 4: SOM-specific metrics ──
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. SOM-Specific Metrics")
    lines.append("")

    if not som_metrics:
        lines.append("> SOM 결과가 아직 없거나 som_metrics 필드가 없습니다.")
    else:
        for cond_name in ["SOM_CT_OFF", "SOM_CT_ON"]:
            if cond_name in som_metrics and cond_name != "comparisons":
                lines.append(f"### {cond_name}")
                lines.append("")
                lines.append("| Metric | Mean | SD | N |")
                lines.append("|--------|------|----|---|")
                for k, v in sorted(som_metrics[cond_name].items()):
                    lines.append(f"| {k} | {fmt(v['mean'], 4)} | {fmt(v['sd'], 4)} | {v['n']} |")
                lines.append("")

        if "comparisons" in som_metrics:
            lines.append("### SOM_CT_OFF vs SOM_CT_ON Metric Comparison")
            lines.append("")
            lines.append("| Metric | OFF (M +/- SD) | ON (M +/- SD) | U | p | Hedges' g |")
            lines.append("|--------|---------------|--------------|---|---|-----------|")
            for k, v in sorted(som_metrics["comparisons"].items()):
                lines.append(
                    f"| {k} | {fmt(v['off_mean'], 4)} +/- {fmt(v['off_sd'], 4)} | "
                    f"{fmt(v['on_mean'], 4)} +/- {fmt(v['on_sd'], 4)} | "
                    f"{fmt(v['U'], 0)} | {fmt_p(v['p'])} | {fmt(v['hedges_g'], 2)} |"
                )
            lines.append("")

    # ── Section 5: SOM history (developmental trajectory) ──
    if som_history:
        lines.append("---")
        lines.append("")
        lines.append("## 5. SOM Developmental Trajectory")
        lines.append("")

        for cond_name in ["SOM_CT_OFF", "SOM_CT_ON"]:
            if cond_name not in som_history:
                continue
            lines.append(f"### {cond_name}")
            lines.append("")

            # Collect all metric names
            all_metrics = set()
            for cp_data in som_history[cond_name].values():
                all_metrics.update(cp_data.keys())
            all_metrics = sorted(all_metrics)

            if all_metrics:
                header = "| Checkpoint | " + " | ".join(all_metrics) + " |"
                sep = "|------------|" + "|".join(["------"] * len(all_metrics)) + "|"
                lines.append(header)
                lines.append(sep)

                for cp in ["100K", "250K", "500K", "750K", "1000K"]:
                    if cp not in som_history[cond_name]:
                        continue
                    cp_data = som_history[cond_name][cp]
                    vals = []
                    for m in all_metrics:
                        if m in cp_data:
                            vals.append(fmt(cp_data[m]["mean"], 4))
                        else:
                            vals.append("-")
                    lines.append(f"| {cp} | " + " | ".join(vals) + " |")

                lines.append("")

    # ── Section 6: Meta-analysis (CT benefit across architectures) ──
    lines.append("---")
    lines.append("")
    lines.append("## 6. CT Benefit Meta-analysis")
    lines.append("")

    ct_benefits = {}
    for arch in ["PPO", "SOM"]:
        off_key = f"{arch}_CT_OFF"
        on_key = f"{arch}_CT_ON"
        if off_key in reward_data and on_key in reward_data:
            off = reward_data[off_key]
            on = reward_data[on_key]
            g, g_lo, g_hi = hedges_g(off, on)
            u, p, r = mann_whitney(off, on)

            benefit_mean = float(np.mean(on) - np.mean(off))
            # Percentage relative to OFF
            if np.mean(off) != 0:
                benefit_pct = benefit_mean / abs(np.mean(off)) * 100
            else:
                benefit_pct = float("nan")

            ct_benefits[arch] = {
                "benefit_mean": benefit_mean,
                "benefit_pct": benefit_pct,
                "hedges_g": g,
                "g_ci": (g_lo, g_hi),
                "p": p,
            }

    if ct_benefits:
        lines.append("| Architecture | CT Benefit (mean) | CT Benefit (%) | Hedges' g | 95% CI | p |")
        lines.append("|-------------|-------------------|----------------|-----------|--------|---|")
        for arch, b in ct_benefits.items():
            lines.append(
                f"| {arch} | {fmt(b['benefit_mean'], 1)} | "
                f"{fmt(b['benefit_pct'], 1)}% | {fmt(b['hedges_g'], 2)} | "
                f"[{fmt(b['g_ci'][0], 2)}, {fmt(b['g_ci'][1], 2)}] | {fmt_p(b['p'])} |"
            )

        lines.append("")

        if len(ct_benefits) == 2:
            ppo_g = ct_benefits["PPO"]["hedges_g"]
            som_g = ct_benefits["SOM"]["hedges_g"]
            if not np.isnan(ppo_g) and not np.isnan(som_g):
                if som_g > ppo_g:
                    lines.append(
                        f"> SOM에서 CT 효과가 더 크다 (g={fmt(som_g, 2)} vs {fmt(ppo_g, 2)}). "
                        f"Interaction p-value로 통계적 유의성 확인 필요."
                    )
                else:
                    lines.append(
                        f"> PPO에서 CT 효과가 더 크거나 유사하다 (PPO g={fmt(ppo_g, 2)}, SOM g={fmt(som_g, 2)})."
                    )
    else:
        lines.append("> CT benefit 비교를 위해 최소 2 conditions (CT_OFF, CT_ON)가 필요합니다.")

    # ── Section 7: Win rate analysis ──
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Win Rate Analysis (Seed-level)")
    lines.append("")

    for comparison_label, a_key, b_key in [
        ("PPO: CT_ON vs CT_OFF", "PPO_CT_OFF", "PPO_CT_ON"),
        ("SOM: CT_ON vs CT_OFF", "SOM_CT_OFF", "SOM_CT_ON"),
        ("CT_OFF: SOM vs PPO", "PPO_CT_OFF", "SOM_CT_OFF"),
        ("CT_ON: SOM vs PPO", "PPO_CT_ON", "SOM_CT_ON"),
    ]:
        if a_key not in conditions or b_key not in conditions:
            continue

        # Match by seed
        a_by_seed = {r["seed"]: r["mean_reward"] for r in conditions[a_key]}
        b_by_seed = {r["seed"]: r["mean_reward"] for r in conditions[b_key]}
        common_seeds = sorted(set(a_by_seed.keys()) & set(b_by_seed.keys()))

        if not common_seeds:
            continue

        wins_b = sum(1 for s in common_seeds if b_by_seed[s] > a_by_seed[s])
        total = len(common_seeds)
        win_pct = wins_b / total * 100

        # Binomial test
        p_binom = stats.binomtest(wins_b, total, 0.5).pvalue

        lines.append(f"- **{comparison_label}**: {wins_b}/{total} ({win_pct:.0f}%) wins, binomial p={fmt_p(p_binom)}")

    # ── Section 8: Body contact comparison (if available) ──
    has_body = any(
        "body_contacts" in run and run["body_contacts"]
        for cond_runs in conditions.values()
        for run in cond_runs
    )
    if has_body:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 8. Body Contact Distribution")
        lines.append("")

        # Collect all body parts
        all_bodies = set()
        for cond_runs in conditions.values():
            for run in cond_runs:
                bc = run.get("body_contacts", {})
                all_bodies.update(bc.keys())

        if all_bodies:
            all_bodies = sorted(all_bodies)
            header = "| Body Part | " + " | ".join(sorted(conditions.keys())) + " |"
            sep = "|-----------|" + "|".join(["------"] * len(conditions)) + "|"
            lines.append(header)
            lines.append(sep)

            for body in all_bodies:
                vals = []
                for cname in sorted(conditions.keys()):
                    body_vals = [
                        run.get("body_contacts", {}).get(body, 0.0)
                        for run in conditions[cname]
                    ]
                    if body_vals:
                        vals.append(fmt(np.mean(body_vals), 1))
                    else:
                        vals.append("-")
                lines.append(f"| {body} | " + " | ".join(vals) + " |")

    # Footer
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Analysis script: `experiments/analyze_som_results.py`*")
    lines.append(f"*Data: PPO from `{PPO_DIR.relative_to(SCRIPT_DIR)}`, SOM from `{SOM_DIR.relative_to(SCRIPT_DIR)}`*")

    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("  SOM × CT 2×2 Factorial Analysis")
    print("=" * 60)

    # ── Load data ──
    print("\n[1] Loading data...")
    conditions = load_all_conditions()

    if not conditions:
        print("\n  No data found in either directory.")
        print(f"  PPO dir: {PPO_DIR}")
        print(f"  SOM dir: {SOM_DIR}")
        print("\n  스크립트는 데이터가 생성된 후 다시 실행하세요.")
        return

    for cname, runs in sorted(conditions.items()):
        rewards = [r["mean_reward"] for r in runs]
        print(f"  {cname}: {len(runs)} seeds, reward = {np.mean(rewards):.1f} +/- {np.std(rewards, ddof=1):.1f}")

    # ── Extract reward arrays ──
    reward_data: dict[str, np.ndarray] = {}
    for cname, runs in conditions.items():
        # Sort by seed for matched comparisons
        sorted_runs = sorted(runs, key=lambda r: r.get("seed", 0))
        reward_data[cname] = np.array([r["mean_reward"] for r in sorted_runs])

    # ── 2x2 Factorial ──
    print("\n[2] 2×2 Factorial analysis...")
    factorial = kruskal_wallis_2x2(reward_data)

    if "error" in factorial:
        print(f"  {factorial['error']}")
    else:
        for eff_name, label in [
            ("architecture", "Architecture"),
            ("ct", "CT"),
            ("interaction", "Interaction"),
        ]:
            eff = factorial[eff_name]
            print(f"  {label}: p={fmt_p(eff['p'])}, g={fmt(eff['hedges_g'], 2)}")

    # ── Pairwise comparisons ──
    print("\n[3] Pairwise comparisons...")
    pairs = pairwise_comparisons(reward_data)
    for pair in pairs:
        print(f"  {pair['comparison']}: p={fmt_p(pair['p'])}, g={fmt(pair['hedges_g'], 2)}")

    # ── SOM-specific ──
    print("\n[4] SOM-specific metrics...")
    som_metrics = analyze_som_metrics(conditions)
    if som_metrics:
        for k in ["SOM_CT_OFF", "SOM_CT_ON"]:
            if k in som_metrics and k != "comparisons":
                print(f"  {k}: {len(som_metrics[k])} metrics")
    else:
        print("  No SOM metrics available yet.")

    # ── SOM history ──
    print("\n[5] SOM developmental trajectory...")
    som_history = analyze_som_history(conditions)
    if som_history:
        for k, v in som_history.items():
            print(f"  {k}: {len(v)} checkpoints")
    else:
        print("  No SOM history available yet.")

    # ── Generate report ──
    print("\n[6] Generating report...")
    report = generate_report(
        conditions, reward_data, factorial, pairs, som_metrics, som_history
    )

    # Print to console
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Save to file
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write(report)
    print(f"\n  Saved: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
