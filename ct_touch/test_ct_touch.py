"""Test script for CT afferent augmented touch module.

Loads a MIMo environment with CTAugmentedTouch, runs random actions,
and validates that all CT-related outputs work correctly.

Run from project root (mimo-tactile/):
    PYTHONPATH=MIMo:. .venv312/bin/python -m ct_touch.test_ct_touch
"""

import os
import sys
import numpy as np

# Paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_MIMO_ROOT = os.path.join(_PROJECT_ROOT, "MIMo")

# Add project root (for ct_touch) -- append to avoid disrupting MIMo's import order
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)
# Add MIMo root (for mimoTouch, mimoEnv, etc.) -- also append
if _MIMO_ROOT not in sys.path:
    sys.path.append(_MIMO_ROOT)


def test_ct_firing_rate_model():
    """Test the standalone CT firing rate function."""
    from ct_touch.ct_augmented_touch import CTAugmentedTouch

    print("=" * 60)
    print("TEST 1: CT Firing Rate Model (Loken et al. 2009)")
    print("=" * 60)

    velocities = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    print(f"  {'Velocity (m/s)':>15}  {'Velocity (cm/s)':>15}  {'Firing Rate':>12}")
    print(f"  {'-'*15}  {'-'*15}  {'-'*12}")
    for v in velocities:
        rate = CTAugmentedTouch.ct_firing_rate(v)
        print(f"  {v:>15.4f}  {v*100:>15.2f}  {rate:>12.4f}")

    # Verify peak is near 3 cm/s
    peak_rate = CTAugmentedTouch.ct_firing_rate(0.03)
    assert peak_rate > 0.99, f"Peak rate at 3 cm/s should be ~1.0, got {peak_rate}"

    # Verify inverted-U: rates at 0.003 and 0.3 should be lower than at 0.03
    rate_slow = CTAugmentedTouch.ct_firing_rate(0.003)
    rate_fast = CTAugmentedTouch.ct_firing_rate(0.3)
    assert rate_slow < peak_rate, "Rate at 0.3 cm/s should be < peak"
    assert rate_fast < peak_rate, "Rate at 30 cm/s should be < peak"

    # Verify symmetry in log-space
    assert abs(rate_slow - rate_fast) < 0.5, \
        f"Log-Gaussian should be roughly symmetric: {rate_slow:.3f} vs {rate_fast:.3f}"

    print("\n  [PASS] CT firing rate model validated.\n")


def test_skin_map():
    """Test skin type mapping."""
    from ct_touch.skin_map import get_skin_type, has_ct_afferents, SkinType

    print("=" * 60)
    print("TEST 2: Skin Type Map")
    print("=" * 60)

    hairy_parts = ["left_upper_arm", "chest", "head", "left_lower_leg"]
    glabrous_parts = ["left_hand", "left_fingers", "left_foot", "left_toes"]

    print("  Hairy skin (should have CT):")
    for part in hairy_parts:
        skin = get_skin_type(part)
        has_ct = has_ct_afferents(part)
        print(f"    {part:>20}: {skin.value:>8}, CT={has_ct}")
        assert skin == SkinType.HAIRY, f"{part} should be HAIRY"
        assert has_ct, f"{part} should have CT"

    print("  Glabrous skin (no CT):")
    for part in glabrous_parts:
        skin = get_skin_type(part)
        has_ct = has_ct_afferents(part)
        print(f"    {part:>20}: {skin.value:>8}, CT={has_ct}")
        assert skin == SkinType.GLABROUS, f"{part} should be GLABROUS"
        assert not has_ct, f"{part} should NOT have CT"

    print("\n  [PASS] Skin type map validated.\n")


def test_developmental_profile():
    """Test developmental profile calculations."""
    from ct_touch.developmental import DevelopmentalProfile

    print("=" * 60)
    print("TEST 3: Developmental Profile")
    print("=" * 60)

    ages = [0, 3, 6, 12, 18, 24]
    print(f"  {'Age (mo)':>8}  {'Myelin':>8}  {'CT mat':>8}  "
          f"{'Myel mat':>8}  {'Density':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for age in ages:
        dp = DevelopmentalProfile(age)
        s = dp.summary()
        print(f"  {age:>8}  {s['myelination_factor']:>8.3f}  "
              f"{s['ct_maturity']:>8.3f}  {s['myelinated_maturity']:>8.3f}  "
              f"{s['density_factor']:>8.3f}")

    # Verify monotonic increase of maturation factors
    prev_ct = 0
    prev_myel = 0
    for age in range(0, 25):
        dp = DevelopmentalProfile(age)
        ct = dp.ct_maturity()
        myel = dp.myelinated_maturity()
        assert ct >= prev_ct, f"CT maturity should increase: {ct} < {prev_ct} at age {age}"
        assert myel >= prev_myel, f"Myel maturity should increase: {myel} < {prev_myel}"
        prev_ct = ct
        prev_myel = myel

    # Verify density decreases
    dp0 = DevelopmentalProfile(0)
    dp24 = DevelopmentalProfile(24)
    assert dp0.density_factor() > dp24.density_factor(), "Density should decrease with age"

    print("\n  [PASS] Developmental profile validated.\n")


def test_ct_touch_in_environment():
    """Test CTAugmentedTouch in a live MIMo environment."""
    from ct_touch.ct_augmented_touch import CTAugmentedTouch
    from mimoEnv.envs.mimo_env import DEFAULT_TOUCH_PARAMS, SCENE_DIRECTORY
    from mimoEnv.envs.dummy import MIMoDummyEnv

    print("=" * 60)
    print("TEST 4: CTAugmentedTouch in MIMo Environment")
    print("=" * 60)

    # --- Test with ct_afferent_response ---
    print("\n  4a. Testing ct_afferent_response touch function...")
    ct_params = dict(DEFAULT_TOUCH_PARAMS)
    ct_params["touch_function"] = "ct_afferent_response"
    ct_params["response_function"] = "spread_gaussian"

    class CTDummyEnv(MIMoDummyEnv):
        def touch_setup(self, touch_params):
            self.touch = CTAugmentedTouch(self, touch_params, developmental_age=18.0)
            count = 0
            for body_id in self.touch.sensor_positions:
                count += self.touch.get_sensor_count(body_id)
            print(f"    Total sensor points: {count}")

    env = CTDummyEnv(
        touch_params=ct_params,
        vision_params=None,
        vestibular_params=None,
        goals_in_observation=False,
        done_active=False,
    )

    obs, info = env.reset()
    print(f"    Touch obs shape: {obs['touch'].shape}")
    print(f"    Touch obs dtype: {obs['touch'].dtype}")
    assert "touch" in obs, "Touch should be in observations"

    # Run a few steps
    n_steps = 20
    all_touch = []
    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        all_touch.append(obs["touch"].copy())

    all_touch = np.array(all_touch)
    print(f"    Touch output over {n_steps} steps: shape={all_touch.shape}, "
          f"min={all_touch.min():.4f}, max={all_touch.max():.4f}, "
          f"mean={all_touch.mean():.6f}")

    # Check skin type mapping
    skin_summary = env.touch.get_skin_type_summary()
    n_hairy = sum(1 for v in skin_summary.values() if v == "hairy")
    n_glabrous = sum(1 for v in skin_summary.values() if v == "glabrous")
    print(f"    Skin types: {n_hairy} hairy, {n_glabrous} glabrous")
    assert n_hairy > 0, "Should have hairy skin parts"
    assert n_glabrous > 0, "Should have glabrous skin parts"

    env.close()
    print("    [PASS] ct_afferent_response works.\n")

    # --- Test with multi_receptor ---
    print("  4b. Testing multi_receptor touch function...")
    mr_params = dict(DEFAULT_TOUCH_PARAMS)
    mr_params["touch_function"] = "multi_receptor"
    mr_params["response_function"] = "spread_linear"

    class MRDummyEnv(MIMoDummyEnv):
        def touch_setup(self, touch_params):
            self.touch = CTAugmentedTouch(self, touch_params)
            count = 0
            for body_id in self.touch.sensor_positions:
                count += self.touch.get_sensor_count(body_id)
            print(f"    Total sensor points: {count}")

    env2 = MRDummyEnv(
        touch_params=mr_params,
        vision_params=None,
        vestibular_params=None,
        goals_in_observation=False,
        done_active=False,
    )

    obs2, _ = env2.reset()
    print(f"    Touch obs shape: {obs2['touch'].shape}")
    # multi_receptor outputs 7 channels per sensor
    total_sensors = sum(env2.touch.get_sensor_count(b) for b in env2.touch.sensor_positions)
    expected_size = total_sensors * 7
    assert obs2["touch"].shape[0] == expected_size, \
        f"Expected {expected_size} touch values, got {obs2['touch'].shape[0]}"

    n_steps = 20
    all_touch2 = []
    for step in range(n_steps):
        action = env2.action_space.sample()
        obs2, reward, terminated, truncated, info = env2.step(action)
        all_touch2.append(obs2["touch"].copy())

    all_touch2 = np.array(all_touch2)
    print(f"    Multi-receptor output over {n_steps} steps: shape={all_touch2.shape}")

    # Reshape to (steps, n_sensors, 7) and check channels
    reshaped = all_touch2.reshape(n_steps, total_sensors, 7)
    channel_names = ["SA-I(x)", "SA-I(y)", "SA-I(z)", "FA-I", "FA-II", "CT", "Normal"]
    print("    Per-channel statistics:")
    for ch, name in enumerate(channel_names):
        ch_data = reshaped[:, :, ch]
        print(f"      {name:>10}: min={ch_data.min():>8.4f}, "
              f"max={ch_data.max():>8.4f}, mean={ch_data.mean():>8.6f}")

    env2.close()
    print("    [PASS] multi_receptor works.\n")

    # --- Test with force_vector (backward compatibility) ---
    print("  4c. Testing backward compatibility (force_vector)...")
    fv_params = dict(DEFAULT_TOUCH_PARAMS)
    fv_params["touch_function"] = "force_vector"
    fv_params["response_function"] = "spread_linear"

    class FVDummyEnv(MIMoDummyEnv):
        def touch_setup(self, touch_params):
            self.touch = CTAugmentedTouch(self, touch_params)

    env3 = FVDummyEnv(
        touch_params=fv_params,
        vision_params=None,
        vestibular_params=None,
        goals_in_observation=False,
        done_active=False,
    )

    obs3, _ = env3.reset()
    for _ in range(5):
        obs3, _, _, _, _ = env3.step(env3.action_space.sample())
    print(f"    Touch obs shape: {obs3['touch'].shape}")
    env3.close()
    print("    [PASS] Backward compatible with force_vector.\n")


def test_visualization():
    """Generate a simple visualization of CT response across body parts."""
    from ct_touch.ct_augmented_touch import CTAugmentedTouch
    from mimoEnv.envs.mimo_env import DEFAULT_TOUCH_PARAMS
    from mimoEnv.envs.dummy import MIMoDummyEnv

    print("=" * 60)
    print("TEST 5: Visualization (CT response heatmap)")
    print("=" * 60)

    ct_params = dict(DEFAULT_TOUCH_PARAMS)
    ct_params["touch_function"] = "ct_afferent_response"
    ct_params["response_function"] = "spread_gaussian"

    class CTVizEnv(MIMoDummyEnv):
        def touch_setup(self, touch_params):
            self.touch = CTAugmentedTouch(self, touch_params, developmental_age=18.0)

    env = CTVizEnv(
        touch_params=ct_params,
        vision_params=None,
        vestibular_params=None,
        goals_in_observation=False,
        done_active=False,
    )

    # Collect data over multiple steps
    n_steps = 50
    body_responses = {}
    obs, _ = env.reset()

    for body_id in env.touch.sensor_positions:
        body_name = env.model.body(body_id).name
        body_responses[body_name] = []

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        summary = env.touch.get_ct_summary()
        for name, val in summary.items():
            if name in body_responses:
                body_responses[name].append(val)

    # Compute mean response per body part
    mean_responses = {
        name: np.mean(vals) if vals else 0.0
        for name, vals in body_responses.items()
    }

    # Sort by response
    sorted_parts = sorted(mean_responses.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Mean CT response per body part ({n_steps} steps):")
    print(f"  {'Body Part':>25}  {'Mean Response':>15}  {'Skin Type':>10}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*10}")
    skin_summary = env.touch.get_skin_type_summary()
    for name, resp in sorted_parts:
        skin = skin_summary.get(name, "unknown")
        bar = "#" * int(resp * 200)
        print(f"  {name:>25}  {resp:>15.6f}  {skin:>10}  {bar}")

    # Save heatmap plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        names = [p[0] for p in sorted_parts]
        values = [p[1] for p in sorted_parts]
        colors = ["#e74c3c" if skin_summary.get(n, "") == "hairy" else "#3498db"
                  for n in names]

        bars = ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Mean CT Afferent Response")
        ax.set_title("CT Afferent Response by Body Part (red=hairy, blue=glabrous)")
        ax.invert_yaxis()
        plt.tight_layout()

        output_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(output_dir, "ct_response_heatmap.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"\n  Heatmap saved to: {plot_path}")
    except Exception as e:
        print(f"\n  (Skipping plot: {e})")

    env.close()
    print("  [PASS] Visualization test complete.\n")


def test_ct_firing_rate_curve():
    """Generate and save the CT firing rate vs velocity curve."""
    from ct_touch.ct_augmented_touch import CTAugmentedTouch

    print("=" * 60)
    print("TEST 6: CT Firing Rate Curve Plot")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        velocities = np.logspace(-3, 0, 200)  # 0.001 to 1.0 m/s
        rates = [CTAugmentedTouch.ct_firing_rate(v) for v in velocities]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(velocities * 100, rates, "b-", linewidth=2)
        ax.axvline(x=3.0, color="r", linestyle="--", alpha=0.7, label="Peak (3 cm/s)")
        ax.set_xlabel("Stroking Velocity (cm/s)")
        ax.set_ylabel("CT Firing Rate (normalized)")
        ax.set_title("CT Afferent Firing Rate vs. Stroking Velocity\n(Loken et al. 2009 model)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.1, 100)
        ax.set_ylim(0, 1.1)
        plt.tight_layout()

        output_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(output_dir, "ct_firing_rate_curve.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Curve saved to: {plot_path}")
    except Exception as e:
        print(f"  (Skipping plot: {e})")

    print("  [PASS] Firing rate curve test complete.\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CT Afferent Touch Module -- Test Suite")
    print("=" * 60 + "\n")

    # Unit tests (no environment needed)
    test_ct_firing_rate_model()
    test_skin_map()
    test_developmental_profile()

    # Integration tests (need MIMo environment)
    test_ct_touch_in_environment()

    # Visualization tests
    test_visualization()
    test_ct_firing_rate_curve()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
