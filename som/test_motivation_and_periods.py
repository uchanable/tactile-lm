"""Tests for intrinsic motivation and critical period modules."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from som.core import SelfOrganizingMap
from som.intrinsic_motivation import IntrinsicMotivation
from som.critical_periods import CriticalPeriodScheduler


# === IntrinsicMotivation tests ===

def test_novelty_decreases_with_learning():
    """Novelty should decrease as SOM learns the data."""
    rng = np.random.default_rng(42)
    # Use wider weight range so initial SOM is far from data
    som = SelfOrganizingMap(grid_size=(10, 10), input_dim=8, decay_steps=5000, rng=rng)
    som.weights *= 5.0  # Push weights away from data
    im = IntrinsicMotivation(alpha=1.0, beta=0.0)

    # Clustered data for clear structure
    centers = [rng.uniform(0, 1, 8) for _ in range(4)]
    data = np.vstack([rng.normal(c, 0.1, (100, 8)) for c in centers])

    # Novelty before training
    early_novelties = []
    for x in data[:30]:
        r = im.compute_reward(som, x, ct_activation=0.0)
        early_novelties.append(r["novelty_raw"])

    # Train SOM
    for _ in range(5000):
        som.update(data[rng.integers(len(data))])

    # Novelty after training
    im2 = IntrinsicMotivation(alpha=1.0, beta=0.0)
    late_novelties = []
    for x in data[:30]:
        r = im2.compute_reward(som, x, ct_activation=0.0)
        late_novelties.append(r["novelty_raw"])

    assert np.mean(late_novelties) < np.mean(early_novelties), (
        f"Novelty should decrease: {np.mean(early_novelties):.3f} -> {np.mean(late_novelties):.3f}"
    )
    print(f"[PASS] test_novelty_decreases (before: {np.mean(early_novelties):.3f}, after: {np.mean(late_novelties):.3f})")


def test_ct_increases_reward():
    """Higher CT activation should increase reward (with beta > 0)."""
    rng = np.random.default_rng(42)
    som = SelfOrganizingMap(grid_size=(5, 5), input_dim=4, rng=rng)
    im = IntrinsicMotivation(alpha=0.0, beta=1.0, ct_window=100)

    x = rng.uniform(0, 1, 4)

    # Warm up running stats
    for _ in range(50):
        im.compute_reward(som, x, ct_activation=0.1)

    r_low = im.compute_reward(som, x, ct_activation=0.01)
    r_high = im.compute_reward(som, x, ct_activation=0.5)

    assert r_high["reward"] > r_low["reward"], (
        f"Higher CT should give higher reward: {r_low['reward']:.3f} vs {r_high['reward']:.3f}"
    )
    print(f"[PASS] test_ct_increases_reward (low: {r_low['reward']:.3f}, high: {r_high['reward']:.3f})")


def test_combined_reward():
    """Combined reward should respond to both novelty and CT."""
    rng = np.random.default_rng(42)
    som = SelfOrganizingMap(grid_size=(5, 5), input_dim=4, rng=rng)
    im = IntrinsicMotivation(alpha=1.0, beta=1.0, novelty_window=50, ct_window=50)

    # Warm up
    for _ in range(50):
        x = rng.uniform(0, 1, 4)
        im.compute_reward(som, x, ct_activation=rng.uniform(0, 0.1))

    result = im.compute_reward(som, rng.uniform(0, 1, 4), ct_activation=0.05)

    assert "reward" in result
    assert "novelty_raw" in result
    assert "novelty_normalized" in result
    assert "ct_raw" in result
    assert "ct_normalized" in result
    print(f"[PASS] test_combined_reward (reward={result['reward']:.3f})")


def test_serialization_im():
    """Test IntrinsicMotivation state save/restore."""
    rng = np.random.default_rng(42)
    som = SelfOrganizingMap(grid_size=(5, 5), input_dim=4, rng=rng)
    im = IntrinsicMotivation()

    for _ in range(20):
        im.compute_reward(som, rng.uniform(0, 1, 4), ct_activation=rng.uniform(0, 0.1))

    state = im.get_state()
    im2 = IntrinsicMotivation()
    im2.set_state(state)

    assert np.allclose(im._novelty_history, im2._novelty_history)
    assert im._novelty_count == im2._novelty_count
    print("[PASS] test_serialization_im")


# === CriticalPeriodScheduler tests ===

def test_critical_period_profile():
    """Test that multipliers follow the expected profile."""
    scheduler = CriticalPeriodScheduler(steps_per_month=10_000)

    # Tactile_disc: onset=0, peak=3, offset=12
    # Before onset (impossible since onset=0)
    # At peak (3 months = 30K steps)
    mult_peak = scheduler.get_multiplier("tactile_disc", 30_000)
    # After offset (12 months = 120K steps)
    mult_after = scheduler.get_multiplier("tactile_disc", 150_000)

    assert mult_peak > mult_after, (
        f"Peak ({mult_peak:.2f}) should be > after offset ({mult_after:.2f})"
    )
    assert mult_peak == 1.5, f"Peak multiplier should be 1.5, got {mult_peak}"
    assert mult_after == 0.1, f"After-offset multiplier should be 0.1, got {mult_after}"
    print(f"[PASS] test_critical_period_profile (peak={mult_peak}, after={mult_after})")


def test_modality_ordering():
    """Earlier-developing modalities should peak before later ones."""
    scheduler = CriticalPeriodScheduler(steps_per_month=10_000)

    # At 2 months: tactile_aff should be near peak, visual should be before onset
    step_2mo = 20_000
    tac_aff = scheduler.get_multiplier("tactile_aff", step_2mo)
    visual = scheduler.get_multiplier("visual", step_2mo)

    assert tac_aff > visual, (
        f"Tactile aff ({tac_aff:.2f}) should be > visual ({visual:.2f}) at 2 months"
    )
    print(f"[PASS] test_modality_ordering (tac_aff={tac_aff:.2f}, visual={visual:.2f} at 2mo)")


def test_is_in_critical_period():
    """Test critical period detection."""
    scheduler = CriticalPeriodScheduler(steps_per_month=10_000)

    # Tactile_disc: onset=0, offset=12
    assert scheduler.is_in_critical_period("tactile_disc", 0)
    assert scheduler.is_in_critical_period("tactile_disc", 60_000)  # 6mo
    assert not scheduler.is_in_critical_period("tactile_disc", 150_000)  # 15mo

    # Visual: onset=4, offset=18
    assert not scheduler.is_in_critical_period("visual", 20_000)  # 2mo
    assert scheduler.is_in_critical_period("visual", 100_000)  # 10mo

    print("[PASS] test_is_in_critical_period")


def test_apply_to_som():
    """Test that scheduler modulates SOM parameters."""
    rng = np.random.default_rng(42)
    som = SelfOrganizingMap(grid_size=(5, 5), input_dim=4, rng=rng)
    scheduler = CriticalPeriodScheduler(steps_per_month=10_000)

    original_lr = som.initial_lr

    # At peak (3 months for tactile_disc)
    scheduler.apply_to_som(som, "tactile_disc", 30_000)
    assert som.initial_lr == original_lr * 1.5, (
        f"LR at peak should be {original_lr * 1.5}, got {som.initial_lr}"
    )

    # After offset
    scheduler.apply_to_som(som, "tactile_disc", 150_000)
    assert som.initial_lr == original_lr * 0.1, (
        f"LR after offset should be {original_lr * 0.1}, got {som.initial_lr}"
    )

    print("[PASS] test_apply_to_som")


def test_developmental_profile():
    """Test profile logging."""
    scheduler = CriticalPeriodScheduler(steps_per_month=10_000)
    profile = scheduler.get_developmental_profile(60_000)  # 6 months

    assert profile["age_months"] == 6.0
    assert "tactile_disc_lr_mult" in profile
    assert "visual_lr_mult" in profile
    assert len(profile) == 1 + len(scheduler.periods)  # age + modalities

    print(f"[PASS] test_developmental_profile (age={profile['age_months']}mo)")
    for k, v in profile.items():
        if k != "age_months":
            print(f"  {k}: {v:.3f}")


def test_unknown_modality():
    """Unknown modality should return 1.0 multiplier."""
    scheduler = CriticalPeriodScheduler()
    assert scheduler.get_multiplier("nonexistent", 0) == 1.0
    assert not scheduler.is_in_critical_period("nonexistent", 0)
    print("[PASS] test_unknown_modality")


if __name__ == "__main__":
    # IntrinsicMotivation
    test_novelty_decreases_with_learning()
    test_ct_increases_reward()
    test_combined_reward()
    test_serialization_im()

    print()

    # CriticalPeriodScheduler
    test_critical_period_profile()
    test_modality_ordering()
    test_is_in_critical_period()
    test_apply_to_som()
    test_developmental_profile()
    test_unknown_modality()

    print("\n=== All motivation & critical period tests passed ===")
