"""Integration tests for SOM wrapper (mock-based, no MuJoCo required).

Tests the full pipeline: preprocessor -> SOM -> Hebbian -> representation,
using mock touch data that mimics MIMo's sensor_outputs structure.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from som.core import SelfOrganizingMap
from som.hebbian import CrossModalNetwork


class MockTouchModule:
    """Mocks CTAugmentedTouch for testing without MuJoCo."""

    def __init__(self, n_bodies=12, sensors_per_body=50, touch_size=7):
        self.touch_size = touch_size
        self.meshes = {i: None for i in range(n_bodies)}
        self.sensor_positions = {
            i: np.random.randn(sensors_per_body, 3)
            for i in range(n_bodies)
        }
        self.sensor_outputs = {
            i: np.zeros((sensors_per_body, touch_size), dtype=np.float32)
            for i in range(n_bodies)
        }

    def simulate_contact(self, body_ids, rng):
        """Simulate random touch contacts on specified bodies."""
        for bid in self.meshes:
            self.sensor_outputs[bid][:] = 0.0

        for bid in body_ids:
            n_sensors = self.sensor_outputs[bid].shape[0]
            # Random number of active sensors
            n_active = rng.integers(1, max(2, n_sensors // 5))
            active = rng.choice(n_sensors, n_active, replace=False)

            for s in active:
                # SA-I: log-compressed force
                self.sensor_outputs[bid][s, 0:3] = rng.exponential(0.5, 3)
                # FA-I: velocity
                self.sensor_outputs[bid][s, 3] = rng.exponential(0.3)
                # FA-II: vibration
                self.sensor_outputs[bid][s, 4] = rng.exponential(0.1)
                # CT: velocity-tuned (sparse, only on hairy skin)
                if bid < 8:  # Mock: first 8 bodies are "hairy"
                    self.sensor_outputs[bid][s, 5] = rng.exponential(0.05)
                # Normal force
                self.sensor_outputs[bid][s, 6] = rng.exponential(1.0)


class MockEnv:
    """Minimal mock of MIMo environment."""

    class MockModel:
        class BodyProxy:
            def __init__(self, name):
                self._name = name
            @property
            def name(self):
                return self._name

        BODY_NAMES = [
            "left_upper_arm", "right_upper_arm",
            "left_lower_arm", "right_lower_arm",
            "upper_body", "chest", "hip", "head",
            "left_hand", "right_hand",
            "left_fingers", "right_fingers",
        ]

        def body(self, bid):
            return self.BodyProxy(self.BODY_NAMES[bid])

    model = MockModel()


def test_preprocessor_with_mock():
    """Test TouchPreprocessor with mock data."""
    from som.preprocessor import TouchPreprocessor

    touch = MockTouchModule(n_bodies=12, sensors_per_body=50, touch_size=7)
    env = MockEnv()
    prep = TouchPreprocessor(touch, env)

    # Check dimensions
    assert prep.n_bodies == 12
    assert prep.disc_dim == 12 * 6  # 72
    assert prep.aff_dim == 12 * 1   # 12
    assert prep.is_multi_receptor is True

    # Process empty touch (no contacts)
    result = prep.process(normalize=False)
    assert result["discriminative"].shape == (72,)
    assert result["affective"].shape == (12,)
    assert np.all(result["discriminative"] == 0)
    assert np.all(result["affective"] == 0)

    # Process with simulated contacts
    rng = np.random.default_rng(42)
    touch.simulate_contact([0, 1, 4, 5], rng)
    result = prep.process(normalize=False)

    # Active bodies should have non-zero discriminative features
    disc_2d = result["disc_2d"]
    assert disc_2d[0].sum() > 0, "Body 0 should be active"
    assert disc_2d[4].sum() > 0, "Body 4 should be active"
    assert disc_2d[10].sum() == 0, "Body 10 should be inactive"

    # CT should be non-zero only for "hairy" bodies (0-7)
    aff_2d = result["aff_2d"]
    assert aff_2d[0, 0] > 0, "Hairy body should have CT"
    # Body 8-11 are glabrous in mock, but CT is based on actual sensor data
    # In mock, bodies 8+ get CT=0 in simulate_contact

    print(f"[PASS] test_preprocessor_with_mock (disc_dim={prep.disc_dim}, aff_dim={prep.aff_dim})")


def test_full_pipeline_mock():
    """Test full SOM pipeline: preprocess -> SOM -> Hebbian -> representation."""
    from som.preprocessor import TouchPreprocessor

    rng = np.random.default_rng(42)

    # Setup
    touch = MockTouchModule(n_bodies=12, sensors_per_body=50, touch_size=7)
    env = MockEnv()
    prep = TouchPreprocessor(touch, env)

    # Create CrossModalNetwork
    network = CrossModalNetwork(
        som_configs={
            "tactile_disc": {
                "grid_size": (8, 8),
                "input_dim": prep.disc_dim,  # 72
                "decay_steps": 5000,
                "rng": np.random.default_rng(42),
            },
            "tactile_aff": {
                "grid_size": (6, 6),
                "input_dim": prep.aff_dim,   # 12
                "decay_steps": 5000,
                "rng": np.random.default_rng(43),
            },
            "proprio": {
                "grid_size": (8, 8),
                "input_dim": 20,  # Mock proprio dim
                "decay_steps": 5000,
                "rng": np.random.default_rng(44),
            },
        },
        hebbian_eta=0.01,
        hebbian_decay=0.001,
    )

    # Simulate 1000 steps
    for step in range(1000):
        # Random contacts on 2-4 bodies
        n_contact = rng.integers(2, 5)
        contact_bodies = rng.choice(12, n_contact, replace=False).tolist()
        touch.simulate_contact(contact_bodies, rng)

        # Preprocess
        features = prep.process(normalize=True)

        # Mock proprioception
        proprio = rng.normal(0, 1, 20)

        # SOM + Hebbian learning
        inputs = {
            "tactile_disc": features["discriminative"],
            "tactile_aff": features["affective"],
            "proprio": proprio,
        }
        network.learn(inputs)

    # Get representation
    touch.simulate_contact([0, 1, 2], rng)
    features = prep.process(normalize=True)
    inputs = {
        "tactile_disc": features["discriminative"],
        "tactile_aff": features["affective"],
        "proprio": rng.normal(0, 1, 20),
    }
    repr_vec = network.get_representation(inputs)

    expected_dim = 64 + 36 + 64  # 8x8 + 6x6 + 8x8
    assert repr_vec.shape == (expected_dim,), f"Shape {repr_vec.shape} != ({expected_dim},)"

    # Check metrics
    metrics = network.get_metrics()
    assert metrics["hebbian_tactile_disc_tactile_aff_binding"] > 0
    assert metrics["som_tactile_disc_step"] == 1000

    print(f"[PASS] test_full_pipeline_mock")
    print(f"  repr_dim: {expected_dim}")
    print(f"  disc SOM steps: {metrics['som_tactile_disc_step']}")
    print(f"  binding (disc-aff): {metrics['hebbian_tactile_disc_tactile_aff_binding']:.4f}")
    print(f"  binding (disc-proprio): {metrics['hebbian_tactile_disc_proprio_binding']:.4f}")
    print(f"  binding (aff-proprio): {metrics['hebbian_tactile_aff_proprio_binding']:.4f}")


def test_som_quality_after_training():
    """Test that SOM quality improves after training with structured data."""
    from som.preprocessor import TouchPreprocessor

    rng = np.random.default_rng(42)

    touch = MockTouchModule(n_bodies=12, sensors_per_body=50, touch_size=7)
    env = MockEnv()
    prep = TouchPreprocessor(touch, env)

    som = SelfOrganizingMap(
        grid_size=(10, 10),
        input_dim=prep.disc_dim,
        decay_steps=5000,
        rng=np.random.default_rng(42),
    )

    # Collect data
    data = []
    for _ in range(500):
        n_contact = rng.integers(1, 4)
        contact_bodies = rng.choice(12, n_contact, replace=False).tolist()
        touch.simulate_contact(contact_bodies, rng)
        features = prep.process(normalize=False)
        data.append(features["discriminative"].copy())
    data = np.array(data)

    # Measure before training
    qe_before = som.quantization_error(data[:100])

    # Train
    for _ in range(5000):
        x = data[rng.integers(len(data))]
        som.update(x)

    qe_after = som.quantization_error(data[:100])
    te = som.topographic_error(data[:100])

    print(f"[PASS] test_som_quality_after_training")
    print(f"  QE: {qe_before:.4f} -> {qe_after:.4f} (ratio: {qe_after/qe_before:.3f})")
    print(f"  TE: {te:.4f}")
    assert qe_after < qe_before, "SOM should improve"


def test_force_vector_mode():
    """Test preprocessor with 3-channel force_vector (CT OFF)."""
    from som.preprocessor import TouchPreprocessor

    touch = MockTouchModule(n_bodies=12, sensors_per_body=50, touch_size=3)
    env = MockEnv()
    prep = TouchPreprocessor(touch, env)

    assert prep.is_multi_receptor is False
    assert prep.disc_dim == 72  # Still 12 * 6, but FA-I/FA-II/normal channels zero
    assert prep.aff_dim == 12   # Still 12 * 1, but all zero

    rng = np.random.default_rng(42)

    # Simulate 3-channel contact
    for bid in [0, 1, 2]:
        n_sensors = touch.sensor_outputs[bid].shape[0]
        for s in range(5):
            touch.sensor_outputs[bid][s, 0:3] = rng.normal(0, 1, 3)

    result = prep.process(normalize=False)
    # SA-I channels should have data, others zero
    disc_2d = result["disc_2d"]
    assert disc_2d[0, 0:3].sum() != 0, "SA-I should have data"
    assert disc_2d[0, 3:6].sum() == 0, "FA-I/FA-II/Normal should be zero in force_vector"
    assert result["aff_2d"].sum() == 0, "CT should be zero in force_vector mode"

    print("[PASS] test_force_vector_mode")


def test_representation_stability():
    """Test that SOM representation is deterministic for same input."""
    from som.preprocessor import TouchPreprocessor

    rng = np.random.default_rng(42)
    touch = MockTouchModule(n_bodies=12, sensors_per_body=50, touch_size=7)
    env = MockEnv()
    prep = TouchPreprocessor(touch, env)

    network = CrossModalNetwork(
        som_configs={
            "disc": {"grid_size": (5, 5), "input_dim": prep.disc_dim,
                     "rng": np.random.default_rng(42)},
        },
        link_pairs=[],
    )

    # Same input -> same output
    touch.simulate_contact([0, 1], rng)
    feat = prep.process(normalize=False)
    r1 = network.get_representation({"disc": feat["discriminative"]})
    r2 = network.get_representation({"disc": feat["discriminative"]})
    assert np.allclose(r1, r2), "Same input should give same representation"

    print("[PASS] test_representation_stability")


if __name__ == "__main__":
    test_preprocessor_with_mock()
    test_full_pipeline_mock()
    test_som_quality_after_training()
    test_force_vector_mode()
    test_representation_stability()

    print("\n=== All integration tests passed ===")
