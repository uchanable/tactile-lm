"""Unit tests for Hebbian cross-modal binding."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from som.core import SelfOrganizingMap
from som.hebbian import HebbianLink, CrossModalNetwork


def test_hebbian_link_basic():
    """Test HebbianLink creation and shapes."""
    link = HebbianLink(100, 64, "disc", "aff")
    assert link.W.shape == (100, 64)
    assert link.binding_strength() == 0.0
    print("[PASS] test_hebbian_link_basic")


def test_hebbian_learning():
    """Test that co-activation strengthens connections."""
    link = HebbianLink(25, 25, "a", "b", eta=0.1, decay=0.0)

    # Create localized activations
    act_a = np.zeros(25)
    act_a[10:13] = 1.0
    act_b = np.zeros(25)
    act_b[15:18] = 1.0

    for _ in range(50):
        link.update(act_a, act_b)

    # Connection between active regions should be strong
    assert link.W[10:13, 15:18].mean() > 0.5, "Active regions not connected"
    # Inactive regions should be near zero
    assert link.W[0:5, 0:5].mean() < 0.01, "Inactive regions too strong"
    print("[PASS] test_hebbian_learning")


def test_hebbian_decay():
    """Test weight decay prevents saturation."""
    link = HebbianLink(10, 10, "a", "b", eta=0.5, decay=0.1)

    act = np.ones(10) * 0.5
    for _ in range(1000):
        link.update(act, act)

    # With decay, weights should plateau below 1.0
    assert link.W.max() <= 1.0, "Weights exceeded clip bound"
    print(f"[PASS] test_hebbian_decay (max W: {link.W.max():.3f})")


def test_cross_modal_prediction():
    """Test that Hebbian links enable cross-modal prediction."""
    link = HebbianLink(25, 25, "a", "b", eta=0.05, decay=0.001)

    rng = np.random.default_rng(42)

    # Train: paired activations (neuron i in A <-> neuron i in B)
    for _ in range(500):
        idx = rng.integers(25)
        act_a = np.zeros(25)
        act_b = np.zeros(25)
        act_a[idx] = 1.0
        act_b[idx] = 1.0
        # Add some neighborhood
        if idx > 0:
            act_a[idx-1] = 0.5
            act_b[idx-1] = 0.5
        if idx < 24:
            act_a[idx+1] = 0.5
            act_b[idx+1] = 0.5
        link.update(act_a, act_b)

    # Test: predict B from A
    correct = 0
    for idx in range(25):
        act_a = np.zeros(25)
        act_a[idx] = 1.0
        pred = link.predict_b(act_a)
        if np.argmax(pred) == idx:
            correct += 1

    accuracy = correct / 25
    assert accuracy > 0.5, f"Prediction accuracy too low: {accuracy}"
    print(f"[PASS] test_cross_modal_prediction (accuracy: {accuracy:.2f})")


def test_cross_modal_network():
    """Test full CrossModalNetwork with 3 SOMs."""
    rng = np.random.default_rng(42)

    network = CrossModalNetwork(
        som_configs={
            "tactile_disc": {"grid_size": (8, 8), "input_dim": 12, "rng": rng},
            "tactile_aff": {"grid_size": (6, 6), "input_dim": 4, "rng": rng},
            "proprio": {"grid_size": (8, 8), "input_dim": 20, "rng": rng},
        },
        hebbian_eta=0.01,
        hebbian_decay=0.001,
    )

    assert len(network.soms) == 3
    assert len(network.links) == 3  # fully connected: 3 pairs
    print("[PASS] test_cross_modal_network (3 SOMs, 3 links)")


def test_network_learn_and_represent():
    """Test learning and representation generation."""
    rng = np.random.default_rng(42)

    network = CrossModalNetwork(
        som_configs={
            "disc": {"grid_size": (5, 5), "input_dim": 6, "rng": rng},
            "aff": {"grid_size": (5, 5), "input_dim": 2, "rng": rng},
        },
        link_pairs=[("disc", "aff")],
    )

    # Train
    for _ in range(500):
        inputs = {
            "disc": rng.uniform(0, 1, 6),
            "aff": rng.uniform(0, 1, 2),
        }
        network.learn(inputs)

    # Get representation
    test_input = {"disc": rng.uniform(0, 1, 6), "aff": rng.uniform(0, 1, 2)}
    repr_vec = network.get_representation(test_input)

    expected_dim = 25 + 25  # Two 5x5 SOMs
    assert repr_vec.shape == (expected_dim,), f"Repr shape {repr_vec.shape} != ({expected_dim},)"
    assert repr_vec.max() <= 1.0 and repr_vec.min() >= 0.0

    # Metrics
    metrics = network.get_metrics()
    assert "hebbian_disc_aff_binding" in metrics
    assert metrics["hebbian_disc_aff_binding"] > 0
    print(f"[PASS] test_network_learn_and_represent (binding: {metrics['hebbian_disc_aff_binding']:.4f})")


def test_specificity():
    """Test binding specificity metric."""
    link = HebbianLink(10, 10, "a", "b")

    # Diffuse: all connections equal
    link.W = np.ones((10, 10)) * 0.5
    spec_diffuse = link.specificity()

    # Specific: one strong connection
    link.W = np.zeros((10, 10))
    link.W[3, 7] = 1.0
    spec_specific = link.specificity()

    assert spec_specific > spec_diffuse, (
        f"Specific ({spec_specific:.3f}) should be > diffuse ({spec_diffuse:.3f})"
    )
    print(f"[PASS] test_specificity (specific={spec_specific:.3f}, diffuse={spec_diffuse:.3f})")


def test_network_serialization():
    """Test CrossModalNetwork state save/restore."""
    rng = np.random.default_rng(42)

    network = CrossModalNetwork(
        som_configs={
            "a": {"grid_size": (5, 5), "input_dim": 4, "rng": rng},
            "b": {"grid_size": (5, 5), "input_dim": 4, "rng": rng},
        },
        link_pairs=[("a", "b")],
    )

    # Train a bit
    for _ in range(100):
        network.learn({"a": rng.uniform(0, 1, 4), "b": rng.uniform(0, 1, 4)})

    state = network.get_state()

    # Create new network and restore
    network2 = CrossModalNetwork(
        som_configs={
            "a": {"grid_size": (5, 5), "input_dim": 4},
            "b": {"grid_size": (5, 5), "input_dim": 4},
        },
        link_pairs=[("a", "b")],
    )
    network2.set_state(state)

    # Check weights match
    assert np.allclose(
        network.soms["a"].weights, network2.soms["a"].weights
    ), "SOM weights not restored"
    assert np.allclose(
        network.links[("a", "b")].W, network2.links[("a", "b")].W
    ), "Hebbian weights not restored"
    print("[PASS] test_network_serialization")


if __name__ == "__main__":
    test_hebbian_link_basic()
    test_hebbian_learning()
    test_hebbian_decay()
    test_cross_modal_prediction()
    test_cross_modal_network()
    test_network_learn_and_represent()
    test_specificity()
    test_network_serialization()

    print("\n=== All Hebbian tests passed ===")
