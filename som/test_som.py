"""Unit tests for SOM core module."""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from som.core import SelfOrganizingMap


def test_basic_creation():
    """Test SOM creation and shapes."""
    som = SelfOrganizingMap(grid_size=(10, 10), input_dim=4)
    assert som.weights.shape == (100, 4)
    assert som.n_neurons == 100
    assert som.grid_h == 10
    assert som.grid_w == 10
    print("[PASS] test_basic_creation")


def test_bmu_finding():
    """Test that BMU is the nearest neuron."""
    som = SelfOrganizingMap(grid_size=(5, 5), input_dim=3, rng=np.random.default_rng(42))

    x = np.array([0.5, 0.5, 0.5])
    bmu = som.find_bmu(x)

    # BMU should have smallest distance
    distances = np.sum((som.weights - x) ** 2, axis=1)
    expected = np.argmin(distances)
    assert bmu == expected, f"BMU {bmu} != expected {expected}"
    print("[PASS] test_bmu_finding")


def test_bmu_position():
    """Test 2D position mapping."""
    som = SelfOrganizingMap(grid_size=(5, 8), input_dim=2)
    som.weights[23] = np.array([0.0, 0.0])  # Neuron at (2, 7)

    r, c = som.get_bmu_position(np.array([0.0, 0.0]))
    assert r == 23 // 8 and c == 23 % 8, f"Position ({r},{c}) wrong for index 23"
    print("[PASS] test_bmu_position")


def test_learning_convergence():
    """Test that SOM converges on clustered data."""
    rng = np.random.default_rng(42)
    som = SelfOrganizingMap(
        grid_size=(10, 10), input_dim=2,
        initial_lr=0.5, final_lr=0.01,
        initial_sigma=5.0, final_sigma=0.5,
        decay_steps=5000, rng=rng,
    )

    # Generate 4 clusters
    centers = np.array([[0.2, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.8]])
    data = []
    for c in centers:
        data.append(rng.normal(c, 0.05, (200, 2)))
    data = np.vstack(data)

    # Train
    qe_before = som.quantization_error(data[:100])
    for _ in range(5000):
        x = data[rng.integers(len(data))]
        som.update(x)
    qe_after = som.quantization_error(data[:100])

    assert qe_after < qe_before, f"QE did not improve: {qe_before:.4f} -> {qe_after:.4f}"
    print(f"[PASS] test_learning_convergence (QE: {qe_before:.4f} -> {qe_after:.4f})")


def test_topographic_error():
    """Test topographic error calculation."""
    rng = np.random.default_rng(42)
    som = SelfOrganizingMap(
        grid_size=(10, 10), input_dim=2,
        initial_lr=0.5, final_lr=0.01,
        decay_steps=3000, rng=rng,
    )
    som.init_from_data(rng.uniform(0, 1, (200, 2)))

    data = rng.uniform(0, 1, (100, 2))
    for _ in range(3000):
        x = data[rng.integers(len(data))]
        som.update(x)

    te = som.topographic_error(data)
    assert 0.0 <= te <= 1.0, f"TE out of range: {te}"
    print(f"[PASS] test_topographic_error (TE: {te:.4f})")


def test_activation_map():
    """Test activation map shape and range."""
    som = SelfOrganizingMap(grid_size=(5, 5), input_dim=3)
    x = np.array([0.5, 0.5, 0.5])
    act = som.get_activation_map(x)

    assert act.shape == (25,), f"Activation shape {act.shape} != (25,)"
    assert act.max() <= 1.0 and act.min() >= 0.0, "Activations out of [0,1]"
    # BMU should have highest activation
    bmu = som.find_bmu(x)
    assert np.argmax(act) == bmu, "BMU doesn't have max activation"
    print("[PASS] test_activation_map")


def test_u_matrix():
    """Test U-matrix shape."""
    som = SelfOrganizingMap(grid_size=(8, 8), input_dim=4)
    u = som.u_matrix()
    assert u.shape == (8, 8), f"U-matrix shape {u.shape} != (8, 8)"
    assert np.all(u >= 0), "U-matrix has negative values"
    print("[PASS] test_u_matrix")


def test_pca_init():
    """Test PCA initialization."""
    rng = np.random.default_rng(42)
    som = SelfOrganizingMap(grid_size=(10, 10), input_dim=3, rng=rng)
    data = rng.normal(0, 1, (500, 3))

    som.init_from_data(data)

    # After PCA init, QE should be reasonable
    qe = som.quantization_error(data[:100])
    assert qe < 10.0, f"QE after PCA init too high: {qe}"
    print(f"[PASS] test_pca_init (QE after init: {qe:.4f})")


def test_serialization():
    """Test state save/restore."""
    rng = np.random.default_rng(42)
    som = SelfOrganizingMap(grid_size=(5, 5), input_dim=3, rng=rng)

    # Train a few steps
    for _ in range(10):
        som.update(rng.uniform(0, 1, 3))

    state = som.get_state()

    # Create new SOM and restore
    som2 = SelfOrganizingMap(grid_size=(5, 5), input_dim=3)
    som2.set_state(state)

    assert np.allclose(som.weights, som2.weights), "Weights not restored"
    assert som._step == som2._step, "Step count not restored"
    print("[PASS] test_serialization")


def test_high_dim_convergence():
    """Test SOM with higher dimensional input (closer to real use case)."""
    rng = np.random.default_rng(42)
    input_dim = 96  # Realistic: 16 bodies x 6 discriminative channels

    som = SelfOrganizingMap(
        grid_size=(15, 15), input_dim=input_dim,
        initial_lr=0.5, final_lr=0.01,
        initial_sigma=7.0, final_sigma=0.5,
        decay_steps=10000, rng=rng,
    )

    # Generate structured data (sparse, like real touch)
    data = np.zeros((1000, input_dim))
    for i in range(1000):
        # Activate 2-4 random "body parts" (each 6 channels)
        n_active = rng.integers(2, 5)
        active_bodies = rng.choice(input_dim // 6, n_active, replace=False)
        for b in active_bodies:
            data[i, b*6:(b+1)*6] = rng.exponential(0.5, 6)

    qe_before = som.quantization_error(data[:200])
    for step in range(10000):
        x = data[rng.integers(len(data))]
        som.update(x)
    qe_after = som.quantization_error(data[:200])

    ratio = qe_after / qe_before
    assert ratio < 0.8, f"QE ratio {ratio:.3f} not enough improvement"
    print(f"[PASS] test_high_dim_convergence (QE ratio: {ratio:.3f})")


if __name__ == "__main__":
    test_basic_creation()
    test_bmu_finding()
    test_bmu_position()
    test_learning_convergence()
    test_topographic_error()
    test_activation_map()
    test_u_matrix()
    test_pca_init()
    test_serialization()
    test_high_dim_convergence()

    print("\n=== All SOM core tests passed ===")
