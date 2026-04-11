"""Real MIMo integration test for SOM wrapper.

Requires MuJoCo and MIMo dependencies (.venv312).
Run: PYTHONPATH=MIMo:. .venv312/bin/python som/test_mimo_integration.py
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "MIMo"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimoEnv.envs.dummy import MIMoDummyEnv
from mimoEnv.envs.selfbody import TOUCH_PARAMS
from ct_touch.ct_augmented_touch import CTAugmentedTouch
from som.preprocessor import TouchPreprocessor
from som.hebbian import CrossModalNetwork
from som.som_wrapper import SOMObservationWrapper


class CTDummyEnv(MIMoDummyEnv):
    """DummyEnv with CT-Touch multi_receptor."""
    def __init__(self, **kwargs):
        super().__init__(
            touch_params=TOUCH_PARAMS,
            vision_params=None,
            vestibular_params=None,
            **kwargs,
        )

    def touch_setup(self, touch_params):
        ct_params = dict(touch_params)
        ct_params['touch_function'] = 'multi_receptor'
        ct_params['response_function'] = 'spread_linear'
        self.touch = CTAugmentedTouch(self, ct_params)
        count = sum(self.touch.get_sensor_count(bid) for bid in self.touch.meshes)
        print(f"CT-Touch: {count} sensors, {self.touch.touch_size} channels")


def test_ct_dummy_env():
    """Test CTDummyEnv observation structure."""
    env = CTDummyEnv()
    obs, _ = env.reset()

    print(f"Proprio: {obs['observation'].shape}")
    print(f"Touch: {obs['touch'].shape}")
    assert env.touch.touch_size == 7, f"Expected 7ch, got {env.touch.touch_size}"

    # Verify sensor_outputs structure
    for bid in sorted(env.touch.meshes):
        name = env.model.body(bid).name
        so = env.touch.sensor_outputs[bid]
        print(f"  {name}: sensor_outputs shape {so.shape}")

    env.close()
    print("[PASS] test_ct_dummy_env")


def test_preprocessor_real():
    """Test TouchPreprocessor with real MIMo data."""
    env = CTDummyEnv()
    obs, _ = env.reset()

    prep = TouchPreprocessor(env.touch, env)
    print(f"Bodies: {prep.n_bodies}")
    print(f"Body names: {prep.body_names}")
    print(f"Skin types: {[st.value for st in prep.skin_types]}")
    print(f"Disc dim: {prep.disc_dim} (= {prep.n_bodies} x 6)")
    print(f"Aff dim: {prep.aff_dim} (= {prep.n_bodies} x 1)")

    # Run a few steps to generate contacts
    for _ in range(50):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)

    result = prep.process(normalize=False)
    print(f"Disc features (non-zero): {np.count_nonzero(result['discriminative'])}/{prep.disc_dim}")
    print(f"Aff features (non-zero): {np.count_nonzero(result['affective'])}/{prep.aff_dim}")

    # CT by region
    ct_map = prep.get_ct_by_region()
    for name, val in ct_map.items():
        if val > 0:
            print(f"  CT active: {name} = {val:.4f}")

    env.close()
    print("[PASS] test_preprocessor_real")


def test_som_wrapper():
    """Test SOMObservationWrapper with real MIMo environment."""
    base_env = CTDummyEnv()
    wrapped = SOMObservationWrapper(base_env, som_config={
        "disc_grid": (10, 10),
        "aff_grid": (6, 6),
        "proprio_grid": (8, 8),
        "decay_steps": 10000,
    }, seed=42)

    obs, _ = wrapped.reset()
    print(f"Wrapped obs keys: {list(obs.keys())}")
    print(f"  observation: {obs['observation'].shape}")
    print(f"  som_repr: {obs['som_repr'].shape}")

    # Run 100 steps
    t0 = time.time()
    for step in range(100):
        action = wrapped.action_space.sample()
        obs, reward, term, trunc, info = wrapped.step(action)
        if term or trunc:
            obs, _ = wrapped.reset()
    dt = time.time() - t0

    print(f"100 steps in {dt:.2f}s ({dt/100*1000:.1f}ms/step)")
    print(f"SOM metrics: {info.get('som_metrics', {})}")

    # Check repr shape is consistent
    assert obs["som_repr"].shape[0] == wrapped._som_repr_dim

    wrapped.close()
    print("[PASS] test_som_wrapper")


def test_som_wrapper_ct_off():
    """Test SOM wrapper with force_vector (CT OFF) mode."""
    base_env = MIMoDummyEnv(
        touch_params=TOUCH_PARAMS,
        vision_params=None,
        vestibular_params=None,
    )
    wrapped = SOMObservationWrapper(base_env, som_config={
        "disc_grid": (10, 10),
        "aff_grid": (6, 6),
        "proprio_grid": (8, 8),
    }, seed=42)

    obs, _ = wrapped.reset()
    print(f"CT OFF - obs keys: {list(obs.keys())}")
    print(f"  som_repr: {obs['som_repr'].shape}")

    # In CT OFF, there should be no tactile_aff SOM
    has_aff = "tactile_aff" in wrapped.network.soms
    print(f"  Has affective SOM: {has_aff}")
    assert not has_aff, "CT OFF should not have affective SOM"

    for _ in range(20):
        action = wrapped.action_space.sample()
        obs, _, term, trunc, _ = wrapped.step(action)
        if term or trunc:
            obs, _ = wrapped.reset()

    wrapped.close()
    print("[PASS] test_som_wrapper_ct_off")


def test_ppo_compatibility():
    """Test that wrapped env works with PPO (quick 1K steps)."""
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("[SKIP] test_ppo_compatibility (stable_baselines3 not found)")
        return

    base_env = CTDummyEnv()
    wrapped = SOMObservationWrapper(base_env, som_config={
        "disc_grid": (8, 8),
        "aff_grid": (5, 5),
        "proprio_grid": (6, 6),
        "decay_steps": 5000,
    }, seed=42)

    obs, _ = wrapped.reset()
    print(f"PPO test - obs space: {wrapped.observation_space}")

    t0 = time.time()
    model = PPO("MultiInputPolicy", wrapped, n_steps=64, batch_size=32, verbose=0)
    model.learn(total_timesteps=256)
    dt = time.time() - t0

    print(f"PPO 256 steps in {dt:.2f}s")

    # Test prediction
    obs, _ = wrapped.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"Action shape: {action.shape}")

    wrapped.close()
    print("[PASS] test_ppo_compatibility")


if __name__ == "__main__":
    test_ct_dummy_env()
    print()
    test_preprocessor_real()
    print()
    test_som_wrapper()
    print()
    test_som_wrapper_ct_off()
    print()
    test_ppo_compatibility()
    print("\n=== All MIMo integration tests passed ===")
