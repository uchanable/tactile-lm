#!/usr/bin/env python3
"""
CT-Touch Channel Ablation Experiment
=====================================

CT-Touch 7채널 중 어떤 채널이 학습에 기여하는지 분리하기 위한 ablation study.

Conditions:
  1. 7ch_full   : SA-I(3) + FA-I(1) + FA-II(1) + CT(1) + Normal(1) -- 기존 CT_ON과 동일
  2. 6ch_no_ct  : SA-I(3) + FA-I(1) + FA-II(1) + Normal(1) -- CT(ch5)만 제거
  3. 4ch_disc   : SA-I(3) + Normal(1) -- discriminative touch만 (ch3,4,5 제거)
  4. 3ch_force  : Force vector (3D) -- 기존 CT_OFF와 동일

Usage:
    cd /path/to/mimo-tactile
    PYTHONPATH=MIMo:. python experiments/run_ablation_experiment.py
    PYTHONPATH=MIMo:. python experiments/run_ablation_experiment.py --conditions 6ch_no_ct 4ch_disc
    PYTHONPATH=MIMo:. python experiments/run_ablation_experiment.py --seeds 5 --steps 500000

Results saved to: experiments/rl_results/ablation_reach/
"""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mimoEnv.envs.reach import MIMoReachEnv
from mimoEnv.envs.selfbody import TOUCH_PARAMS
from ct_touch.ct_augmented_touch import CTAugmentedTouch


# =============================================================================
# Ablation conditions
# =============================================================================

# Channel layout of multi_receptor (7ch):
#   [0:3] SA-I  (sustained pressure, 3D force vector)
#   [3]   FA-I  (velocity-proportional)
#   [4]   FA-II (vibration/acceleration)
#   [5]   CT    (velocity-tuned inverted-U)
#   [6]   Normal force magnitude

ABLATION_CONDITIONS = {
    "7ch_full":   None,           # no masking -- full 7ch
    "6ch_no_ct":  [5],            # zero out CT channel
    "4ch_disc":   [3, 4, 5],      # zero out FA-I, FA-II, CT
    "3ch_force":  "force_vector", # use ReachWithTouchEnv (3ch force_vector)
}

ALL_CONDITIONS = list(ABLATION_CONDITIONS.keys())


# =============================================================================
# Base environment classes (from run_reach_experiment.py)
# =============================================================================

class ReachWithTouchEnv(MIMoReachEnv):
    """Reach env with standard touch (force_vector, 3ch)."""
    def __init__(self, **kwargs):
        super().__init__(touch_params=TOUCH_PARAMS, **kwargs)


class CTReachEnv(MIMoReachEnv):
    """Reach env with CT-Touch multi_receptor (7ch)."""
    def __init__(self, **kwargs):
        super().__init__(touch_params=TOUCH_PARAMS, **kwargs)

    def touch_setup(self, touch_params):
        ct_params = dict(touch_params)
        ct_params['touch_function'] = 'multi_receptor'
        ct_params['response_function'] = 'spread_linear'
        self.touch = CTAugmentedTouch(self, ct_params)


# =============================================================================
# Channel masking wrapper
# =============================================================================

class ChannelMaskWrapper(gym.Wrapper):
    """Gym wrapper that zeros out specified channels in touch observations.

    touch obs는 (n_sensors * touch_size,) 형태의 flat vector로 들어온다.
    touch_size=7 (multi_receptor)일 때, (n_sensors, 7)로 reshape한 뒤
    지정된 채널을 0으로 마스킹하고 다시 flatten한다.
    """

    def __init__(self, env, masked_channels):
        """
        Args:
            env: CTReachEnv (7ch multi_receptor)
            masked_channels: list of int, 0으로 설정할 채널 인덱스 (e.g. [5] for CT)
        """
        super().__init__(env)
        self.masked_channels = masked_channels
        self.touch_size = 7  # multi_receptor output size

    def _mask_touch(self, obs):
        """obs dict 안의 'touch' 키를 마스킹한다."""
        if 'touch' not in obs:
            return obs
        touch = obs['touch'].copy()
        n_sensors = len(touch) // self.touch_size
        if n_sensors > 0 and len(touch) == n_sensors * self.touch_size:
            reshaped = touch.reshape(n_sensors, self.touch_size)
            reshaped[:, self.masked_channels] = 0.0
            obs['touch'] = reshaped.flatten()
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._mask_touch(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._mask_touch(obs)
        return obs, reward, terminated, truncated, info


# =============================================================================
# Callbacks
# =============================================================================

class RewardLogger(BaseCallback):
    """Logs per-episode rewards during training."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self):
        self.current_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True


# =============================================================================
# Environment factory
# =============================================================================

def make_env(condition):
    """Create environment for the given ablation condition.

    Args:
        condition: one of "7ch_full", "6ch_no_ct", "4ch_disc", "3ch_force"

    Returns:
        Gymnasium environment
    """
    spec = ABLATION_CONDITIONS[condition]

    if spec == "force_vector":
        # 3ch_force: standard force_vector touch
        return ReachWithTouchEnv()

    # All other conditions start from the 7ch CTReachEnv
    env = CTReachEnv()

    if spec is not None:
        # Apply channel masking
        env = ChannelMaskWrapper(env, masked_channels=spec)

    return env


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, env, n_episodes=10):
    """Evaluate trained model.

    Returns dict with reward stats, touch/CT activation info, and
    per-body contact summary.
    """
    rewards = []
    touch_activations = []
    ct_activations = []
    body_contacts = {}

    # Unwrap to base env for touch introspection
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env

    is_7ch = hasattr(base_env, 'touch') and base_env.touch and base_env.touch.touch_size == 7

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0

        for step_i in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward

            # Collect touch activation from base env
            if hasattr(base_env, 'touch') and base_env.touch:
                touch_obs = base_env.touch.sensor_outputs
                total_act = sum(
                    float(np.sum(np.abs(v)))
                    for v in touch_obs.values()
                )
                touch_activations.append(total_act)

                # CT activation (channel 5) -- raw, before masking
                if is_7ch:
                    ct_act = sum(
                        float(np.sum(np.abs(v[:, 5])))
                        for v in touch_obs.values()
                    )
                    ct_activations.append(ct_act)

                # Per-body contacts
                for bid in sorted(touch_obs.keys()):
                    name = base_env.model.body(bid).name
                    act = float(np.sum(np.abs(touch_obs[bid])))
                    body_contacts.setdefault(name, []).append(act)

            if terminated or truncated:
                break
        rewards.append(total_r)

    results = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'rewards_per_episode': [float(r) for r in rewards],
    }

    if touch_activations:
        results['mean_touch'] = float(np.mean(touch_activations))
    if ct_activations:
        results['mean_ct'] = float(np.mean(ct_activations))

    results['body_contacts'] = {
        name: float(np.mean(vals))
        for name, vals in body_contacts.items()
    }

    return results


# =============================================================================
# Single run
# =============================================================================

def run_single(condition, total_timesteps, seed, save_dir):
    """Run a single ablation experiment.

    Args:
        condition: ablation condition name
        total_timesteps: training steps
        seed: random seed
        save_dir: directory for results

    Returns:
        dict: results
    """
    print(f"\n{'='*60}")
    print(f"  ABLATION | {condition} | {total_timesteps//1000}K | seed={seed}")
    print(f"{'='*60}")

    env = make_env(condition)

    callback = RewardLogger()
    model = PPO(
        "MultiInputPolicy", env,
        verbose=0,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        seed=seed,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0

    print(f"  Training: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Evaluate
    results = evaluate(model, env)
    results['condition'] = condition
    results['total_timesteps'] = total_timesteps
    results['training_time'] = elapsed
    results['seed'] = seed
    results['learning_curve'] = [float(x) for x in callback.episode_rewards]

    print(f"  Reward: {results['mean_reward']:.1f} +/- {results['std_reward']:.1f}")
    if 'mean_ct' in results:
        print(f"  CT activation: {results['mean_ct']:.4f}")
    if 'mean_touch' in results:
        print(f"  Touch activation: {results['mean_touch']:.1f}")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{condition}_{total_timesteps//1000}K_seed{seed}.json"
    fpath = os.path.join(save_dir, fname)

    # Serialize numpy types
    save_data = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_data[k] = {
                sk: float(sv) if isinstance(sv, (np.floating, np.integer)) else sv
                for sk, sv in v.items()
            }
        elif isinstance(v, list):
            save_data[k] = [
                float(x) if isinstance(x, (np.floating, np.integer)) else x
                for x in v
            ]
        elif isinstance(v, (np.floating, np.integer)):
            save_data[k] = float(v)
        else:
            save_data[k] = v

    with open(fpath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved: {fpath}")

    # Save model
    model_path = os.path.join(save_dir, f"model_{condition}_{total_timesteps//1000}K_seed{seed}")
    model.save(model_path)

    env.close()
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CT-Touch Channel Ablation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation conditions:
  7ch_full   SA-I(3) + FA-I(1) + FA-II(1) + CT(1) + Normal(1)  [full 7ch]
  6ch_no_ct  SA-I(3) + FA-I(1) + FA-II(1) + Normal(1)          [CT removed]
  4ch_disc   SA-I(3) + Normal(1)                                [discriminative only]
  3ch_force  Force vector (3D)                                   [baseline]
""")
    parser.add_argument(
        '--conditions', nargs='+',
        default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help='Ablation conditions to run (default: all 4)',
    )
    parser.add_argument('--seeds', type=int, default=30, help='Number of seeds (default: 30)')
    parser.add_argument('--steps', type=int, default=1_000_000, help='Training steps (default: 1M)')
    parser.add_argument('--start-seed', type=int, default=0, help='Starting seed index')
    args = parser.parse_args()

    # 30 fixed seeds for reproducibility (same as other experiments)
    ALL_SEEDS = [
        42, 123, 7, 256, 512, 1024, 2048, 314, 777, 55,
        101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
        1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
    ]
    seeds = ALL_SEEDS[args.start_seed:args.start_seed + args.seeds]

    save_dir = "experiments/rl_results/ablation_reach"
    all_results = []

    total_runs = len(args.conditions) * len(seeds)
    run_idx = 0

    for condition in args.conditions:
        for seed in seeds:
            run_idx += 1
            fname = f"{condition}_{args.steps//1000}K_seed{seed}.json"
            fpath = os.path.join(save_dir, fname)

            # Skip if already completed
            if os.path.exists(fpath):
                print(f"\n  [{run_idx}/{total_runs}] Skipping {fname} (exists)")
                with open(fpath) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n  [{run_idx}/{total_runs}]", end="")
            result = run_single(condition, args.steps, seed, save_dir)
            all_results.append(result)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY | {args.steps//1000}K steps | {len(seeds)} seeds")
    print(f"{'='*70}")
    print(f"  {'Condition':<12} | {'Reward':>14} | {'Touch':>10} | {'CT':>10} | {'N':>3}")
    print(f"  {'-'*12}-+-{'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*3}")

    for condition in args.conditions:
        cond_results = [r for r in all_results if r.get('condition') == condition]
        if not cond_results:
            continue
        rewards = [r['mean_reward'] for r in cond_results]
        n = len(cond_results)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)

        touch_str = "N/A"
        ct_str = "N/A"
        if any('mean_touch' in r for r in cond_results):
            touches = [r['mean_touch'] for r in cond_results if 'mean_touch' in r]
            touch_str = f"{np.mean(touches):.1f}"
        if any('mean_ct' in r for r in cond_results):
            cts = [r['mean_ct'] for r in cond_results if 'mean_ct' in r]
            ct_str = f"{np.mean(cts):.4f}"

        print(f"  {condition:<12} | {mean_r:>6.1f} +/- {std_r:<4.1f} | {touch_str:>10} | {ct_str:>10} | {n:>3}")

    # Save summary
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'experiment': 'ablation_reach',
            'conditions': args.conditions,
            'steps': args.steps,
            'seeds': seeds,
            'n_results': len(all_results),
        }, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
