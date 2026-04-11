#!/usr/bin/env python3
"""
SOM + CT-Touch 2x2 Factorial Experiment
========================================

Conditions:
  1. PPO + CT OFF  (baseline, force_vector 3ch)
  2. PPO + CT ON   (multi_receptor 7ch)
  3. SOM + CT OFF  (SOM wrapper + force_vector)
  4. SOM + CT ON   (SOM wrapper + multi_receptor)

PPO conditions (1,2) can reuse existing 30-seed data.
SOM conditions (3,4) are new.

Usage:
    cd /path/to/mimo-tactile
    PYTHONPATH=MIMo:. python experiments/run_som_experiment.py [--task reach|selfbody] [--seeds N]

Results saved to: experiments/rl_results/som_{task}/
"""

import argparse
import json
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mimoEnv.envs.reach import MIMoReachEnv
from mimoEnv.envs.selfbody import MIMoSelfBodyEnv, TOUCH_PARAMS
from ct_touch.ct_augmented_touch import CTAugmentedTouch
from som.som_wrapper import SOMObservationWrapper
from som.preprocessor import TouchPreprocessor


# =============================================================================
# Environment classes
# =============================================================================

class ReachCTOff(MIMoReachEnv):
    """Reach with standard force_vector touch (3ch)."""
    def __init__(self, **kwargs):
        super().__init__(touch_params=TOUCH_PARAMS, **kwargs)


class ReachCTOn(MIMoReachEnv):
    """Reach with CT-Touch multi_receptor (7ch)."""
    def __init__(self, **kwargs):
        super().__init__(touch_params=TOUCH_PARAMS, **kwargs)

    def touch_setup(self, touch_params):
        ct_params = dict(touch_params)
        ct_params['touch_function'] = 'multi_receptor'
        ct_params['response_function'] = 'spread_linear'
        self.touch = CTAugmentedTouch(self, ct_params)


class SelfbodyCTOff(MIMoSelfBodyEnv):
    """Selfbody with standard force_vector touch (3ch)."""
    pass


class SelfbodyCTOn(MIMoSelfBodyEnv):
    """Selfbody with CT-Touch multi_receptor (7ch)."""
    def touch_setup(self, touch_params):
        ct_params = dict(touch_params)
        ct_params['touch_function'] = 'multi_receptor'
        ct_params['response_function'] = 'spread_linear'
        self.touch = CTAugmentedTouch(self, ct_params)


# =============================================================================
# SOM configuration
# =============================================================================

SOM_CONFIG = {
    "disc_grid": (15, 15),      # 225 neurons
    "aff_grid": (10, 10),       # 100 neurons
    "proprio_grid": (10, 10),   # 100 neurons
    "initial_lr": 0.5,
    "final_lr": 0.01,
    "initial_sigma": None,
    "final_sigma": 0.5,
    "decay_steps": 200_000,
    "hebbian_eta": 0.01,
    "hebbian_decay": 0.001,
    "include_proprio_som": True,
    "include_raw_proprio": True,
}


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


class SOMMetricsLogger(BaseCallback):
    """Logs SOM metrics periodically."""
    def __init__(self, log_interval=10_000):
        super().__init__()
        self.log_interval = log_interval
        self.som_metrics_history = []

    def _on_step(self):
        if self.num_timesteps % self.log_interval == 0:
            info = self.locals.get('infos', [{}])
            if info and 'som_metrics' in info[0]:
                entry = {'step': self.num_timesteps}
                entry.update(info[0]['som_metrics'])
                self.som_metrics_history.append(entry)
        return True


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, env, n_episodes=10):
    """Evaluate trained model.

    Returns dict with reward stats, touch activation, CT activation,
    SOM metrics, and per-body contact summary.
    """
    rewards = []
    touch_activations = []
    ct_activations = []
    som_metrics_list = []
    body_contacts = {}  # body_name -> list of activations

    is_som = isinstance(env, SOMObservationWrapper)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0

        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward

            # Collect SOM metrics from last step
            if is_som and 'som_metrics' in info:
                if ep == n_episodes - 1 and step == 0:
                    som_metrics_list.append(info['som_metrics'])

            # Collect touch activation from base env
            base_env = env.env if is_som else env
            if hasattr(base_env, 'touch') and base_env.touch:
                touch_obs = base_env.touch.sensor_outputs
                total_act = sum(
                    float(np.sum(np.abs(v)))
                    for v in touch_obs.values()
                )
                touch_activations.append(total_act)

                # CT activation (channel 5)
                if base_env.touch.touch_size == 7:
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

    # Body contact summary
    results['body_contacts'] = {
        name: float(np.mean(vals))
        for name, vals in body_contacts.items()
    }

    # SOM metrics
    if som_metrics_list:
        results['som_metrics'] = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in som_metrics_list[-1].items()
        }

    return results


# =============================================================================
# Main experiment runner
# =============================================================================

def make_env(task, ct_mode, use_som, seed):
    """Create environment for given condition.

    Args:
        task: "reach" or "selfbody"
        ct_mode: "CT_ON" or "CT_OFF"
        use_som: Whether to wrap with SOMObservationWrapper
        seed: Random seed

    Returns:
        Gymnasium environment.
    """
    env_classes = {
        ("reach", "CT_OFF"): ReachCTOff,
        ("reach", "CT_ON"): ReachCTOn,
        ("selfbody", "CT_OFF"): SelfbodyCTOff,
        ("selfbody", "CT_ON"): SelfbodyCTOn,
    }
    base_env = env_classes[(task, ct_mode)]()

    if use_som:
        return SOMObservationWrapper(base_env, som_config=SOM_CONFIG, seed=seed)
    return base_env


def run_single(task, condition, total_timesteps, seed, save_dir):
    """Run a single experiment.

    Args:
        task: "reach" or "selfbody"
        condition: "PPO_CT_OFF", "PPO_CT_ON", "SOM_CT_OFF", "SOM_CT_ON"
        total_timesteps: Training steps
        seed: Random seed
        save_dir: Directory for results

    Returns:
        dict: Results.
    """
    # Parse condition
    parts = condition.split('_', 1)
    use_som = parts[0] == 'SOM'
    ct_mode = parts[1] if not use_som else '_'.join(parts[1:])
    # Handle "SOM_CT_ON" -> use_som=True, ct_mode="CT_ON"
    if condition.startswith('SOM_'):
        use_som = True
        ct_mode = condition[4:]  # "CT_ON" or "CT_OFF"
    else:
        use_som = False
        ct_mode = condition[4:]  # "CT_ON" or "CT_OFF"

    print(f"\n{'='*60}")
    print(f"  {task.upper()} | {condition} | {total_timesteps//1000}K | seed={seed}")
    print(f"{'='*60}")

    env = make_env(task, ct_mode, use_som, seed)

    callbacks = [RewardLogger()]
    if use_som:
        callbacks.append(SOMMetricsLogger(log_interval=10_000))

    model = PPO(
        "MultiInputPolicy", env,
        verbose=0,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        seed=seed,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    elapsed = time.time() - t0

    print(f"  Training: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Evaluate
    results = evaluate(model, env)
    results['condition'] = condition
    results['task'] = task
    results['total_timesteps'] = total_timesteps
    results['training_time'] = elapsed
    results['seed'] = seed
    results['learning_curve'] = [float(x) for x in callbacks[0].episode_rewards]

    if use_som and len(callbacks) > 1:
        results['som_metrics_history'] = callbacks[1].som_metrics_history

    print(f"  Reward: {results['mean_reward']:.1f} +/- {results['std_reward']:.1f}")
    if 'mean_ct' in results:
        print(f"  CT activation: {results['mean_ct']:.4f}")
    if 'mean_touch' in results:
        print(f"  Touch activation: {results['mean_touch']:.1f}")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{condition}_{total_timesteps//1000}K_seed{seed}.json"
    fpath = os.path.join(save_dir, fname)

    # Serialize (convert numpy types)
    save_data = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_data[k] = {
                sk: float(sv) if isinstance(sv, (np.floating, np.integer)) else sv
                for sk, sv in v.items()
            }
        elif isinstance(v, list):
            save_data[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
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


def main():
    parser = argparse.ArgumentParser(description="SOM + CT-Touch 2x2 Factorial Experiment")
    parser.add_argument('--task', choices=['reach', 'selfbody'], default='reach')
    parser.add_argument('--seeds', type=int, default=30, help='Number of seeds')
    parser.add_argument('--steps', type=int, default=1_000_000, help='Training steps')
    parser.add_argument('--conditions', nargs='+',
                        default=['SOM_CT_OFF', 'SOM_CT_ON'],
                        choices=['PPO_CT_OFF', 'PPO_CT_ON', 'SOM_CT_OFF', 'SOM_CT_ON'],
                        help='Conditions to run')
    parser.add_argument('--start-seed', type=int, default=0, help='Starting seed index')
    args = parser.parse_args()

    # 30 fixed seeds for reproducibility (same as PPO experiments)
    ALL_SEEDS = [
        42, 123, 7, 256, 512, 1024, 2048, 314, 777, 55,
        101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
        1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
    ]
    seeds = ALL_SEEDS[args.start_seed:args.start_seed + args.seeds]

    save_dir = f"experiments/rl_results/som_{args.task}"
    all_results = []

    total_runs = len(args.conditions) * len(seeds)
    run_idx = 0

    for condition in args.conditions:
        for seed in seeds:
            run_idx += 1
            fname = f"{condition}_{args.steps//1000}K_seed{seed}.json"
            fpath = os.path.join(save_dir, fname)

            if os.path.exists(fpath):
                print(f"\n  [{run_idx}/{total_runs}] Skipping {fname} (exists)")
                with open(fpath) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n  [{run_idx}/{total_runs}]", end="")
            result = run_single(args.task, condition, args.steps, seed, save_dir)
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {args.task} | {args.steps//1000}K steps | {len(seeds)} seeds")
    print(f"{'='*60}")

    for condition in args.conditions:
        cond_results = [r for r in all_results if r.get('condition') == condition]
        if not cond_results:
            continue
        rewards = [r['mean_reward'] for r in cond_results]
        print(f"\n  {condition}:")
        print(f"    Reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
        print(f"    N seeds: {len(cond_results)}")

        if 'mean_touch' in cond_results[0]:
            touches = [r['mean_touch'] for r in cond_results if 'mean_touch' in r]
            print(f"    Touch: {np.mean(touches):.1f}")
        if 'mean_ct' in cond_results[0]:
            cts = [r['mean_ct'] for r in cond_results if 'mean_ct' in r]
            print(f"    CT: {np.mean(cts):.4f}")

    # Save summary
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'task': args.task,
            'steps': args.steps,
            'conditions': args.conditions,
            'seeds': seeds,
            'n_results': len(all_results),
        }, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
