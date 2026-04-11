#!/usr/bin/env python3
"""
Developmental Trajectory Experiment (Persona Simulation)
========================================================

Varies maturation parameters (tau_CT, tau_Abeta) to test robustness
of CT effects across different developmental profiles.

This answers the reviewer concern: "maturation schedules are first
approximations, not fitted to empirical data."

Design:
  - tau_CT ∈ [4, 8, 12] months (CT maturation speed)
  - tau_Abeta ∈ [8, 12, 16] months (myelination speed)
  - 3×3 = 9 persona conditions × 2 CT modes (ON/OFF) = 18 conditions
  - N seeds per condition (default 10)
  - SOM + PPO, reach task, 1M steps

Usage:
    cd /path/to/mimo-tactile
    PYTHONPATH=MIMo:. python experiments/run_developmental_experiment.py

    # Quick test
    PYTHONPATH=MIMo:. python experiments/run_developmental_experiment.py --seeds 2 --steps 100000

    # Specific persona only
    PYTHONPATH=MIMo:. python experiments/run_developmental_experiment.py --tau-ct 8 --tau-ab 12
"""

import argparse
import json
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mimoEnv.envs.reach import MIMoReachEnv
from mimoEnv.envs.selfbody import TOUCH_PARAMS
from ct_touch.ct_augmented_touch import CTAugmentedTouch
from ct_touch.developmental import DevelopmentalProfile
from som.som_wrapper import SOMObservationWrapper
from som.critical_periods import CriticalPeriodScheduler


# =============================================================================
# Environment classes with configurable developmental profile
# =============================================================================

class DevReachCTOn(MIMoReachEnv):
    """Reach with CT-Touch and configurable developmental maturation."""

    # Class-level config set before instantiation
    _dev_age: float = 18.0

    def __init__(self, **kwargs):
        super().__init__(touch_params=TOUCH_PARAMS, **kwargs)

    def touch_setup(self, touch_params):
        ct_params = dict(touch_params)
        ct_params['touch_function'] = 'multi_receptor'
        ct_params['response_function'] = 'spread_linear'
        self.touch = CTAugmentedTouch(
            self, ct_params,
            developmental_age=self._dev_age,
        )


class DevReachCTOff(MIMoReachEnv):
    """Reach with standard touch (no CT)."""
    def __init__(self, **kwargs):
        super().__init__(touch_params=TOUCH_PARAMS, **kwargs)


# =============================================================================
# Custom DevelopmentalProfile with configurable tau
# =============================================================================

class ConfigurableDevelopmentalProfile(DevelopmentalProfile):
    """DevelopmentalProfile with configurable time constants."""

    def __init__(self, age_months=18.0, tau_ct=8.0, tau_abeta=12.0):
        super().__init__(age_months)
        self.tau_ct = tau_ct
        self.tau_abeta = tau_abeta

    def ct_maturity(self):
        return 0.4 + 0.6 * (1.0 - np.exp(-self.age_months / self.tau_ct))

    def myelinated_maturity(self):
        return 0.2 + 0.8 * (1.0 - np.exp(-self.age_months / (self.tau_abeta / 12.0 * 7.0)))

    def myelination_factor(self):
        return 0.3 + 0.7 * (1.0 - np.exp(-self.age_months / self.tau_abeta))


# =============================================================================
# Callbacks
# =============================================================================

class RewardLogger(BaseCallback):
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
# Evaluation
# =============================================================================

def evaluate(model, env, n_episodes=10):
    """Evaluate model and collect metrics."""
    rewards = []
    touch_acts = []
    ct_acts = []

    is_som = isinstance(env, SOMObservationWrapper)
    som_metrics = None

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_r += reward

            base = env.env if is_som else env
            if hasattr(base, 'touch') and base.touch:
                so = base.touch.sensor_outputs
                touch_acts.append(sum(float(np.sum(np.abs(v))) for v in so.values()))
                if base.touch.touch_size == 7:
                    ct_acts.append(sum(float(np.sum(np.abs(v[:, 5]))) for v in so.values()))

            if is_som and 'som_metrics' in info and ep == n_episodes - 1 and step == 0:
                som_metrics = info['som_metrics']

            if term or trunc:
                break
        rewards.append(total_r)

    result = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
    }
    if touch_acts:
        result['mean_touch'] = float(np.mean(touch_acts))
    if ct_acts:
        result['mean_ct'] = float(np.mean(ct_acts))
    if som_metrics:
        result['som_metrics'] = {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in som_metrics.items()
        }
    return result


# =============================================================================
# Main
# =============================================================================

def run_single(tau_ct, tau_ab, ct_mode, total_timesteps, seed, save_dir, use_som=True):
    """Run one persona experiment."""
    persona = f"tauCT{tau_ct}_tauAB{tau_ab}"
    condition = f"{'SOM_' if use_som else ''}{ct_mode}_{persona}"

    print(f"\n{'='*60}")
    print(f"  REACH | {condition} | {total_timesteps//1000}K | seed={seed}")
    print(f"{'='*60}")

    # Create environment
    if ct_mode == "CT_ON":
        DevReachCTOn._dev_age = 18.0
        env = DevReachCTOn()
        # Replace developmental profile with configurable one
        if hasattr(env.touch, 'developmental') and env.touch.developmental is not None:
            env.touch.developmental = ConfigurableDevelopmentalProfile(
                age_months=18.0, tau_ct=tau_ct, tau_abeta=tau_ab,
            )
    else:
        env = DevReachCTOff()

    if use_som:
        env = SOMObservationWrapper(env, som_config={
            "disc_grid": (15, 15),
            "aff_grid": (10, 10),
            "proprio_grid": (10, 10),
            "decay_steps": 200_000,
        }, seed=seed)

    callback = RewardLogger()
    model = PPO(
        "MultiInputPolicy", env,
        verbose=0, n_steps=512, batch_size=64,
        learning_rate=3e-4, seed=seed,
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - t0

    results = evaluate(model, env)
    results.update({
        'condition': condition,
        'ct_mode': ct_mode,
        'tau_ct': tau_ct,
        'tau_abeta': tau_ab,
        'persona': persona,
        'total_timesteps': total_timesteps,
        'training_time': elapsed,
        'seed': seed,
        'learning_curve': [float(x) for x in callback.episode_rewards],
        'use_som': use_som,
    })

    print(f"  Reward: {results['mean_reward']:.1f} +/- {results['std_reward']:.1f}")
    print(f"  Time: {elapsed:.0f}s")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{condition}_{total_timesteps//1000}K_seed{seed}.json"
    fpath = os.path.join(save_dir, fname)
    with open(fpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {fpath}")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Developmental Persona Experiment")
    parser.add_argument('--tau-ct', type=float, nargs='+', default=[4, 8, 12],
                        help='CT maturation tau values (months)')
    parser.add_argument('--tau-ab', type=float, nargs='+', default=[8, 12, 16],
                        help='Abeta myelination tau values (months)')
    parser.add_argument('--ct-modes', nargs='+', default=['CT_ON', 'CT_OFF'],
                        choices=['CT_ON', 'CT_OFF'])
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--no-som', action='store_true', help='Use PPO without SOM')
    parser.add_argument('--start-seed', type=int, default=0)
    args = parser.parse_args()

    ALL_SEEDS = [42, 123, 7, 256, 512, 1024, 2048, 314, 777, 55,
                 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]
    seeds = ALL_SEEDS[args.start_seed:args.start_seed + args.seeds]
    use_som = not args.no_som

    save_dir = "experiments/rl_results/developmental"
    all_results = []

    total_runs = len(args.tau_ct) * len(args.tau_ab) * len(args.ct_modes) * len(seeds)
    run_idx = 0

    for tau_ct in args.tau_ct:
        for tau_ab in args.tau_ab:
            for ct_mode in args.ct_modes:
                for seed in seeds:
                    run_idx += 1
                    persona = f"tauCT{tau_ct}_tauAB{tau_ab}"
                    condition = f"{'SOM_' if use_som else ''}{ct_mode}_{persona}"
                    fname = f"{condition}_{args.steps//1000}K_seed{seed}.json"
                    fpath = os.path.join(save_dir, fname)

                    if os.path.exists(fpath):
                        print(f"  [{run_idx}/{total_runs}] Skip {fname}")
                        with open(fpath) as f:
                            all_results.append(json.load(f))
                        continue

                    print(f"  [{run_idx}/{total_runs}]", end="")
                    result = run_single(
                        tau_ct, tau_ab, ct_mode,
                        args.steps, seed, save_dir, use_som,
                    )
                    all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  DEVELOPMENTAL EXPERIMENT SUMMARY")
    print(f"  {len(args.tau_ct)}×{len(args.tau_ab)} personas × {len(args.ct_modes)} CT modes × {len(seeds)} seeds")
    print(f"{'='*60}")

    for tau_ct in args.tau_ct:
        for tau_ab in args.tau_ab:
            persona = f"tauCT{tau_ct}_tauAB{tau_ab}"
            for ct_mode in args.ct_modes:
                cond_results = [r for r in all_results
                                if r.get('persona') == persona and r.get('ct_mode') == ct_mode]
                if cond_results:
                    rewards = [r['mean_reward'] for r in cond_results]
                    print(f"  {persona} {ct_mode}: {np.mean(rewards):.1f} ± {np.std(rewards):.1f} (n={len(cond_results)})")

    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'tau_ct_values': args.tau_ct,
            'tau_ab_values': args.tau_ab,
            'ct_modes': args.ct_modes,
            'seeds': seeds,
            'n_results': len(all_results),
            'use_som': use_som,
        }, f, indent=2)


if __name__ == '__main__':
    main()
