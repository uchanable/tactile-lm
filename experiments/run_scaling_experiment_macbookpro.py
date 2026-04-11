#!/usr/bin/env python3
"""
CT-Touch Scaling Experiment: 50K / 100K / 500K / 1000K steps
Run on Mac Studio (M1 Max, 64GB) for faster training.

Usage:
    cd /path/to/mimo-tactile
    PYTHONPATH=MIMo:. python experiments/run_scaling_experiment.py

Results saved to: experiments/rl_results/scaling/
"""

import numpy as np
import time
import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# ---- Environments ----
from mimoEnv.envs.selfbody import MIMoSelfBodyEnv
from ct_touch.ct_augmented_touch import CTAugmentedTouch


class CTSelfBodyEnv(MIMoSelfBodyEnv):
    """SelfBody env with CT-Touch multi_receptor"""
    def touch_setup(self, touch_params):
        ct_params = dict(touch_params)
        ct_params['touch_function'] = 'multi_receptor'
        ct_params['response_function'] = 'spread_linear'
        self.touch = CTAugmentedTouch(self, ct_params)


class RewardLogger(BaseCallback):
    """Log rewards during training for learning curves"""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0

    def _on_step(self):
        # Accumulate reward
        self.current_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0
        return True


def evaluate(model, env, n_episodes=10):
    """Evaluate trained model"""
    rewards = []
    touch_activations = []
    ct_activations = []
    hairy_contacts = 0
    glabrous_contacts = 0
    total_contacts = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            touch = obs['touch']
            touch_activations.append(float(np.sum(np.abs(touch))))

            # Parse CT channel if multi_receptor (7ch per sensor)
            if len(touch) > 2000:  # multi_receptor has more channels
                n_sensors = len(touch) // 7
                reshaped = touch.reshape(n_sensors, 7)
                ct_val = float(np.sum(np.abs(reshaped[:, 3])))
                ct_activations.append(ct_val)

            if terminated or truncated:
                break
        rewards.append(total_r)

    results = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_touch': float(np.mean(touch_activations)),
    }
    if ct_activations:
        results['mean_ct'] = float(np.mean(ct_activations))

    return results


def run_experiment(condition, total_timesteps, seed=42):
    """Run single experiment"""
    print(f"\n{'='*60}")
    print(f"  {condition} | {total_timesteps//1000}K steps | seed={seed}")
    print(f"{'='*60}")

    # Create environment
    if condition == 'CT_OFF':
        env = MIMoSelfBodyEnv()
    else:
        env = CTSelfBodyEnv()

    # Train
    callback = RewardLogger()
    model = PPO("MultiInputPolicy", env, verbose=0, n_steps=512,
                batch_size=64, learning_rate=3e-4, seed=seed)

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    elapsed = time.time() - start

    print(f"  Training time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Evaluate
    results = evaluate(model, env)
    results['condition'] = condition
    results['total_timesteps'] = total_timesteps
    results['training_time'] = elapsed
    results['learning_curve'] = callback.episode_rewards
    results['seed'] = seed

    print(f"  Mean reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    if 'mean_ct' in results:
        print(f"  Mean CT activation: {results['mean_ct']:.4f}")

    # Save
    save_dir = 'experiments/rl_results/scaling'
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{condition}_{total_timesteps//1000}K_seed{seed}.json"

    # Convert learning curve to list for JSON
    save_results = {k: v for k, v in results.items()}
    save_results['learning_curve'] = [float(x) for x in results['learning_curve']]

    with open(f'{save_dir}/{fname}', 'w') as f:
        json.dump(save_results, f, indent=2)

    # Save model
    model.save(f'{save_dir}/model_{condition}_{total_timesteps//1000}K_seed{seed}')

    env.close()
    return results


def main():
    steps_list = [50_000, 100_000, 500_000, 1_000_000]
    conditions = ['CT_OFF', 'CT_ON']
    seeds = [42, 123, 7]

    all_results = []

    for total_steps in steps_list:
        for condition in conditions:
            for seed in seeds:
                result = run_experiment(condition, total_steps, seed)
                all_results.append({
                    'condition': condition,
                    'steps': total_steps,
                    'seed': seed,
                    'reward': result['mean_reward'],
                    'touch': result['mean_touch'],
                    'ct': result.get('mean_ct', 0),
                    'time': result['training_time'],
                })

    # Print summary table
    print("\n" + "="*80)
    print("SCALING EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Steps':>8} | {'Condition':>8} | {'Reward':>12} | {'Touch':>10} | {'CT':>10} | {'Time':>8}")
    print("-"*80)

    for steps in steps_list:
        for cond in conditions:
            subset = [r for r in all_results if r['steps'] == steps and r['condition'] == cond]
            rewards = [r['reward'] for r in subset]
            touches = [r['touch'] for r in subset]
            cts = [r['ct'] for r in subset]
            times = [r['time'] for r in subset]
            print(f"{steps//1000:>6}K | {cond:>8} | {np.mean(rewards):>6.1f}±{np.std(rewards):>4.1f} | "
                  f"{np.mean(touches):>10.1f} | {np.mean(cts):>10.4f} | {np.mean(times):>6.0f}s")
        print("-"*80)

    # Save summary
    save_dir = 'experiments/rl_results/scaling'
    with open(f'{save_dir}/summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_dir}/")


if __name__ == '__main__':
    main()
