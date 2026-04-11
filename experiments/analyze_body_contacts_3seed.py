#!/usr/bin/env python3
"""
Body region별 접촉 패턴 분석: CT_ON vs CT_OFF PPO 모델 비교 (3-seed 평균)

3개 seed(7, 42, 123)로 학습된 모델을 각각 평가하여 seed 간 평균/표준편차로
body region별 접촉 빈도, hairy/glabrous 비율, CT 활성화 top 5를 분석.

Usage:
    cd /path/to/mimo-tactile
    PYTHONPATH=MIMo:. .venv312/bin/python experiments/analyze_body_contacts_3seed.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

from mimoEnv.envs.reach import MIMoReachEnv
from mimoEnv.envs.selfbody import TOUCH_PARAMS
from ct_touch.ct_augmented_touch import CTAugmentedTouch
from ct_touch.skin_map import get_skin_type, SkinType


# ── 환경 정의 ──────────────────────────────────────────────

class ReachWithTouchEnv(MIMoReachEnv):
    """CT_OFF: standard touch (force_vector, 3ch)"""
    def __init__(self, **kwargs):
        super().__init__(touch_params=TOUCH_PARAMS, **kwargs)


class CTReachEnv(MIMoReachEnv):
    """CT_ON: multi_receptor touch (7ch)"""
    def __init__(self, **kwargs):
        super().__init__(touch_params=TOUCH_PARAMS, **kwargs)

    def touch_setup(self, touch_params):
        ct_params = dict(touch_params)
        ct_params['touch_function'] = 'multi_receptor'
        ct_params['response_function'] = 'spread_linear'
        self.touch = CTAugmentedTouch(self, ct_params)


# ── 분석 함수 ──────────────────────────────────────────────

SEEDS = [7, 42, 123]
N_EPISODES = 10


def analyze_model(model_path, condition, n_episodes=N_EPISODES):
    """모델을 로드하고 n_episodes만큼 평가하며 body별 접촉 데이터 수집."""
    print(f"\n{'='*60}")
    print(f"  Loading: {os.path.basename(model_path)}")
    print(f"  Condition: {condition}")
    print(f"{'='*60}")

    # 환경 생성
    if condition == 'CT_OFF':
        env = ReachWithTouchEnv()
    else:
        env = CTReachEnv()

    model = PPO.load(model_path, env=env)

    # body_id -> body_name 매핑 구축
    body_id_to_name = {}
    for body_id in env.touch.sensor_outputs:
        body_name = env.model.body(body_id).name
        body_id_to_name[body_id] = body_name

    print(f"  Touch bodies: {list(body_id_to_name.values())}")
    touch_size = env.touch.touch_size
    print(f"  Touch channels per sensor: {touch_size}")

    # 데이터 수집용 딕셔너리
    body_activation_counts = defaultdict(int)      # body별 활성 스텝 수
    body_activation_sum = defaultdict(float)        # body별 총 활성화 합
    body_ct_sum = defaultdict(float)                # body별 CT 채널 합 (CT_ON only)
    body_ct_counts = defaultdict(int)               # body별 CT 활성 스텝 수

    total_steps = 0
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0

        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            total_steps += 1

            for body_id, sensor_data in env.touch.sensor_outputs.items():
                body_name = body_id_to_name.get(body_id, f"body_{body_id}")
                activation = np.sum(np.abs(sensor_data))

                if activation > 1e-6:
                    body_activation_counts[body_name] += 1
                    body_activation_sum[body_name] += float(activation)

                    # CT_ON 모델: channel 5가 CT
                    if condition == 'CT_ON' and touch_size == 7:
                        ct_values = sensor_data[:, 5]
                        ct_activation = np.sum(np.abs(ct_values))
                        if ct_activation > 1e-8:
                            body_ct_sum[body_name] += float(ct_activation)
                            body_ct_counts[body_name] += 1

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        print(f"  Episode {ep+1}/{n_episodes}: reward={ep_reward:.1f}, steps={step+1}")

    env.close()

    # 결과 정리
    results = {
        'condition': condition,
        'model_path': model_path,
        'n_episodes': n_episodes,
        'total_steps': total_steps,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'body_regions': {},
    }

    all_bodies = set(body_id_to_name.values())
    for body_name in sorted(all_bodies):
        skin_type = get_skin_type(body_name).value
        count = body_activation_counts.get(body_name, 0)
        total_act = body_activation_sum.get(body_name, 0.0)
        freq = count / total_steps if total_steps > 0 else 0.0
        mean_act = total_act / count if count > 0 else 0.0

        entry = {
            'skin_type': skin_type,
            'activation_steps': count,
            'activation_freq': round(freq, 4),
            'mean_activation': round(mean_act, 4),
            'total_activation': round(total_act, 4),
        }

        if condition == 'CT_ON':
            ct_total = body_ct_sum.get(body_name, 0.0)
            ct_count = body_ct_counts.get(body_name, 0)
            entry['ct_total'] = round(ct_total, 6)
            entry['ct_active_steps'] = ct_count
            entry['ct_mean'] = round(ct_total / ct_count, 6) if ct_count > 0 else 0.0

        results['body_regions'][body_name] = entry

    return results


def aggregate_seeds(all_seed_results):
    """여러 seed 결과를 평균/표준편차로 집계.

    Parameters
    ----------
    all_seed_results : list[dict]
        각 seed의 analyze_model() 반환값 리스트.

    Returns
    -------
    dict  집계된 결과 (평균, 표준편차).
    """
    condition = all_seed_results[0]['condition']
    n_seeds = len(all_seed_results)

    rewards = [r['mean_reward'] for r in all_seed_results]
    total_steps_list = [r['total_steps'] for r in all_seed_results]

    # 모든 body 합치기
    all_bodies = set()
    for r in all_seed_results:
        all_bodies.update(r['body_regions'].keys())
    all_bodies = sorted(all_bodies)

    body_agg = {}
    for body_name in all_bodies:
        freqs = []
        mean_acts = []
        act_steps_list = []
        skin_type = None
        ct_totals = []
        ct_means = []
        ct_active_steps_list = []

        for r in all_seed_results:
            bd = r['body_regions'].get(body_name)
            if bd is None:
                freqs.append(0.0)
                mean_acts.append(0.0)
                act_steps_list.append(0)
                if condition == 'CT_ON':
                    ct_totals.append(0.0)
                    ct_means.append(0.0)
                    ct_active_steps_list.append(0)
                continue

            skin_type = bd['skin_type']
            freqs.append(bd['activation_freq'])
            mean_acts.append(bd['mean_activation'])
            act_steps_list.append(bd['activation_steps'])

            if condition == 'CT_ON':
                ct_totals.append(bd.get('ct_total', 0.0))
                ct_means.append(bd.get('ct_mean', 0.0))
                ct_active_steps_list.append(bd.get('ct_active_steps', 0))

        if skin_type is None:
            skin_type = get_skin_type(body_name).value

        entry = {
            'skin_type': skin_type,
            'freq_mean': round(float(np.mean(freqs)), 4),
            'freq_std': round(float(np.std(freqs)), 4),
            'mean_act_mean': round(float(np.mean(mean_acts)), 4),
            'mean_act_std': round(float(np.std(mean_acts)), 4),
            'act_steps_mean': round(float(np.mean(act_steps_list)), 1),
            'act_steps_list': act_steps_list,
        }

        if condition == 'CT_ON':
            entry['ct_total_mean'] = round(float(np.mean(ct_totals)), 6)
            entry['ct_total_std'] = round(float(np.std(ct_totals)), 6)
            entry['ct_mean_mean'] = round(float(np.mean(ct_means)), 6)
            entry['ct_mean_std'] = round(float(np.std(ct_means)), 6)
            entry['ct_active_steps_mean'] = round(float(np.mean(ct_active_steps_list)), 1)

        body_agg[body_name] = entry

    agg = {
        'condition': condition,
        'n_seeds': n_seeds,
        'seeds': SEEDS,
        'n_episodes_per_seed': all_seed_results[0]['n_episodes'],
        'reward_mean': round(float(np.mean(rewards)), 2),
        'reward_std': round(float(np.std(rewards)), 2),
        'reward_per_seed': [round(r, 2) for r in rewards],
        'total_steps_per_seed': total_steps_list,
        'body_regions': body_agg,
    }
    return agg


def print_comparison_3seed(ct_on_agg, ct_off_agg):
    """3-seed 평균 비교 테이블 콘솔 출력."""
    print("\n" + "="*100)
    print("  BODY REGION별 접촉 빈도 비교 - 3 seeds 평균 (CT_ON vs CT_OFF)")
    print("="*100)
    print(f"  CT_ON  reward: {ct_on_agg['reward_mean']:.2f} +/- {ct_on_agg['reward_std']:.2f}  "
          f"(per seed: {ct_on_agg['reward_per_seed']})")
    print(f"  CT_OFF reward: {ct_off_agg['reward_mean']:.2f} +/- {ct_off_agg['reward_std']:.2f}  "
          f"(per seed: {ct_off_agg['reward_per_seed']})")
    print()

    all_bodies = sorted(set(
        list(ct_on_agg['body_regions'].keys()) +
        list(ct_off_agg['body_regions'].keys())
    ))

    header = (f"{'Body Region':<22} {'Skin':<9} "
              f"{'ON freq':>9} {'(sd)':>7} "
              f"{'OFF freq':>9} {'(sd)':>7} "
              f"{'ON act':>9} {'OFF act':>9} "
              f"{'CT mean':>10}")
    print(header)
    print("-" * 100)

    hairy_on_steps = []
    hairy_off_steps = []
    glabrous_on_steps = []
    glabrous_off_steps = []

    for body_name in all_bodies:
        on_d = ct_on_agg['body_regions'].get(body_name, {})
        off_d = ct_off_agg['body_regions'].get(body_name, {})

        skin = on_d.get('skin_type', off_d.get('skin_type', '?'))
        on_freq = on_d.get('freq_mean', 0)
        on_freq_sd = on_d.get('freq_std', 0)
        off_freq = off_d.get('freq_mean', 0)
        off_freq_sd = off_d.get('freq_std', 0)
        on_act = on_d.get('mean_act_mean', 0)
        off_act = off_d.get('mean_act_mean', 0)
        ct_val = on_d.get('ct_mean_mean', 0)

        if skin == 'hairy':
            hairy_on_steps.append(on_d.get('act_steps_list', [0]*3))
            hairy_off_steps.append(off_d.get('act_steps_list', [0]*3))
        else:
            glabrous_on_steps.append(on_d.get('act_steps_list', [0]*3))
            glabrous_off_steps.append(off_d.get('act_steps_list', [0]*3))

        print(f"{body_name:<22} {skin:<9} "
              f"{on_freq:>9.4f} {on_freq_sd:>7.4f} "
              f"{off_freq:>9.4f} {off_freq_sd:>7.4f} "
              f"{on_act:>9.4f} {off_act:>9.4f} "
              f"{ct_val:>10.6f}")

    print("-" * 100)

    # Hairy vs Glabrous 비율 (seed별 합 -> seed별 비율 -> 평균)
    hairy_on_per_seed = [sum(s[i] for s in hairy_on_steps) for i in range(len(SEEDS))]
    hairy_off_per_seed = [sum(s[i] for s in hairy_off_steps) for i in range(len(SEEDS))]
    glab_on_per_seed = [sum(s[i] for s in glabrous_on_steps) for i in range(len(SEEDS))]
    glab_off_per_seed = [sum(s[i] for s in glabrous_off_steps) for i in range(len(SEEDS))]

    total_on_per_seed = [h + g for h, g in zip(hairy_on_per_seed, glab_on_per_seed)]
    total_off_per_seed = [h + g for h, g in zip(hairy_off_per_seed, glab_off_per_seed)]

    hairy_on_pct = [h / t * 100 if t > 0 else 0 for h, t in zip(hairy_on_per_seed, total_on_per_seed)]
    hairy_off_pct = [h / t * 100 if t > 0 else 0 for h, t in zip(hairy_off_per_seed, total_off_per_seed)]
    glab_on_pct = [g / t * 100 if t > 0 else 0 for g, t in zip(glab_on_per_seed, total_on_per_seed)]
    glab_off_pct = [g / t * 100 if t > 0 else 0 for g, t in zip(glab_off_per_seed, total_off_per_seed)]

    print(f"\n  Hairy skin 접촉 비율 (3-seed 평균):")
    print(f"    CT_ON:  {np.mean(hairy_on_pct):.1f}% +/- {np.std(hairy_on_pct):.1f}%")
    print(f"    CT_OFF: {np.mean(hairy_off_pct):.1f}% +/- {np.std(hairy_off_pct):.1f}%")
    print(f"\n  Glabrous skin 접촉 비율 (3-seed 평균):")
    print(f"    CT_ON:  {np.mean(glab_on_pct):.1f}% +/- {np.std(glab_on_pct):.1f}%")
    print(f"    CT_OFF: {np.mean(glab_off_pct):.1f}% +/- {np.std(glab_off_pct):.1f}%")

    # CT 활성화 top 5
    print(f"\n  CT 활성화 Top 5 Body Regions (3-seed 평균):")
    ct_ranking = []
    for body_name, data in ct_on_agg['body_regions'].items():
        ct_total = data.get('ct_total_mean', 0)
        if ct_total > 0:
            ct_ranking.append((
                body_name,
                ct_total,
                data.get('ct_total_std', 0),
                data.get('ct_mean_mean', 0),
                data.get('ct_mean_std', 0),
                data.get('skin_type', '?'),
            ))

    ct_ranking.sort(key=lambda x: x[1], reverse=True)
    for i, (name, total, total_sd, mean, mean_sd, skin) in enumerate(ct_ranking[:5]):
        print(f"    {i+1}. {name:<22} total_CT={total:.6f}+/-{total_sd:.6f}  "
              f"mean_CT={mean:.6f}+/-{mean_sd:.6f}  ({skin})")

    if not ct_ranking:
        print("    (CT 활성화 없음)")

    return {
        'hairy_on_pct_mean': round(float(np.mean(hairy_on_pct)), 1),
        'hairy_on_pct_std': round(float(np.std(hairy_on_pct)), 1),
        'hairy_off_pct_mean': round(float(np.mean(hairy_off_pct)), 1),
        'hairy_off_pct_std': round(float(np.std(hairy_off_pct)), 1),
        'glab_on_pct_mean': round(float(np.mean(glab_on_pct)), 1),
        'glab_on_pct_std': round(float(np.std(glab_on_pct)), 1),
        'glab_off_pct_mean': round(float(np.mean(glab_off_pct)), 1),
        'glab_off_pct_std': round(float(np.std(glab_off_pct)), 1),
        'hairy_on_per_seed': [int(x) for x in hairy_on_per_seed],
        'hairy_off_per_seed': [int(x) for x in hairy_off_per_seed],
        'glab_on_per_seed': [int(x) for x in glab_on_per_seed],
        'glab_off_per_seed': [int(x) for x in glab_off_per_seed],
        'ct_ranking': ct_ranking,
    }


def generate_markdown_3seed(ct_on_agg, ct_off_agg, comparison):
    """3-seed 분석 결과를 Markdown으로 생성."""
    lines = []
    lines.append("---")
    lines.append("type: research")
    lines.append("created: 2026-04-09")
    lines.append("updated: 2026-04-09")
    lines.append("tags: [mimo, tactile, CT-touch, body-contact, PPO, 3-seed]")
    lines.append("---")
    lines.append("")
    lines.append("# Body Region별 접촉 패턴 분석 (CT_ON vs CT_OFF, 3-Seed 평균)")
    lines.append("")
    lines.append("## 개요")
    lines.append("")
    lines.append("학습된 PPO 모델 (1000K steps)을 seed 7, 42, 123으로 각각 평가하여 "
                 "body region별 접촉 패턴을 3-seed 평균으로 분석한 결과.")
    lines.append("")
    lines.append(f"- **Seeds**: {SEEDS}")
    lines.append(f"- **평가 에피소드**: seed당 {ct_on_agg['n_episodes_per_seed']}회 (총 {ct_on_agg['n_episodes_per_seed'] * len(SEEDS)}회)")
    lines.append("")

    # 평가 성능
    lines.append("## 평가 성능")
    lines.append("")
    lines.append("| 조건 | Mean Reward (3-seed) | SD | Seed별 |")
    lines.append("|------|---------------------|-----|--------|")
    lines.append(f"| CT_ON | {ct_on_agg['reward_mean']:.2f} | {ct_on_agg['reward_std']:.2f} | {ct_on_agg['reward_per_seed']} |")
    lines.append(f"| CT_OFF | {ct_off_agg['reward_mean']:.2f} | {ct_off_agg['reward_std']:.2f} | {ct_off_agg['reward_per_seed']} |")
    lines.append("")

    # Body region별 비교 테이블
    lines.append("## Body Region별 접촉 빈도 (3-Seed 평균)")
    lines.append("")
    lines.append("| Body Region | Skin | CT_ON freq | (SD) | CT_OFF freq | (SD) | CT_ON act | CT_OFF act |")
    lines.append("|-------------|------|-----------|------|------------|------|----------|-----------|")

    all_bodies = sorted(set(
        list(ct_on_agg['body_regions'].keys()) +
        list(ct_off_agg['body_regions'].keys())
    ))

    for body_name in all_bodies:
        on_d = ct_on_agg['body_regions'].get(body_name, {})
        off_d = ct_off_agg['body_regions'].get(body_name, {})
        skin = on_d.get('skin_type', off_d.get('skin_type', '?'))
        on_freq = on_d.get('freq_mean', 0)
        on_freq_sd = on_d.get('freq_std', 0)
        off_freq = off_d.get('freq_mean', 0)
        off_freq_sd = off_d.get('freq_std', 0)
        on_act = on_d.get('mean_act_mean', 0)
        off_act = off_d.get('mean_act_mean', 0)
        lines.append(f"| {body_name} | {skin} | {on_freq:.4f} | {on_freq_sd:.4f} | "
                     f"{off_freq:.4f} | {off_freq_sd:.4f} | {on_act:.4f} | {off_act:.4f} |")

    lines.append("")

    # Hairy vs Glabrous
    lines.append("## Hairy vs Glabrous Skin 접촉 비율 (3-Seed 평균)")
    lines.append("")
    lines.append("| 피부 유형 | CT_ON (%) | CT_ON SD | CT_OFF (%) | CT_OFF SD |")
    lines.append("|----------|----------|---------|-----------|----------|")
    lines.append(f"| Hairy | {comparison['hairy_on_pct_mean']:.1f}% | {comparison['hairy_on_pct_std']:.1f}% | "
                 f"{comparison['hairy_off_pct_mean']:.1f}% | {comparison['hairy_off_pct_std']:.1f}% |")
    lines.append(f"| Glabrous | {comparison['glab_on_pct_mean']:.1f}% | {comparison['glab_on_pct_std']:.1f}% | "
                 f"{comparison['glab_off_pct_mean']:.1f}% | {comparison['glab_off_pct_std']:.1f}% |")
    lines.append("")

    # Seed별 raw steps
    lines.append("### Seed별 접촉 스텝 수")
    lines.append("")
    lines.append("| Seed | CT_ON Hairy | CT_ON Glabrous | CT_OFF Hairy | CT_OFF Glabrous |")
    lines.append("|------|-----------|---------------|-------------|----------------|")
    for i, seed in enumerate(SEEDS):
        lines.append(f"| {seed} | {comparison['hairy_on_per_seed'][i]} | "
                     f"{comparison['glab_on_per_seed'][i]} | "
                     f"{comparison['hairy_off_per_seed'][i]} | "
                     f"{comparison['glab_off_per_seed'][i]} |")
    lines.append("")

    # CT Top 5
    lines.append("## CT 활성화 Top 5 Body Regions (3-Seed 평균)")
    lines.append("")
    if comparison['ct_ranking']:
        lines.append("| Rank | Body Region | Skin | Total CT (mean) | Total CT (SD) | Mean CT (mean) | Mean CT (SD) |")
        lines.append("|------|-------------|------|----------------|--------------|---------------|-------------|")
        for i, (name, total, total_sd, mean, mean_sd, skin) in enumerate(comparison['ct_ranking'][:5]):
            lines.append(f"| {i+1} | {name} | {skin} | {total:.6f} | {total_sd:.6f} | "
                         f"{mean:.6f} | {mean_sd:.6f} |")
    else:
        lines.append("CT 활성화가 관측되지 않음.")
    lines.append("")

    # CT_ON vs CT_OFF 행동 차이 분석
    lines.append("## CT_ON vs CT_OFF 행동 차이 분석")
    lines.append("")

    # 빈도 차이가 큰 body region 찾기
    diff_list = []
    for body_name in all_bodies:
        on_d = ct_on_agg['body_regions'].get(body_name, {})
        off_d = ct_off_agg['body_regions'].get(body_name, {})
        on_freq = on_d.get('freq_mean', 0)
        off_freq = off_d.get('freq_mean', 0)
        diff = on_freq - off_freq
        skin = on_d.get('skin_type', off_d.get('skin_type', '?'))
        diff_list.append((body_name, diff, on_freq, off_freq, skin))

    diff_list.sort(key=lambda x: abs(x[1]), reverse=True)

    lines.append("접촉 빈도 차이가 큰 상위 body regions (|CT_ON - CT_OFF|):")
    lines.append("")
    lines.append("| Body Region | Skin | CT_ON freq | CT_OFF freq | Diff | Direction |")
    lines.append("|-------------|------|-----------|------------|------|-----------|")
    for name, diff, on_f, off_f, skin in diff_list[:10]:
        direction = "CT_ON > CT_OFF" if diff > 0 else "CT_OFF > CT_ON"
        if abs(diff) < 0.0001:
            direction = "same"
        lines.append(f"| {name} | {skin} | {on_f:.4f} | {off_f:.4f} | {diff:+.4f} | {direction} |")
    lines.append("")

    # 해석
    lines.append("## 해석")
    lines.append("")
    lines.append("- CT 채널 (channel 5)은 hairy skin에서만 활성화되며, "
                 "glabrous skin (손, 발, 손가락)에서는 0")
    lines.append("- 3-seed 평균으로 random seed에 의한 변동을 감안한 안정적 추정")
    lines.append("- CT_ON 모델은 multi-receptor (7ch) 관측으로 학습되어 "
                 "접촉 패턴이 CT_OFF (3ch)와 다를 수 있음")
    lines.append("- Hairy/glabrous 비율 차이는 CT 신호가 행동 전략에 미치는 영향을 시사")
    lines.append("")

    return "\n".join(lines)


# ── 메인 ──────────────────────────────────────────────────

def main():
    base_dir = "experiments/from_macstudio/reach"

    # 모델 경로 확인
    ct_on_paths = []
    ct_off_paths = []
    for seed in SEEDS:
        on_path = os.path.join(base_dir, f"model_CT_ON_1000K_seed{seed}")
        off_path = os.path.join(base_dir, f"model_CT_OFF_1000K_seed{seed}")
        for p in [on_path, off_path]:
            if not os.path.exists(p + ".zip"):
                print(f"ERROR: {p}.zip not found!")
                sys.exit(1)
        ct_on_paths.append(on_path)
        ct_off_paths.append(off_path)

    print(f"Models found: {len(ct_on_paths)} CT_ON, {len(ct_off_paths)} CT_OFF")
    print(f"Seeds: {SEEDS}")

    # 각 seed별 분석
    ct_off_all = []
    ct_on_all = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'#'*60}")
        print(f"  SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'#'*60}")

        ct_off_results = analyze_model(ct_off_paths[i], 'CT_OFF', n_episodes=N_EPISODES)
        ct_off_all.append(ct_off_results)

        ct_on_results = analyze_model(ct_on_paths[i], 'CT_ON', n_episodes=N_EPISODES)
        ct_on_all.append(ct_on_results)

    # 3-seed 집계
    ct_on_agg = aggregate_seeds(ct_on_all)
    ct_off_agg = aggregate_seeds(ct_off_all)

    # 비교 출력
    comparison = print_comparison_3seed(ct_on_agg, ct_off_agg)

    # JSON 저장
    output_json = "experiments/body_contact_3seed.json"
    with open(output_json, 'w') as f:
        json.dump({
            'ct_on_agg': ct_on_agg,
            'ct_off_agg': ct_off_agg,
            'ct_on_per_seed': [r for r in ct_on_all],
            'ct_off_per_seed': [r for r in ct_off_all],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON saved: {output_json}")

    # Markdown 저장
    md_content = generate_markdown_3seed(ct_on_agg, ct_off_agg, comparison)
    md_path = "/Users/uchanable_m1/obsidian/30_Research/09_TactileLM/02_IEEE_TCDS/body_contact_3seed.md"
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"  Markdown saved: {md_path}")


if __name__ == '__main__':
    main()
