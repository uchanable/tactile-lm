#!/usr/bin/env python3
"""
Body region별 접촉 패턴 분석: CT_ON vs CT_OFF PPO 모델 비교

학습된 PPO 모델을 로드하여 10 에피소드 평가 → body region별 접촉 빈도,
hairy/glabrous 비율, CT 활성화 top 5 분석.

Usage:
    cd /path/to/mimo-tactile
    PYTHONPATH=MIMo:. .venv312/bin/python experiments/analyze_body_contacts.py
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

def analyze_model(model_path, condition, n_episodes=10):
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

    # body_id → body_name 매핑 구축
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

            # sensor_outputs는 get_touch_obs() 호출 후 업데이트됨 (step 안에서 호출됨)
            for body_id, sensor_data in env.touch.sensor_outputs.items():
                body_name = body_id_to_name.get(body_id, f"body_{body_id}")
                activation = np.sum(np.abs(sensor_data))

                if activation > 1e-6:
                    body_activation_counts[body_name] += 1
                    body_activation_sum[body_name] += float(activation)

                    # CT_ON 모델: channel 5가 CT
                    if condition == 'CT_ON' and touch_size == 7:
                        ct_values = sensor_data[:, 5]  # (n_sensors, 7) → 5번 채널
                        ct_activation = np.sum(np.abs(ct_values))
                        if ct_activation > 1e-8:
                            body_ct_sum[body_name] += float(ct_activation)
                            body_ct_counts[body_name] += 1

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        print(f"  Episode {ep+1}/{n_episodes}: reward={ep_reward:.1f}, "
              f"steps={step+1}")

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

    # body별 정리
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


def print_comparison(ct_on_results, ct_off_results):
    """CT_ON vs CT_OFF 비교 테이블 출력."""
    print("\n" + "="*90)
    print("  BODY REGION별 접촉 빈도 비교 (CT_ON vs CT_OFF)")
    print("="*90)
    print(f"  CT_ON  reward: {ct_on_results['mean_reward']:.1f} ± {ct_on_results['std_reward']:.1f}")
    print(f"  CT_OFF reward: {ct_off_results['mean_reward']:.1f} ± {ct_off_results['std_reward']:.1f}")
    print()

    # 모든 body 합치기
    all_bodies = sorted(set(
        list(ct_on_results['body_regions'].keys()) +
        list(ct_off_results['body_regions'].keys())
    ))

    header = (f"{'Body Region':<22} {'Skin':<9} "
              f"{'CT_ON freq':>10} {'CT_OFF freq':>11} "
              f"{'CT_ON act':>10} {'CT_OFF act':>11} "
              f"{'CT val':>10}")
    print(header)
    print("-" * 90)

    hairy_on = 0
    hairy_off = 0
    glabrous_on = 0
    glabrous_off = 0

    for body_name in all_bodies:
        on_data = ct_on_results['body_regions'].get(body_name, {})
        off_data = ct_off_results['body_regions'].get(body_name, {})

        skin = on_data.get('skin_type', off_data.get('skin_type', '?'))
        on_freq = on_data.get('activation_freq', 0)
        off_freq = off_data.get('activation_freq', 0)
        on_act = on_data.get('mean_activation', 0)
        off_act = off_data.get('mean_activation', 0)
        ct_val = on_data.get('ct_mean', 0)

        if skin == 'hairy':
            hairy_on += on_data.get('activation_steps', 0)
            hairy_off += off_data.get('activation_steps', 0)
        else:
            glabrous_on += on_data.get('activation_steps', 0)
            glabrous_off += off_data.get('activation_steps', 0)

        print(f"{body_name:<22} {skin:<9} "
              f"{on_freq:>10.4f} {off_freq:>11.4f} "
              f"{on_act:>10.4f} {off_act:>11.4f} "
              f"{ct_val:>10.6f}")

    print("-" * 90)

    # Hairy vs Glabrous 비율
    total_on = hairy_on + glabrous_on
    total_off = hairy_off + glabrous_off

    print(f"\n  Hairy skin 접촉 비율:")
    if total_on > 0:
        print(f"    CT_ON:  {hairy_on}/{total_on} = {hairy_on/total_on*100:.1f}%")
    else:
        print(f"    CT_ON:  0/0 = N/A")
    if total_off > 0:
        print(f"    CT_OFF: {hairy_off}/{total_off} = {hairy_off/total_off*100:.1f}%")
    else:
        print(f"    CT_OFF: 0/0 = N/A")

    print(f"\n  Glabrous skin 접촉 비율:")
    if total_on > 0:
        print(f"    CT_ON:  {glabrous_on}/{total_on} = {glabrous_on/total_on*100:.1f}%")
    else:
        print(f"    CT_ON:  0/0 = N/A")
    if total_off > 0:
        print(f"    CT_OFF: {glabrous_off}/{total_off} = {glabrous_off/total_off*100:.1f}%")
    else:
        print(f"    CT_OFF: 0/0 = N/A")

    # CT 활성화 top 5
    print(f"\n  CT 활성화 Top 5 Body Regions:")
    ct_ranking = []
    for body_name, data in ct_on_results['body_regions'].items():
        ct_total = data.get('ct_total', 0)
        if ct_total > 0:
            ct_ranking.append((body_name, ct_total, data.get('ct_mean', 0),
                               data.get('skin_type', '?')))

    ct_ranking.sort(key=lambda x: x[1], reverse=True)
    for i, (name, total, mean, skin) in enumerate(ct_ranking[:5]):
        print(f"    {i+1}. {name:<22} total_CT={total:.6f}  mean_CT={mean:.6f}  ({skin})")

    if not ct_ranking:
        print("    (CT 활성화 없음)")

    return {
        'hairy_on': hairy_on, 'hairy_off': hairy_off,
        'glabrous_on': glabrous_on, 'glabrous_off': glabrous_off,
        'ct_ranking': ct_ranking,
    }


def generate_markdown(ct_on_results, ct_off_results, comparison):
    """분석 결과를 Markdown 형식으로 생성."""
    lines = []
    lines.append("---")
    lines.append("type: research")
    lines.append("created: 2026-04-09")
    lines.append("updated: 2026-04-09")
    lines.append("tags: [mimo, tactile, CT-touch, body-contact, PPO]")
    lines.append("---")
    lines.append("")
    lines.append("# Body Region별 접촉 패턴 분석 (CT_ON vs CT_OFF)")
    lines.append("")
    lines.append("## 개요")
    lines.append("")
    lines.append("학습된 PPO 모델 (1000K steps)을 사용하여 reach task에서 "
                 "body region별 접촉 패턴을 분석한 결과.")
    lines.append("")
    lines.append(f"- **CT_ON 모델**: `{os.path.basename(ct_on_results['model_path'])}`")
    lines.append(f"- **CT_OFF 모델**: `{os.path.basename(ct_off_results['model_path'])}`")
    lines.append(f"- **평가 에피소드**: {ct_on_results['n_episodes']}")
    lines.append("")

    # 보상 비교
    lines.append("## 평가 성능")
    lines.append("")
    lines.append("| 조건 | Mean Reward | Std |")
    lines.append("|------|------------|-----|")
    lines.append(f"| CT_ON | {ct_on_results['mean_reward']:.1f} | {ct_on_results['std_reward']:.1f} |")
    lines.append(f"| CT_OFF | {ct_off_results['mean_reward']:.1f} | {ct_off_results['std_reward']:.1f} |")
    lines.append("")

    # Body region별 비교 테이블
    lines.append("## Body Region별 접촉 빈도")
    lines.append("")
    lines.append("| Body Region | Skin Type | CT_ON freq | CT_OFF freq | CT_ON mean_act | CT_OFF mean_act |")
    lines.append("|-------------|-----------|-----------|------------|---------------|----------------|")

    all_bodies = sorted(set(
        list(ct_on_results['body_regions'].keys()) +
        list(ct_off_results['body_regions'].keys())
    ))

    for body_name in all_bodies:
        on_data = ct_on_results['body_regions'].get(body_name, {})
        off_data = ct_off_results['body_regions'].get(body_name, {})
        skin = on_data.get('skin_type', off_data.get('skin_type', '?'))
        on_freq = on_data.get('activation_freq', 0)
        off_freq = off_data.get('activation_freq', 0)
        on_act = on_data.get('mean_activation', 0)
        off_act = off_data.get('mean_activation', 0)
        lines.append(f"| {body_name} | {skin} | {on_freq:.4f} | {off_freq:.4f} | {on_act:.4f} | {off_act:.4f} |")

    lines.append("")

    # Hairy vs Glabrous
    lines.append("## Hairy vs Glabrous Skin 접촉 비율")
    lines.append("")

    total_on = comparison['hairy_on'] + comparison['glabrous_on']
    total_off = comparison['hairy_off'] + comparison['glabrous_off']

    lines.append("| 피부 유형 | CT_ON (steps) | CT_ON (%) | CT_OFF (steps) | CT_OFF (%) |")
    lines.append("|----------|--------------|----------|---------------|-----------|")

    if total_on > 0:
        h_on_pct = comparison['hairy_on'] / total_on * 100
        g_on_pct = comparison['glabrous_on'] / total_on * 100
    else:
        h_on_pct = g_on_pct = 0

    if total_off > 0:
        h_off_pct = comparison['hairy_off'] / total_off * 100
        g_off_pct = comparison['glabrous_off'] / total_off * 100
    else:
        h_off_pct = g_off_pct = 0

    lines.append(f"| Hairy | {comparison['hairy_on']} | {h_on_pct:.1f}% | "
                 f"{comparison['hairy_off']} | {h_off_pct:.1f}% |")
    lines.append(f"| Glabrous | {comparison['glabrous_on']} | {g_on_pct:.1f}% | "
                 f"{comparison['glabrous_off']} | {g_off_pct:.1f}% |")
    lines.append("")

    # CT Top 5
    lines.append("## CT 활성화 Top 5 Body Regions")
    lines.append("")
    if comparison['ct_ranking']:
        lines.append("| Rank | Body Region | Skin Type | Total CT | Mean CT |")
        lines.append("|------|-------------|-----------|---------|---------|")
        for i, (name, total, mean, skin) in enumerate(comparison['ct_ranking'][:5]):
            lines.append(f"| {i+1} | {name} | {skin} | {total:.6f} | {mean:.6f} |")
    else:
        lines.append("CT 활성화가 관측되지 않음.")
    lines.append("")

    # 해석
    lines.append("## 해석")
    lines.append("")
    lines.append("- CT 채널 (channel 5)은 hairy skin에서만 활성화되며, "
                 "glabrous skin (손, 발, 손가락)에서는 0")
    lines.append("- Reach task에서는 주로 팔/손 부위에서 접촉이 발생")
    lines.append("- CT_ON 모델은 multi-receptor (7ch) 관측으로 학습되어 "
                 "접촉 패턴이 CT_OFF (3ch)와 다를 수 있음")
    lines.append("")

    return "\n".join(lines)


# ── 메인 ──────────────────────────────────────────────────

def main():
    base_dir = "experiments/from_macstudio/reach"

    # 1000K 모델, seed 42 사용
    ct_on_path = os.path.join(base_dir, "model_CT_ON_1000K_seed42")
    ct_off_path = os.path.join(base_dir, "model_CT_OFF_1000K_seed42")

    # .zip 확인
    for p in [ct_on_path, ct_off_path]:
        if not os.path.exists(p + ".zip"):
            print(f"ERROR: {p}.zip not found!")
            sys.exit(1)

    # 분석 실행
    ct_off_results = analyze_model(ct_off_path, 'CT_OFF', n_episodes=10)
    ct_on_results = analyze_model(ct_on_path, 'CT_ON', n_episodes=10)

    # 비교 출력
    comparison = print_comparison(ct_on_results, ct_off_results)

    # JSON 저장
    output_json = "experiments/body_contact_analysis.json"
    with open(output_json, 'w') as f:
        json.dump({
            'ct_on': ct_on_results,
            'ct_off': ct_off_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON saved: {output_json}")

    # Markdown 저장
    md_content = generate_markdown(ct_on_results, ct_off_results, comparison)
    md_path = "/Users/uchanable_m1/obsidian/30_Research/09_TactileLM/02_IEEE_TCDS/body_contact_analysis.md"
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"  Markdown saved: {md_path}")


if __name__ == '__main__':
    main()
