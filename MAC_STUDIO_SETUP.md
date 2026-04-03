# Mac Studio 셋업 가이드 (CT-Touch 실험)

## 1. 프로젝트 복사

```bash
# 맥북 프로에서 맥 스튜디오로 복사 (Tailscale IP 사용)
scp -r /Users/uchanable_m1/obsidian/40_Ideas/project/mimo-tactile/ \
  user@100.x.x.x:/path/to/mimo-tactile/
```

또는 Git으로 관리하는 경우:
```bash
cd /path/to/mimo-tactile
git init && git add -A && git commit -m "initial"
# 맥 스튜디오에서 clone
```

## 2. Python 환경 세팅

```bash
# Homebrew Python 3.12 설치
brew install python@3.12

# venv 생성
cd /path/to/mimo-tactile
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv312

# 활성화
source .venv312/bin/activate

# setuptools 다운그레이드 (pkg_resources 호환)
pip install 'setuptools<81'

# 의존성 설치
pip install -r MIMo/requirements.txt
```

## 3. 동작 확인

```bash
source .venv312/bin/activate
PYTHONPATH=MIMo:. python -c "
from mimoEnv.envs.dummy import MIMoDummyEnv
env = MIMoDummyEnv()
env.reset()
print('MIMo OK! Sensors:', env.observation_space['touch'].shape)
env.close()
"
```

## 4. CT-Touch 테스트

```bash
PYTHONPATH=MIMo:. python -m ct_touch.test_ct_touch
```

6개 테스트 전부 PASS 되면 OK.

## 5. 스케일링 실험 실행

```bash
# 전체 실험 (50K/100K/500K/1000K × CT ON/OFF × 3 seeds = 24 runs)
# 예상 시간: M1 Max 64GB에서 약 3-4시간
PYTHONPATH=MIMo:. python experiments/run_scaling_experiment.py

# 또는 특정 조건만 (tmux 사용 권장)
tmux new -s ct_experiment
PYTHONPATH=MIMo:. python experiments/run_scaling_experiment.py
# Ctrl+B, D로 detach → 나중에 tmux attach -t ct_experiment
```

## 6. 결과 확인

```bash
cat experiments/rl_results/scaling/summary.json
```

## 7. 결과를 맥북으로 복사

```bash
scp -r /path/to/mimo-tactile/experiments/rl_results/ \
  uchanable_m1@100.x.x.x:/Users/uchanable_m1/obsidian/40_Ideas/project/mimo-tactile/experiments/
```

## 예상 소요 시간 (M1 Max 64GB)

| Steps | CT OFF | CT ON | 합계 (×3 seeds) |
|-------|--------|-------|----------------|
| 50K   | ~1min  | ~1min | ~6min          |
| 100K  | ~2min  | ~3min | ~15min         |
| 500K  | ~10min | ~13min| ~69min         |
| 1000K | ~20min | ~26min| ~138min        |
| **합계** | | | **~4시간**     |
