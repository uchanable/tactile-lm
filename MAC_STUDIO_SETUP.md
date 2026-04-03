# Mac Studio 원클릭 세팅 가이드

## 한번에 실행 (복붙용)

```bash
# ---- 1. 작업 디렉토리 생성 ----
mkdir -p ~/tactile-lm-workspace && cd ~/tactile-lm-workspace

# ---- 2. 레포 클론 ----
git clone https://github.com/trieschlab/MIMo.git
git clone https://github.com/uchanable/tactile-lm.git

# ---- 3. Python 3.12 설치 (없으면) ----
brew install python@3.12

# ---- 4. venv 생성 + 의존성 설치 ----
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv312
source .venv312/bin/activate
pip install 'setuptools<81'
pip install -r MIMo/requirements.txt

# ---- 5. 동작 확인 ----
PYTHONPATH=MIMo:tactile-lm python -c "
from mimoEnv.envs.dummy import MIMoDummyEnv
env = MIMoDummyEnv(); env.reset()
print('MIMo OK! Sensors:', env.observation_space['touch'].shape)
env.close()
"

# ---- 6. CT-Touch 테스트 ----
PYTHONPATH=MIMo:tactile-lm python -m ct_touch.test_ct_touch

# ---- 7. 스케일링 실험 실행 (tmux 권장) ----
tmux new -s ct_experiment
PYTHONPATH=MIMo:tactile-lm python tactile-lm/experiments/run_scaling_experiment.py
# Ctrl+B, D로 detach → 나중에 tmux attach -t ct_experiment
```

## 디렉토리 구조 (세팅 후)

```
~/tactile-lm-workspace/
├── MIMo/                  ← trieschlab/MIMo (영아 시뮬레이터)
├── tactile-lm/            ← uchanable/tactile-lm (CT-Touch 모듈)
│   ├── ct_touch/          ← 핵심 코드
│   ├── experiments/       ← 실험 스크립트 + 결과
│   └── ...
└── .venv312/              ← Python 환경
```

## 실험 내용

`run_scaling_experiment.py`는 다음을 실행:
- **4 step sizes**: 50K, 100K, 500K, 1M
- **2 conditions**: CT OFF (force_vector) / CT ON (multi_receptor)  
- **3 seeds**: 42, 123, 7
- **합계: 24 runs**

## 예상 소요 시간 (M1 Max 64GB)

| Steps | 1 run | × 2조건 × 3 seeds | 누적 |
|-------|-------|-------------------|------|
| 50K   | ~1min | ~6min | 6min |
| 100K  | ~2min | ~12min | 18min |
| 500K  | ~8min | ~48min | 66min |
| 1M    | ~15min | ~90min | 156min |
| **합계** | | | **~2.5시간** |

## 결과 확인

```bash
cat tactile-lm/experiments/rl_results/scaling/summary.json
```

## 결과를 맥북으로 가져오기 (git)

```bash
# 맥 스튜디오에서
cd ~/tactile-lm-workspace/tactile-lm
git add -A && git commit -m "Add Mac Studio scaling results" && git push

# 맥북에서
cd /path/to/tactile-lm && git pull
```
