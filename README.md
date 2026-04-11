# Tactile-LM: CT Afferent-Level Tactile Modeling for Developmental Simulation

**CT-Touch** module for the [MIMo](https://github.com/trieschlab/MIMo) infant simulation platform.

> Does Affective Touch Matter for Development?  
> CT Afferent Modeling in the MIMo Infant Simulation  
> *ICDL 2026 (Under Review)*

**[Project Page](https://uchanable.github.io/project/ct-touch/)** | **[Paper (PDF)](#)** 

## Overview

CT-Touch extends MIMo's force-only tactile system with biologically grounded mechanoreceptor models:

- **4 receptor types**: SA-I (Merkel), FA-I (Meissner), FA-II (Pacinian), CT afferents
- **Body-site-specific distributions**: hairy skin (CT present) vs glabrous skin (no CT)
- **Developmental maturation**: A-beta myelination + CT sensitivity schedule (0-24 months)
- **7-channel output** per sensor point (vs MIMo's 3-channel force vector)

## Quick Start

```bash
# 1. Clone MIMo
git clone https://github.com/trieschlab/MIMo.git

# 2. Clone this repo
git clone https://github.com/uchanable/tactile-lm.git

# 3. Setup environment (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate
pip install 'setuptools<81'
pip install -r MIMo/requirements.txt

# 4. Run tests
PYTHONPATH=MIMo:. python -m ct_touch.test_ct_touch

# 5. Run CT ON vs OFF comparison
PYTHONPATH=MIMo:. python experiments/ct_comparison.py
```

## Repository Structure

```
tactile-lm/
├── ct_touch/                    # CT-Touch module
│   ├── ct_augmented_touch.py    # Main module (extends TrimeshTouch)
│   ├── skin_map.py              # Body-site receptor mapping
│   ├── developmental.py         # Maturation schedule
│   └── test_ct_touch.py         # Tests (6 tests)
├── som/                         # SOM + Hebbian cross-modal network
│   ├── core.py                  # Self-Organizing Map implementation
│   ├── hebbian.py               # Hebbian cross-modal learning
│   ├── preprocessor.py          # Tactile signal preprocessing
│   ├── critical_periods.py      # Developmental critical periods
│   ├── intrinsic_motivation.py  # Curiosity-driven exploration
│   └── som_wrapper.py           # MIMo integration wrapper
├── experiments/                 # Experiment scripts & results
│   ├── ct_comparison.py         # CT ON vs OFF analysis
│   ├── run_som_experiment.py    # SOM factorial experiment
│   ├── run_ablation_experiment.py # Ablation study
│   ├── run_developmental_experiment.py # Developmental persona
│   ├── analyze_som_results.py   # SOM results analysis
│   ├── analyze_body_contacts.py # Body contact analysis
│   ├── generate_figures.py      # Paper figure generation
│   └── rl_results/              # Training results
├── figures/                     # Paper figures
├── videos/                      # Demo videos
└── MAC_STUDIO_SETUP.md          # Setup guide for Mac Studio
```

## Key Results

### Sensor Validation (Random Motor Babbling)

| Metric | CT OFF (3ch) | CT ON (7ch) |
|--------|-------------|-------------|
| Channels/sensor | 3 | 7 |
| CT on hairy skin | N/A | 0.050 |
| CT on glabrous skin | N/A | 0.000 |
| SA-I contribution | — | 12.7% |
| FA-I contribution | — | 17.4% |
| CT contribution | — | 0.01% |

### RL Experiments (PPO, n=30 seeds, 4 training horizons)

**Reach Task** — CT ON shows consistent advantage:

| Metric | Value |
|--------|-------|
| Effect size (Hedges' g, 1M steps) | 0.37 (small) |
| Meta-analysis (4 time steps combined) | θ = 9.69, **p = .016** |
| Combined Bayes Factor | **BF₁₀ = 453 (Extreme evidence)** |
| CT ON win rate (1M) | 67% (20/30) |
| Touch activation ratio (CT ON / CT OFF) | **0.49** (half the touch for higher reward) |

**Self-body Task** — No difference (as expected):

| Metric | Value |
|--------|-------|
| Combined Bayes Factor | BF₁₀ = 0.111 (moderate evidence for H₀) |
| Interpretation | Binary reward is orthogonal to contact quality |

### SOM + Hebbian Cross-Modal Architecture (Ongoing)

Post-ICDL implementation of the neural architecture proposed in the paper but not originally implemented:

- **Discriminative SOM**: 13 body parts × 6 channels = 78D → body-topographic map (SA-I, FA-I, FA-II, normal force)
- **Affective SOM**: 13 body parts × 1 CT channel = 13D → CT-specific map (hairy skin only)
- **Proprioceptive SOM**: joint angles + velocities → motor context map
- **Hebbian cross-modal binding**: 3 pairs (Disc↔Aff, Disc↔Proprio, Aff↔Proprio), co-activation strengthening, weight decay (96% cross-modal prediction accuracy)
- **Critical period scheduler**: high SOM plasticity early, gradual decay
- **Intrinsic motivation**: curiosity-driven exploration via SOM prediction error
- Implemented as Gym observation wrapper (zero MIMo modification, 2.2 ms/step overhead)

2×2 factorial experiment (PPO/SOM+PPO × CT OFF/CT ON, n=30 seeds) in progress.
Hypothesis: SOM body-topographic representation unlocks CT information that raw PPO cannot exploit.

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Number 25K24420.  
CT-Touch builds upon the [MIMo](https://github.com/trieschlab/MIMo) platform (CC BY 4.0).
