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
├── experiments/                 # Experiment scripts & results
│   ├── ct_comparison.py         # CT ON vs OFF analysis
│   ├── run_scaling_experiment.py # 50K-1000K scaling
│   └── rl_results/              # Training results
├── figures/                     # Paper figures
├── videos/                      # Demo videos
└── MAC_STUDIO_SETUP.md          # Setup guide for Mac Studio
```

## Key Results

| Metric | CT OFF (3ch) | CT ON (7ch) |
|--------|-------------|-------------|
| Channels/sensor | 3 | 7 |
| CT on hairy skin | N/A | 0.050 |
| CT on glabrous skin | N/A | 0.000 |
| SA-I contribution | - | 12.7% |
| FA-I contribution | - | 17.4% |
| CT contribution | - | 0.01% |

## Citation

```bibtex
@inproceedings{yim2026ct-touch,
  title     = {Does Affective Touch Matter for Development?
               CT Afferent Modeling in the MIMo Infant Simulation},
  author    = {Yim, Youchan},
  booktitle = {Proc. IEEE International Conference on
               Development and Learning (ICDL)},
  year      = {2026}
}
```

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Number 25K24420.  
CT-Touch builds upon the [MIMo](https://github.com/trieschlab/MIMo) platform (CC BY 4.0).
