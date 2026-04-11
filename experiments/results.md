# CT Afferent ON vs OFF: Experiment Results

**Date**: 2026-04-04 11:42
**Steps per condition**: 1000
**Seeds**: [42, 123, 7]
**Policy**: Random (uniform action sampling)

## 1. Contact Frequency

| Condition | Contact steps (mean +/- SD) | % of total |
|-----------|---------------------------|------------|
| CT OFF (force_vector) | 987.7 +/- 0.9 | 98.8% |
| CT ON (multi_receptor) | 991.3 +/- 1.2 | 99.1% |

## 2. Total Sensor Activation

| Condition | Mean activation (mean +/- SD) |
|-----------|-------------------------------|
| CT OFF | 357.6529 +/- 21.2205 |
| CT ON  | 297.4780 +/- 20.9603 |

## 3. Per-Body Contact Frequency

| Body Part | Skin Type | CT OFF (%) | CT ON (%) |
|-----------|-----------|------------|-----------|
| head | hairy | 67.5 | 69.4 |
| hip | hairy | 39.2 | 32.4 |
| left_foot | glabrous | 23.3 | 24.7 |
| left_hand | glabrous | 13.3 | 25.7 |
| left_lower_arm | hairy | 70.5 | 75.8 |
| left_lower_leg | hairy | 0.3 | 0.2 |
| left_upper_arm | hairy | 8.1 | 8.4 |
| left_upper_leg | hairy | 3.1 | 10.6 |
| lower_body | hairy | 4.7 | 9.6 |
| right_foot | glabrous | 24.1 | 21.9 |
| right_hand | glabrous | 10.2 | 11.8 |
| right_lower_arm | hairy | 73.0 | 71.5 |
| right_lower_leg | hairy | 0.0 | 0.2 |
| right_upper_arm | hairy | 30.8 | 18.8 |
| right_upper_leg | hairy | 11.5 | 7.3 |
| upper_body | hairy | 30.1 | 22.1 |

## 4. Multi-receptor Channel Analysis (CT ON only)

| Channel | Mean activation | % of total |
|---------|-----------------|------------|
| SA-I | 42.160572 | 14.2% |
| FA-I | 59.184425 | 19.9% |
| FA-II | 21.013332 | 7.1% |
| CT | 0.044726 | 0.02% |
| Normal | 175.074914 | 58.9% |

## 5. CT Response by Skin Type

| Skin Type | Mean CT activation |
|-----------|-------------------|
| Hairy (CT present) | 0.044726 |
| Glabrous (no CT) | 0.000000 |

## 6. Key Findings

1. **Multi-receptor output provides richer information**: 7 channels per sensor vs 3, 
   decomposing force into physiologically meaningful receptor types.
2. **CT afferents respond only on hairy skin**: Consistent with neurophysiology 
   (Vallbo et al. 1999). Glabrous areas (hands, feet) show zero CT activation.
3. **CT firing rate follows inverted-U velocity tuning**: Peak at ~3 cm/s 
   (Loken et al. 2009), providing velocity-dependent affective touch signals.
4. **CT channel constitutes 0.015% of total multi-receptor output**: 
   While small in magnitude (consistent with CT's role as a gentle-touch sensor),
   this represents a qualitatively new velocity-tuned signal absent in force-only mode.

## Figures

- `fig1_overall_comparison.png`: Overall comparison of CT ON vs OFF
- `fig2_channel_analysis.png`: Multi-receptor channel breakdown
- `fig3_ct_model.png`: CT model properties and information content
