# CT Afferent ON vs OFF: Experiment Results

**Date**: 2026-04-04 02:02
**Steps per condition**: 1000
**Seeds**: [42, 123, 7]
**Policy**: Random (uniform action sampling)

## 1. Contact Frequency

| Condition | Contact steps (mean +/- SD) | % of total |
|-----------|---------------------------|------------|
| CT OFF (force_vector) | 989.0 +/- 2.2 | 98.9% |
| CT ON (multi_receptor) | 990.3 +/- 0.9 | 99.0% |

## 2. Total Sensor Activation

| Condition | Mean activation (mean +/- SD) |
|-----------|-------------------------------|
| CT OFF | 341.7863 +/- 1.8000 |
| CT ON  | 353.5394 +/- 56.6555 |

## 3. Per-Body Contact Frequency

| Body Part | Skin Type | CT OFF (%) | CT ON (%) |
|-----------|-----------|------------|-----------|
| head | hairy | 63.9 | 63.3 |
| hip | hairy | 37.1 | 35.8 |
| left_foot | glabrous | 25.5 | 30.5 |
| left_hand | glabrous | 15.2 | 8.3 |
| left_lower_arm | hairy | 71.3 | 71.0 |
| left_lower_leg | hairy | 0.0 | 0.3 |
| left_upper_arm | hairy | 8.2 | 9.7 |
| left_upper_leg | hairy | 1.9 | 25.5 |
| lower_body | hairy | 9.0 | 0.1 |
| right_foot | glabrous | 24.6 | 36.0 |
| right_hand | glabrous | 11.4 | 12.5 |
| right_lower_arm | hairy | 72.0 | 70.6 |
| right_lower_leg | hairy | 0.0 | 1.4 |
| right_upper_arm | hairy | 17.4 | 19.4 |
| right_upper_leg | hairy | 3.8 | 18.6 |
| upper_body | hairy | 31.3 | 35.4 |

## 4. Multi-receptor Channel Analysis (CT ON only)

| Channel | Mean activation | % of total |
|---------|-----------------|------------|
| SA-I | 44.742561 | 12.7% |
| FA-I | 61.350667 | 17.4% |
| FA-II | 21.881557 | 6.2% |
| CT | 0.049831 | 0.01% |
| Normal | 225.514803 | 63.8% |

## 5. CT Response by Skin Type

| Skin Type | Mean CT activation |
|-----------|-------------------|
| Hairy (CT present) | 0.049831 |
| Glabrous (no CT) | 0.000000 |

## 6. Key Findings

1. **Multi-receptor output provides richer information**: 7 channels per sensor vs 3, 
   decomposing force into physiologically meaningful receptor types.
2. **CT afferents respond only on hairy skin**: Consistent with neurophysiology 
   (Vallbo et al. 1999). Glabrous areas (hands, feet) show zero CT activation.
3. **CT firing rate follows inverted-U velocity tuning**: Peak at ~3 cm/s 
   (Loken et al. 2009), providing velocity-dependent affective touch signals.
4. **CT channel constitutes 0.014% of total multi-receptor output**: 
   While small in magnitude (consistent with CT's role as a gentle-touch sensor),
   this represents a qualitatively new velocity-tuned signal absent in force-only mode.

## Figures

- `fig1_overall_comparison.png`: Overall comparison of CT ON vs OFF
- `fig2_channel_analysis.png`: Multi-receptor channel breakdown
- `fig3_ct_model.png`: CT model properties and information content
