# 45-Minute Opening Move Predictive Analysis (NQ)

## Sanity Check
Unique dates: 1289
BigMove45 baseline: 3.7%
ExtremeMove45 baseline: 2.7%
Dump45 baseline: 5.0%
Pump45 baseline: 2.6%

---

## Top Predictors – BigMove45

- **h4_below_lower_kc**: Correlation = 0.160 (increases BigMove45 probability)
- **h1_kc_pos_prev**: Correlation = -0.148 (decreases BigMove45 probability)
- **h4_kc_pos_prev**: Correlation = -0.140 (decreases BigMove45 probability)
- **overnight_range_vs_10d_avg**: Correlation = 0.128 (increases BigMove45 probability)
- **h1_rsi_14_prev**: Correlation = -0.110 (decreases BigMove45 probability)
- **overnight_range**: Correlation = 0.107 (increases BigMove45 probability)
- **gap_from_yclose**: Correlation = -0.093 (decreases BigMove45 probability)
- **rolling_5d_return**: Correlation = -0.090 (decreases BigMove45 probability)
- **overnight_open**: Correlation = 0.088 (increases BigMove45 probability)
- **overnight_high**: Correlation = 0.088 (increases BigMove45 probability)

### Conditional Probabilities
- **h4_below_lower_kc** = Yes: 11.3% BigMove45 probability (vs 3.7% baseline, n=177)
- **h1_kc_pos_prev** = Q1: 9.0% BigMove45 probability (vs 3.7% baseline, n=322)
- **overnight_range_vs_10d_avg** = Q4: 8.7% BigMove45 probability (vs 3.7% baseline, n=323)
- **h4_kc_pos_prev** = Q1: 8.1% BigMove45 probability (vs 3.7% baseline, n=320)
- **h1_rsi_14_prev** = Q1: 7.5% BigMove45 probability (vs 3.7% baseline, n=322)

---

## Top Predictors – Dump45

- **h1_kc_pos_prev**: Correlation = -0.289 (decreases Dump45 probability)
- **h4_kc_pos_prev**: Correlation = -0.224 (decreases Dump45 probability)
- **h1_rsi_14_prev**: Correlation = -0.223 (decreases Dump45 probability)
- **h4_below_lower_kc**: Correlation = 0.220 (increases Dump45 probability)
- **h4_rsi_14_prev**: Correlation = -0.159 (decreases Dump45 probability)
- **overnight_range_vs_10d_avg**: Correlation = 0.139 (increases Dump45 probability)
- **h4_above_upper_kc**: Correlation = -0.121 (decreases Dump45 probability)
- **overnight_open**: Correlation = 0.105 (increases Dump45 probability)
- **overnight_high**: Correlation = 0.105 (increases Dump45 probability)
- **preopen_price**: Correlation = 0.104 (increases Dump45 probability)

### Conditional Probabilities
- **h4_below_lower_kc** = Yes: 16.9% Dump45 probability (vs 5.0% baseline, n=177)
- **h1_kc_pos_prev** = Q1: 15.8% Dump45 probability (vs 5.0% baseline, n=322)
- **h1_rsi_14_prev** = Q1: 12.7% Dump45 probability (vs 5.0% baseline, n=322)
- **h4_kc_pos_prev** = Q1: 11.9% Dump45 probability (vs 5.0% baseline, n=320)
- **overnight_range_vs_10d_avg** = Q4: 11.5% Dump45 probability (vs 5.0% baseline, n=323)

---

## Top Predictors – Pump45

- **overnight_range**: Correlation = 0.142 (increases Pump45 probability)
- **overnight_range_vs_10d_avg**: Correlation = 0.138 (increases Pump45 probability)
- **h1_kc_pos_prev**: Correlation = 0.124 (increases Pump45 probability)
- **gap_from_yclose**: Correlation = -0.116 (decreases Pump45 probability)
- **open_vs_yday_low**: Correlation = -0.113 (decreases Pump45 probability)
- **h1_rsi_14_prev**: Correlation = 0.100 (increases Pump45 probability)
- **week_range_pos**: Correlation = -0.098 (decreases Pump45 probability)
- **prev_day_return_pts**: Correlation = -0.097 (decreases Pump45 probability)
- **prev_day_return_pct**: Correlation = -0.092 (decreases Pump45 probability)
- **rolling_5d_return**: Correlation = -0.085 (decreases Pump45 probability)

### Conditional Probabilities
- **overnight_range_vs_10d_avg** = Q4: 6.2% Pump45 probability (vs 2.6% baseline, n=323)
- **week_range_pos** = Q1: 5.3% Pump45 probability (vs 2.6% baseline, n=321)
- **open_below_prev_low** = Yes: 5.2% Pump45 probability (vs 2.6% baseline, n=211)
- **distance_to_yday_low_normalized** = Q4: 4.7% Pump45 probability (vs 2.6% baseline, n=320)
- **h1_kc_pos_prev** = Q4: 4.7% Pump45 probability (vs 2.6% baseline, n=322)

---

## Observations

- When **h4_below_lower_kc** = Yes, Dump45 probability rises from 5.0% → 16.9%.
- When **overnight_range_vs_10d_avg** = Q4, Pump45 probability rises from 2.6% → 6.2%.
- ExtremeMove45 (≥ 200 pts) occurs 2.7% of days; 54% of those overlap with BigMove45.

---

## Notes

All features computed using data available before 09:30 ET.
Labels measured over 09:30 – 10:15 ET only.
