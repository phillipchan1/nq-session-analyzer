# VIX Correlation and NY Session Range Prediction Analysis

## Overview
This analysis examines how VIX (VIXY ETF) correlates with NY session range for both the first 45 minutes (9:30-10:15 AM ET) and full session (9:30 AM-4:00 PM ET). It covers 1,209 trading days from December 2020 to December 2025.

## Key Findings

### 1. Overall Statistics

| Metric | Mean | Median | Std Dev | Min | Max |
|--------|------|--------|---------|-----|-----|
| **VIX at 9:30 AM** | 21.36 | 16.58 | 13.52 | 5.42 | 85.43 |
| **Overnight Range (NQ)** | 144.9 pts | 124.8 pts | 89.1 pts | 21.8 pts | 841.5 pts |
| **First 45m Range** | 129.3 pts | 115.3 pts | 72.0 pts | 33.8 pts | 1,448.8 pts |
| **Full Session Range** | 250.8 pts | 218.5 pts | 145.6 pts | 58.5 pts | 2,184.5 pts |

### 2. Correlation Analysis

#### Strongest Correlations

**First 45-Minute Range:**
- **Overnight Range**: r=0.564 (Pearson), ρ=0.567 (Spearman) - **STRONGEST PREDICTOR**
- **VIX Overnight Range**: r=0.468 (Pearson), ρ=0.412 (Spearman)
- **VIX × Overnight Range Interaction**: r=0.557 (Pearson), ρ=0.506 (Spearman)
- **VIX at 9:30 AM**: r=0.305 (Pearson), ρ=0.206 (Spearman) - Moderate

**Full Session Range:**
- **Overnight Range**: r=0.561 (Pearson), ρ=0.514 (Spearman) - **STRONGEST PREDICTOR**
- **VIX × Overnight Range Interaction**: r=0.583 (Pearson), ρ=0.427 (Spearman)
- **VIX Overnight Range**: r=0.534 (Pearson), ρ=0.369 (Spearman)
- **VIX at 9:30 AM**: r=0.293 (Pearson), ρ=0.141 (Spearman) - Moderate

**Key Insight**: Overnight range is the **strongest predictor** of both first 45-minute and full session ranges, with correlations around 0.56-0.57. VIX adds predictive power, especially when combined with overnight range.

### 3. Binning Analysis

#### By VIX Level

| VIX Level | Days | Avg First 45m Range | Avg Full Session Range | vs Baseline |
|-----------|------|---------------------|------------------------|-------------|
| **<15** (Low Vol) | 514 | 114.9 pts | 227.7 pts | Baseline |
| **15-20** | 248 | 139.5 pts | 275.5 pts | +21% / +21% |
| **20-25** | 176 | 120.4 pts | 226.6 pts | +5% / -0.5% |
| **25-30** | 49 | 98.8 pts | 176.5 pts | -14% / -22% |
| **30-50** | 158 | 147.6 pts | 260.9 pts | +29% / +15% |
| **50+** (High Vol) | 64 | 208.3 pts | 440.4 pts | **+81% / +93%** |

**Key Findings:**
- VIX <15: Lower volatility days with average ranges
- VIX 15-20: Slightly elevated ranges (+21%)
- VIX 50+: **Extreme volatility** - ranges are **81% higher** in first 45m and **93% higher** for full session
- Interestingly, VIX 25-30 shows lower ranges than expected (possibly mean reversion days)

#### By Overnight Range

| Overnight Range | Days | Avg First 45m Range | Avg Full Session Range | vs Baseline |
|-----------------|------|---------------------|------------------------|-------------|
| **<50 pts** | 41 | 75.1 pts | 147.1 pts | Baseline |
| **50-100 pts** | 368 | 96.6 pts | 195.3 pts | +29% / +33% |
| **100-150 pts** | 386 | 123.8 pts | 229.0 pts | +65% / +56% |
| **150-200 pts** | 202 | 145.6 pts | 287.8 pts | +94% / +96% |
| **200+ pts** | 212 | 191.0 pts | 371.9 pts | **+154% / +153%** |

**Key Finding**: Overnight range is a **very strong predictor**. When overnight range exceeds 200 points, first 45-minute ranges average 191 points (2.5x baseline) and full session ranges average 372 points (2.5x baseline).

### 4. Regression Models

#### Model 1: First 45-Minute Range Prediction
```
first_45m_range = 50.58 + 0.84 × vix_930_open + 0.42 × overnight_range
R² = 0.341, RMSE = 58.4 points
```

**Interpretation:**
- Each 1-point increase in VIX at 9:30 AM → +0.84 points in first 45m range
- Each 1-point increase in overnight range → +0.42 points in first 45m range
- Model explains **34% of variance** in first 45-minute ranges

#### Model 2: Full Session Range Prediction
```
full_session_range = 94.09 + 1.57 × vix_930_open + 0.85 × overnight_range
R² = 0.335, RMSE = 118.7 points
```

**Interpretation:**
- Each 1-point increase in VIX at 9:30 AM → +1.57 points in full session range
- Each 1-point increase in overnight range → +0.85 points in full session range
- Model explains **33% of variance** in full session ranges

**Model Performance**: Both models achieve R² around 0.33-0.34, meaning VIX and overnight range together explain about one-third of the variance in session ranges. The remaining variance is likely due to other factors (events, news, market structure, etc.).

### 5. Predictive Insights

#### High Range Days (Predictors)
1. **Overnight Range >200 points**: Expect first 45m range ~191 pts, full session ~372 pts
2. **VIX >50**: Expect first 45m range ~208 pts, full session ~440 pts
3. **Combined**: VIX >50 AND Overnight Range >200 → Expect extreme ranges

#### Low Range Days (Predictors)
1. **Overnight Range <50 points**: Expect first 45m range ~75 pts, full session ~147 pts
2. **VIX <15**: Expect first 45m range ~115 pts, full session ~228 pts
3. **Combined**: VIX <15 AND Overnight Range <50 → Expect quiet days

### 6. Trading Implications

1. **Overnight Range is the Primary Predictor**: Monitor overnight range (6 PM prev day to 9:30 AM) - it's the strongest indicator of session volatility.

2. **VIX Adds Context**: VIX level at 9:30 AM provides additional context, especially at extremes (VIX >50 or VIX <15).

3. **Combined Signal**: The interaction of VIX × Overnight Range shows strong correlation (r=0.557-0.583), suggesting these factors work together.

4. **Prediction Accuracy**: Models explain ~33-34% of variance, meaning there's still significant uncertainty. Use these as **probabilistic guides**, not certain predictions.

5. **Risk Management**: On days with VIX >50 and overnight range >200, expect extreme volatility and adjust position sizing accordingly.

### 7. Limitations

- Models explain only ~33% of variance - many other factors influence ranges
- VIXY ETF may not perfectly track VIX index (though correlation should be high)
- Sample size varies by bin (some bins have <50 days)
- Market regime changes over time not accounted for
- No adjustment for market events/news (though these may be reflected in VIX)

## Conclusion

**Overnight range is the strongest predictor** of NY session range, with correlations around 0.56-0.57. VIX adds predictive power, especially at extreme levels (VIX >50). Combined, these two factors explain about one-third of the variance in session ranges, providing useful probabilistic guidance for trading decisions.

The analysis shows clear patterns:
- High overnight range (>200 pts) → High session ranges (2.5x baseline)
- High VIX (>50) → High session ranges (1.8-2.0x baseline)
- Low overnight range (<50 pts) → Low session ranges (baseline)
- Low VIX (<15) → Moderate ranges (baseline)

Use these insights to set expectations and manage risk, but remember that ~67% of variance remains unexplained by these factors alone.


