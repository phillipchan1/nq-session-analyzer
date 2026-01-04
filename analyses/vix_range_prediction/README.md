# VIX Correlation and NY Session Range Prediction Analysis

## Overview

This analysis examines how VIX (VIXY ETF) correlates with NY session range for both:
- **First 45 minutes** (9:30-10:15 AM ET) - Active trading period
- **Full session** (9:30 AM-4:00 PM ET) - Complete trading day

The analysis builds predictive models using VIX metrics and overnight range to predict upcoming session ranges.

## Data Sources

- **NQ Futures**: `data/glbx-mdp3-20200927-20250926.ohlcv-1m.csv` (1-minute bars)
- **VIXY ETF**: `data/xnas-itch-20201203-20251203.ohlcv-1m.csv` (1-minute bars)

**Analysis Period**: December 2020 to December 2025 (1,209 trading days)

## Files

### Script
- `vix_range_analysis.py` - Main analysis script

### Outputs
- `vix_daily_metrics.csv` - Per-day metrics with all VIX values, overnight range, and session ranges
- `vix_correlations.csv` - Correlation analysis (Pearson and Spearman) between VIX metrics and ranges
- `vix_bin_analysis.csv` - Binning analysis grouped by VIX levels and overnight range bins
- `vix_regression_results.csv` - Regression model results with coefficients and performance metrics
- `analysis.md` - Summary report with key findings and trading implications

## VIX Metrics Calculated

All metrics are calculated **before** the NY session starts (available for prediction):

1. **vix_930_open**: VIX at 9:30 AM ET (or closest bar)
2. **vix_overnight_close**: VIX close at end of overnight session (just before 9:30 AM)
3. **vix_prev_day_close**: VIX close from previous trading day (4:00 PM ET)
4. **vix_overnight_range**: VIX high - low during overnight session (6 PM prev day to 9:30 AM)
5. **vix_overnight_change**: VIX change during overnight (close - open)

## Range Metrics Calculated

1. **overnight_range**: NQ range from 6 PM previous day to 9:30 AM current day
2. **first_45m_range**: NQ range from 9:30-10:15 AM ET
3. **full_session_range**: NQ range from 9:30 AM-4:00 PM ET

## Key Findings

### Strongest Predictors

1. **Overnight Range** (r=0.56-0.57): Strongest predictor of both first 45m and full session ranges
2. **VIX × Overnight Range Interaction** (r=0.56-0.58): Combined signal shows strong correlation
3. **VIX Overnight Range** (r=0.47-0.53): VIX volatility during overnight session
4. **VIX at 9:30 AM** (r=0.29-0.30): Moderate correlation

### Predictive Models

**First 45-Minute Range:**
```
first_45m_range = 50.58 + 0.84 × vix_930_open + 0.42 × overnight_range
R² = 0.341, RMSE = 58.4 points
```

**Full Session Range:**
```
full_session_range = 94.09 + 1.57 × vix_930_open + 0.85 × overnight_range
R² = 0.335, RMSE = 118.7 points
```

### Binning Insights

- **VIX >50**: First 45m range ~208 pts (+81% vs baseline), Full session ~440 pts (+93%)
- **Overnight Range >200 pts**: First 45m range ~191 pts (+154% vs baseline), Full session ~372 pts (+153%)
- **VIX <15**: First 45m range ~115 pts (baseline), Full session ~228 pts (baseline)
- **Overnight Range <50 pts**: First 45m range ~75 pts (baseline), Full session ~147 pts (baseline)

## Usage

### Running the Analysis

```bash
python analyses/vix_range_prediction/vix_range_analysis.py
```

### Output Files

1. **vix_daily_metrics.csv**: Use for detailed day-by-day analysis
2. **vix_correlations.csv**: Review correlation strengths and statistical significance
3. **vix_bin_analysis.csv**: See average ranges by VIX/overnight range bins
4. **vix_regression_results.csv**: Model coefficients and performance metrics
5. **analysis.md**: Read summary findings and trading implications

## Methodology

1. **Data Loading**: Chunked reading of large CSV files for memory efficiency
2. **Front-Month Filtering**: NQ data filtered to front-month contracts using liquidity proxy (max range in 9:30-10:15 window)
3. **Timezone Handling**: All timestamps converted to US/Eastern timezone
4. **Weekday Filtering**: Only trading days (Monday-Friday) included
5. **Missing Data**: Days with missing VIX or NQ data excluded from analysis
6. **Statistical Tests**: Pearson (linear) and Spearman (rank) correlations calculated
7. **Regression**: Multiple linear regression using numpy least squares

## Dependencies

- pandas
- numpy
- scipy
- pytz

## Notes

- VIXY ETF is used as proxy for VIX index (should track closely)
- Models explain ~33-34% of variance - other factors (events, news, market structure) account for remaining variance
- Use predictions as **probabilistic guides**, not certain forecasts
- Consider market regime changes over time period (2020-2025)

## Trading Implications

1. **Monitor Overnight Range**: Strongest predictor - check 6 PM prev day to 9:30 AM range
2. **Check VIX at Open**: Provides additional context, especially at extremes
3. **Combined Signal**: High VIX (>50) + High Overnight Range (>200) → Expect extreme volatility
4. **Risk Management**: Adjust position sizing based on predicted ranges
5. **Expectation Setting**: Use binning analysis to set realistic range expectations

See `analysis.md` for detailed findings and insights.


