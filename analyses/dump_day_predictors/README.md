# Dump Day Predictors Analysis

## Overview

This analysis identifies pre-market and prior-session conditions that statistically correlate with major NY session downside moves ("dump days").

## Definition of Dump Day

A "dump day" is defined as a NY session (09:30–16:00 ET) where:

- `NY_Close < NY_Open` (downward close), AND
- `(NY_Open - NY_Low) >= 1.5 * NY_ATR_20` OR `(NY_Open - NY_Low) >= 150 points`

This captures days with significant downside moves during the NY session.

## Features Computed

### Gap Features
- `gap_from_yclose`: Raw gap from yesterday's close
- `gap_from_yclose_normalized`: Gap normalized by ADR (Average Daily Range)
- `gap_pct`: Gap as percentage of previous close

### Overnight Session Features
- `overnight_range`: Range during overnight session (6 PM previous day to 9:30 AM current day)
- `overnight_range_normalized`: Overnight range normalized by ADR
- `overnight_range_vs_10d_avg`: Overnight range vs 10-day average
- `overnight_direction`: Direction of overnight move (-1 = down, 1 = up, 0 = neutral)
- `overnight_return`: Overnight close - overnight open

### Open Position Features
- `open_vs_yday_low`: NY open vs yesterday's low
- `open_below_prev_low`: Binary flag (1 if open < yesterday's low)
- `distance_to_yday_low`: Absolute distance to yesterday's low
- `distance_to_yday_low_normalized`: Distance normalized by ADR

### Early Session Features
- `first_5m_bar_return`: Return of first 5-minute bar
- `first_5m_volume`: Volume in first 5 minutes
- `first_5m_volume_ratio`: First 5m volume vs 10-day average
- `first_5m_range`: Range in first 5 minutes

### Trend Features
- `daily_trend`: Daily trend based on SMA relationship (50 vs 200)
- `four_h_trend`: 4-hour trend direction

### Structure Features
- `equal_lows_within_3d`: Binary flag if current low is within 0.1% of any low in past 3 days

### Volatility Features
- `atr_percentile_20d`: ATR percentile over 20-day window
- `volatility_percentile`: Volatility percentile (legacy)

### Event Features
- `has_red_news_830`: Binary flag for negative news at 8:30 AM
- `has_pre_session_event`: Binary flag for pre-session events
- `has_during_session_event`: Binary flag for during-session events
- `event_count`: Number of events on the day

### Calendar Features
- `day_of_week`: Day of week (0=Monday, 4=Friday)
- `day_name`: Day name (Monday, Tuesday, etc.)

## Outputs

### 1. `dump_day_features.csv`
Complete dataset with all features and labels for each trading day.

### 2. `dump_day_correlations.csv`
Correlation coefficients between each feature and the `big_dump` label, ranked by absolute correlation.

### 3. `dump_day_conditional_probs.csv`
Conditional probabilities showing dump day probability for different feature bins (quartiles for continuous features, categories for discrete features).

### 4. `dump_day_hypotheses.md`
Natural language summary of findings, including:
- Baseline dump day probability
- Top predictive features by correlation
- Conditional probability analysis
- Top 5 most significant preconditions

### 5. `dump_day_visualizations.png`
Visualizations showing:
- Gap distribution (dump days vs normal days)
- Overnight range distribution
- Dump probability by open position
- Dump probability by day of week

## How to Run

```bash
cd analyses/dump_day_predictors
python dump_day_analysis.py
```

## Requirements

- Python 3.10+
- pandas, numpy, scipy
- matplotlib, seaborn
- pytz, python-dateutil
- tqdm

## Data Requirements

- `data/glbx-mdp3-20200927-20250926.ohlcv-1m.csv`: 1-minute OHLCV data
- `data/us_high_impact_events_2020_to_2025.csv`: Event calendar data

## Analysis Methodology

1. **Data Loading**: Loads NQ futures data in chunks, filters to front-month contracts, processes weekdays only
2. **Feature Engineering**: Computes all features for each trading day
3. **Label Engineering**: Identifies dump days based on the defined criteria
4. **Rolling Features**: Computes features requiring rolling windows (ADR, percentiles, etc.)
5. **Statistical Analysis**:
   - Correlation analysis between features and dump days
   - Conditional probability analysis by feature bins
6. **Hypothesis Generation**: Identifies top predictive patterns
7. **Visualization**: Creates distribution plots

## Key Insights

The analysis will identify:
- Which pre-open conditions increase dump day probability
- Feature combinations that are most predictive
- Statistical significance of each feature
- Natural language hypotheses for trading decisions

## Notes

- The script processes data in chunks for memory efficiency
- Front-month contract is determined by maximum range in 9:30-10:15 window
- All timestamps are converted to Eastern Time
- Missing data is handled gracefully (NaN values)

