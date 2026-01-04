# 4-Hour Candle Sweep Analysis

## Purpose
This analysis includes two complementary studies:

1. **Pre-Open Sweep Analysis**: Analyzes the probability that 4-hour candle highs/lows get swept in the first 45 minutes (9:30-10:15 ET) when they haven't been swept before the NY open. Focuses on the 6am candle (6am-10am ET) with expansion to 2am and 10pm candles.

2. **Post-Formation Sweep Analysis**: Analyzes which 15-minute windows after a candle completes are most likely to sweep its high/low, conditional on that side not being swept yet. Tracks sweeps from `candle_end` until end of session (16:00 ET).

## Questions Answered

### Pre-Open Analysis
1. **If the 6am 4h candle high/low is NOT swept before 9:30am, what's the probability it gets swept in the first 45 minutes?**
2. **Same analysis for 2am (2am-6am) and 10pm (10pm-2am previous day) candles**
3. **Breakdown by sweep type:**
   - Both high AND low swept
   - Only high swept
   - Only low swept
   - Neither swept
4. **Additional metrics:** Distance from 9:30am open, time to sweep, points beyond level

### Post-Formation Analysis
1. **Which 15-minute windows after candle completion are most likely to sweep the high/low?**
2. **Does high or low tend to get swept first?**
3. **How does distance from candle_end close affect sweep probability?**
4. **What's the time-to-sweep distribution for each candle type?**

## Key Focus
The **6am candle** (6am-10am ET) is the primary focus, as it's often the most relevant candle at the NY open. The 2am and 10pm candles are also analyzed but are often already swept before 9:30am.

## How to Run
```bash
python 4h_candle_sweep_analysis.py
```

## Inputs
- `glbx-mdp3-20200927-20250926.ohlcv-1m.csv` (minute-level OHLCV data)

## Outputs

### Pre-Open Analysis Outputs

1. **4h_candle_sweep_detailed.csv** - Per-day results with:
   - Candle identification (time, start, end)
   - High/low levels
   - Pre-open sweep status
   - First 45-minute sweep status
   - Sweep category (both/high_only/low_only/neither)
   - Time to sweep and points beyond level

2. **4h_candle_sweep_summary.csv** - Summary statistics by candle time:
   - Total candles analyzed
   - Pre-open sweep counts
   - First 45-minute sweep probabilities (conditional on not swept pre-open)
   - Average distances and times to sweep

3. **4h_candle_sweep_distance_buckets.csv** - Distance-based sweep probabilities

### Post-Formation Analysis Outputs

1. **4h_candle_post_formation_windows.csv** - Window summary (main deliverable):
   - One row per candle_type + window_index
   - Window start/end times
   - Eligible counts and sweep counts per window
   - Conditional probabilities: P(swept in window | not swept at window start)
   - Tracks high, low, and either side independently

2. **4h_candle_post_formation_timing.csv** - Time-to-sweep distribution:
   - One row per day + candle_type + side
   - Minutes after candle_end to sweep (null if never swept)
   - Time bucket (0-15, 15-30, 30-60, 60-120, 120-240, 240+)
   - Distance to level and distance bucket

3. **4h_candle_post_formation_distance.csv** - Distance bucket analysis:
   - Sweep probabilities by distance bucket (0-10, 10-20, 20-40, 40-80, 80+)
   - Separate statistics for high and low sides

## Methodology

### 4-Hour Candle Alignment
Candles are aligned to specific ET start times:
- **2am candle**: 2:00 AM - 6:00 AM ET
- **6am candle**: 6:00 AM - 10:00 AM ET (PRIMARY FOCUS)
- **10am candle**: 10:00 AM - 2:00 PM ET
- **2pm candle**: 2:00 PM - 6:00 PM ET
- **6pm candle**: 6:00 PM - 10:00 PM ET
- **10pm candle**: 10:00 PM - 2:00 AM ET (spans two days)

### Sweep Detection (Strict Takeout)
- **High swept**: Price high > candle high + TICK (strict takeout, ≥1 tick)
- **Low swept**: Price low < candle low - TICK (strict takeout, ≥1 tick)
- TICK = 0.25 (NQ tick size)
- Tracks first occurrence timestamp and points beyond level

### Pre-Open Analysis Windows
- **Pre-open**: From candle start (or candle_end for completed candles) until 9:30 AM ET
- **First 45 minutes**: 9:30 AM - 10:15 AM ET

### Post-Formation Analysis Windows
- **Windows**: 15-minute windows starting at `candle_end`
- **Horizon**: 
  - For candles ending during RTH: until 16:00 ET same day
  - For overnight candles (22:00-02:00): until 16:00 ET next day
- **Eligibility**: Tracks state through windows - once a side is swept, it's marked ineligible for future windows

### Probability Calculation
- **Pre-open**: Conditional probabilities - given NOT swept before 9:30am, probability swept in first 45min
- **Post-formation**: Conditional probabilities per window - given NOT swept at window start, probability swept in this window

## Key Metrics

### Pre-Open Analysis
- **Both sides swept**: Both high and low swept in first 45min
- **Only high swept**: Only high swept, low not swept
- **Only low swept**: Only low swept, high not swept
- **Neither swept**: Neither high nor low swept in first 45min
- **Average time to sweep**: Minutes from 9:30am to sweep
- **Average distance**: Distance from 9:30am open to candle high/low

### Post-Formation Analysis
- **Window probabilities**: P(swept in window | not swept at window start) for each 15-minute window
- **Time-to-sweep distribution**: Minutes after candle_end until sweep, bucketed by time ranges
- **Distance effects**: How distance from candle_end close affects sweep probability
- **Side precedence**: Which side (high/low) tends to get swept first
- **Horizon completion**: P(both sides swept by end of horizon | neither swept at candle_end)

