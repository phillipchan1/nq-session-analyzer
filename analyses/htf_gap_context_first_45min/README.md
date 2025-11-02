# HTF Gap Context Analysis - First 45 Minutes

## Purpose
Analyzes how Higher Timeframe (HTF) gap contexts at 9:30 AM correlate with:
- First 45 minutes behavior (direction, range, volatility)
- 15-minute candle patterns
- Gap fill probabilities
- Opening range behavior

## Questions Answered

1. **Where is price relative to HTF gaps at 9:30 AM?**
   - Within gap (near top, middle, or bottom)
   - Above gap (delivering off gap high)
   - Below gap (delivering off gap low)
   - Between gaps
   - No gaps present

2. **How do HTF gap contexts correlate with first 45 min behavior?**
   - Directional bias (up vs down)
   - Range size and volatility
   - Gap fill probabilities
   - Opening range behavior

3. **What are the 15-minute candle patterns for each context?**
   - Candle sequences (C1/C2/C3 directions)
   - Candle size patterns
   - Reversal frequencies

4. **Which patterns have 70%+ probability?**
   - High-probability directional biases
   - High-probability gap fill scenarios
   - Reliable opening range behaviors

## HTF Gaps Analyzed

- **1-hour gaps**: Gaps of 60+ minutes duration
- **4-hour gaps**: Gaps of 240+ minutes duration
- **Daily gaps**: Overnight gaps between previous day close and current day open

## How to Run

```bash
python htf_gap_context_analysis.py
```

## Inputs
- `glbx-mdp3-20200927-20250926.ohlcv-1m.csv` (minute-level OHLCV data)

## Outputs

1. **htf_gap_context_detailed.csv** - Detailed daily results
   - Price position relative to each HTF gap
   - First 45 min behavior metrics
   - Gap fill status
   - Candle patterns

2. **htf_gap_context_summary.csv** - Aggregate statistics by position
   - Counts and percentages for each position type
   - Average first 45 min behavior
   - Gap fill probabilities
   - Candle pattern frequencies

3. **htf_gap_context_high_prob_patterns.csv** - High-probability patterns (70%+)
   - Directional biases
   - Gap fill scenarios
   - Alignment with daily bias

## Position Classifications

### Single HTF Context
- `within_gap_near_top` - Price within gap, closer to top
- `within_gap_near_bottom` - Price within gap, closer to bottom
- `within_gap_middle` - Price in middle of gap
- `above_gap` - Price above gap (delivering off gap high)
- `below_gap` - Price below gap (delivering off gap low)
- `between_gaps_closer_to_above` - Between gaps, closer to above gap
- `between_gaps_closer_to_below` - Between gaps, closer to below gap
- `no_gap` - No gap present at this timeframe

### Multi-HTf Context
- `all_three` - 1h, 4h, and daily gaps all present
- `1h_4h` - 1h and 4h gaps present
- `1h_daily` - 1h and daily gaps present
- `4h_daily` - 4h and daily gaps present
- `1h_only` - Only 1h gap present
- `4h_only` - Only 4h gap present
- `daily_only` - Only daily gap present
- `no_gaps` - No gaps at any timeframe

