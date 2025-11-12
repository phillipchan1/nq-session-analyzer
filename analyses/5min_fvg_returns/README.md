# 5-Minute Fair Value Gap Returns Analysis

## Question
"If a 5-minute fair value gap is created in the first 45 minutes, what are the chances that price will return to it within the first 45 minutes?"

## Methodology

### Fair Value Gap Definition (3-Candle Pattern)
A 5-minute Fair Value Gap (FVG) is a **3-candle pattern** defined as:
- **Bullish FVG**: Candle 3's low > Candle 1's high (gap between candles 1 and 3, with candle 2 in between)
- **Bearish FVG**: Candle 3's high < Candle 1's low
- The gap must **not fill** in the next 2 candles (10 minutes) to be considered a valid FVG
- The gap zone is the imbalance area between Candle 1 and Candle 3

### Time Windows
- **Creation window**: 9:30 AM - 10:15 AM ET (first 45 minutes)
- **Exclusion**: FVGs created at 10:10 AM or later are excluded (since trading session ends at 10:15 AM)
- **Return check**: Price must touch the gap zone within 9:30 AM - 10:15 AM ET

### Data Processing
1. Load 1-minute OHLCV data for NQ futures
2. Resample to 5-minute candles using pandas `resample("5min")`
3. Detect FVGs created in the first 45 minutes (before 10:10 AM cutoff)
4. For each FVG, check if price returns to (touches) the gap zone within the first 45 minutes
5. Track results per day

### Return Definition
Price "returns" to an FVG if any 1-minute candle's price range overlaps with the gap zone:
- Gap zone: `[fvg_low - epsilon, fvg_high + epsilon]`
- Touch condition: `candle["low"] <= fvg_high + epsilon AND candle["high"] >= fvg_low - epsilon`
- Epsilon: 0.1 points (handles floating point precision)

## Output Files

### `fvg_5min_daily_summary.csv`
Day-by-day summary with:
- Date
- Symbol
- Open price
- First 45min high/low
- Number of FVGs created
- Number of FVGs that returned
- Return rate per day

### `fvg_5min_detailed.csv`
Detailed tracking of each FVG with:
- Date and symbol
- FVG type (bullish/bearish)
- FVG bounds (low/high)
- Gap size
- Creation time
- Whether it returned
- First touch time (if returned)
- Minutes after creation when first touched

## Key Findings

Analyzed 1,290 trading days (2020-2025), found **1,168 Fair Value Gaps** across 719 days:

### Overall Statistics
- **Total FVGs created**: 1,168 
- **Days with FVGs**: 719 out of 1,290 (55.7%)
- **Overall return rate**: **58.48%**
- **Average time to return**: 5.47 minutes after creation
- **Bullish FVGs**: 621 total, 56.52% return rate
- **Bearish FVGs**: 547 total, 60.69% return rate

### Return Rate by Gap Size

| Gap Size (pts) | Count | Returned | Return Rate |
|----------------|-------|----------|-------------|
| **50-100** | 42 | 33 | **78.57%** ⭐ |
| **30-50** | 133 | 87 | **65.41%** |
| **20-30** | 203 | 132 | **65.02%** |
| **15-20** | 139 | 87 | **62.59%** |
| 10-15 | 166 | 90 | 54.22% |
| 5-10 | 227 | 130 | 57.27% |
| 0-5 | 256 | 124 | 48.44% |
| 100+ | 2 | 0 | 0.00% |

### Key Trading Insights

1. **Larger FVGs are more reliable**: FVGs of 30+ points have a **65-78% probability** of being touched within the first 45 minutes
2. **Quick fills**: When FVGs do return, they're touched within an average of 5.47 minutes
3. **Bearish FVGs slightly more reliable**: 60.69% vs 56.52% for bullish FVGs
4. **Small FVGs less reliable**: Gaps under 10 points have only 48-57% return rates

## Usage

```bash
python analyses/5min_fvg_returns/fvg_5min_returns_analysis.py
```

## Trading Applications

Based on these findings:

1. **High-probability entries**: Target 30-50 point FVGs (65% return rate) for entries
2. **Wait for the touch**: Average 5.47 minutes to return means patience pays off
3. **Larger gaps are magnets**: 50-100 point FVGs have 78.57% return probability
4. **Avoid tiny gaps**: Sub-10 point FVGs are coin flips (48-57% return rate)
5. **Consider bearish FVGs slightly more**: 60.69% vs 56.52% for bullish

## Further Research

Consider exploring:
1. **Return depth**: How far into the FVG does price penetrate?
2. **Multiple touches**: Do FVGs that return once get touched again later?
3. **Context matters**: Does trend/bias affect return rates?
4. **Timeframe comparison**: 15-min and 30-min FVG return rates
5. **Session timing**: Do FVGs created at different times show different return rates?

