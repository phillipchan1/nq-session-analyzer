# 15-Minute Candle Behavior Analysis
This analysis examines 15-minute candle patterns in NQ RTH sessions from 2020-2025.
**Dataset**: 1,290 trading days, 33,022 total 15-minute candles

---

# 🎯 FIRST 3 CANDLES DEEP DIVE (9:30-10:15)
**Focus: Trading Window Analysis**

## First 3 Candles: Individual Statistics

### Candle 1 (09:30:00)

- **Mean range**: 80.6 points
- **Median range**: 71.8 points
- **10th percentile**: 42.0 points
- **90th percentile**: 127.2 points
- **% Bullish**: 52.2%
- **% Bearish**: 47.8%
- **Mean volume**: 41,307
- **Median volume**: 41,318
- **Mean wick-to-body ratio**: 6.87

### Candle 2 (09:45:00)

- **Mean range**: 65.8 points
- **Median range**: 58.0 points
- **10th percentile**: 31.8 points
- **90th percentile**: 109.1 points
- **% Bullish**: 51.6%
- **% Bearish**: 48.4%
- **Mean volume**: 29,141
- **Median volume**: 28,780
- **Mean wick-to-body ratio**: 5.68

### Candle 3 (10:00:00)

- **Mean range**: 66.9 points
- **Median range**: 57.8 points
- **10th percentile**: 31.2 points
- **90th percentile**: 112.8 points
- **% Bullish**: 50.1%
- **% Bearish**: 49.9%
- **Mean volume**: 28,467
- **Median volume**: 27,748
- **Mean wick-to-body ratio**: 6.91

## First 3 Candles: Key Patterns

- **All Same Direction**: 29.1% (375 occurrences)
- **Reversal After C1**: 47.4% (611 occurrences)
- **Reversal After C2**: 45.0% (580 occurrences)
- **Opening Range High Is Session High**: 33.9% (437 occurrences)
- **Opening Range Low Is Session Low**: 40.0% (516 occurrences)
- **First 3 Aligned With Close**: 70.6% (911 occurrences)
- **Mean Sweeps In First 3**: 0.49 (1290 days)
- **Mean Opening Range Size**: 125.90 (1290 days)
- **Median Opening Range Size**: 112.75 (1290 days)
- **C2 Smaller Than C1**: 44.5% (574 occurrences)
- **C3 Smaller Than C2**: 26.3% (339 occurrences)

## First 3 Candles: Sequence Probabilities

Most common directional sequences:

- **bullish-bullish-bullish**: 15.6% (201 days)
- **bearish-bearish-bearish**: 13.5% (174 days)
- **bullish-bearish-bearish**: 13.2% (170 days)
- **bearish-bullish-bullish**: 12.8% (165 days)
- **bullish-bullish-bearish**: 12.6% (163 days)
- **bearish-bearish-bullish**: 10.9% (141 days)
- **bullish-bearish-bullish**: 10.8% (139 days)
- **bearish-bullish-bearish**: 10.6% (137 days)

## First 3 Candles: Trading Insights

### Opening Range (First 3 Candles High-Low)

- **Mean opening range**: 125.9 points
- **Median opening range**: 112.8 points
- **Opening range high = session high**: 33.9% of days
- **Opening range low = session low**: 40.0% of days
- **Avg distance to session high**: 56.2 points
- **Avg distance to session low**: 61.9 points

### Directional Patterns

- **All 3 candles same direction**: 29.1% of days
- **Reversal after candle 1**: 47.4% of days
- **Reversal after candle 2**: 45.0% of days

### Candle Size Patterns

- **Candle 2 smaller than Candle 1** (75% threshold): 44.5% of days
- **Candle 3 smaller than Candle 2** (75% threshold): 26.3% of days
- **Mean candle 1 size**: 80.6 points
- **Mean candle 2 size**: 65.8 points
- **Mean candle 3 size**: 66.9 points

### Liquidity Sweeps in First 3 Candles

- **Average sweeps**: 0.49 per day
- **Days with 0 sweeps**: 56.9%
- **Days with 2+ sweeps**: 6.4%

### Predictive Power

- **First 3 direction aligns with daily close**: 70.6% of days
- **Mean net move in first 3 candles**: 0.1 points
- **Median net move**: 0.0 points

---

# COMPREHENSIVE ANALYSIS (All Sessions)

## 1. Candle Structure & Volatility

### Overall Statistics
- **Median candle size**: 37.0 points
- **Mean candle size**: 45.1 points
- **25.0% exceed 1.5× median** (55.5 points)
- **12.9% exceed 2× median** (74.0 points)

### Session Block Comparison

**Opening Drive**: Mean=73.2, Median=65.0 points

**Mid Morning**: Mean=53.3, Median=46.0 points

**Midday**: Mean=38.4, Median=32.5 points

**Afternoon**: Mean=39.5, Median=31.8 points

## 2-7. Additional Analysis Modules

See CSV files for detailed analysis of:
- **Sequence Behavior**: Continuation/reversal patterns
- **Liquidity & High/Low**: Sweeps and timing
- **Session Phase Comparison**: Volatility by time block
- **Bias & Context**: Daily bias correlations
- **Reversion & Mean**: Mean reversion patterns
- **Volume & Microstructure**: Volume analysis and FVGs

## Key Takeaways for 9:30-10:15 Trading Window

1. **First candle is largest**: Average 80.6 points vs 65.8 for candle 2 and 66.9 for candle 3
2. **Opening range forms session extremes**: ~33.9% of days have opening range high/low equal to session high/low
3. **Reversals are common**: 47.4% see reversal after candle 1, 45.0% after candle 2
4. **Liquidity sweeps occur frequently**: Average 0.49 sweeps in first 3 candles
5. **Size typically decreases**: 44.5% of days see candle 2 smaller than candle 1
6. **Directional alignment**: 70.6% of days see first 3 candles align with daily close direction
