# 6am 4H Candle Post-Formation Sweep Analysis

## Overview

At 10:00 AM ET the 6am 4H candle (6:00-10:00) is fully formed. This analysis asks: **if the 10:00 close is at least 20 points from one side of the candle, how often does price sweep (strict takeout, >= 1 NQ tick beyond) at least one side in the 10:00-10:15 macro?**

The study covers **1,388 qualifying days** from September 2020 through February 2026. A day qualifies when `max(dist_to_high, dist_to_low) >= 20` at 10:00 AM.

---

## Key Findings

### 1. Overall Sweep Probabilities

| Metric | Count | % |
|--------|-------|---|
| **Qualifying Days** | 1,388 | — |
| **Either Side Swept** | 800 | **57.6%** |
| High Swept | 415 | 29.9% |
| Low Swept | 393 | 28.3% |
| Both Swept | 8 | 0.6% |
| Neither Swept | 588 | 42.4% |

**Key Insight**: More than half the time, at least one side of the fully-formed 6am 4H candle gets swept in just 15 minutes (10:00-10:15).

---

### 2. Distance Buckets — Proximity Drives Sweep Probability

The closer price is to a side at 10:00, the more likely that side gets swept by 10:15.

| Side | Distance | Days | Swept | % |
|------|----------|------|-------|---|
| **High** | 20-40 pts | 277 | 101 | **36.5%** |
| High | 40-60 | 181 | 35 | 19.3% |
| High | 60-80 | 138 | 15 | 10.9% |
| High | 80-100 | 113 | 5 | 4.4% |
| High | 100+ | 357 | 2 | 0.6% |
| **Low** | 20-40 pts | 257 | 124 | **48.2%** |
| Low | 40-60 | 206 | 48 | 23.3% |
| Low | 60-80 | 166 | 22 | 13.3% |
| Low | 80-100 | 149 | 11 | 7.4% |
| Low | 100+ | 381 | 8 | 2.1% |

**Key Insight**: The 20-40 pt bucket is the sweet spot — 36.5% for highs, 48.2% for lows. Beyond 60 pts, sweep probability drops sharply.

---

### 3. Close Position in Candle — Strongest Predictor

`close_position = (candle_close - candle_low) / (candle_high - candle_low)` — where the 10:00 close sits within the 4H range (0 = at low, 1 = at high).

| Close Position | Days | High Swept % | Low Swept % | Either % |
|----------------|------|--------------|-------------|----------|
| **0-0.25 (near low)** | 346 | 0.3% | **71.7%** | **72.0%** |
| 0.25-0.50 | 302 | 6.6% | 32.8% | 39.1% |
| 0.50-0.75 | 298 | 23.2% | 11.7% | 34.6% |
| **0.75-1.0 (near high)** | 442 | **73.5%** | 2.5% | **74.7%** |

**Key Insight**: When the close is in the top quartile (0.75-1.0), there's a **74.7%** chance at least one side gets swept — and it's almost always the high (73.5%). Same pattern inverted for the bottom quartile: close near low → 72% chance the low gets swept. Mid-range closes (0.25-0.75) drop to 35-39%.

---

### 4. Last 15m Candle (9:45-10:00) — Direction and Body Strength

| Factor | Days | Either Swept % |
|--------|------|----------------|
| **Bullish** | 717 | **59.1%** |
| Bearish | 671 | 56.0% |

| Body Strength | Days | Either Swept % |
|---------------|------|----------------|
| 0-25% (doji-like) | 356 | 45.2% |
| 25-50% | 416 | 52.2% |
| 50-75% | 400 | **67.5%** |
| 75-100% (strong body) | 216 | **70.4%** |

**Key Insight**: Strong-bodied 9:45-10:00 candles predict sweeps. Dojis (indecision) = lower sweep rate (~45%). Strong body (50%+) = 67-70% sweep probability.

---

### 5. When the Extreme Was Formed — Recency Matters

| Factor | Days | Either Swept % |
|--------|------|----------------|
| **High set in RTH (9:30-10:00)** | 727 | **63.4%** |
| High set pre-market | 661 | 51.3% |
| Low set in RTH | 725 | 57.2% |
| Low set pre-market | 663 | 58.1% |

**Key Insight**: When the high was set during RTH (fresh momentum), it's ~12 percentage points more likely to be swept.

---

### 6. RTH Range as % of Candle Range — Momentum Confirmation

| RTH Range % | Days | Either Swept % |
|-------------|------|----------------|
| 0-25% | 7 | 28.6% |
| 25-50% | 124 | 40.3% |
| 50-75% | 507 | 54.0% |
| **75-100%** | 750 | **63.2%** |

**Key Insight**: When the first 30 minutes of RTH (9:30-10:00) drove most of the candle's range, there's strong directional momentum that carries through into 10:00-10:15.

---

### 7. Other Factors

| Factor | Finding |
|--------|---------|
| **10:00 AM event** | 61.2% sweep rate vs 55.8% without — modest ~5 pt lift |
| **8:30 AM event** | Minimal differentiation |
| **Unswept session level near boundary** | Mixed; unswept near high = 53.5%, near low = 55.3% |
| **Candle range** | Wider ranges (160+) slightly lower sweep rate (55%) |
| **Pre-RTH range** | 80-120 pts = 65% sweep; 120+ = 54% (exhaustion?) |
| **Day of week** | Minimal differentiation (55-60%) |
| **VIXY prev close** | Not strongly predictive |

---

## Trading Implications

### Best Combination Signal

**Close position + last 15m body strength + RTH range %**

When all three align:
- Close in top/bottom quartile (0.75+ or 0-0.25)
- 9:45-10:00 candle has strong body (50%+)
- RTH drove 75%+ of the candle range

→ **~70-75% sweep probability** in just 15 minutes.

### Continuation Trade Setup

1. At 10:00 AM, identify the 6am 4H candle high/low
2. Check: close position (near high or near low?), last 15m body (strong?), RTH range %
3. If close near high + bullish last 15m + high RTH % → bias toward high sweep; consider continuation long toward the high
4. If close near low + bearish last 15m + high RTH % → bias toward low sweep; consider continuation short toward the low
5. Time window: 10:00-10:15

---

## Output Files

| File | Description |
|------|-------------|
| `4h_6am_sweep_summary.csv` | Overall sweep counts and percentages |
| `4h_6am_sweep_distance_buckets.csv` | Sweep rate by distance bucket (high/low) |
| `4h_6am_sweep_detailed.csv` | Per-day rows with all 34 factor columns for backtesting |
| `4h_6am_sweep_factor_analysis.csv` | Sweep rates bucketed by each predictive factor |

Filter `4h_6am_sweep_detailed.csv` for `either_swept == 1` to get the 800 days for manual backtesting on FXReplay/Tradezella.
