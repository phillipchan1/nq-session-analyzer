# HTF Gap Context Analysis - First 45 Minutes - Key Findings

## Overview
This analysis examines how Higher Timeframe (HTF) gap contexts at 9:30 AM correlate with first 45 minutes behavior (9:30-10:15 ET), including directional bias, range size, volatility, and 15-minute candle patterns.

**Dataset**: 1,290 trading days (2020-2025)

## Main Findings

### 1. Most Days Have No HTF Gaps (88.1%)

**Position Distribution**:
- **No gaps at any HTF**: 1,137 days (88.1%)
- **1h gap present**: 149 days (11.6%)
- **4h gap present**: 4 days (0.3%)
- **Daily gap present**: 0 days (0.0%)

**Key Insight**: HTF gaps are relatively rare occurrences. Most trading days open without any active HTF gaps, making gap-based setups infrequent but potentially high-value when they occur.

### 2. When Price Opens Above 1h Gap → 71% Downward Bias (HIGH PROBABILITY)

**The Setup**:
- Price opens **above** a 1-hour gap at 9:30 AM
- Gap remains unfilled (not mitigated before NY open)

**First 45 Minutes Behavior**:
- **Downward bias**: **71.1%** (71% of days)
- **Upward bias**: 28.9%
- **Average net move**: -43.6 points (downward)
- **Median net move**: -32.0 points (downward)
- **Average range**: 141.8 points (larger than average)
- **Median range**: 128.3 points

**Candle Patterns**:
- **All same direction**: 22.4% (lower than average 29.1%)
- **Reversal after C1**: 55.3% (higher than average 47.4%)

**Key Insight**: This is a **high-probability reversal setup**. When price opens above a 1h gap, there's a **71% chance** the first 45 minutes moves down, with an average move of **-43.6 points**. This suggests price is "delivering off" the gap high and retracing.

### 3. When Price Opens Below 1h Gap → Moderate Upward Bias (60%)

**The Setup**:
- Price opens **below** a 1-hour gap at 9:30 AM
- Gap remains unfilled

**First 45 Minutes Behavior**:
- **Upward bias**: **60.3%** (60% of days)
- **Downward bias**: 39.7%
- **Average net move**: +34.0 points (upward)
- **Median net move**: +5.8 points (upward)
- **Average range**: 146.7 points (larger than average)
- **Median range**: 112.8 points

**Candle Patterns**:
- **All same direction**: 17.8% (lower than average)
- **Reversal after C1**: 68.5% (much higher than average 47.4%)

**Key Insight**: Opening below a 1h gap shows **moderate upward bias** (60%), but with **high reversal frequency** after C1 (68.5%). This suggests initial upward movement toward gap fill, followed by potential reversal.

### 4. No Gaps Context → Balanced Behavior

**The Setup**:
- No HTF gaps present at any timeframe (88.1% of days)

**First 45 Minutes Behavior**:
- **Upward bias**: 50.6%
- **Downward bias**: 49.4%
- **Average net move**: +0.6 points (essentially flat)
- **Median net move**: +0.8 points
- **Average range**: 123.7 points (baseline)
- **Median range**: 111.3 points

**Candle Patterns**:
- **All same direction**: 30.2% (baseline)
- **Reversal after C1**: 45.5% (baseline)

**Key Insight**: Days without HTF gaps show **balanced, random behavior** with no directional bias. This is the "normal" state of the market.

### 5. Gap Fill Probabilities Are Very Low

**Gap Fill Rates in First 45 Minutes**:
- **1h gap fill**: 1.3% (when above gap) to 1.4% (when below gap)
- **4h gap fill**: 50.0% (sample size: 4 days)

**Key Insight**: Most HTF gaps are **already filled before 9:30 AM** or remain unfilled during the first 45 minutes. This suggests:
- Gaps that survive to NY open are either very large or very recent
- Price often "delivers off" gap edges rather than filling them immediately
- Gap fill trades may need longer timeframes (beyond 45 minutes)

### 6. Gap Context Affects Range Size

**Range Comparison**:
- **No gaps**: 123.7 points average (baseline)
- **Below 1h gap**: 146.7 points average (**+18.6% larger**)
- **Above 1h gap**: 141.8 points average (**+14.7% larger**)

**Key Insight**: Days with HTF gaps show **larger opening ranges** (14-19% increase), suggesting increased volatility when gaps are present. This could be due to:
- Increased order flow around gap levels
- More aggressive moves to fill or reject gaps
- Higher uncertainty and trading activity

### 7. Reversal Patterns Differ by Gap Context

**Reversal After C1**:
- **No gaps**: 45.5% (baseline)
- **Below 1h gap**: 68.5% (**+50% higher**)
- **Above 1h gap**: 55.3% (**+22% higher**)

**Key Insight**: Days with HTF gaps show **higher reversal frequency** after C1, especially when opening below a gap (68.5%). This suggests:
- Initial move toward gap creates momentum
- Gap level acts as resistance/support causing reversals
- Fade strategies may be effective after initial gap moves

### 8. Multi-HTf Context Analysis

**1h + 4h Gaps Combined**:
- Only 4 days (0.3%) had both 1h and 4h gaps
- Sample size too small for reliable statistics

**Key Insight**: Multiple HTF gaps occurring simultaneously is extremely rare. Most gap setups involve only a single HTF gap (typically 1h).

## Trading Strategies

### Strategy 1: Gap Reversal Fade (71% Probability)

**The Setup**:
- Price opens **above** a 1h gap at 9:30 AM
- Gap remains unfilled

**Entry Criteria**:
1. Identify 1h gap pre-market
2. Confirm price opens above gap level
3. Wait for initial upward move (first 5-10 minutes)
4. Enter short on reversal signals
5. Target: Gap fill or extension zone (30-40 points)
6. Stop: 15-20 points above entry

**Expected Performance**:
- **Downward bias**: 71.1% probability
- **Average move**: -43.6 points
- **Median move**: -32.0 points
- **Risk/Reward**: 1:2+ (15 pt stop vs 32 pt target)

**Example**:
- 1h gap: 20,000 (gap high)
- NY open: 20,050 (50 points above gap)
- Initial move up to 20,075
- Enter short at 20,070
- Target: 20,040 (gap fill) or 20,030 (extension)
- Stop: 20,085
- Expected downward move: 71% probability

### Strategy 2: Gap Fill Scalp (60% Probability)

**The Setup**:
- Price opens **below** a 1h gap at 9:30 AM
- Gap remains unfilled

**Entry Criteria**:
1. Identify 1h gap pre-market
2. Confirm price opens below gap level
3. Enter long at open or on pullback
4. Target: Gap fill (gap low level)
5. Stop: 10-15 points below entry
6. Time window: First 45 minutes

**Expected Performance**:
- **Upward bias**: 60.3% probability
- **Average move**: +34.0 points
- **Median move**: +5.8 points
- **Note**: High reversal frequency (68.5%) suggests quick exits may be better

**Example**:
- 1h gap: 20,100 (gap low)
- NY open: 20,050 (50 points below gap)
- Enter long at 20,050
- Target: 20,100 (gap fill)
- Stop: 20,035
- Expected upward move: 60% probability
- **Important**: Exit quickly if reversal occurs (68.5% chance after C1)

### Strategy 3: Avoid No-Gap Days (Baseline Trading)

**The Setup**:
- No HTF gaps present (88.1% of days)

**Trading Approach**:
- Use standard trading strategies
- No directional bias (50/50)
- Normal range expectations (123.7 points average)
- Standard reversal patterns (45.5% after C1)

**Key Insight**: Most days fall into this category. Gap-based strategies are **specialty setups** that occur only 11.6% of the time.

## Risk Management

### Position Sizing

**High-Probability Setups (70%+)**:
- Standard position size
- Example: Above gap reversal (71% down)

**Moderate-Probability Setups (60-70%)**:
- Slightly reduced size
- Example: Below gap fill (60% up)

**Baseline Setups (50%)**:
- Standard position size
- Example: No-gap days (50/50)

### Stop Placement

**Above Gap Reversal**:
- Stop: 15-20 points above entry
- Average move: -43.6 points
- Median move: -32.0 points
- Use trailing stops after initial move

**Below Gap Fill**:
- Stop: 10-15 points below entry
- Average move: +34.0 points
- Median move: +5.8 points
- **Important**: High reversal frequency (68.5%) - exit quickly if reversal occurs

### Time-Based Risk Management

**First 15 Minutes (9:30-9:45)**:
- Focus on gap-based setups
- Monitor for reversal signals
- High reversal frequency after C1 (55-68%)

**15-45 Minutes (9:45-10:15)**:
- Continue monitoring gap levels
- Be aware of reversal patterns
- Gap fill probabilities are low (1-2%)

**After 45 Minutes**:
- Gap-based setups may lose edge
- Consider closing positions
- Standard trading strategies apply

## Statistical Robustness

### Sample Sizes

| Position Type | Count | Statistical Confidence |
|--------------|-------|----------------------|
| No gaps | 1,137 days | **VERY HIGH** ⭐⭐⭐⭐⭐ |
| Below 1h gap | 73 days | **MODERATE** ⭐⭐⭐ |
| Above 1h gap | 76 days | **MODERATE** ⭐⭐⭐ |
| 4h gaps | 4 days | **TOO SMALL** ⚠️ |

**Key Insight**: The **above gap reversal pattern** (71% down) has sufficient sample size (76 days) to be statistically reliable, though not as robust as patterns with 1,000+ samples.

## Key Insights

### Insight 1: Gaps Create Directional Bias

**Finding**: HTF gaps create measurable directional bias in first 45 minutes.

**Implication**:
- Above gap → 71% down (strong bias)
- Below gap → 60% up (moderate bias)
- No gaps → 50/50 (no bias)

**Trading Application**:
- Use gap context to filter trade direction
- Above gap setups favor shorts
- Below gap setups favor longs

### Insight 2: Gap Delivery vs Gap Fill

**Finding**: Gap fill probabilities are very low (1-2%), but gap "delivery" (price moving away from gap) is common.

**Implication**:
- Gaps act more like **support/resistance levels** than fill targets
- Price "delivers off" gap edges rather than filling immediately
- Gap fill trades may need longer timeframes

**Trading Application**:
- Focus on **reversal trades** off gap edges rather than fill trades
- Above gap → fade the move up (71% down)
- Below gap → ride the move up (60% up) but exit quickly (68.5% reversal)

### Insight 3: Increased Volatility with Gaps

**Finding**: Days with HTF gaps show 14-19% larger opening ranges.

**Implication**:
- Gaps create increased volatility
- More trading opportunities but also more risk
- Larger stops may be needed

**Trading Application**:
- Adjust position sizing for increased volatility
- Use wider stops (15-20 points vs 10-15 points)
- Expect larger moves but also larger drawdowns

### Insight 4: Reversal Frequency Increases with Gaps

**Finding**: Reversal after C1 increases from 45.5% (no gaps) to 55-68% (with gaps).

**Implication**:
- Gaps create momentum that reverses
- Initial move toward gap followed by reversal
- Fade strategies may be effective

**Trading Application**:
- Wait for initial move, then fade
- Above gap: Wait for up move, then short
- Below gap: Ride up move, but exit quickly (68.5% reversal)

### Insight 5: Gaps Are Rare but High-Value

**Finding**: Only 11.6% of days have HTF gaps, but they show strong directional bias.

**Implication**:
- Gap-based setups are **specialty plays**
- Not available every day
- When available, they offer strong edge (71% down for above gap)

**Trading Application**:
- Monitor for gap setups daily
- When available, increase position size due to high probability
- Don't force trades when gaps aren't present

## Practical Implementation

### Pre-Market Checklist (Before 9:30 ET)

1. **Identify HTF Gaps**:
   - Check for 1h gaps (60+ minute time gaps)
   - Check for 4h gaps (240+ minute time gaps)
   - Check for daily gaps (overnight gaps)

2. **Classify Price Position**:
   - Is price above gap? → 71% down bias
   - Is price below gap? → 60% up bias
   - Is price within gap? → Analyze position
   - No gaps? → Standard trading

3. **Prepare Trade Plan**:
   - Above gap: Prepare short setup
   - Below gap: Prepare long setup
   - No gaps: Use standard strategies

### During First 45 Minutes (9:30-10:15 ET)

1. **Above Gap Setup**:
   - Monitor for initial upward move
   - Enter short on reversal signals
   - Target: -30 to -40 points
   - Stop: 15-20 points above entry

2. **Below Gap Setup**:
   - Enter long at open or pullback
   - Target: Gap fill level
   - Stop: 10-15 points below entry
   - **Important**: Exit quickly if reversal occurs (68.5% chance)

3. **No Gap Setup**:
   - Use standard trading strategies
   - No directional bias
   - Normal range expectations

### After 45 Minutes

1. **Gap Fill Status**:
   - Check if gap was filled (unlikely - only 1-2%)
   - If not filled, gap may act as support/resistance later

2. **Position Management**:
   - Consider closing gap-based positions
   - Gap edge may diminish after 45 minutes
   - Standard trading strategies apply

## Recommendations

1. **Focus on Above Gap Reversal** (71% probability) - Highest edge
2. **Use Below Gap Fill Selectively** (60% probability) - Moderate edge, high reversal risk
3. **Monitor for Gap Setups Daily** - They're rare but high-value
4. **Adjust Position Size for Gap Context** - Higher probability = larger size
5. **Use Wider Stops for Gap Setups** - Increased volatility (14-19% larger ranges)
6. **Exit Quickly on Below Gap Setups** - High reversal frequency (68.5%)
7. **Don't Force Gap Trades** - Only 11.6% of days have gaps

## Conclusion

HTF gap contexts provide **measurable directional bias** for the first 45 minutes:

- **Above 1h gap**: **71% downward bias** (high-probability reversal setup)
- **Below 1h gap**: **60% upward bias** (moderate-probability fill setup)
- **No gaps**: **50/50** (balanced, baseline behavior)

**The Edge**: 
- Gap setups are rare (11.6% of days) but offer strong edge when available
- Above gap reversal has **71% probability** of downward move
- Below gap fill has **60% probability** but with high reversal risk

**Best Use Case**: Traders looking for high-probability directional setups in the first 45 minutes, specifically when price opens above a 1h gap (71% down bias).

**Final Rating**: ⭐⭐⭐⭐ **HIGH RECOMMENDATION** (for above gap setup)

The above gap reversal pattern offers one of the strongest directional biases found in HTF gap analysis, with 71% probability and sufficient sample size (76 days) for statistical reliability.

**Limitations**:
- Gap setups are rare (only 11.6% of days)
- Below gap setup has moderate probability (60%) with high reversal risk
- Gap fill probabilities are very low (1-2%) - gaps act more as support/resistance than fill targets

