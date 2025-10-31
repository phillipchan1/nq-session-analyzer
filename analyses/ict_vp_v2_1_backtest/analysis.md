# ICT Value Profile V2.1 Backtest - Key Findings

## Overview
This analysis backtests an ICT (Inner Circle Trader) style strategy combining liquidity sweeps with prior-day value area confluence, momentum filters, and volume confirmation. The strategy uses laddered exits (TP1 and TP2) to maximize profits while protecting capital.

## Main Findings

### 1. Overall Performance: LOW FREQUENCY, POSITIVE EXPECTANCY

**Critical Result**: The strategy generated only **39 trades over 5 years** (2020-2025), averaging **7.8 trades per year**. Despite a **43.59% win rate**, the strategy is **profitable** due to excellent risk/reward management.

| Metric | Value |
|--------|-------|
| Total Trades | 39 |
| Win Rate | **43.59%** (17 wins) |
| Loss Rate | 56.41% (22 losses) |
| Average R per Trade | **+0.18R** |
| Total Points | **+191.65 points** |
| Avg Points per Trade | **+4.91 points** |
| **Expectancy** | **+0.177R** ✅ |

### 2. Risk/Reward Excellence

Despite a sub-50% win rate, the strategy is profitable due to superior risk/reward management:

| Outcome | Avg R-Multiple | Avg Points | Count |
|---------|---------------|------------|-------|
| **Winners** | **+1.47R** | **+40.02 pts** | 17 |
| **Losers** | **-0.82R** | **-22.22 pts** | 22 |

**Key Insight**: The strategy achieves an **average win of 1.47R** while keeping average losses to only **-0.82R**. This asymmetric payoff profile creates positive expectancy:

**Expectancy Calculation:**
```
Expectancy = (Win Rate × Avg Win) + (Loss Rate × Avg Loss)
           = (0.4359 × 1.47R) + (0.5641 × -0.82R)
           = 0.641R - 0.463R
           = +0.177R per trade
```

This means you make **17.7% of your risk** on average per trade, which is excellent for a discretionary/selective strategy.

### 3. Laddered Exit Performance

The strategy uses two profit targets (TP1 and TP2):

| Metric | Value |
|--------|-------|
| **TP1 Hit Rate** | **53.85%** |
| Full TP (TP2) Hit Rate | 43.59% |

**Key Insight**: TP1 (first partial exit) is hit **53.85% of the time**, slightly better than coin-flip odds. This provides:
1. **Early profit taking** to secure gains
2. **Risk reduction** by reducing position size
3. **Runner potential** for the remaining position to reach TP2

The 10-percentage-point gap between TP1 and full TP suggests that the strategy successfully captures partial profits even when trades don't go all the way.

### 4. Directional Bias: Almost Exclusively Short

| Direction | Trades | Win Rate | Avg Points | Total Points |
|-----------|--------|----------|------------|--------------|
| **Short** | 37 (94.9%) | 40.54% | +3.36 | +124.40 |
| Long | 2 (5.1%) | 100.00% | +33.62 | +67.25 |

**Key Insight**: The strategy is **overwhelmingly short-biased** (95% of trades). This makes sense for ICT-style setups looking for:
- Liquidity sweeps above highs followed by reversals
- Value area fades at market tops
- Momentum exhaustion signals

The two long trades both won, but with only 2 samples, this isn't statistically significant.

### 5. Year-Over-Year Performance (HIGH VARIABILITY)

| Year | Trades | Win Rate | Total Points | Notes |
|------|--------|----------|--------------|-------|
| 2020 | 4 | 25.00% | -41.50 | **Worst year** |
| 2021 | 6 | 50.00% | +29.53 | Breakeven |
| **2022** | 10 | **60.00%** | **+228.75** | **Best year** 🏆 |
| **2023** | 2 | **100.00%** | **+83.75** | Perfect (small sample) |
| 2024 | 8 | 50.00% | +30.12 | Decent |
| **2025** | 9 | **11.11%** | **-139.00** | **Current year struggling** ⚠️ |

**Critical Observations:**

1. **2022 was exceptional**: 60% win rate, +228.75 points
   - This was a volatile year with clear trends and reversals
   - ICT concepts worked particularly well in 2022's market structure

2. **2023 was perfect but rare**: Only 2 trades, both winners
   - Extreme selectivity paid off
   - Very low frequency

3. **2025 is concerning**: Only 1 win out of 9 trades (11.11%)
   - This year is dragging down overall performance significantly
   - -139 points YTD
   - Market conditions may have changed or strategy needs adaptation

4. **Overall variability is high**: Win rates range from 11% to 100% year-over-year
   - Low trade frequency amplifies variance
   - Few trades per year mean luck/randomness plays a bigger role

### 6. Trade Frequency and Selectivity

**7.8 trades per year** = 1 trade every 6-7 weeks on average

| Pros of Low Frequency | Cons of Low Frequency |
|----------------------|----------------------|
| ✅ Highly selective setups | ❌ High variance between years |
| ✅ Less commission/slippage | ❌ Hard to evaluate strategy performance |
| ✅ Minimal screen time | ❌ One bad year can wipe out years of gains |
| ✅ Focus on best opportunities | ❌ Requires patience and discipline |

**Key Insight**: This is a "sniper" strategy, not a "machine gun" strategy. You're waiting for perfect ICT setups with all filters aligned. The upside is high-quality trades; the downside is insufficient sample size for robust statistical confidence.

### 7. Loss Characteristics

With 22 losses out of 39 trades, let's examine the loss profile:

- **Average loss**: -22.22 points (-0.82R)
- **Losses are smaller than wins**: Good for long-term profitability
- **Most losses are -1R**: Proper stop loss discipline
- **Some breakeven exits**: A few trades show 0.0R or -0.0, suggesting breakeven exits when TP1 is hit but position is closed early

### 8. Statistical Significance Warning

⚠️ **IMPORTANT**: With only **39 trades over 5 years**, this strategy has **insufficient sample size** for high statistical confidence.

- **Minimum recommended**: 100+ trades for reliable statistics
- **Current sample**: 39 trades
- **Confidence level**: LOW to MODERATE

**What this means:**
- The positive expectancy could be luck
- One bad year (like 2025) has outsized impact
- Win rate could vary widely with more data
- True performance may differ from observed 43.59%

### 9. Strengths of the Strategy

✅ **Positive expectancy** (+0.177R) despite sub-50% win rate
✅ **Excellent risk management** (1.47R avg win vs -0.82R avg loss)
✅ **Laddered exits** capture partial profits effectively
✅ **Highly selective** - only takes best setups
✅ **Strong performance in trending/volatile years** (2022, 2023)

### 10. Weaknesses of the Strategy

❌ **Very low trade frequency** (7.8 trades/year) leads to high variance
❌ **2025 performance is terrible** (11% win rate, -139 points)
❌ **Insufficient sample size** (39 trades) for statistical confidence
❌ **Almost exclusively short** - misses long opportunities
❌ **Year-to-year variability is extreme** (11% to 100% win rates)
❌ **Recent performance suggests possible drift** or changing market conditions

## Trading Implications

### Should You Trade This Strategy?

**Cautiously YES**, with heavy caveats:

1. ✅ **Use it as a "bonus" strategy**: Don't rely on it as your primary income source due to low frequency
2. ✅ **Good risk/reward profile**: The 1.47R wins vs -0.82R losses are attractive
3. ⚠️ **Be prepared for losing streaks**: 2025's 8 losses in 9 trades shows it can go cold
4. ⚠️ **Requires discretion**: ICT setups need experienced eye, not purely mechanical
5. ❌ **Don't expect consistent income**: 1 trade every 6-7 weeks won't pay the bills

### Risk Management Rules

Given the strategy's characteristics:

1. **Position sizing**: Risk only **0.5-1% per trade** (due to low frequency and variance)
2. **Stop loss discipline**: Follow the -1R stops religiously
3. **Laddered exits**: Always scale out at TP1 (hit 53.85% of time)
4. **Patience required**: Don't force trades - wait for perfect setups
5. **Hedge with other strategies**: Combine with higher-frequency strategies for smoother equity curve

### When to Use This Strategy

**Best conditions:**
- High volatility environments (like 2022)
- Clear market structure with defined highs/lows
- Strong trending markets with exhaustion signals
- When prior-day value area is well-defined

**Avoid when:**
- Low volatility / tight ranges
- Unclear market structure
- After extended losing streaks (like current 2025)

## Recommendations for Improvement

1. **Increase sample size**: Run backtest further back (pre-2020) to get 100+ trades
2. **Add long setups**: Capture upside opportunities, not just shorts
3. **Investigate 2025 underperformance**: What changed? Market regime shift?
4. **Add filters for market regime**: Perhaps only trade during high-volatility periods
5. **Consider wider stops in low-volatility regimes**: -0.82R avg loss suggests some stops are too tight
6. **Document discretionary criteria**: What makes a setup "valid"? Codify the rules
7. **Monte Carlo simulation**: With 39 trades, run Monte Carlo to understand true expectancy range

## Conclusion

The ICT Value Profile V2.1 strategy is a **low-frequency, high-selectivity approach** with:

- **Positive expectancy** (+0.177R per trade)
- **Excellent risk/reward** (1.47R wins vs 0.82R losses)  
- **Profitable over 5 years** (+191.65 points)
- **But extremely variable** year-to-year performance
- **And concerning recent drawdown** (2025 YTD)

**Final Rating**: 🟡 **CAUTIOUSLY RECOMMENDED** 

**Use as supplementary/opportunistic strategy, not primary income source. The 2025 underperformance requires investigation before committing significant capital.**

### Best Use Case:
A discretionary trader with other income sources who can patiently wait 6-7 weeks between trades and wants to capitalize on high-quality ICT setups with excellent risk/reward profiles.

**Not suitable for:**
- Traders needing consistent income
- Pure systematic/mechanical traders (requires discretion)
- Traders unable to handle 8-9 trade losing streaks
- Those needing frequent trading activity

