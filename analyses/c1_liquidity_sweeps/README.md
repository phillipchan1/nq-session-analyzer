# C1 Liquidity Sweeps Analysis

## Purpose
Comprehensive statistical analysis of liquidity sweep behavior during the first 15-minute candle (C1, 9:30-9:45 ET) of the regular trading session. Tracks which liquidity levels get swept vs untouched, and predicts when untouched levels will be hit later in the session.

## Questions Answered

1. **Which liquidity levels are most frequently swept during C1?**
2. **How far does price typically extend beyond each level when swept?**
3. **Which untouched levels are most likely to be hit later in the session?**
4. **How long does it take for untouched levels to be hit?**
5. **What are the conditional probabilities?** (e.g., if overnight low swept first → % chance day closes up)
6. **Which confluence patterns lead to continuation vs reversal?**

## How to Run
```bash
python c1_liquidity_sweeps.py
```

## Inputs
- `glbx-mdp3-20200927-20250926.ohlcv-1m.csv` (minute-level OHLCV data)

## Outputs

### CSV Files

1. **c1_sweep_statistics.csv** - Phase A results
   - Sweep frequency by liquidity type
   - Average points beyond each level
   - Continuation probabilities

2. **untouched_levels_tracking.csv** - Phase B results
   - Probability each untouched level is hit later
   - Average time-to-hit in minutes
   - Whether price continued after touch

3. **confluence_sequences.csv** - Phase C results
   - Number of levels swept vs untouched
   - Swept combinations
   - Sequence patterns (first side swept → C2/C3 direction → daily bias)

4. **context_splits.csv** - Phase D results
   - Results segmented by daily bias, volatility regime, trend vs range, day of week

5. **summary_tables.csv** - Aggregated Table A/B/C format

## Liquidity Types Tracked

**Session-based:**
- Previous-day high/low
- Previous RTH high/low
- Overnight (Globex) high/low
- Premarket high/low
- Asian session high/low
- London session high/low + London close
- NY kill-zone high/low
- Current-day 00:00 open
- Daily midpoint

**Timeframe-based:**
- Previous 1-hour candle high/low
- Previous 4-hour candle high/low
- Current daily/weekly opens/highs/lows

**Swing levels:**
- Nearest 1-minute swing highs/lows before 9:30
- Nearest 5-minute swing highs/lows before 9:30
- Nearest 15-minute swing highs/lows before 9:30

**FVG levels:**
- Fair Value Gaps at 1min, 5min, 15min timeframes (with 50% midpoint thresholds)

**Volume Profile levels:**
- Previous day VPOC, value-area high/low
- Overnight VPOC
- Composite 5-day POC

## Analysis Phases

**Phase A:** C1 Sweep Statistics - Which levels get swept and how far
**Phase B:** Untouched Liquidity Tracking - Which untouched levels get hit later
**Phase C:** Confluence & Sequence Analysis - Patterns and combinations
**Phase D:** Context Splits - Segment by bias, volatility, trend/range, day of week

