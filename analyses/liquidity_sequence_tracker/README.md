# Liquidity Sequence Tracker

## Purpose
Tracks the order of liquidity hits/sweeps during the first 45 minutes (9:30-10:15 ET), analyzing sequences like "SSL sweep → failed rally → 930 low → London high".

## Questions Answered

1. **What sequences occur most frequently?**
2. **What are the transition probabilities between liquidity areas?**
3. **How long does it take to move between liquidity areas?**
4. **What are the most common paths price takes?**

## How to Run
```bash
python liquidity_sequence_analysis.py
```

## Inputs
- `glbx-mdp3-20200927-20260221.ohlcv-1m.csv` (minute-level OHLCV data)

## Outputs

1. **liquidity_sequence_detailed.csv** - Detailed daily results with full sequences
2. **liquidity_sequence_frequency.csv** - Most common sequences ranked by frequency
3. **liquidity_transition_matrix.csv** - Probability matrix showing transitions between liquidity areas





