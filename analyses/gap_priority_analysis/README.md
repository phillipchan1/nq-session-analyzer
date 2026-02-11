# Gap Priority Analysis

## Purpose
When multiple gaps exist (15m, 1h, 4h, daily), which gets hit first? Analyzes gap fill priority based on distance, size, and timeframe.

## Questions Answered

1. **Which timeframe gaps get filled first?**
2. **Does distance matter more than gap size?**
3. **Do larger timeframe gaps get priority over smaller ones?**
4. **What's the typical order of gap fills?**

## How to Run
```bash
python gap_priority_analysis.py
```

## Inputs
- `glbx-mdp3-20200927-20250926.ohlcv-1m.csv` (minute-level OHLCV data)

## Outputs

1. **gap_priority_detailed.csv** - Detailed daily results with gap fill order
2. **gap_priority_summary.csv** - Summary statistics by timeframe





