# Friday Lower High Setup

Backtest of the claim: when Friday's RTH high fails to exceed Thursday's high, Friday's low tends to be visited on Monday.

## Usage

```bash
python friday_lower_high_analysis.py
```

## Outputs

- `friday_lower_high_summary.csv` - Overall sweep rates
- `friday_lower_high_timing.csv` - When sweeps occur (minutes from open)
- `friday_lower_high_cumulative.csv` - Cumulative sweep by window
- `friday_lower_high_detailed.csv` - Per-Friday rows
- `friday_lower_high_distance_buckets.csv` - Sweep rate by distance
- `friday_lower_high_baseline.csv` - Qualifying vs non-qualifying comparison
- `friday_lower_high_symmetric.csv` - Fri low > Thu low → Fri high visited
- `analysis.md` - Full findings and trading implications
