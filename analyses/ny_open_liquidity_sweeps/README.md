# NY Open Liquidity Sweeps Analysis

Analyzes Asia H/L, London H/L, and Previous Day NY (RTH) H/L as liquidity targets for the first 45 minutes of NY (9:30–10:15 ET).

## Session Definitions

| Session | Time (ET) |
|---------|-----------|
| Asia | 7:00pm – 2:00am |
| London | 2:00am – 8:00am |
| NY first 45 min | 9:30am – 10:15am |

## What It Tracks

- **Availability:** % of days with ≥1 level available (not swept pre-open) within 50/100/150/200 pts
- **Distance as measuring variable:** Sweep probability by distance bucket (0–25, 25–50, 50–75, … pts)
- **Level clustering:** When 2+ levels within 50 pts ("super magnet"), hit rate for that zone
- **First 45 min range:** Correlation of sweep rate with session range (9:30–10:15 high–low)
- **Red folder days:** Sweep rate on high-impact event days vs neutral days

## Usage

```bash
python ny_open_liquidity_analysis.py
```

## Outputs

| File | Description |
|------|--------------|
| `availability_summary.csv` | % of days with ≥1 level within 50/100/150/200 pts |
| `sweep_by_distance.csv` | Sweep rate by level type and distance bucket |
| `clustering_magnet.csv` | Hit rate for confluence zones (2+ levels within 50 pts) |
| `level_clustering_2025.csv` | 2025 daily list: which levels cluster (within 50 pts) and available at NY open |
| `sweep_by_range_redfolder.csv` | Sweep rate by range quartile and red folder vs neutral |
| `daily_detail.csv` | Per-level rows with distance, range_45m, is_red_folder_day |

## Dependencies

- `data/glbx-mdp3-*.ohlcv-1m.csv` – 1m OHLCV data
- `data/us_high_impact_events_2020_to_2025.csv` – Red folder event dates
