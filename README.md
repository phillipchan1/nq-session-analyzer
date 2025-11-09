## NQ Session Analyzer

Research toolkit to analyze NASDAQ futures (NQ) behavior around the New York open, focusing on the first 45 minutes (9:30–10:15 ET) and related liquidity structures, macro events, gaps, reversals, and indicator-driven edges.

### What this repo aims to answer
- Which 15-minute windows contribute the most range during RTH?
- How do macro event days (timing/type) impact the first 45-minute range and volume?
- How often do London swing levels, hourly highs/lows, and time gaps get hit shortly after the open, conditional on distance?
- Do clustered liquidity levels (confluence) increase hit probabilities?
- Which level types produce the strongest, sustained reversals upon first touch (and which combos amplify that)?
- Which indicator deciles show directional edge and favorable excursions for a given TP/SL/horizon?
- How does an ICT-style sweep + value-area confluence with momentum/volume filters perform with laddered exits?

## Repository structure
- `analyses/`
  - `ny_open_45m_range_vs_events/`: First-45m range/volume vs macro event timing/type and day of week
  - `confluence_zones_hits/`: Confluence zones (clustered levels) and early hit rates
  - `gaps_15m_plus_hits/`: ≥15-minute time gaps that survive to NY open and their hit probabilities
  - `liquidity_levels_hit_prob/`: London swing H/L, hourly H/L, unmitigated gaps – hit probability by distance
  - `london_swing_hits_at_open/`: Hit rates for available London swing levels after the NY open
  - `range_extremes_and_15m_slots/`: Extreme-day correlates and 15m slot contributions across RTH
  - `indicator_bins_vs_outcomes/`: Indicator deciles vs outcomes (win-rate, excursions) for given TP/SL/horizon
  - `ema_stretch_high_precision/`: EMA-stretch extremes for short-horizon, fixed TP/SL mean reversion
  - `reversal_power_across_levels/`: Per-touch reversal metrics and combos leaderboard
  - `ict_vp_v2_1_backtest/`: ICT + prior value area backtest with momentum/volume filters and laddered exits
- `data/`: Centralized raw inputs (committed here), e.g. `glbx-mdp3-*.ohlcv-1m.csv`/`.zst`, `us_high_impact_events_2020_to_2025.csv`, `symbology.*`
- `tools/`: Utilities (`decompress.py`, `inspect_data.py`, `debug_day_final.py`)
- `run_all.sh`: Convenience runner (select scripts; heavy ones commented by default)

All scripts read inputs from `data/` and write outputs into their own folder under `analyses/`.

## Quickstart
### 1) Environment
Use Python 3.10+ (tested with 3.13). Install the common scientific stack:
```bash
pip install pandas numpy scipy pyarrow pytz python-dateutil tqdm zstandard
```
Note: You already have a `venv/` here; you can also activate and reuse it.

### 2) Data
Place your 1-minute OHLCV CSV or ZST files in `data/`. This repo expects filenames like:
- `glbx-mdp3-20200927-20250926.ohlcv-1m.csv`
- `glbx-mdp3-20240926-20250925.ohlcv-1m.csv.zst`

Macro calendar (for event-based analyses):
- `data/us_high_impact_events_2020_to_2025.csv`

If you have a compressed ZST file and want a plain CSV for some scripts, use:
```bash
python tools/decompress.py
```

### 3) Run analyses
Option A — curated common runs:
```bash
bash run_all.sh
```

Option B — run a specific analysis. Examples:
```bash
# First-45m range and event correlations
python analyses/ny_open_45m_range_vs_events/backtest.py
python analyses/ny_open_45m_range_vs_events/analyze_volume.py

# London swing availability & hit rates
python analyses/london_swing_hits_at_open/london_hl.py

# Extreme-day correlates and 15m slot stats
python analyses/range_extremes_and_15m_slots/range_and_intervals_analysis.py

# Heavier (uncomment as needed)
# python analyses/confluence_zones_hits/confluence_analysis.py
# python analyses/gaps_15m_plus_hits/gap_analysis.py
# python analyses/liquidity_levels_hit_prob/liquidity_hit_analysis.py
# python analyses/indicator_bins_vs_outcomes/indicators_backtest.py
# python analyses/ema_stretch_high_precision/ema_stretch_backtest.py
# python analyses/reversal_power_across_levels/reversal_analysis.py
# python analyses/reversal_power_across_levels/reversal_top_filter.py
# python analyses/reversal_power_across_levels/top_reversal_categorizer.py
# python analyses/ict_vp_v2_1_backtest/ict_vp_backtest.py
```

Each script writes its CSV outputs into its own analysis folder and prints key insights to the console. See each folder’s `README.md` for details.

## Notes on performance and reproducibility
- Large input files: most scripts stream with chunking (1,000,000 rows) and filter to outright NQ contracts.
- Timezone handling: timestamps are read as UTC and converted to `America/New_York` for session logic.
- Front-month selection: where relevant, scripts isolate the front-month per day (e.g., by early RTH range or volume).
- Some scripts are compute-intensive (confluence/gaps/reversals). Consider running them overnight or limiting date ranges if you modify them.

## Adding a new analysis
1) Create a new folder under `analyses/<your_topic>/`.
2) Place your script(s) there. Read inputs from `data/` using a `Path(__file__).resolve()` pattern.
3) Write outputs (CSVs) to the same folder.
4) Add a short `README.md` with:
   - Question being answered
   - How to run
   - Inputs (from `data/`)
   - Outputs

## Support
Open an issue/PR or extend the analyses with additional questions you want to explore.











