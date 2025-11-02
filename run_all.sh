#!/usr/bin/env bash
set -euo pipefail

# Lightweight runner (skips heaviest scripts by default).
# Adjust as needed.

root_dir="$(cd "$(dirname "$0")" && pwd)"

python "$root_dir/analyses/ny_open_45m_range_vs_events/backtest.py"
python "$root_dir/analyses/ny_open_45m_range_vs_events/analyze_volume.py"

# Medium
python "$root_dir/analyses/london_swing_hits_at_open/london_hl.py"
python "$root_dir/analyses/range_extremes_and_15m_slots/range_and_intervals_analysis.py"

# Heavy (uncomment to run)
# python "$root_dir/analyses/confluence_zones_hits/confluence_analysis.py"
# python "$root_dir/analyses/gaps_15m_plus_hits/gap_analysis.py"
# python "$root_dir/analyses/liquidity_levels_hit_prob/liquidity_hit_analysis.py"
# python "$root_dir/analyses/indicator_bins_vs_outcomes/indicators_backtest.py"
# python "$root_dir/analyses/ema_stretch_high_precision/ema_stretch_backtest.py"
# python "$root_dir/analyses/reversal_power_across_levels/reversal_analysis.py"
# python "$root_dir/analyses/reversal_power_across_levels/reversal_top_filter.py"
# python "$root_dir/analyses/reversal_power_across_levels/top_reversal_categorizer.py"
# python "$root_dir/analyses/ict_vp_v2_1_backtest/ict_vp_backtest.py"

echo "All selected analyses completed."






