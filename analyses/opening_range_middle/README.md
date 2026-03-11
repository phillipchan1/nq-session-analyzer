## Opening Range Middle Analysis

### Question
When price sits "in the middle" of the opening range (first 15 minutes, 9:30-9:45 AM ET) at 9:45 AM, what's the probability that one side (high or low) gets hit before 10:15 AM?

### Methodology

**Opening Range Definition:**
- Time window: 9:30 AM - 9:45 AM ET
- Calculates the high and low during this period

**"In the Middle" Definition:**
- Price at 9:45 AM must be at least **20 points** away from BOTH:
  - Opening range high
  - Opening range low
- This ensures price is truly positioned in the middle, not near either boundary

**Hit Detection:**
- Tracks whether the opening range high or low gets hit in the 9:45 AM - 10:15 AM window
- Uses epsilon tolerance (0.1 points) for hit detection
- Records which side hits first (or if both hit, or neither)

**Data Processing:**
- Uses front-month NQ contracts (determined by max volume in first 15 minutes)
- Filters to weekdays only
- Processes all available trading days in the dataset

### How to Run
```bash
python opening_range_middle_analysis.py
```

### Inputs (from data/)
- `glbx-mdp3-20200927-20260221.ohlcv-1m.csv`

### Outputs

1. **opening_range_middle_summary.csv**
   - Summary statistics including:
     - Total days analyzed
     - Days where price was in middle at 9:45
     - Hit rates (high hit, low hit, both hit, neither hit)
     - First hit statistics
     - Average distances and opening range sizes

2. **opening_range_middle_days.csv**
   - List of the last 100 days where price was in the middle at 9:45 AM
   - Includes:
     - Date, symbol
     - Opening range high/low and size
     - Price at 9:45 AM
     - Distance to high/low
     - Which side hit (high/low/both/neither)
     - Time of first hit (if any)
     - **minutes_after_formation** - minutes from 9:45 to first sweep
     - **macro_15m** - which 15m candle (9:45, 10:00) the first sweep occurred in
   - Use this list for manual verification and backtesting setups

3. **opening_range_middle_sweep_timing.csv**
   - Distribution of when the first sweep occurs (minutes after 9:45 formation)
   - 5-minute buckets (0-5, 5-10, 10-15, etc.) with count and percentage

4. **opening_range_middle_detailed.csv**
   - Complete results for all trading days analyzed
   - Includes both days where price was in middle and days where it wasn't
   - Useful for deeper analysis and filtering

### Key Metrics

- **Hit Probability**: Percentage of "middle" days where high/low gets hit before 10:15 AM
- **First Hit Rate**: Which side tends to get hit first when both sides eventually get hit
- **Average Distances**: Typical positioning within the range when price is "in the middle"
- **Opening Range Size**: Average size of the opening range for context

### Configuration

Key parameters in the script:
- `MIN_DISTANCE_POINTS = 20` - Minimum distance from each boundary (configurable)
- `OR_START = time(9, 30)` - Opening range start time
- `OR_END = time(9, 45)` - Opening range end time
- `CHECK_TIME = time(9, 45)` - Time to check if price is in middle
- `HIT_WINDOW_END = time(10, 15)` - End of hit tracking window
- `RECENT_DAYS = 45` - Number of recent days to include in qualifying list

### Use Cases

1. **Probability Analysis**: Understand the likelihood of range expansion when price starts in the middle
2. **Directional Bias**: Determine if there's a bias toward hitting high vs low first
3. **Manual Backtesting**: Use the qualifying days list to manually review setups and test trading strategies
4. **Range Behavior**: Study how opening ranges behave when price is positioned neutrally
