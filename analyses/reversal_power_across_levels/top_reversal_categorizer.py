import pandas as pd
import numpy as np
from pathlib import Path

# --- Load all reversal events ---
print("Loading reversal events...")
OUT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(str(OUT_DIR / "reversal_events.csv"), parse_dates=["touch_ts", "created_ts", "et_date"])

print(f"Total reversal events: {len(df):,}")

# --- Define the TOP 10 most powerful reversal types ---
top_reversals = {
    "1_SMA200_POC": {
        "description": "SMA200 + Overnight POC",
        "patterns": ["sma200+vp_on_poc"],
        "context": "Intra-day trend meets volume equilibrium"
    },
    "2_SMA200_SMA50": {
        "description": "SMA200 + SMA50", 
        "patterns": ["sma200+sma50"],
        "context": "Dual moving-average cluster"
    },
    "3_IB_High_Low": {
        "description": "IB High / Low",
        "patterns": ["ib_high", "ib_low"],
        "context": "Post-open range extremes (first hour)"
    },
    "4_Daily_High_Low": {
        "description": "Daily (D1) High / Low",
        "patterns": ["d1_high", "d1_low"],
        "context": "Prior-day extremes"
    },
    "5_H4_High_Low": {
        "description": "H4 High / Low",
        "patterns": ["h4_high", "h4_low"],
        "context": "4-hour structure levels"
    },
    "6_VP_Value_Area": {
        "description": "VP Value Area Edges (VAH/VAL)",
        "patterns": ["vp_on_vah", "vp_on_val"],
        "context": "Overnight auction edges"
    },
    "7_SMA50_Alone": {
        "description": "SMA50 alone",
        "patterns": ["sma50"],
        "context": "1-minute × 50 = hourly trendline"
    },
    "8_London_High_Low": {
        "description": "London High / Low",
        "patterns": ["london_high", "london_low"],
        "context": "Session liquidity from Europe"
    },
    "9_D1_Low_London_Low": {
        "description": "D1 Low + London Low Combo",
        "patterns": ["d1_low+london_low"],
        "context": "Dual-session & multi-day liquidity"
    },
    "10_BB_Upper_D1_High": {
        "description": "BB Upper + D1 High",
        "patterns": ["bb_up+d1_high"],
        "context": "Volatility band + prior-day extreme"
    }
}

# --- Function to check if an event matches a pattern ---
def matches_pattern(event_row, patterns):
    """Check if an event matches any of the given patterns"""
    level_type = str(event_row['level_type']).lower()
    combo_10pt = str(event_row['combo_10pt']).lower()
    
    for pattern in patterns:
        pattern_lower = pattern.lower()
        # Check if it's a single level type match
        if '+' not in pattern_lower and level_type == pattern_lower:
            return True
        # Check if it's a combo match
        elif '+' in pattern_lower and pattern_lower in combo_10pt:
            return True
    return False

# --- Categorize all events ---
print("\nCategorizing events by top reversal types...")

categorized_events = {}
summary_stats = {}

for reversal_id, reversal_info in top_reversals.items():
    print(f"Processing {reversal_info['description']}...")
    
    # Filter events that match this reversal type
    mask = df.apply(lambda row: matches_pattern(row, reversal_info['patterns']), axis=1)
    matching_events = df[mask].copy()
    
    if len(matching_events) > 0:
        # Add metadata
        matching_events['reversal_rank'] = reversal_id
        matching_events['reversal_description'] = reversal_info['description']
        matching_events['reversal_context'] = reversal_info['context']
        
        # Calculate stats
        stats = {
            'total_events': len(matching_events),
            'avg_disp_atr': matching_events['disp_atr'].mean(),
            'median_disp_atr': matching_events['disp_atr'].median(),
            'avg_reversal_ratio': matching_events['reversal_ratio'].mean(),
            'median_reversal_ratio': matching_events['reversal_ratio'].median(),
            'avg_time_to_peak': matching_events['time_to_peak_min'].mean(),
            'median_time_to_peak': matching_events['time_to_peak_min'].median(),
            'date_range': f"{matching_events['et_date'].min().strftime('%Y-%m-%d')} to {matching_events['et_date'].max().strftime('%Y-%m-%d')}"
        }
        
        categorized_events[reversal_id] = matching_events
        summary_stats[reversal_id] = stats
        
        print(f"  Found {len(matching_events):,} events")
        print(f"  Avg displacement: {stats['avg_disp_atr']:.2f} ATR")
        print(f"  Avg reversal ratio: {stats['avg_reversal_ratio']:.2f}")
    else:
        print(f"  No events found for this pattern")

# --- Save detailed results for each category ---
print("\nSaving detailed results...")

for reversal_id, events in categorized_events.items():
    if len(events) > 0:
        filename = OUT_DIR / f"top_reversal_{reversal_id}_{reversal_info['description'].replace(' ', '_').replace('/', '_')}.csv"
        events.to_csv(str(filename), index=False)
        print(f"Saved {len(events):,} events to {filename}")

# --- Create master summary ---
print("\nCreating master summary...")

summary_data = []
for reversal_id, stats in summary_stats.items():
    reversal_info = top_reversals[reversal_id]
    summary_data.append({
        'Rank': reversal_id.split('_')[0],
        'Reversal_Type': reversal_info['description'],
        'Context': reversal_info['context'],
        'Total_Events': stats['total_events'],
        'Avg_Displacement_ATR': round(stats['avg_disp_atr'], 2),
        'Median_Displacement_ATR': round(stats['median_disp_atr'], 2),
        'Avg_Reversal_Ratio': round(stats['avg_reversal_ratio'], 2),
        'Median_Reversal_Ratio': round(stats['median_reversal_ratio'], 2),
        'Avg_Time_to_Peak_Min': round(stats['avg_time_to_peak'], 1),
        'Median_Time_to_Peak_Min': round(stats['median_time_to_peak'], 1),
        'Date_Range': stats['date_range']
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(str(OUT_DIR / "top_reversal_summary.csv"), index=False)

# --- Create combined detailed file ---
print("Creating combined detailed file...")
all_top_events = pd.concat(categorized_events.values(), ignore_index=True)
all_top_events = all_top_events.sort_values(['reversal_rank', 'et_date', 'touch_ts'])
all_top_events.to_csv(str(OUT_DIR / "all_top_reversal_events.csv"), index=False)

print(f"\n✅ Analysis complete!")
print(f"Total top reversal events found: {len(all_top_events):,}")
print(f"Summary saved to: top_reversal_summary.csv")
print(f"All events saved to: all_top_reversal_events.csv")
print(f"Individual category files saved as: top_reversal_*.csv")


