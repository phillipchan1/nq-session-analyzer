import pandas as pd
import numpy as np

# ==== 1. Load Raw Data ====
df = pd.read_csv(
    "glbx-mdp3-20200927-20250926.ohlcv-1m.csv",
)

# ==== 2. Data Cleaning ====
# Remove anomalous prices
df = df[(df["open"] > 10000) & (df["high"] > 10000) & (df["low"] > 10000) & (df["close"] > 10000)]
# Keep only outright NQ contracts
df = df[df["symbol"].str.startswith("NQ") & ~df["symbol"].str.contains("-")]

# Convert timestamp to New York time and get date
df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert("America/New_York")
df["date"] = df["ts_event"].dt.date

# ==== 3. Load and Merge US High Impact Events ====
events_df = pd.read_csv("us_high_impact_events_2020_to_2025.csv")
events_df["date"] = pd.to_datetime(events_df["date"], format='%Y-%m-%d', errors='coerce')
events_df.dropna(subset=['date'], inplace=True)
events_df["date"] = events_df["date"].dt.date

# Create comprehensive event aggregation by date
events_agg = events_df.groupby("date").agg({
    'event_name': lambda x: ', '.join(x),
    'event_type': lambda x: ', '.join(x),
    'session': lambda x: ', '.join(x),
    'time_et': lambda x: ', '.join(x)
}).reset_index()

# Rename columns for clarity
events_agg.rename(columns={
    'event_name': 'event_names',
    'event_type': 'event_types', 
    'session': 'event_sessions',
    'time_et': 'event_times'
}, inplace=True)

# Merge event data into the main dataframe
df = pd.merge(df, events_agg, on="date", how="left")

# Fill missing values
df['event_names'] = df['event_names'].fillna('None')
df['event_types'] = df['event_types'].fillna('None')
df['event_sessions'] = df['event_sessions'].fillna('None')
df['event_times'] = df['event_times'].fillna('None')


# ==== 4. Determine and Filter for Front-Month Contract ====
# Isolate a liquid period to determine the most active contract
rth_mask_for_front_month = (df["ts_event"].dt.time >= pd.to_datetime("09:30").time()) & \
                           (df["ts_event"].dt.time < pd.to_datetime("10:15").time())
rth_df_for_front_month = df[rth_mask_for_front_month]

# Determine front-month by the largest range during this liquid period
rth_agg = rth_df_for_front_month.groupby(["date", "symbol"]).agg(
    high=("high", "max"),
    low=("low", "min")
).reset_index()
rth_agg["range"] = rth_agg["high"] - rth_agg["low"]
front_month_idx = rth_agg.groupby("date")["range"].idxmax()
front_month_symbols = rth_agg.loc[front_month_idx][["date", "symbol"]]

# Merge and filter to keep only front-month data
df = pd.merge(df, front_month_symbols.rename(columns={'symbol': 'front_month_symbol'}), on="date")
df = df[df["symbol"] == df["front_month_symbol"]].drop(columns=['front_month_symbol'])


# ==== 5. Overnight Session Analysis ====
overnight_analyses = []
for trade_date in sorted(df['date'].unique()):
    # Define overnight session from 6 PM previous day to 9:30 AM current day
    start_ts = (pd.Timestamp(trade_date) - pd.Timedelta(days=1)).replace(hour=18, minute=0)
    start_ts = start_ts.tz_localize("America/New_York", nonexistent='NaT')
    end_ts = pd.Timestamp(trade_date).replace(hour=9, minute=30)
    end_ts = end_ts.tz_localize("America/New_York", nonexistent='NaT')

    # Filter for the overnight session
    mask = (df['ts_event'] >= start_ts) & (df['ts_event'] < end_ts)
    overnight_df = df.loc[mask]

    if not overnight_df.empty:
        overnight_high = overnight_df['high'].max()
        overnight_low = overnight_df['low'].min()
        overnight_range = overnight_high - overnight_low
        
        overnight_open = overnight_df.iloc[0]['open']
        overnight_close = overnight_df.iloc[-1]['close']
        overnight_direction = 'Up' if overnight_close > overnight_open else 'Down'

        overnight_analyses.append({
            'date': trade_date,
            'overnight_range': overnight_range,
            'overnight_direction': overnight_direction
        })

overnight_stats_df = pd.DataFrame(overnight_analyses)
overnight_stats_df['date'] = pd.to_datetime(overnight_stats_df['date'])


# ==== 6. RTH Session Analysis ====
# Define the first 45-minute session (9:30 AM to 10:15 AM ET)
session_mask = (df["ts_event"].dt.time >= pd.to_datetime("09:30").time()) & \
           (df["ts_event"].dt.time < pd.to_datetime("10:15").time())
session_45m = df.loc[session_mask].copy() # Use .copy() to avoid SettingWithCopyWarning


# --- New Correlation Analysis ---
# Define time intervals for initial periods
intervals = [5, 10, 15, 30] # 45 is the whole period, so no remainder
daily_analyses = []

# Group by date to process each day
for date, daily_data in session_45m.groupby(session_45m['ts_event'].dt.date):
    day_results = {'date': date}

    # Calculate initial ranges and remainder ranges
    start_time_obj = pd.to_datetime(str(date) + " 09:30").tz_localize("America/New_York")
    for minutes in intervals:
        # Initial period
        end_time_initial = (start_time_obj + pd.Timedelta(minutes=minutes)).time()
        initial_period = daily_data[daily_data['ts_event'].dt.time < end_time_initial]
        day_results[f'range_{minutes}m'] = 0
        if not initial_period.empty:
            day_results[f'range_{minutes}m'] = initial_period['high'].max() - initial_period['low'].min()

        # Remainder of the 45-minute period
        start_time_remainder = end_time_initial
        remainder_period = daily_data[daily_data['ts_event'].dt.time >= start_time_remainder]
        day_results[f'remainder_range_vs_{minutes}m'] = 0
        if not remainder_period.empty:
            day_results[f'remainder_range_vs_{minutes}m'] = remainder_period['high'].max() - remainder_period['low'].min()

    # --- 15-Minute Interval Analysis ---
    # 9:30 - 9:45
    end_time_1 = (start_time_obj + pd.Timedelta(minutes=15)).time()
    period_1 = daily_data[daily_data['ts_event'].dt.time < end_time_1]
    day_results['range_930_945'] = 0
    if not period_1.empty:
        day_results['range_930_945'] = period_1['high'].max() - period_1['low'].min()

    # 9:45 - 10:00
    start_time_2 = end_time_1
    end_time_2 = (start_time_obj + pd.Timedelta(minutes=30)).time()
    period_2 = daily_data[(daily_data['ts_event'].dt.time >= start_time_2) & (daily_data['ts_event'].dt.time < end_time_2)]
    day_results['range_945_1000'] = 0
    if not period_2.empty:
        day_results['range_945_1000'] = period_2['high'].max() - period_2['low'].min()

    # 10:00 - 10:15
    start_time_3 = end_time_2
    end_time_3 = (start_time_obj + pd.Timedelta(minutes=45)).time()
    period_3 = daily_data[(daily_data['ts_event'].dt.time >= start_time_3) & (daily_data['ts_event'].dt.time < end_time_3)]
    day_results['range_1000_1015'] = 0
    if not period_3.empty:
        day_results['range_1000_1015'] = period_3['high'].max() - period_3['low'].min()
        
    # Aggregate full 45m session stats
    day_results['range_45m'] = daily_data['high'].max() - daily_data['low'].min()
    day_results['total_volume_45m'] = daily_data['volume'].sum()
    
    # Event information
    day_results['event_names'] = daily_data['event_names'].iloc[0]
    day_results['event_types'] = daily_data['event_types'].iloc[0]
    day_results['event_sessions'] = daily_data['event_sessions'].iloc[0]
    day_results['event_times'] = daily_data['event_times'].iloc[0]
    
    # Create event timing flags
    day_results['has_pre_session_event'] = 'pre' in str(daily_data['event_sessions'].iloc[0]).lower()
    day_results['has_during_session_event'] = 'during' in str(daily_data['event_sessions'].iloc[0]).lower()
    day_results['has_any_event'] = daily_data['event_names'].iloc[0] != 'None'
    
    # Count events by type
    event_types_str = str(daily_data['event_types'].iloc[0])
    day_results['fomc_events'] = event_types_str.count('Monetary Policy')
    day_results['employment_events'] = event_types_str.count('Employment')
    day_results['inflation_events'] = event_types_str.count('Inflation')
    day_results['gdp_events'] = event_types_str.count('GDP')
    day_results['consumer_events'] = event_types_str.count('Consumer')
    day_results['business_events'] = event_types_str.count('Business')
    day_results['housing_events'] = event_types_str.count('Housing')
    day_results['manufacturing_events'] = event_types_str.count('Manufacturing')
    day_results['labor_events'] = event_types_str.count('Labor')
    day_results['production_events'] = event_types_str.count('Production')

    daily_analyses.append(day_results)

# Create the results dataframe from the daily analyses
result = pd.DataFrame(daily_analyses)
result['date'] = pd.to_datetime(result['date'])

# Merge overnight stats
result = pd.merge(result, overnight_stats_df, on='date', how='left')

result["day_of_week"] = result["date"].dt.day_name()
result["year"] = result["date"].dt.year
result["month"] = result["date"].dt.month


# ==== 7. Print Analysis to Console ====
print("--- Correlation Analysis: Initial Range vs. Remainder of 45-Min Session ---")
for minutes in intervals:
    range_col = f'range_{minutes}m'
    remainder_range_col = f'remainder_range_vs_{minutes}m'

    # Use quantile-based cut for more evenly distributed bins (quartiles)
    try:
        result[f'{range_col}_bin'] = pd.qcut(result[range_col], q=4, labels=False, duplicates='drop')
    except ValueError:
        # Fallback for data that can't be split into quantiles
        result[f'{range_col}_bin'] = pd.cut(result[range_col], bins=4, labels=False, duplicates='drop')


    # Analyze the relationship
    if not result[f'{range_col}_bin'].isnull().all():
        try:
            # Group by the bins and calculate stats for both initial and remainder ranges
            summary = result.groupby(f'{range_col}_bin').agg(
                avg_initial_range=(range_col, 'mean'),
                avg_remainder_range=(remainder_range_col, 'mean'),
                median_remainder_range=(remainder_range_col, 'median'),
                std_remainder_range=(remainder_range_col, 'std'),
                count=(range_col, 'size')
            )

            print(f"\n---> Analysis for first {minutes} minutes (vs. next {45-minutes} mins):")
            print("       (Initial ranges grouped into quartiles from smallest [0] to largest [3])")
            print(summary.to_string(float_format='{:,.2f}'.format))

        except Exception as e:
            print(f"\nCould not generate summary for first {minutes} minutes: {e}")

print("\n--- Average Range (First 45 mins) by Day of Week ---")
day_of_week_summary = result[result['range_45m'] > 0].groupby('day_of_week')['range_45m'].agg(['mean', 'median', 'count', 'std'])
# Order the days of the week for logical display
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
day_of_week_summary = day_of_week_summary.reindex(days_order)
print(day_of_week_summary.to_string(float_format='{:,.2f}'.format))


print("\n--- Average Range (First 45 mins) by Event Timing ---")
# Analysis by event session timing
timing_summary = result[result['range_45m'] > 0].groupby(['has_pre_session_event', 'has_during_session_event'])['range_45m'].agg(['mean', 'median', 'count', 'std'])
print("Pre-Session | During-Session | Mean Range | Median | Count | Std")
print("-" * 70)
for (pre, during), stats in timing_summary.iterrows():
    pre_str = "Yes" if pre else "No"
    during_str = "Yes" if during else "No"
    print(f"{pre_str:11} | {during_str:13} | {stats['mean']:8.2f} | {stats['median']:6.2f} | {stats['count']:5.0f} | {stats['std']:6.2f}")

print("\n--- Average Range (First 45 mins) by Event Type ---")
# Filter out days with no trading range before grouping
event_summary = result[result['range_45m'] > 0].groupby('event_names')['range_45m'].agg(['mean', 'median', 'count', 'std'])
print(event_summary.sort_values(by='mean', ascending=False).head(20).to_string(float_format='{:,.2f}'.format))

print("\n--- Average Range (First 45 mins) by Event Category Count ---")
# Analysis by number of events by category
event_categories = ['fomc_events', 'employment_events', 'inflation_events', 'gdp_events', 
                   'consumer_events', 'business_events', 'housing_events', 'manufacturing_events', 
                   'labor_events', 'production_events']

for category in event_categories:
    if category in result.columns:
        category_summary = result[result['range_45m'] > 0].groupby(category)['range_45m'].agg(['mean', 'median', 'count'])
        if len(category_summary) > 1:  # Only show if there's variation
            print(f"\n{category.replace('_events', '').title()} Events:")
            print(category_summary.to_string(float_format='{:,.2f}'.format))


print("\n--- Average Range (First 45 mins) by Year ---")
yearly_summary = result[result['range_45m'] > 0].groupby('year')['range_45m'].agg(['mean', 'median', 'count', 'std'])
print(yearly_summary.sort_index().to_string(float_format='{:,.2f}'.format))

print("\n--- Average Range (First 45 mins) by Month ---")
monthly_summary = result[result['range_45m'] > 0].groupby('month')['range_45m'].agg(['mean', 'median', 'count', 'std'])
print(monthly_summary.sort_index().to_string(float_format='{:,.2f}'.format))


# --- Analysis of 15-Minute Macro Intervals ---
print("\n--- Analysis of 15-Minute Macro Intervals ---")
interval_cols = ['range_930_945', 'range_945_1000', 'range_1000_1015']
interval_summary = result[interval_cols].agg(['mean', 'median', 'std', 'min', 'max'])
print("Summary statistics for each 15-minute interval range:")
print(interval_summary.to_string(float_format='{:,.2f}'.format))

print("\nCorrelation matrix for 15-minute interval ranges:")
correlation_matrix = result[interval_cols].corr()
print(correlation_matrix.to_string(float_format='{:,.2f}'.format))


# --- Event Timing and Market Impact Analysis ---
print("\n--- Event Timing and Market Impact Analysis ---")

# Analysis of pre-session events impact
print("\nPre-Session Events Impact on 45-Min RTH Range:")
pre_session_analysis = result[result['range_45m'] > 0].groupby('has_pre_session_event')['range_45m'].agg(['mean', 'median', 'count', 'std'])
print(pre_session_analysis.to_string(float_format='{:,.2f}'.format))

# Analysis of during-session events impact
print("\nDuring-Session Events Impact on 45-Min RTH Range:")
during_session_analysis = result[result['range_45m'] > 0].groupby('has_during_session_event')['range_45m'].agg(['mean', 'median', 'count', 'std'])
print(during_session_analysis.to_string(float_format='{:,.2f}'.format))

# Analysis of event combinations
print("\nEvent Combination Analysis:")
event_combinations = result[result['range_45m'] > 0].groupby(['has_pre_session_event', 'has_during_session_event', 'has_any_event'])['range_45m'].agg(['mean', 'median', 'count'])
print(event_combinations.to_string(float_format='{:,.2f}'.format))

# High-impact event analysis (FOMC, NFP, CPI)
print("\nHigh-Impact Event Analysis:")
high_impact_events = ['FOMC Statement', 'NFP', 'CPI', 'GDP Release']
for event in high_impact_events:
    event_mask = result['event_names'].str.contains(event, na=False)
    if event_mask.any():
        event_data = result[event_mask & (result['range_45m'] > 0)]
        if not event_data.empty:
            print(f"\n{event} Events (n={len(event_data)}):")
            print(f"  Mean Range: {event_data['range_45m'].mean():.2f}")
            print(f"  Median Range: {event_data['range_45m'].median():.2f}")
            print(f"  Std Range: {event_data['range_45m'].std():.2f}")

# --- Overnight vs. RTH Analysis ---
print("\n--- Overnight Session vs. RTH Analysis ---")
if 'overnight_range' in result.columns:
    # Binning overnight range into quartiles
    try:
        result['overnight_range_bin'] = pd.qcut(result['overnight_range'].dropna(), q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'], duplicates='drop')
        
        print("\n45-Min RTH Range based on Overnight Range Quartiles:")
        overnight_summary = result.groupby('overnight_range_bin')['range_45m'].agg(['mean', 'median', 'count', 'std'])
        print(overnight_summary.to_string(float_format='{:,.2f}'.format))

    except Exception as e:
        print(f"\nCould not generate overnight range summary: {e}")

    # Analysis by overnight direction
    if 'overnight_direction' in result.columns:
        print("\n45-Min RTH Range based on Overnight Direction:")
        direction_summary = result.groupby('overnight_direction')['range_45m'].agg(['mean', 'median', 'count', 'std'])
        print(direction_summary.to_string(float_format='{:,.2f}'.format))

    # Correlation between overnight range and RTH range
    print("\nCorrelation between Overnight Range and 45-Min RTH Range:")
    correlation = result[['overnight_range', 'range_45m']].corr().iloc[0, 1]
    print(f"{correlation:.2f}")

# --- Event Timing vs Overnight Analysis ---
print("\n--- Event Timing vs Overnight Session Analysis ---")
if 'overnight_range' in result.columns:
    print("\nOvernight Range by Event Timing:")
    overnight_event_analysis = result.groupby(['has_pre_session_event', 'has_during_session_event'])['overnight_range'].agg(['mean', 'median', 'count', 'std'])
    print(overnight_event_analysis.to_string(float_format='{:,.2f}'.format))


# ==== 8. Save Output ====
result.to_csv("nq_session_summary.csv", index=False)
print("\n✅ Done! Detailed session analysis saved to nq_session_summary.csv")
