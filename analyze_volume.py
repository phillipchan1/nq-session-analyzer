import pandas as pd
from scipy.stats import ttest_ind

def perform_and_print_analysis(group_name, group_df, control_df, metric='total_volume'):
    """Helper function to perform t-test and print the results for a given metric."""
    
    if group_df.empty:
        print(f"\n--- {group_name} ---")
        print("No data available for this group.")
        return

    # Calculate averages
    avg_metric_group = group_df[metric].mean()
    avg_metric_control = control_df[metric].mean()

    # Perform independent t-test
    ttest_result = ttest_ind(group_df[metric], control_df[metric], equal_var=False)

    # Print results
    metric_name = metric.replace('_', ' ').title()
    print(f"\n--- Analysis for {group_name} vs. Neutral Days ---")
    print(f"Average {metric_name}: {avg_metric_group:,.2f}")
    
    if avg_metric_group > avg_metric_control:
        print(f"Conclusion: {group_name} tend to have higher {metric_name.lower()}.")
    else:
        print(f"Conclusion: {group_name} do not tend to have higher {metric_name.lower()}.")
    
    print("\nStatistical Significance (T-test):")
    print(f"T-statistic: {ttest_result.statistic:.4f}")
    print(f"P-value: {ttest_result.pvalue:.4f}")

    if ttest_result.pvalue < 0.05:
        print(f"The difference in {metric_name.lower()} is statistically significant (p < 0.05).")
    else:
        print(f"The difference in {metric_name.lower()} is not statistically significant (p >= 0.05).")


def analyze_session_data(file_path):
    """
    Analyzes trading data on, before, and after red folder days vs. neutral days.
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Identify days before and after red folder days
    df['is_before_red_folder'] = df['redFolder'].shift(-1).fillna(False)
    df['is_after_red_folder'] = df['redFolder'].shift(1).fillna(False)

    # Create distinct groups for analysis
    red_folder_days = df[df['redFolder']]
    
    # Use bitwise AND (&) and NOT (~) for boolean indexing
    before_red_folder_days = df[df['is_before_red_folder'] & ~df['redFolder']]
    after_red_folder_days = df[df['is_after_red_folder'] & ~df['redFolder']]
    
    neutral_days = df[~df['redFolder'] & ~df['is_before_red_folder'] & ~df['is_after_red_folder']]

    # --- Start Printing Results ---
    print("Correlative Analysis of NQ Trading Data")
    print("="*50)
    
    # --- Volume Analysis ---
    print("\nVolume Analysis")
    print("-" * 50)
    print(f"Average Volume on Neutral Days: {neutral_days['total_volume'].mean():,.2f}")
    
    perform_and_print_analysis("Red Folder Days", red_folder_days, neutral_days, 'total_volume')
    perform_and_print_analysis("Days Before Red Folder", before_red_folder_days, neutral_days, 'total_volume')
    perform_and_print_analysis("Days After Red Folder", after_red_folder_days, neutral_days, 'total_volume')

    # --- Range Analysis ---
    print("\n\n" + "="*50)
    print("\nRange Analysis (in Points)")
    print("-" * 50)
    print(f"Average Range on Neutral Days: {neutral_days['point_range'].mean():,.2f}")
    
    perform_and_print_analysis("Red Folder Days", red_folder_days, neutral_days, 'point_range')
    perform_and_print_analysis("Days Before Red Folder", before_red_folder_days, neutral_days, 'point_range')
    perform_and_print_analysis("Days After Red Folder", after_red_folder_days, neutral_days, 'point_range')

    # --- Detailed Analysis by Event Name ---
    print("\n\n" + "="*50)
    print("\nDetailed Analysis by Event Name")
    print("-" * 50)
    
    for event_name in red_folder_days['event_name'].dropna().unique():
        event_specific_days = red_folder_days[red_folder_days['event_name'] == event_name]
        
        # Analyze volume for this specific event
        perform_and_print_analysis(f"{event_name} Days", event_specific_days, neutral_days, 'total_volume')
        
        # Analyze range for this specific event
        perform_and_print_analysis(f"{event_name} Days", event_specific_days, neutral_days, 'point_range')

    # --- Detailed Analysis by Event Timing ---
    print("\n\n" + "="*50)
    print("\nDetailed Analysis by Event Timing")
    print("-" * 50)

    for event_timing in red_folder_days['event_timing'].dropna().unique():
        timing_specific_days = red_folder_days[red_folder_days['event_timing'] == event_timing]

        # Analyze volume for this specific timing
        perform_and_print_analysis(f"'{event_timing.title()}' Events", timing_specific_days, neutral_days, 'total_volume')
        
        # Analyze range for this specific timing
        perform_and_print_analysis(f"'{event_timing.title()}' Events", timing_specific_days, neutral_days, 'point_range')


if __name__ == "__main__":
    analyze_session_data('nq_session_summary.csv')
