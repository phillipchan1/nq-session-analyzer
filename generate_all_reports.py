#!/usr/bin/env python3
"""
Generate All Reports
--------------------

Generates summary reports from all analysis outputs.
Run this after all analyses have completed.
"""

from pathlib import Path
import pandas as pd
from datetime import datetime

ANALYSES_DIR = Path(__file__).parent / "analyses"
OUTPUT_DIR = Path(__file__).parent / "reports"

def generate_sequence_report():
    """Generate report from liquidity sequence tracker."""
    seq_dir = ANALYSES_DIR / "liquidity_sequence_tracker"
    
    detailed_file = seq_dir / "liquidity_sequence_detailed.csv"
    frequency_file = seq_dir / "liquidity_sequence_frequency.csv"
    transition_file = seq_dir / "liquidity_transition_matrix.csv"
    
    if not detailed_file.exists():
        print(f"⚠️  Sequence analysis not complete: {detailed_file} not found")
        return
    
    print("📊 Generating Sequence Report...")
    
    df_detailed = pd.read_csv(detailed_file)
    df_freq = pd.read_csv(frequency_file) if frequency_file.exists() else None
    df_trans = pd.read_csv(transition_file) if transition_file.exists() else None
    
    report_lines = [
        "# Liquidity Sequence Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Total Days Analyzed:** {len(df_detailed)}",
        "",
    ]
    
    if df_freq is not None and not df_freq.empty:
        report_lines.extend([
            "## Top 10 Most Common Sequences",
            "",
        ])
        for i, row in df_freq.head(10).iterrows():
            report_lines.append(f"{i+1}. {row['sequence']}: {row['count']} times ({row['frequency_pct']:.1f}%)")
        report_lines.append("")
    
    if df_trans is not None and not df_trans.empty:
        report_lines.extend([
            "## Transition Probabilities",
            "",
            "Most common transitions (>10% probability):",
            "",
        ])
        # Extract high-probability transitions
        for _, row in df_trans.iterrows():
            from_level = row['from_level']
            transitions = []
            for col in df_trans.columns:
                if col not in ['from_level', 'total_count'] and not col.endswith('_count'):
                    prob = row[col]
                    if prob > 10:
                        transitions.append(f"{col}: {prob:.1f}%")
            if transitions:
                report_lines.append(f"- **{from_level}** → {', '.join(transitions)}")
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_file = OUTPUT_DIR / "sequence_report.md"
    report_file.write_text(report_text)
    print(f"✅ Saved: {report_file}")


def generate_failed_rally_report():
    """Generate report from failed rally detector."""
    rally_dir = ANALYSES_DIR / "failed_rally_detector"
    
    detailed_file = rally_dir / "failed_rally_detailed.csv"
    stats_file = rally_dir / "failed_rally_statistics.csv"
    
    if not detailed_file.exists():
        print(f"⚠️  Failed rally analysis not complete: {detailed_file} not found")
        return
    
    print("📊 Generating Failed Rally Report...")
    
    df_detailed = pd.read_csv(detailed_file)
    df_stats = pd.read_csv(stats_file) if stats_file.exists() else None
    
    failed_count = len(df_detailed[df_detailed["failed"] == True])
    total_count = len(df_detailed)
    
    report_lines = [
        "# Failed Rally Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Total Rally Attempts:** {total_count}",
        f"**Failed Rallies:** {failed_count} ({failed_count/total_count*100:.1f}%)",
        f"**Successful Rallies:** {total_count - failed_count} ({(total_count-failed_count)/total_count*100:.1f}%)",
        "",
    ]
    
    if df_stats is not None and not df_stats.empty:
        report_lines.extend([
            "## Statistics by Liquidity Type",
            "",
            "| Liquidity Type | Total Rallies | Failed | Failed % | Avg Rally Distance | Avg Time to Extreme |",
            "|----------------|---------------|--------|----------|-------------------|---------------------|",
        ])
        
        for _, row in df_stats.iterrows():
            report_lines.append(
                f"| {row['liquidity_type']} | {row['total_rallies']} | {row['failed_count']} | "
                f"{row['failed_pct']:.1f}% | {row['avg_rally_distance']:.1f} | {row['avg_time_to_extreme']:.1f} |"
            )
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_file = OUTPUT_DIR / "failed_rally_report.md"
    report_file.write_text(report_text)
    print(f"✅ Saved: {report_file}")


def generate_gap_priority_report():
    """Generate report from gap priority analysis."""
    gap_dir = ANALYSES_DIR / "gap_priority_analysis"
    
    detailed_file = gap_dir / "gap_priority_detailed.csv"
    summary_file = gap_dir / "gap_priority_summary.csv"
    
    if not detailed_file.exists():
        print(f"⚠️  Gap priority analysis not complete: {detailed_file} not found")
        return
    
    print("📊 Generating Gap Priority Report...")
    
    df_detailed = pd.read_csv(detailed_file)
    df_summary = pd.read_csv(summary_file) if summary_file.exists() else None
    
    report_lines = [
        "# Gap Priority Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Total Days Analyzed:** {len(df_detailed)}",
        "",
    ]
    
    if df_summary is not None and not df_summary.empty:
        report_lines.extend([
            "## First Fill by Timeframe",
            "",
            "| Timeframe | First Fill Count | First Fill % |",
            "|-----------|-----------------|---------------|",
        ])
        
        for _, row in df_summary.iterrows():
            report_lines.append(
                f"| {row['timeframe']} | {row['first_fill_count']} | {row['first_fill_pct']:.1f}% |"
            )
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_file = OUTPUT_DIR / "gap_priority_report.md"
    report_file.write_text(report_text)
    print(f"✅ Saved: {report_file}")


def main():
    """Generate all reports."""
    print("=" * 60)
    print("Generating Analysis Reports")
    print("=" * 60)
    print()
    
    generate_sequence_report()
    print()
    generate_failed_rally_report()
    print()
    generate_gap_priority_report()
    print()
    
    print("=" * 60)
    print("✅ All reports generated!")
    print(f"Reports saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()





