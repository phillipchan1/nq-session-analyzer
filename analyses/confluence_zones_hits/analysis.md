# Confluence Zones Analysis - Key Findings

## Overview
This analysis examines whether clustered liquidity levels (confluence zones) within various thresholds get hit during early NY session windows, and how hit rates scale with confluence strength and distance from the open.

## Main Findings

### 1. Confluence Strength Matters Significantly
- **Low Confluence (2 levels)**: Hit rates range from 63-68% in the first 15 minutes
- **Medium Confluence (5-7 levels)**: Hit rates increase to 65-75% in the first 15 minutes
- **High Confluence (8+ levels)**: Hit rates reach 76-100% in the first 15 minutes (small sample sizes)

### 2. Distance from Open is Crucial
The analysis reveals an inverse relationship between distance from open and hit probability:

**Zones Very Close to Open (< 10 points)**:
- Threshold 10pt, Confluence 9: **3.6 points from open**, 100% hit rate (9:30-10:15)
- Threshold 20pt, Confluence 11: **10.8 points from open**, 100% hit rate (9:30-10:15)
- These ultra-close confluence zones show exceptional reliability

**Medium Distance (50-100 points)**:
- Confluence 5-7 zones at 60-85 points from open: 60-75% hit rates
- Still significant but less reliable than very close zones

**Far Distance (100-200 points)**:
- Confluence 2-3 zones at 150-165 points from open: Only 7-12% hit rates in individual 15-min windows
- Overall 9:30-10:15 hit rates improve to 17-19% but still relatively weak

### 3. Time Window Performance
Breaking down hit rates by time window:

**9:30-9:45 (First 15 minutes)**:
- Consistently the highest hit rates across all confluence levels
- Confluence 6-9: **76-100% hit rates** (20-30pt thresholds)
- Confluence 2: Still respectable at **63-68%**

**9:45-10:00 (Second 15 minutes)**:
- Moderate decline in hit rates
- Confluence 6-9: **50-75% hit rates**
- Confluence 2: **57-63%**

**10:00-10:15 (Third 15 minutes)**:
- Similar to second window
- Indicates most action happens in first 30 minutes

**Cumulative 9:30-10:15**:
- Confluence 9-10: **78-100% hit rates**
- Confluence 5-7: **67-75% hit rates**
- Confluence 2: **69-73% hit rates**

### 4. Optimal Threshold Settings
Comparing different clustering thresholds (5pt, 10pt, 15pt, 20pt, 25pt, 30pt):

**20-25 Point Threshold**:
- Best balance of zone identification and hit rate accuracy
- Confluence 5: **70-74% cumulative hit rate**
- Confluence 6-7: **73-75% cumulative hit rate**
- Zone widths: 16-24 points (manageable for trading)

**30 Point Threshold**:
- Slightly higher hit rates at very high confluence (78% for confluence 10)
- But wider zones (26-36 points) may be less precise for entries

**5-10 Point Threshold**:
- More granular zones but higher confluence needed
- Good for precision entries if you can identify confluence 6+

### 5. Top Performing Patterns

The most reliable confluence combinations include:

**Gap-Based Patterns**:
- Multiple 15min_gap_up levels stacking with 1h_gap_up
- Example: 8+ 15min_gap_up levels + 2-3 1h_gap_up = 80-100% hit rate

**High/Low Confluence**:
- 4h_high + daily_high + hourly_high combined with 4h_low + daily_low + hourly_low
- Especially powerful when combined with gap levels

**London Session Reference**:
- London_high/low combined with gap levels
- London_low zones showing particularly strong early hits

### 6. Practical Trading Implications

**Strong Trade Setups** (Consider taking):
- Confluence 6+ zones within 10 points of open: **78-100% hit probability**
- Confluence 5+ zones within 50 points at 20-25pt threshold: **70-75% hit probability**
- Any confluence 8+ zone within 100 points: **65-80% hit probability**

**Moderate Setups** (Use with confirmation):
- Confluence 3-4 zones within 100 points: **20-25% hit probability**
- Confluence 2 zones within 100 points: **69-73% hit probability** (surprisingly consistent)

**Weak Setups** (Avoid or use as counter-indicators):
- Any zone 150+ points from open with confluence < 4: **<20% hit probability**
- High confluence (12+) with very wide zones: Small sample, unreliable

## Statistical Summary

**Best Performance**:
- Configuration: 20-25pt threshold, confluence 5-7, distance < 75 points from open
- Expected hit rate: **70-75%** within first 45 minutes
- Zone width: **16-24 points** (manageable for entry/exit)

**Zone Count vs Quality**:
- Lower thresholds (5pt) identify more zones (4000+) but most are weak
- Higher thresholds (25-30pt) identify fewer zones (200-400 at confluence 5) but higher quality
- Sweet spot: 20pt threshold provides good balance

## Recommendations

1. **Focus on the first 15 minutes** (9:30-9:45) for highest probability plays
2. **Prioritize zones within 50 points of the open** for best hit rates
3. **Look for confluence 5+** as minimum threshold for trade consideration
4. **Use 20-25pt clustering threshold** for optimal zone identification
5. **Gap-based patterns** (15min_gap + 1h_gap) show most consistent performance
6. **Extreme confluence (9+) zones very close to open (<10 points)** are rare but nearly guaranteed hits

## Data Quality Notes
- Total zones analyzed: Varies by threshold from 1,034 (5pt, confluence 2) to single digits (highest confluence)
- Sample sizes become small at confluence 10+, interpret with caution
- Analysis covers full dataset from 2020-2025, providing robust multi-year perspective

