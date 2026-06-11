# Friday Lower High Setup: Monday Visit Probability of Friday's Low

## Overview

This analysis backtests the claim from a YouTube transcript:

> If Friday's regular trading session high is **lower** than Thursday's high, then Friday's low will be visited on Monday with "overwhelmingly high" odds.

**Definitions:**
- **Friday/Thursday high**: RTH (9:30–16:00 ET) session high
- **Friday low**: RTH session low (the level to visit)
- **Visited**: Strict sweep = price goes beyond level by ≥1 NQ tick (low < friday_low - 0.25)
- **Monday**: Next trading day (skipping weekends and holidays)

**Critical context**: The trader operates only in the **first 45 minutes** (9:30–10:15 ET), so the key question is *when* the visit occurs.

**Data**: 1,393 trading days, Sept 2020 – Feb 2026. Front-month NQ by volume in 9:30–9:45.

---

## Key Findings

### 1. Setup Has Predictive Power vs Baseline

| Group | Days | Sweep 45min | Sweep 45min % | Sweep EOD | Sweep EOD % |
|-------|------|-------------|---------------|-----------|-------------|
| **Qualifying** (Fri high < Thu high) | 126 | 49 | **38.9%** | 65 | **51.6%** |
| **Non-qualifying** (Fri high ≥ Thu high) | 133 | 25 | **18.8%** | 34 | **25.6%** |
| All Fridays | 259 | 74 | 28.6% | 99 | 38.2% |

**Key Insight**: When Friday fails to make a new high (Fri high < Thu high), Friday's low is swept on Monday **about twice as often** as when Friday makes a new high. The setup is statistically meaningful: 38.9% vs 18.8% within 45 min, and 51.6% vs 25.6% by EOD.

---

### 2. First 45 Minutes: ~39% Visit Rate

| Metric | Value |
|--------|-------|
| Qualifying Fridays | 126 |
| Swept within 9:30–10:15 | 49 |
| **Sweep rate (45 min)** | **38.9%** |
| Swept by 16:00 | 65 |
| **Sweep rate (EOD)** | **51.6%** |

**Key Insight**: The transcript's "overwhelmingly high" odds apply more to the **full day** (52%) than to the first 45 minutes (39%). For a first-45-min trader, about 4 in 10 qualifying setups see Friday's low swept in the trading window.

---

### 3. When the Visit Happens (of those swept)

| Minutes from 9:30 | Count | % of Sweeps |
|-------------------|-------|-------------|
| **0–5** | 38 | **58.5%** |
| 5–10 | 1 | 1.5% |
| 10–15 | 2 | 3.1% |
| 15–20 | 3 | 4.6% |
| 20–25 | 4 | 6.2% |
| 25–30 | 0 | 0.0% |
| 30–45 | 1 | 1.5% |
| After 45 | 16 | 24.6% |

**Key Insight**: When Friday's low *is* swept on Monday, **58.5% of sweeps occur in the first 5 minutes** (9:30–9:35). If it doesn't happen early, a large share (25%) occurs after 10:15. For first-45-min trading, the best window is the opening 5–15 minutes.

---

### 4. Cumulative Sweep by Window

| Window End | Swept | % |
|------------|-------|---|
| 10:15 | 49 | 38.9% |
| 10:30 | 50 | 39.7% |
| 11:00 | 53 | 42.1% |
| 12:00 | 58 | 46.0% |
| 16:00 | 65 | 51.6% |

**Key Insight**: Most of the additional sweeps after 10:15 happen between 10:15 and 12:00. By noon, 46% have been swept; by 4 PM, 52%.

---

### 5. Distance from Monday Open to Friday Low

| Distance (pts) | Days | Sweep 45min % | Sweep EOD % |
|----------------|------|---------------|-------------|
| **0–25** | 9 | **88.9%** | **100%** |
| **25–50** | 13 | **69.2%** | **84.6%** |
| **50–100** | 26 | **57.7%** | **73.1%** |
| 100–200 | 37 | 24.3% | 37.8% |
| 200+ | 41 | 19.5% | 29.3% |

**Key Insight**: **Distance is the main driver.** When Friday's low is within 25 points of Monday's open, 89% get swept in the first 45 min and 100% by EOD. Beyond 100 points, sweep rates drop to 19–24% in 45 min. Prioritize setups where Friday's low is within 50 points of the expected Monday open.

---

### 6. Symmetric Setup: Friday Low > Thursday Low → Friday High Visited?

When Friday's low is *higher* than Thursday's low (Friday failed to make a new low), does Friday's *high* get visited on Monday?

| Metric | Value |
|--------|-------|
| Qualifying Fridays | 31 |
| Sweep 45 min | 15 |
| **Sweep 45 min %** | **48.4%** |
| Sweep EOD | 21 |
| **Sweep EOD %** | **67.7%** |

**Key Insight**: The symmetric setup (Fri low > Thu low) also shows elevated visit rates (48% in 45 min, 68% by EOD), though the sample is smaller (31 days). Worth tracking as a complementary pattern.

---

## Trading Implications

### For First-45-Minute Traders

1. **Use the setup as a filter**: When Friday high < Thursday high, treat Friday's low as a liquidity target on Monday.
2. **Focus on distance**: Best edge when Friday's low is within 50 pts of Monday open (69–89% sweep in 45 min).
3. **Time window**: Most sweeps occur in the first 5 minutes. If no sweep by 9:35, probability drops; after 10:15, many visits happen later in the day.
4. **Baseline matters**: Non-qualifying Fridays (Fri high ≥ Thu high) have only 19% sweep rate in 45 min—avoid treating Friday's low as a target on those days.

### Suggested Entry Logic

- **Pre-market**: Identify Friday's low and Monday's expected open (e.g., from futures).
- **Filter**: Only if Friday RTH high < Thursday RTH high.
- **Distance**: Prefer setups where Friday low is within 50 pts of open.
- **Entry**: Consider a limit buy near Friday's low at the open, targeting a sweep (1–2 pts beyond).
- **Time stop**: If no sweep by 10:15, exit or tighten stop.

---

## Output Files

| File | Description |
|------|-------------|
| `friday_lower_high_summary.csv` | Overall sweep counts and rates |
| `friday_lower_high_timing.csv` | Minutes-from-open distribution of sweeps |
| `friday_lower_high_cumulative.csv` | Cumulative sweep % by window |
| `friday_lower_high_detailed.csv` | Per-Friday rows for manual review |
| `friday_lower_high_distance_buckets.csv` | Sweep rate by distance from Monday open |
| `friday_lower_high_baseline.csv` | Qualifying vs non-qualifying vs all Fridays |
| `friday_lower_high_symmetric.csv` | Fri low > Thu low → Fri high visited |

---

## Statistical Summary

| Metric | Value |
|--------|-------|
| Total Fridays with complete data | 259 |
| Qualifying (Fri high < Thu high) | 126 (48.6%) |
| Non-qualifying | 133 (51.4%) |
| Data period | Sept 2020 – Feb 2026 |
