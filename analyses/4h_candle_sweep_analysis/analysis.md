# NQ 4H Candle Post-Formation Sweep Study

## Overview
This study analyzes **post-formation liquidity sweeps** of completed **4-hour candles** on NQ using 30s / 1m data over ~5 years.

The goal is to identify **when** and **under what conditions** the market is most likely to sweep the **high and/or low** of a completed 4H candle — with an emphasis on **tradable, non-tautological edges**.

---

## Candle Definitions

We analyze the following 4H candles (ET):

- **22:00–02:00** (10pm candle)
- **02:00–06:00** (2am candle)
- **06:00–10:00** (6am candle)

A candle is considered **formed** at its end time.  
Only price action **after candle_end** is used to test sweeps.

---

## Sweep Definition (Strict Takeout)

A sweep requires **≥ 1 NQ tick beyond the level**.

TICK = 0.25

High swept if price_high >= candle_high + TICK
Low swept  if price_low  <= candle_low  - TICK
Touches do not count.

---

## Study Design

### Post-Formation Windows

After candle formation, the session is divided into 15-minute windows:

- **Window 1**: 0–15 minutes
- **Window 2**: 15–30 minutes
- **Window 3**: 30–45 minutes
- …
- Through end of RTH (16:00 ET)

### Conditional Logic

- High and low are tracked independently
- A side is only eligible if it has not yet been swept
- Once swept, it is excluded from later windows
- Window probabilities are conditional:

  P(sweep in window | not swept at window start)

---

## Key Finding #1 — Timing Dominance (6–10 Candle)

The 6:00–10:00 candle shows the strongest post-formation urgency.

### First 3 Windows (6–10 Candle)

| Window | Either Side Swept |
|--------|-------------------|
| 10:00–10:15 | ~58% |
| 10:15–10:30 (conditional) | ~31% |
| 10:30–10:45 (conditional) | ~29% |

### Interpretation

- Liquidity resolution is front-loaded
- If unresolved by ~10:45, urgency collapses

## Key Finding #2 — Distance Is the Primary Control Variable

Distance from price at candle formation strongly predicts sweep probability.

### Distance Buckets (Window 1 — 10:00–10:15)

| Distance to Level | Either Side Swept |
|------------------|-------------------|
| 0–5 pts | 90–95% |
| 5–10 pts | 85–88% |
| 10–20 pts | 70–75% |
| 20–40 pts | 45–55% |
| 40–80 pts | 25–30% |
| 80+ pts | <15% |

This is the strongest regime discovered in the entire study.

## Key Finding #3 — State × Distance × Window (The Core Edge)

### Forced Resolution Regime

- **Candle**: 6–10
- **Window**: 10:00–10:15
- **State**: neither side swept
- **Distance**: ≤ 5 pts

➡️ **~90–95% probability** that one side is swept.

This regime is:

- Mechanical
- Time-bounded
- Direction-agnostic
- Repeatable

### Distance Decay After Window 1

If a ≤5pt level fails to sweep in Window 1:

| Window | Sweep Probability |
|--------|-------------------|
| 10:15–10:30 | ~70% |
| 10:30–10:45 | <40% |

Failure to sweep early is meaningful information.

## Candle Comparison Summary

| Candle | Behavior |
|--------|----------|
| 06–10 | Strong, urgent resolution |
| 02–06 | Slower, distributed resolution |
| 22–02 | Weak, narrative-dependent |

---

## Practical Implications

- Close liquidity resolves quickly or not at all
- The first 45 minutes post-formation matter most
- Distance is more important than candle type
- After ~10:45 ET, sweep expectancy drops sharply

---

## High-Value Follow-Up Questions

### Asymmetric distance
- One side ≤5 pts, other ≥20 pts
- Does the close side sweep at >90%?

### Window-1 failure regimes
- How probability decays after first failure

### Trend-day conditioning
- Does early sweep probability increase on trend days?

---

## Non-Goals (Intentionally Excluded)

- Pre-open range sweeps
- Touch-based definitions
- Late-day sweep chasing
- Both-sides sweep bias

---

## Summary

This study isolates a high-confidence liquidity regime driven by:

- Candle completion
- Distance at formation
- Early post-formation timing

It provides a clean statistical foundation for post-10:00 NY execution models on NQ.