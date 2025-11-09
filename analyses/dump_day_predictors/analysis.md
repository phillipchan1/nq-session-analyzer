# 📈 45-Minute Opening Move Predictive Interpretation (NQ)

_This document interprets the 45-minute predictive analysis model.  
It is designed for both human readers (traders) and automated premarket agents._

---

## 1. Overview

This model forecasts the likelihood of a **large directional move** during the **first 45 minutes of the NY session (09:30–10:15 ET)**.

It evaluates pre-open conditions (HTF structure, volatility, overnight activity) to determine probabilities of:

| Label | Definition | Formula |
|-------|-------------|----------|
| **BigMove45** | Absolute move ≥ 0.75 × prior 20-day RTH ATR | `abs(close_10:15 - open_09:30) ≥ 0.75 × ATR20_RTH_prev` |
| **ExtremeMove45** | Absolute move ≥ 200 pts | `abs(close_10:15 - open_09:30) ≥ 200` |
| **Dump45** | Downward open drive ≥ 0.75 × ATR20_RTH_prev | `close_10:15 < open_09:30 AND (open_09:30 - low_10:15) ≥ 0.75 × ATR20_RTH_prev` |
| **Pump45** | Upward open drive ≥ 0.75 × ATR20_RTH_prev | `close_10:15 > open_09:30 AND (high_10:15 - open_09:30) ≥ 0.75 × ATR20_RTH_prev` |

---

## 2. Baseline Probabilities

| Label | Probability | Frequency | Description |
|-------|--------------|------------|--------------|
| **BigMove45** | **3.7 %** | ~1 in 27 sessions | Large 45-min impulse. |
| **ExtremeMove45** | **2.7 %** | ~1 in 37 sessions | “God candle” open drives ≥ 200 pts. |
| **Dump45** | **5.0 %** | ~1 in 20 sessions | Hard open-drive down. |
| **Pump45** | **2.6 %** | ~1 in 38 sessions | Hard open-drive up. |

These represent unconditional (baseline) probabilities — the chance of each event on a random trading day.

---

## 3. Conditional Probabilities (Key Predictors)

### 3.1 Dump45 — Downward 45-Minute Drives

| Condition | Probability | Baseline Δ | Interpretation |
|------------|--------------|-------------|----------------|
| **4H below lower Keltner** | 16.9 % | +11.9 pp / 3.4× lift | Bearish continuation regime. |
| **1H Keltner position Q1** | 15.8 % | +10.8 pp / 3.2× lift | Hourly stretched low. |
| **1H RSI Q1 (< 30)** | 12.7 % | +7.7 pp / 2.5× lift | Momentum exhaustion → follow-through. |
| **4H Keltner position Q1** | 11.9 % | +6.9 pp / 2.4× lift | 4H oversold. |
| **Overnight range vs 10d avg Q4** | 11.5 % | +6.5 pp / 2.3× lift | High-volatility overnight sets up continuation. |

> **Summary Insight:**  
> When the 4H and 1H are oversold and overnight volatility is high,  
> the odds of a dump triple from 5 % → 15–17 %.  
> These are **high-energy continuation mornings** — expect fast liquidity sweeps and follow-through.

---

### 3.2 Pump45 — Upward 45-Minute Drives

| Condition | Probability | Baseline Δ | Interpretation |
|------------|--------------|-------------|----------------|
| **Overnight range vs 10d avg Q4** | 6.2 % | +3.6 pp / 2.4× lift | Volatility regime expansion. |
| **1H RSI Q4 (> 70)** | 4.7 % | +2.1 pp / 1.8× lift | Strong bullish momentum. |
| **1H Keltner position Q4** | 4.7 % | +2.1 pp / 1.8× lift | Hourly compression breakout. |
| **Open below yesterday’s low** | 5.2 % | +2.6 pp / 2× lift | Fake-break → short-covering drive. |
| **Week-range position Q1** | 5.3 % | +2.7 pp / 2× lift | Market near weekly lows → squeeze regime. |

> **Summary Insight:**  
> Pump days are **short-covering squeezes**.  
> Look for large overnight ranges, pre-open dips below prior lows,  
> and HTF RSI/Keltner strength → open-drive reversal.

---

### 3.3 BigMove45 — Any Large 45-Minute Drive (Direction-Agnostic)

| Condition | Probability | Baseline Δ | Interpretation |
|------------|--------------|-------------|----------------|
| **4H below lower Keltner** | 11.3 % | +7.6 pp / 3× lift | Pre-open imbalance primed for expansion. |
| **1H Keltner position Q1** | 9.0 % | +5.3 pp / 2.4× lift | 1H oversold / volatility compression. |
| **Overnight range vs 10d avg Q4** | 8.7 % | +5 pp / 2.3× lift | High overnight volatility. |
| **4H Keltner position Q1** | 8.1 % | +4.4 pp / 2.2× lift | 4H structure imbalance. |
| **1H RSI Q1** | 7.5 % | +3.8 pp / 2× lift | Momentum-driven expansion. |

---

## 4. Probability Tiering (for Agents)

### Tier Boundaries

| Tier | Definition | Probability Range | Agent Interpretation |
|------|-------------|-------------------|----------------------|
| **Low Energy** | No significant pre-open stretch or volatility | 0–7 % | Expect chop; normal sizing. |
| **Moderate Energy** | One predictive feature active | 8–14 % | Normal trade selection; tighten risk. |
| **High Energy** | ≥ 2 strong predictors aligned | 15–20 %+ | Expansion likely; widen targets / allow follow-through. |

Agents should use **relative lift vs baseline** as the key metric:  
even +5 pp (e.g., 5 % → 10 %) represents a **2× increase in expected event rate**.

---

## 5. Interpretation Framework

1. **Baseline Context**
   - Random days produce 45-minute moves ≥ 0.75×ATR only ~4 % of the time.  
     Anything that triples that probability is significant.

2. **Relative Lift > Absolute Probability**
   - Don’t dismiss “15 %” as small — it’s 3× baseline; that’s meaningful signal power.

3. **Use as Context Filter, not Trigger**
   - Signals define **when to expect energy**, not **where to enter**.  
     They inform bias, sizing, and patience level at the open.

4. **Practical Application**
   - If **High-Energy Dump45** → expect drive selling, avoid early longs, press shorts on structure.  
   - If **High-Energy Pump45** → anticipate reversal squeeze, be ready for strong long bias.  
   - If **Low-Energy** → avoid overtrading; focus on scalps or sit out.

5. **Extremes (200-Point “God Candle” Days)**
   - Occur ~2–3 % of sessions.  
   - 54 % overlap with BigMove45 → tail of same distribution.  
   - Best treated as **exceptional extensions**, not separate regime.

---

## 6. Agent Decision Schema (JSON-Friendly Outline)

```json
{
  "labels": {
    "BigMove45": "abs(close_10:15 - open_09:30) >= 0.75 * ATR20_RTH_prev",
    "ExtremeMove45": "abs(close_10:15 - open_09:30) >= 200",
    "Dump45": "close_10:15 < open_09:30 AND (open_09:30 - low_10:15) >= 0.75 * ATR20_RTH_prev",
    "Pump45": "close_10:15 > open_09:30 AND (high_10:15 - open_09:30) >= 0.75 * ATR20_RTH_prev"
  },
  "baselines": {
    "BigMove45": 0.037,
    "ExtremeMove45": 0.027,
    "Dump45": 0.050,
    "Pump45": 0.026
  },
  "key_predictors": {
    "Dump45": ["h4_below_lower_kc", "h1_kc_pos_prev_Q1", "h1_rsi_14_prev_Q1", "overnight_range_vs_10d_avg_Q4"],
    "Pump45": ["overnight_range_vs_10d_avg_Q4", "open_below_prev_low", "week_range_pos_Q1", "h1_rsi_14_prev_Q4"],
    "BigMove45": ["h4_below_lower_kc", "h1_kc_pos_prev_Q1", "overnight_range_vs_10d_avg_Q4"]
  },
  "probability_tiers": {
    "low_energy": [0, 0.07],
    "moderate_energy": [0.08, 0.14],
    "high_energy": [0.15, 1.00]
  }
}


7. Trader Notes

Statistical lift (×3) is meaningful even if absolute probability < 20 %.

Use it as a filter: trade harder when conditions align, stay small otherwise.

Stack predictors: overlapping conditions improve reliability dramatically.

Integrate volume metrics next for even sharper regime detection.

Expect asymmetry: dumps occur roughly twice as often as pumps.

Summary Statement for Agents & Humans

“When 4H and 1H structure are stretched and overnight volatility is high,
the probability of a 45-minute open-drive (Dump45) rises from 5 % → ~17 %.
That’s a 3× lift, indicating a high-energy continuation morning.
Treat probabilities as context — not certainties — and size accordingly.”