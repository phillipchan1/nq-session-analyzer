# NY Open Liquidity Sweeps – Key Insights

## Availability (from availability_summary.csv)

| Within | % of Days |
|--------|-----------|
| 50 pts | ~59% |
| 100 pts | ~86% |
| 150 pts | ~95% |
| 200 pts | ~97% |

**Takeaway:** Most days (97%) have at least one target level within 200 pts. About 59% have one within 50 pts (very tradeable).

---

## Sweep Rate by Distance (from sweep_by_distance.csv)

**Distance = measuring variable.** Closer levels get hit very commonly; farther levels less so.

| Distance | Typical sweep rate |
|----------|---------------------|
| 0–25 pts | 67–92% (PD H/L highest) |
| 25–50 pts | 47–71% |
| 50–75 pts | 32–58% |
| 75–100 pts | 22–48% |
| 100–150 pts | 10–27% |
| 150–200 pts | 4–18% |
| 200–300 pts | 2–11% |
| 300+ pts | 0–10% |

**Takeaway:** Within 50 pts, sweep rates are 50–92%. Beyond 150 pts, rates drop to &lt;20%. PD H/L at 0–25 pts: 91–92% sweep rate.

---

## First 45 Min Range (from sweep_by_range_redfolder.csv)

| Range quartile | Avg range (pts) | Sweep rate |
|----------------|-----------------|------------|
| Q1 (narrowest) | ~64 | ~70% |
| Q2 | ~100 | ~74% |
| Q3 | ~135 | ~81% |
| Q4 (widest) | ~223 | ~85% |

**Takeaway:** Wider first 45 min range correlates with higher sweep rate. On wide-range days, levels get hit more often.

---

## Red Folder vs Neutral

| Day type | Sweep rate |
|----------|------------|
| Neutral | ~77% |
| Red folder | ~78% |

**Takeaway:** Red folder days behave similarly to neutral days for liquidity sweeps. No strong edge to avoid or favor them.

---

## Actionable Rules

1. **Prioritize levels within 50 pts** – 50%+ sweep rate; within 25 pts, 67–92%.
2. **Avoid levels 150+ pts away** – &lt;20% sweep rate; not worth targeting.
3. **PD H/L near open** – 91–92% sweep rate at 0–25 pts; strongest setup.
4. **Wide first 45 min range** – Higher sweep probability; consider larger targets.
5. **Level clustering (super magnet)** – See clustering_magnet.csv for confluence zone hit rates.
