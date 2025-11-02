## Questions
What are the patterns and behaviors of 15-minute candles in NQ RTH sessions?

This analysis answers 7 major categories of questions:

1. **Candle Structure & Volatility**: Average/median candle sizes, wick-to-body ratios, directional bias
2. **Candle Sequence Behavior**: Continuation/reversal probabilities after specific patterns
3. **Liquidity & High/Low Behavior**: Liquidity sweeps, session high/low timing, opening range persistence
4. **Session Phase Comparison**: Volatility patterns across different time blocks
5. **Bias & Context Integration**: How daily bias affects first candle behavior, VWAP correlation
6. **Reversion & Mean Behavior**: Mean reversion patterns after extremes
7. **Microstructure & Volume**: Volume analysis, FVG detection, absorption patterns

## How to run
```bash
python candle_behavior_analysis.py
```

## Inputs (from data/)
- glbx-mdp3-20200927-20250926.ohlcv-1m.csv

## Outputs
- candle_structure_stats.csv: Overall and per-block candle statistics
- candle_sequence_behavior.csv: Sequence patterns and conditional probabilities
- liquidity_high_low_behavior.csv: Liquidity sweep and high/low timing analysis
- session_phase_comparison.csv: Phase comparisons and volatility expansion times
- bias_context_integration.csv: Daily bias correlations and context integration
- reversion_mean_behavior.csv: Mean reversion patterns after extremes
- microstructure_volume.csv: Volume analysis, FVG detection, and efficiency metrics

