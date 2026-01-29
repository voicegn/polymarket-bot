# 📈 Trading Strategies

Comprehensive guide to the trading strategies implemented in the Polymarket Trading Bot.

## Table of Contents

1. [Strategy Overview](#strategy-overview)
2. [Core Strategy: Edge-Based Trading](#core-strategy-edge-based-trading)
3. [Compound Growth Strategy](#compound-growth-strategy)
4. [Copy Trading](#copy-trading)
5. [Advanced Strategies](#advanced-strategies)
6. [Risk Management Integration](#risk-management-integration)
7. [Strategy Selection Guide](#strategy-selection-guide)
8. [Performance Metrics](#performance-metrics)

---

## Strategy Overview

The bot implements multiple trading strategies that can be used independently or combined:

| Strategy | Risk Level | Capital Requirement | Best For |
|----------|-----------|---------------------|----------|
| Edge-Based | Medium | $500+ | General trading |
| Compound Growth | Medium-High | $1000+ | Aggressive growth |
| Copy Trading | Low-Medium | $500+ | Following experts |
| Arbitrage | Low | $5000+ | Market inefficiencies |
| Trend Detection | Medium | $500+ | Momentum plays |

---

## Core Strategy: Edge-Based Trading

### Overview

The fundamental strategy uses LLM-powered probability estimation to identify mispriced markets.

### How It Works

```
1. SCAN: Monitor active markets
     ↓
2. ANALYZE: LLM estimates "true" probability
     ↓
3. COMPARE: Calculate edge = model_prob - market_price
     ↓
4. FILTER: Check edge > threshold AND confidence > threshold
     ↓
5. SIZE: Kelly criterion position sizing
     ↓
6. EXECUTE: Place limit order
```

### Edge Calculation

```
Edge = Model Probability - Market Price

Example:
- Market: "Will BTC hit $100k by March?"
- Market Price: 45% ($0.45 for Yes)
- LLM Assessment: 53% probability
- Edge: 53% - 45% = +8%

If edge > min_edge (6%), generate BUY signal
```

### Kelly Criterion Sizing

The Kelly formula determines optimal bet size:

```
Kelly % = (p × b - q) / b

Where:
- p = win probability (model confidence)
- q = 1 - p (loss probability)
- b = odds (payout ratio)

Example:
- Model confidence: 60%
- Market price: 45% (odds = 0.55/0.45 = 1.22)
- Kelly: (0.60 × 1.22 - 0.40) / 1.22 = 27%
- Fractional Kelly (0.35): 27% × 0.35 = 9.5% of portfolio
```

### Configuration

```toml
[strategy]
min_edge = 0.06           # 6% minimum edge to trade
min_confidence = 0.60     # 60% model confidence required
kelly_fraction = 0.35     # Use 35% of Kelly (conservative)
scan_interval_secs = 180  # Check markets every 3 minutes
```

### When to Use

✅ **Good for:**
- General market trading
- Diverse market coverage
- Consistent, systematic approach

⚠️ **Watch out for:**
- LLM hallucination on obscure topics
- Markets with hidden information
- Very short timeframes

---

## Compound Growth Strategy

### Overview

Enhanced strategy that dynamically adjusts position sizing based on performance, optimized for long-term capital growth.

### Key Features

1. **Dynamic Kelly Multiplier** (0.5x - 2.0x)
   - Increases on winning streaks
   - Decreases after losses
   - Prevents overconfidence and tilt

2. **Sqrt Balance Scaling**
   - Position size grows with √(balance/initial)
   - 4x your capital → only 2x position size
   - Prevents overexposure as you grow

3. **Drawdown Protection**
   - -10% drawdown: Reduce to 0.7x sizing
   - -20% drawdown: Reduce to 0.5x sizing
   - Prevents catastrophic losses

### Dynamic Kelly Logic

```rust
// Win streak bonus
if win_streak >= 3 {
    kelly_mult = min(1.0 + (win_streak as f64 * 0.1), 2.0)
}

// Lose streak penalty  
if lose_streak >= 2 {
    kelly_mult = max(1.0 - (lose_streak as f64 * 0.15), 0.5)
}

// Drawdown adjustment
let drawdown = (peak_balance - current_balance) / peak_balance;
if drawdown > 0.20 {
    kelly_mult *= 0.5;
} else if drawdown > 0.10 {
    kelly_mult *= 0.7;
}
```

### Balance Scaling

```
Growth Factor = √(current_balance / initial_balance)

Example:
- Initial: $1,000
- Current: $4,000
- Factor: √(4000/1000) = 2.0

A $100 position at $1k becomes $200 position at $4k (not $400)
```

### Configuration

```toml
[strategy]
compound_enabled = true
kelly_fraction = 0.35

[compound]
min_kelly_mult = 0.5      # Floor multiplier
max_kelly_mult = 2.0      # Ceiling multiplier
drawdown_threshold_1 = 0.10
drawdown_threshold_2 = 0.20
```

### When to Use

✅ **Good for:**
- Long-term capital growth
- Accounts with $1000+
- Patient traders

⚠️ **Watch out for:**
- Can be slow during drawdowns
- Requires discipline to trust the system

---

## Copy Trading

### Overview

Follow successful traders by monitoring their positions and replicating their trades with configurable parameters.

### How It Works

```
1. IDENTIFY: List of top traders (username/address)
     ↓
2. MONITOR: Poll for new positions
     ↓
3. FILTER: Check trader's recent performance
     ↓
4. DELAY: Wait configurable time (avoid front-running detection)
     ↓
5. SCALE: Apply copy ratio to their position size
     ↓
6. EXECUTE: Place matching trade
```

### Finding Top Traders

Top traders can be found on:
- Polymarket leaderboard
- Third-party analytics (PolymarketAnalytics, etc.)
- On-chain analysis tools

Look for:
- High win rate (>60%)
- Consistent profits over time
- Reasonable position sizes
- Active recent trading

### Configuration

```toml
[copy_trade]
enabled = true
follow_users = ["CRYINGLITTLEBABY", "leocm", "paspor"]
copy_ratio = 0.5          # Copy 50% of their position
delay_secs = 30           # Wait 30s before copying
min_trader_profit = 1000  # Only copy traders with $1k+ profit
```

### Copy Ratio Guidelines

| Ratio | Risk | Use Case |
|-------|------|----------|
| 0.1-0.3 | Low | Testing, small account |
| 0.3-0.5 | Medium | Standard following |
| 0.5-0.8 | High | High-conviction traders |
| 0.8-1.0 | Very High | Maximum replication |

### Delay Strategy

- **No delay:** Maximum correlation but detectable
- **30s:** Good balance
- **60s+:** Lower detection risk, may miss fast moves

### When to Use

✅ **Good for:**
- Learning trading patterns
- Limited time for analysis
- Diversifying signal sources

⚠️ **Watch out for:**
- Trader performance can change
- Front-running by others
- Different capital = different risk

---

## Advanced Strategies

### Arbitrage Detection

Finds markets where Yes + No prices sum to less than $1.

```rust
pub struct ArbitrageOpportunity {
    market_id: String,
    yes_price: Decimal,
    no_price: Decimal,
    gap: Decimal,  // 1.0 - (yes + no)
    expected_profit: Decimal,
}
```

**Configuration:**
```toml
[arbitrage]
enabled = true
min_gap = 0.02  # 2% minimum arbitrage gap
```

### Trend Detection

Identifies momentum and reversal patterns in market prices.

**Signals:**
- Breakout (price crosses moving average)
- Reversal (oversold/overbought conditions)
- Momentum (sustained price movement)

```toml
[trend]
enabled = true
lookback_periods = 10
breakout_threshold = 0.05
```

### Volatility-Adaptive Exits

Adjusts take-profit and stop-loss based on market volatility.

```rust
pub enum VolatilityRegime {
    Low,     // Tight stops, quick profits
    Normal,  // Standard parameters  
    High,    // Wide stops, larger targets
}
```

### Signal Aggregation

Combines signals from multiple sources with weighted confidence.

```
Final Signal = Σ(source_weight × source_confidence × source_direction)

Sources:
- LLM Analysis (weight: 1.0)
- Copy Trades (weight: 0.7)
- Trend Detection (weight: 0.5)
- Arbitrage (weight: 0.3)
```

---

## Risk Management Integration

Every strategy passes through the risk management layer:

### Pre-Trade Checks

```
1. Daily loss limit not exceeded
2. Position count < max_open_positions
3. Position size < max_position_pct
4. Total exposure < max_exposure_pct
5. Market quality score acceptable
6. Correlation check (not too similar to existing positions)
```

### Position Sizing Flow

```
Signal Suggested Size
        ↓
    [Kelly Calculation]
        ↓
    [Compound Adjustment]
        ↓
    [Volatility Adjustment]
        ↓
    [Risk Limit Caps]
        ↓
    [Balance Check]
        ↓
Final Position Size
```

### Risk Configuration

```toml
[risk]
max_position_pct = 0.05    # 5% max per position
max_exposure_pct = 0.50    # 50% max total exposure
max_daily_loss_pct = 0.10  # 10% daily loss limit
min_balance_reserve = 100  # Keep $100 reserve
max_open_positions = 10    # Max concurrent positions
```

---

## Strategy Selection Guide

### By Account Size

| Balance | Recommended Strategy |
|---------|---------------------|
| <$500 | Copy Trade (0.3x) |
| $500-$2000 | Edge-Based |
| $2000-$10000 | Compound Growth |
| >$10000 | Multi-strategy |

### By Risk Tolerance

| Risk Appetite | Configuration |
|--------------|---------------|
| Conservative | kelly=0.25, max_pos=3%, compound=off |
| Moderate | kelly=0.35, max_pos=5%, compound=on |
| Aggressive | kelly=0.50, max_pos=8%, compound=on |

### By Time Commitment

| Time Available | Strategy |
|---------------|----------|
| None (set & forget) | Copy Trade |
| Daily check | Edge-Based + Compound |
| Active monitoring | All strategies |

---

## Performance Metrics

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Win Rate | % of profitable trades | >55% |
| Average Edge | Mean edge of trades | >5% |
| Sharpe Ratio | Risk-adjusted return | >1.5 |
| Max Drawdown | Largest peak-to-trough | <20% |
| Profit Factor | Gross profit / gross loss | >1.5 |

### Tracking Performance

```bash
# View recent trades
polymarket-bot trades --limit 20

# Performance summary
polymarket-bot status

# Export for analysis
sqlite3 data/trades.db "SELECT * FROM trades;" > trades.csv
```

### Performance Calculations

```rust
// Win Rate
win_rate = winning_trades / total_trades

// Sharpe Ratio
sharpe = (avg_return - risk_free) / std_dev(returns)

// Maximum Drawdown
max_dd = max(peak - trough) / peak
```

---

## Strategy Development

### Adding a New Strategy

1. Create module in `src/strategy/`
2. Implement signal generation
3. Add configuration options
4. Write tests (required)
5. Integrate with strategy engine

### Backtesting

```rust
// Use backtest module
let backtest = Backtest::new(historical_data);
let results = backtest.run(&strategy, &config);

println!("Win Rate: {:.1}%", results.win_rate * 100.0);
println!("Total Return: {:.1}%", results.total_return * 100.0);
```

### Strategy Evaluation Checklist

- [ ] Edge calculation is sound
- [ ] Risk management integrated
- [ ] Tested on historical data
- [ ] Works in dry-run mode
- [ ] Handles edge cases (no liquidity, closed markets)

---

## References

- [Kelly Criterion Paper](https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf)
- [Polymarket Documentation](https://docs.polymarket.com)
- [Prediction Market Theory](https://en.wikipedia.org/wiki/Prediction_market)

---

*Last updated: 2026-01-29*
