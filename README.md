# 🎲 Polymarket Trading Bot

> ⚠️ **DISCLAIMER: FOR RESEARCH PURPOSES ONLY**
> 
> 🇺🇸 **EN**: This project is strictly for educational and research purposes. Users are solely responsible for compliance with all applicable laws and regulations in their jurisdiction.
> 
> 🇨🇳 **中文**: 本项目仅供教育和研究目的。用户须自行负责遵守所在地区的法律法规。
> 
> 🇯🇵 **日本語**: 本プロジェクトは教育・研究目的のみ。ユーザーは現地の法規制を遵守する責任を負います。
> 
> 🇪🇸 **ES**: Solo para fines educativos e investigación. Los usuarios deben cumplir con las leyes locales.
> 
> 🇫🇷 **FR**: Uniquement à des fins éducatives et de recherche. Respectez les lois locales.
> 
> 🇩🇪 **DE**: Nur für Bildungs- und Forschungszwecke. Lokale Gesetze beachten.
> 
> 🇷🇺 **RU**: Только для исследовательских и образовательных целей.
> 
> 🇸🇦 **AR**: للأغراض البحثية والتعليمية فقط.
> 
> 📜 **[Full Disclaimer in 16 Languages → LICENSE](LICENSE)**
> 
> ---
> 
> **Support This Research / 赞助研究 / 研究支援:**
> 
> | Network | Address |
> |---------|---------|
> | EVM (ETH/Polygon/BSC) | `0x5b8A5c95e3C74b6673cAda74649264242EbEe077` |
> | Solana | `3gxSjqv154cDysYuoMxUcMMZ1wnGFDtLnT21w3xueiuf` |
> | TRON | `TQL1dgCxMUYiqnhYL5VSzKZCdsXTdzeJ7S` |
> | Bitcoin | `bc1qrngacl69znhujy6m83cpzsyf5j9lzdd5qdxenv` |

---

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1144%20passing-green.svg)]()

A high-performance, institutional-grade automated trading system for [Polymarket](https://polymarket.com) prediction markets. Built in Rust for speed, safety, and reliability.

## ✨ Features

### 🤖 Intelligent Trading
- **LLM-Powered Analysis** - DeepSeek, Claude, GPT, or local Ollama for market probability estimation
- **Kelly Criterion Sizing** - Mathematically optimal position sizing based on edge and confidence
- **Signal Generation** - Automatic edge detection when model predictions diverge from market prices
- **Multi-Source Signals** - Aggregate insights from Telegram, Twitter/X, and on-chain data

### 📈 Advanced Strategies
- **Compound Growth** - Dynamic Kelly with sqrt scaling (4x balance → 2x sizing)
- **Copy Trading** - Follow top traders with configurable ratio and delay
- **Trend Detection** - Real-time momentum and reversal signals
- **Take Profit/Stop Loss** - Automated exit strategies

### 🛡️ Enterprise Risk Management
- **Position Limits** - Max 5-10% per position, 50% total exposure
- **Daily Loss Limits** - Auto-stop at configurable drawdown
- **Drawdown Protection** - Auto-reduce sizing at -10% and -20%
- **Smart Execution** - Depth analysis, limit orders, retry logic

### 📊 Monitoring & Alerts
- **Telegram Notifications** - Real-time signals, trades, and daily reports
- **Performance Tracking** - Win rate, PnL, Sharpe ratio
- **Dry Run Mode** - Paper trading for strategy validation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POLYMARKET TRADING BOT                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                     │
│  │   INGESTER   │   │  COPY TRADE  │   │   SCANNER    │                     │
│  │  TG/X/Chain  │   │  Top Traders │   │   Markets    │                     │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                     │
│         │                  │                  │                              │
│         ▼                  ▼                  ▼                              │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │              LLM PROCESSOR (DeepSeek/Claude)         │                    │
│  │         Signal Extraction / Probability Modeling     │                    │
│  └──────────────────────────┬──────────────────────────┘                    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │                 STRATEGY ENGINE                       │                    │
│  │   ┌─────────┐  ┌──────────┐  ┌────────────────┐     │                    │
│  │   │ Signal  │  │ Compound │  │ Risk Manager   │     │                    │
│  │   │  Gen    │→ │  Growth  │→ │ Kelly + Limits │     │                    │
│  │   └─────────┘  └──────────┘  └────────────────┘     │                    │
│  └──────────────────────────┬──────────────────────────┘                    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────┐                    │
│  │              SMART EXECUTOR                           │                    │
│  │    Depth Analysis → Limit Orders → Retry Logic       │                    │
│  └──────────────────────────┬──────────────────────────┘                    │
│                             │                                                │
│                             ▼                                                │
│  ┌───────────────┐   ┌─────────────┐   ┌───────────────┐                    │
│  │   POLYMARKET  │   │   STORAGE   │   │   NOTIFIER    │                    │
│  │   CLOB API    │   │   SQLite    │   │   Telegram    │                    │
│  └───────────────┘   └─────────────┘   └───────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Rust 1.75+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- Polymarket wallet with USDC on Polygon
- LLM API key (DeepSeek recommended for cost-effectiveness)

### Installation

```bash
# Clone the repository
git clone https://github.com/voicegn/polymarket-bot.git
cd polymarket-bot

# Build release binary
cargo build --release

# Copy and configure
cp config.example.toml config.toml
cp .env.example .env

# Edit configuration (see Configuration section)
nano config.toml
nano .env
```

### Running

```bash
# Start the bot (dry run mode first!)
./target/release/polymarket-bot run --dry-run

# When ready for live trading
./target/release/polymarket-bot run

# Or use the start script
./start.sh
```

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Required: LLM API Key
DEEPSEEK_API_KEY=sk-xxx

# Required: Polymarket wallet
POLYMARKET_PRIVATE_KEY=your_wallet_private_key_without_0x

# Required: Telegram notifications
TELEGRAM_BOT_TOKEN=123456:ABC-xxx
TELEGRAM_CHAT_ID=your_chat_id
```

### Main Configuration (`config.toml`)

```toml
# LLM Configuration
[llm]
provider = "deepseek"          # deepseek | anthropic | openai | ollama
model = "deepseek-chat"

# Strategy Settings
[strategy]
min_edge = 0.06                # 6% minimum edge to trade
min_confidence = 0.60          # 60% model confidence threshold
kelly_fraction = 0.35          # 35% Kelly (conservative)
compound_enabled = true        # Enable compound growth
scan_interval_secs = 180       # Scan markets every 3 minutes

# Risk Management
[risk]
max_position_pct = 0.05        # 5% max per position
max_exposure_pct = 0.50        # 50% max total exposure
max_daily_loss_pct = 0.10      # 10% daily loss limit
min_balance_reserve = 100      # Keep $100 reserve
max_open_positions = 10        # Max concurrent positions

# Copy Trading (optional)
[copy_trade]
enabled = true
follow_users = ["CRYINGLITTLEBABY", "leocm"]
copy_ratio = 0.5               # 50% of their size
delay_secs = 30                # Delay to avoid detection
```

📖 See [docs/MANUAL.md](docs/MANUAL.md) for complete configuration reference.

## 📁 Project Structure

```
polymarket-bot/
├── src/
│   ├── main.rs              # CLI entry point & main loop
│   ├── lib.rs               # Library exports
│   ├── config.rs            # Configuration management
│   ├── types.rs             # Core types (Market, Signal, Trade)
│   ├── error.rs             # Error handling
│   │
│   ├── client/              # Polymarket API clients
│   │   ├── clob.rs          # Order book & trading
│   │   ├── gamma.rs         # Market data
│   │   ├── websocket.rs     # Real-time streaming
│   │   └── auth.rs          # Signing & authentication
│   │
│   ├── model/               # Probability models
│   │   ├── llm.rs           # LLM providers (DeepSeek, Claude, etc.)
│   │   └── sentiment.rs     # Sentiment analysis
│   │
│   ├── strategy/            # Trading strategies
│   │   ├── mod.rs           # SignalGenerator (Kelly criterion)
│   │   ├── compound.rs      # Compound growth strategy
│   │   ├── copy_trade.rs    # Copy trading
│   │   ├── crypto_hf.rs     # Crypto high-frequency
│   │   ├── trend_detector.rs
│   │   └── take_profit.rs
│   │
│   ├── executor/            # Trade execution
│   │   ├── mod.rs           # Base executor with risk checks
│   │   ├── smart_executor.rs # Advanced execution (depth, retry)
│   │   └── gradual_exit.rs  # Gradual position unwinding
│   │
│   ├── ingester/            # Signal collection
│   │   ├── telegram.rs      # Telegram channel monitoring
│   │   ├── twitter.rs       # Twitter/X monitoring
│   │   ├── binance.rs       # Crypto price feeds
│   │   └── processor.rs     # LLM signal extraction
│   │
│   ├── analysis/            # Pattern recognition
│   │   └── pattern.rs       # Trading pattern detection
│   │
│   ├── notify/              # Notifications
│   │   └── mod.rs           # Telegram notifier
│   │
│   ├── storage/             # Persistence
│   │   └── mod.rs           # SQLite database
│   │
│   └── monitor/             # Performance tracking
│       └── mod.rs           # Trade monitoring & stats
│
├── docs/
│   ├── MANUAL.md            # Operations manual
│   ├── API.md               # API reference
│   └── STRATEGY_ANALYSIS.md # Strategy deep dive
│
├── config.example.toml      # Configuration template
├── .env.example             # Environment template
└── Cargo.toml               # Dependencies
```

## 🖥️ CLI Commands

```bash
# Run the trading bot
polymarket-bot run [--dry-run] [--config <path>]

# List active markets
polymarket-bot markets [--limit <n>] [--min-volume <usd>]

# Analyze a specific market
polymarket-bot analyze <market_id>

# Check bot status and positions
polymarket-bot status

# View recent trades
polymarket-bot trades [--limit <n>]

# Get help
polymarket-bot --help
```

### 📝 Paper Trading CLI

Simulate trades with real market data without risking funds:

```bash
# Check paper trading account status
cargo run --bin paper_cli -- status

# Buy shares (search by keyword)
cargo run --bin paper_cli -- buy -m "trump" -s yes -a 50

# Buy shares (by market ID)
cargo run --bin paper_cli -- buy --id -m 517310 -s yes -a 50

# View open positions
cargo run --bin paper_cli -- positions

# Sell a position
cargo run --bin paper_cli -- sell -p <position_id>

# View trade history
cargo run --bin paper_cli -- history --limit 10
```

State is persisted to `paper_trading_state.json`.

## 📊 Trading Strategies

### 1. Edge-Based Trading (Default)
- LLM estimates "true" probability
- Compares to market price
- Trades when edge > 6% with confidence > 60%
- Position sized by Kelly criterion

### 2. Compound Growth
- Dynamic Kelly multiplier (0.5x - 2.0x)
- Increases on win streaks, decreases on losses
- Sqrt scaling: 4x balance → 2x position size
- Drawdown protection at -10% and -20%

### 3. Copy Trading
- Follow successful traders by address
- Configurable copy ratio (10% - 100%)
- Delay execution to avoid front-running detection

### 4. Signal Aggregation
- Monitor Telegram alpha channels
- Follow Twitter/X KOLs
- Aggregate and weight signals by source trust

## ⚠️ Risk Warning

**This bot trades real money. Use at your own risk.**

- 💸 Start with small amounts you can afford to lose
- 🧪 Always test in dry-run mode first
- 👀 Monitor closely, especially initially
- 📉 Prediction markets can be highly volatile
- 🔒 Never share your private key

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific module tests
cargo test strategy::
cargo test executor::

# Run integration tests
cargo test --test integration
```

**Current test coverage: 544 tests passing**

## 📈 Performance

| Metric | Value |
|--------|-------|
| Build Time | ~45s (release) |
| Memory Usage | ~50MB idle |
| API Latency | <100ms avg |
| Scan Cycle | 3 min default |

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 Documentation

- [Operations Manual](docs/OPERATIONS.md) - Deployment, monitoring, and troubleshooting
- [Configuration Guide](docs/MANUAL.md) - Complete configuration reference
- [Trading Strategies](docs/STRATEGY.md) - Strategy implementation guide
- [API Reference](docs/API.md) - Public modules and functions

---

<div align="center">
  <b>Built with 🦀 Rust for maximum performance and safety</b>
  <br>
  <sub>Not financial advice. Trade responsibly.</sub>
</div>
