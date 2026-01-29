# 📖 Polymarket Bot Operations Manual

Complete guide for deploying, configuring, and operating the Polymarket Trading Bot.

## Table of Contents

1. [Installation & Deployment](#installation--deployment)
2. [Configuration Reference](#configuration-reference)
3. [Strategy Guide](#strategy-guide)
4. [Risk Management](#risk-management)
5. [Monitoring & Alerts](#monitoring--alerts)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Installation & Deployment

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 2 GB | 4+ GB |
| Storage | 1 GB | 10+ GB |
| OS | Linux/macOS | Ubuntu 22.04+ |
| Rust | 1.75+ | Latest stable |

### Step 1: Install Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version  # Should be 1.75+
```

### Step 2: Clone & Build

```bash
git clone https://github.com/voicegn/polymarket-bot.git
cd polymarket-bot

# Build release binary (optimized)
cargo build --release

# Binary will be at ./target/release/polymarket-bot
```

### Step 3: Configure

```bash
# Copy templates
cp config.example.toml config.toml
cp .env.example .env

# Edit configuration files
nano config.toml
nano .env
```

### Step 4: Set Up Wallet

1. **Create/Import Wallet**
   - Use MetaMask or any Ethereum wallet
   - Export private key (without `0x` prefix)
   - Add to `.env` as `POLYMARKET_PRIVATE_KEY`

2. **Fund Wallet**
   - Transfer USDC to your wallet on **Polygon mainnet**
   - Ensure you have MATIC for gas (0.1 MATIC is plenty)

3. **Approve Polymarket**
   - Go to [polymarket.com](https://polymarket.com)
   - Connect your wallet
   - Complete any required approvals

### Step 5: Configure Telegram Bot

1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot` and follow prompts
3. Save the bot token to `.env` as `TELEGRAM_BOT_TOKEN`
4. Message [@userinfobot](https://t.me/userinfobot) to get your chat ID
5. Save to `.env` as `TELEGRAM_CHAT_ID`

### Step 6: Get LLM API Key

**DeepSeek (Recommended - Best cost/performance)**
1. Visit [platform.deepseek.com](https://platform.deepseek.com)
2. Create account and get API key
3. Add to `.env` as `DEEPSEEK_API_KEY`

**Alternatives:**
- Anthropic Claude: [console.anthropic.com](https://console.anthropic.com)
- OpenAI: [platform.openai.com](https://platform.openai.com)
- Ollama (Local): No key needed, install Ollama locally

### Step 7: Test & Deploy

```bash
# Test with dry run (no real trades)
./target/release/polymarket-bot run --dry-run

# When ready for live trading
./target/release/polymarket-bot run

# Or run as service (see below)
```

### Running as Systemd Service

```bash
# Copy service file
sudo cp polymarket-bot.service /etc/systemd/system/

# Edit paths if needed
sudo nano /etc/systemd/system/polymarket-bot.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable polymarket-bot
sudo systemctl start polymarket-bot

# Check status
sudo systemctl status polymarket-bot
journalctl -u polymarket-bot -f
```

---

## Configuration Reference

### Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes* | DeepSeek API key |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic API key (if using Claude) |
| `OPENAI_API_KEY` | Yes* | OpenAI API key (if using GPT) |
| `POLYMARKET_PRIVATE_KEY` | Yes | Wallet private key (no 0x prefix) |
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Yes | Your Telegram chat ID |

*One LLM key required depending on provider choice.

### Main Configuration (`config.toml`)

#### `[polymarket]` - Exchange Connection

```toml
[polymarket]
# API endpoints (don't change unless you know what you're doing)
clob_url = "https://clob.polymarket.com"
gamma_url = "https://gamma-api.polymarket.com"

# Your private key (use env var reference)
private_key = "${POLYMARKET_PRIVATE_KEY}"

# Chain ID: 137 = Polygon Mainnet
chain_id = 137

# Signature type:
# 0 = EOA (standard wallet)
# 1 = Magic (Magic.link wallet)
# 2 = Proxy (Polymarket proxy wallet)
signature_type = 0

# Optional: Funder address for proxy wallets
# funder_address = "0x..."
```

#### `[llm]` - LLM Provider

```toml
[llm]
# Provider: deepseek | anthropic | openai | ollama | compatible
provider = "deepseek"

# API key (use env var reference for security)
api_key = "${DEEPSEEK_API_KEY}"

# Model name (optional, uses provider default)
model = "deepseek-chat"

# For OpenAI-compatible APIs (e.g., local LLM servers)
# base_url = "http://localhost:8080/v1"
```

**Provider-specific models:**

| Provider | Default Model | Notes |
|----------|---------------|-------|
| deepseek | deepseek-chat | Best cost/performance |
| anthropic | claude-sonnet-4-20250514 | High quality |
| openai | gpt-4o | Expensive |
| ollama | llama2 | Free, local |

#### `[strategy]` - Trading Strategy

```toml
[strategy]
# Minimum edge required to trade (0.06 = 6%)
# Higher = fewer trades, more selective
# Lower = more trades, potentially more noise
min_edge = 0.06

# Minimum model confidence (0.60 = 60%)
# Higher = only trade high-conviction signals
min_confidence = 0.60

# Kelly criterion fraction (0.35 = 35% Kelly)
# Full Kelly is aggressive; 0.25-0.50 is typical
# Lower = more conservative, slower growth
# Higher = more aggressive, higher volatility
kelly_fraction = 0.35

# Market scan interval (seconds)
# How often to check for new opportunities
scan_interval_secs = 180

# Model update interval (seconds)
# How often to refresh probability estimates
model_update_interval_secs = 900

# Enable compound growth strategy
# Dynamically adjusts sizing based on performance
compound_enabled = true

# Use sqrt scaling for compound (safer)
# If enabled: 4x balance growth → 2x position size
# If disabled: 4x balance → 4x position size
compound_sqrt_scaling = true
```

#### `[risk]` - Risk Management

```toml
[risk]
# Maximum single position size (fraction of portfolio)
# 0.05 = max 5% in any single market
max_position_pct = 0.05

# Maximum total exposure (fraction of portfolio)
# 0.50 = max 50% of portfolio in open positions
max_exposure_pct = 0.50

# Maximum daily loss before stopping
# 0.10 = stop trading if down 10% in a day
max_daily_loss_pct = 0.10

# Minimum balance to keep in reserve (USD)
# For gas, emergencies, etc.
min_balance_reserve = 100

# Maximum number of concurrent positions
max_open_positions = 10
```

**Risk Setting Presets:**

| Profile | max_position | max_exposure | max_daily_loss | kelly_fraction |
|---------|--------------|--------------|----------------|----------------|
| Conservative | 3% | 30% | 5% | 0.20 |
| Moderate | 5% | 50% | 10% | 0.35 |
| Aggressive | 8% | 60% | 15% | 0.50 |

#### `[database]` - Storage

```toml
[database]
# SQLite database path
# Stores trades, positions, market cache
path = "data/polymarket.db"
```

#### `[telegram]` - Notifications

```toml
[telegram]
bot_token = "${TELEGRAM_BOT_TOKEN}"
chat_id = "${TELEGRAM_CHAT_ID}"

# Notification settings (all default to true)
notify_signals = true   # Signal found notifications
notify_trades = true    # Trade execution notifications
notify_errors = true    # Error notifications
notify_daily = true     # Daily performance reports
```

#### `[copy_trade]` - Copy Trading (Optional)

```toml
[copy_trade]
enabled = true

# Usernames to follow (Polymarket usernames)
follow_users = ["CRYINGLITTLEBABY", "leocm", "rename"]

# Wallet addresses to follow (alternative to usernames)
follow_addresses = ["0x..."]

# Copy ratio: how much of their position to mirror
# 0.5 = copy 50% of their position size
copy_ratio = 0.5

# Delay before copying (seconds)
# Avoids front-running detection
delay_secs = 30
```

#### `[ingester]` - Signal Ingestion (Optional)

```toml
[ingester]
enabled = true

# Telegram userbot for private channel monitoring
[ingester.telegram_userbot]
api_id = 12345678
api_hash = "your_api_hash"
session_file = "data/telegram.session"
watch_chats = [-1001234567890]  # Channel IDs

# Twitter/X monitoring
[ingester.twitter]
bearer_token = "your_twitter_bearer_token"
user_ids = ["44196397"]  # Elon Musk
keywords = ["polymarket", "prediction market"]

# Signal processing settings
[ingester.processing]
aggregation_window_secs = 300  # 5 minutes
min_confidence = 0.5
min_agg_score = 0.6

# Author trust scores (0.0 - 1.0)
[ingester.author_trust]
"@elonmusk" = 0.7
"TradingChannel" = 0.8
```

---

## Strategy Guide

### Edge-Based Trading (Core Strategy)

The bot's fundamental strategy:

1. **Probability Estimation**: LLM analyzes market question and estimates "true" probability
2. **Edge Detection**: Compare model estimate to market price
3. **Signal Generation**: If edge > threshold AND confidence > threshold → generate signal
4. **Position Sizing**: Kelly criterion determines optimal bet size
5. **Execution**: Smart executor places limit order with depth analysis

**Example:**
- Market: "Will BTC hit $150k by 2026?" trading at 45%
- Model estimate: 58% with 70% confidence
- Edge: 58% - 45% = 13% → Above 6% threshold ✓
- Kelly size: (0.58 - 0.45) / (1 - 0.45) × 0.35 × 0.70 = 5.8%
- Action: BUY YES at ~$0.45, position = 5.8% of portfolio

### Compound Growth Strategy

Optimizes for long-term growth:

**Dynamic Kelly Multiplier:**
- Win streak ≥3 + high confidence → Kelly × 1.0-2.0
- Lose streak ≥2 → Kelly × 0.5-0.9
- Adjusts based on actual vs expected win rate

**Sqrt Scaling:**
- Balance growth compounds position size
- 4x balance → 2x sizing (not 4x)
- Balances growth with risk management

**Drawdown Protection:**
- -10% drawdown → reduce sizing 25%
- -20% drawdown → reduce sizing 50%
- Preserves capital during losing streaks

### Copy Trading Strategy

Follow successful traders:

1. Monitor trader activity via Polymarket API
2. Detect when followed trader opens/closes position
3. Wait `delay_secs` to avoid detection
4. Mirror trade at `copy_ratio` of their size

**Selection Tips:**
- Check trader history on Polymarket leaderboards
- Look for consistent profitability, not one-time wins
- Diversify across multiple traders
- Start with low copy_ratio (0.25) and increase

### Signal Aggregation

Combine signals from multiple sources:

1. **Collection**: Monitor Telegram channels, Twitter accounts
2. **Extraction**: LLM extracts signal (direction, confidence, timeframe)
3. **Aggregation**: Weight signals by source trust score
4. **Threshold**: Only act when aggregate score > min_agg_score

---

## Risk Management

### Position Sizing Rules

| Rule | Purpose | Default |
|------|---------|---------|
| Max position size | Limit single-market risk | 5% |
| Max exposure | Limit total portfolio risk | 50% |
| Min balance reserve | Keep funds for emergencies | $100 |
| Max open positions | Diversification | 10 |

### Daily Loss Limits

- Bot tracks daily P&L
- If daily loss exceeds `max_daily_loss_pct`, trading stops
- Resets at midnight UTC
- Can be overridden with `--force` flag (not recommended)

### Drawdown Protection

| Drawdown | Action |
|----------|--------|
| 0-10% | Normal operation |
| 10-20% | Reduce position sizes 25% |
| 20%+ | Reduce position sizes 50% |

### Smart Execution

Before placing orders, the executor:

1. **Depth Analysis**: Check orderbook liquidity
2. **Slippage Check**: Ensure < 2% expected slippage
3. **Minimum Liquidity**: Require $50+ available
4. **Limit Orders**: Place at expected fill price + buffer
5. **Retry Logic**: Up to 3 attempts with exponential backoff

### Emergency Stop

```bash
# Graceful shutdown
kill -SIGTERM $(pgrep polymarket-bot)

# Or via systemctl
sudo systemctl stop polymarket-bot

# Emergency: kill all and cancel orders manually
kill -9 $(pgrep polymarket-bot)
# Then cancel orders on polymarket.com
```

---

## Monitoring & Alerts

### Telegram Notifications

| Notification | Trigger | Example |
|--------------|---------|---------|
| Signal Found | New trading opportunity | "🟢 Signal Found: BTC $150k - Edge: +13%" |
| Trade Executed | Order filled | "✅ BOUGHT 100 shares @ $0.45" |
| Error Alert | API error, risk limit | "⚠️ Error: API rate limited" |
| Risk Alert | Limit exceeded | "🚨 Daily loss limit reached: -10.5%" |
| Daily Report | End of day | "📊 Trades: 5, Win Rate: 60%, PnL: +$45" |
| Startup/Shutdown | Bot lifecycle | "🤖 Bot Started (DRY RUN)" |

### Log Files

```bash
# View live logs
tail -f bot.log

# Or via journalctl if running as service
journalctl -u polymarket-bot -f

# Log levels: error, warn, info, debug, trace
# Set via RUST_LOG environment variable
RUST_LOG=polymarket_bot=debug ./target/release/polymarket-bot run
```

### Health Checks

```bash
# Check bot status
./target/release/polymarket-bot status

# Output includes:
# - Current positions
# - Today's P&L
# - Open orders
# - Last activity
# - Error count
```

---

## Troubleshooting

### Common Issues

#### Bot won't start

```
Error: No configuration file found
```
**Solution:** Ensure `config.toml` exists in current directory or specify path:
```bash
./polymarket-bot run --config /path/to/config.toml
```

#### API Authentication Failed

```
Error: Signature verification failed
```
**Solutions:**
1. Check private key is correct (no `0x` prefix)
2. Ensure signature_type matches your wallet type
3. Verify wallet has approved Polymarket contracts

#### LLM Errors

```
Error: LLM request failed: 401 Unauthorized
```
**Solutions:**
1. Check API key is correct
2. Verify key has sufficient credits
3. Check provider status page

#### Order Rejected

```
Error: Order rejected: Insufficient balance
```
**Solutions:**
1. Check USDC balance on Polygon
2. Ensure `min_balance_reserve` leaves room for trade
3. Reduce position size

#### Rate Limiting

```
Error: Rate limit exceeded
```
**Solutions:**
1. Increase `scan_interval_secs`
2. Reduce number of markets monitored
3. Wait 1 minute before retrying

### Debug Mode

```bash
# Enable verbose logging
RUST_LOG=debug ./target/release/polymarket-bot run

# Even more verbose
RUST_LOG=trace ./target/release/polymarket-bot run
```

### Reset State

```bash
# Clear database (loses trade history)
rm data/polymarket.db

# Reset daily P&L (bot will recreate)
sqlite3 data/polymarket.db "DELETE FROM trades WHERE timestamp LIKE '$(date +%Y-%m-%d)%'"
```

---

## Best Practices

### 1. Start Conservative

```toml
[strategy]
min_edge = 0.10        # 10% edge (higher threshold)
kelly_fraction = 0.25  # Quarter Kelly

[risk]
max_position_pct = 0.03  # 3% max position
max_exposure_pct = 0.30  # 30% max exposure
```

### 2. Always Dry Run First

```bash
# Run for at least 24-48 hours in dry run
./polymarket-bot run --dry-run

# Check hypothetical performance before going live
```

### 3. Monitor Actively (Initially)

- First week: Check every few hours
- First month: Check daily
- After validation: Weekly review is fine

### 4. Diversify Signal Sources

- Don't rely on single LLM
- Combine with copy trading
- Use signal aggregation from multiple channels

### 5. Regular Maintenance

```bash
# Weekly: Check logs for errors
journalctl -u polymarket-bot --since "1 week ago" | grep -i error

# Monthly: Review performance
./polymarket-bot trades --limit 100

# Quarterly: Review and adjust parameters
```

### 6. Backup Configuration

```bash
# Backup config (exclude secrets)
cp config.toml config.backup.toml

# Never commit .env to git!
echo ".env" >> .gitignore
```

### 7. Gradual Scaling

| Phase | Duration | Capital | Risk Settings |
|-------|----------|---------|---------------|
| Testing | 1-2 weeks | $100-500 | Conservative |
| Validation | 1 month | $500-2000 | Moderate |
| Scaling | Ongoing | $2000+ | Based on results |

---

## Support

- **Documentation**: [docs/](.)
- **Issues**: [GitHub Issues](https://github.com/voicegn/polymarket-bot/issues)
- **Updates**: `git pull && cargo build --release`

---

*Last updated: 2026-01-29*
