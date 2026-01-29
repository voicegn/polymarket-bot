# 🔧 Operations Manual

Complete guide for deploying, monitoring, and maintaining the Polymarket Trading Bot in production.

## Table of Contents

1. [Deployment](#deployment)
2. [Systemd Service](#systemd-service)
3. [Monitoring](#monitoring)
4. [Logging](#logging)
5. [Troubleshooting](#troubleshooting)
6. [Backup & Recovery](#backup--recovery)
7. [Security](#security)
8. [Maintenance](#maintenance)

---

## Deployment

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 2 GB | 4+ GB |
| Storage | 1 GB | 10+ GB |
| Network | 10 Mbps | 100+ Mbps |
| OS | Linux (x64) | Ubuntu 22.04+ |

### Installation Steps

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev sqlite3

# 2. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 3. Clone repository
git clone https://github.com/voicegn/polymarket-bot.git
cd polymarket-bot

# 4. Build release binary
cargo build --release

# 5. Verify build
./target/release/polymarket-bot --version
```

### Configuration

```bash
# Copy templates
cp config.example.toml config.toml
cp .env.example .env

# Edit configuration
nano config.toml  # See Configuration Reference section
nano .env         # Set API keys
```

### Pre-flight Checklist

Before starting the bot in production:

- [ ] API keys configured (`.env`)
- [ ] Wallet funded with USDC on Polygon
- [ ] Telegram bot configured and tested
- [ ] `dry_run = true` for initial testing
- [ ] Log directory writable (`data/`)
- [ ] Database initialized (`data/trades.db`)

---

## Systemd Service

### Install Service

```bash
# Copy service file
sudo cp polymarket-bot.service /etc/systemd/system/

# Edit paths if needed
sudo nano /etc/systemd/system/polymarket-bot.service

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start
sudo systemctl enable polymarket-bot
```

### Service Commands

```bash
# Start the bot
sudo systemctl start polymarket-bot

# Stop the bot
sudo systemctl stop polymarket-bot

# Restart the bot
sudo systemctl restart polymarket-bot

# Check status
sudo systemctl status polymarket-bot

# View logs
sudo journalctl -u polymarket-bot -f
```

### Service File Reference

```ini
[Unit]
Description=Polymarket Trading Bot
After=network.target

[Service]
Type=simple
User=bot
WorkingDirectory=/home/bot/clawd/projects/polymarket-bot
ExecStart=/home/bot/clawd/projects/polymarket-bot/target/release/polymarket-bot run
Restart=always
RestartSec=10
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

---

## Monitoring

### Health Checks

#### 1. Process Status

```bash
# Check if bot is running
pgrep -f polymarket-bot

# Check resource usage
ps aux | grep polymarket-bot
htop -p $(pgrep -f polymarket-bot)
```

#### 2. Log Monitoring

```bash
# Real-time logs
tail -f bot.log | grep -E "(ERROR|WARN|trade|signal)"

# Recent errors
grep ERROR bot.log | tail -20

# Trade activity
grep "trade executed" bot.log | tail -10
```

#### 3. Database Health

```bash
# Check database size
ls -lh data/trades.db

# Recent trades
sqlite3 data/trades.db "SELECT * FROM trades ORDER BY created_at DESC LIMIT 10;"

# Trade count by day
sqlite3 data/trades.db "SELECT date(created_at), count(*) FROM trades GROUP BY date(created_at);"
```

### Metrics to Watch

| Metric | Normal Range | Alert Threshold |
|--------|--------------|-----------------|
| Memory | 50-100 MB | >500 MB |
| CPU | <5% idle | >80% sustained |
| API Latency | <200ms | >1s |
| Error Rate | <1/hr | >10/hr |
| Daily P&L | Varies | >-10% |

### Telegram Alerts

The bot sends automatic alerts for:

- 🚀 **Trade Executed** - Every successful trade
- ⚠️ **Risk Warning** - Approaching limits
- 🛑 **Trading Halted** - Daily loss limit reached
- 📊 **Daily Report** - End-of-day summary
- ❌ **Error Alert** - Critical errors

---

## Logging

### Log Levels

```bash
# Set log level via environment
export RUST_LOG=info    # Production
export RUST_LOG=debug   # Development
export RUST_LOG=trace   # Full debugging

# Module-specific logging
export RUST_LOG=polymarket_bot::strategy=debug,polymarket_bot::executor=info
```

### Log Format

```
2026-01-29T12:34:56.789Z INFO  [polymarket_bot::executor] Trade executed: BUY 100 YES @ 0.65
2026-01-29T12:34:57.123Z DEBUG [polymarket_bot::strategy] Signal generated: edge=0.08, confidence=0.75
```

### Log Rotation

```bash
# Create logrotate config
sudo tee /etc/logrotate.d/polymarket-bot << EOF
/home/bot/clawd/projects/polymarket-bot/bot.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 bot bot
}
EOF
```

---

## Troubleshooting

### Common Issues

#### 1. Bot Won't Start

**Symptoms:** Service fails to start, immediate exit

**Check:**
```bash
# Check for config errors
./target/release/polymarket-bot validate-config

# Check permissions
ls -la config.toml .env

# Run manually to see errors
./target/release/polymarket-bot run --dry-run 2>&1
```

**Solutions:**
- Fix configuration syntax errors
- Ensure API keys are set
- Check file permissions

#### 2. No Trades Executing

**Symptoms:** Bot running but no trades

**Check:**
```bash
# Check market scanning
grep "scanning markets" bot.log | tail -5

# Check signals generated
grep "signal" bot.log | tail -10

# Check risk limits
grep "risk" bot.log | tail -10
```

**Solutions:**
- Lower `min_edge` threshold
- Lower `min_confidence` threshold
- Check if daily loss limit reached
- Verify wallet has funds

#### 3. API Errors

**Symptoms:** Repeated 401/403/429 errors

**Check:**
```bash
# Check API connectivity
curl -I https://clob.polymarket.com/health

# Check rate limits
grep "rate limit" bot.log
```

**Solutions:**
- Verify API keys are valid
- Increase `scan_interval_secs`
- Check IP not blocked

#### 4. High Memory Usage

**Symptoms:** Memory growing over time

**Check:**
```bash
# Monitor memory
watch -n 5 'ps -o rss= -p $(pgrep -f polymarket-bot) | awk "{print \$1/1024 \" MB\"}"'
```

**Solutions:**
- Restart the bot
- Check for memory leaks in logs
- Update to latest version

#### 5. Telegram Not Working

**Symptoms:** No notifications received

**Check:**
```bash
# Test Telegram directly
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe"

# Check chat ID
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates"
```

**Solutions:**
- Verify bot token
- Start chat with bot
- Check chat ID is correct

### Error Reference

| Error | Meaning | Action |
|-------|---------|--------|
| `InsufficientBalance` | Not enough USDC | Fund wallet |
| `RateLimitExceeded` | API throttled | Wait/reduce frequency |
| `SignatureInvalid` | Auth failed | Check private key |
| `MarketClosed` | Market not trading | Skip market |
| `DailyLimitReached` | Loss limit hit | Wait until tomorrow |

---

## Backup & Recovery

### What to Backup

| Item | Location | Frequency |
|------|----------|-----------|
| Database | `data/trades.db` | Daily |
| Config | `config.toml` | On change |
| Secrets | `.env` | On change |
| Logs | `bot.log` | Weekly |

### Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/polymarket-bot"
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR

# Backup database
cp data/trades.db $BACKUP_DIR/trades_$DATE.db

# Backup config (without secrets)
cp config.toml $BACKUP_DIR/config_$DATE.toml

# Compress old backups
find $BACKUP_DIR -name "*.db" -mtime +7 -exec gzip {} \;

echo "Backup completed: $DATE"
```

### Recovery

```bash
# Stop bot
sudo systemctl stop polymarket-bot

# Restore database
cp /backup/polymarket-bot/trades_YYYYMMDD.db data/trades.db

# Restore config
cp /backup/polymarket-bot/config_YYYYMMDD.toml config.toml

# Start bot
sudo systemctl start polymarket-bot
```

---

## Security

### Principle of Least Privilege

```bash
# Create dedicated user
sudo useradd -r -s /bin/false polymarket-bot

# Set ownership
sudo chown -R polymarket-bot:polymarket-bot /opt/polymarket-bot

# Restrict config permissions
chmod 600 .env
chmod 644 config.toml
```

### Secret Management

**DO:**
- Use environment variables for secrets
- Never commit `.env` to git
- Rotate API keys periodically
- Use separate wallets for trading

**DON'T:**
- Put private key in config.toml
- Share API keys
- Run as root
- Expose API endpoints

### Firewall Rules

```bash
# Allow only outbound HTTPS
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw enable
```

---

## Maintenance

### Regular Tasks

| Task | Frequency | Command |
|------|-----------|---------|
| Check logs | Daily | `tail -100 bot.log` |
| Verify trades | Daily | `sqlite3 data/trades.db "SELECT ..."` |
| Backup DB | Daily | `cp data/trades.db backup/` |
| Update bot | Weekly | `git pull && cargo build --release` |
| Rotate logs | Weekly | Auto via logrotate |
| Check disk | Monthly | `df -h` |

### Updating the Bot

```bash
# 1. Pull updates
git pull origin main

# 2. Run tests
cargo test

# 3. Build new binary
cargo build --release

# 4. Restart service
sudo systemctl restart polymarket-bot

# 5. Verify running
sudo systemctl status polymarket-bot
```

### Performance Tuning

```toml
# config.toml optimizations

[strategy]
scan_interval_secs = 300  # Increase if hitting rate limits

[risk]
max_open_positions = 5    # Reduce for safety
max_position_pct = 0.03   # Smaller positions = less risk
```

### Database Maintenance

```bash
# Vacuum database (reclaim space)
sqlite3 data/trades.db "VACUUM;"

# Analyze for query optimization
sqlite3 data/trades.db "ANALYZE;"

# Export trades to CSV
sqlite3 -header -csv data/trades.db "SELECT * FROM trades;" > trades_export.csv
```

---

## Emergency Procedures

### Kill Switch

To immediately stop all trading:

```bash
# Option 1: Stop service
sudo systemctl stop polymarket-bot

# Option 2: Kill process
pkill -9 -f polymarket-bot

# Option 3: Enable dry-run mode
sed -i 's/dry_run = false/dry_run = true/' config.toml
sudo systemctl restart polymarket-bot
```

### Position Exit

If you need to exit all positions manually:

1. Stop the bot
2. Log into Polymarket web interface
3. Close all open positions
4. Withdraw funds if needed

### Incident Response

1. **Detect** - Alert received or issue observed
2. **Contain** - Stop the bot if necessary
3. **Investigate** - Check logs, trades, positions
4. **Fix** - Apply configuration or code fix
5. **Recover** - Restart bot in dry-run mode
6. **Review** - Document incident and prevent recurrence

---

## Support

- **GitHub Issues:** https://github.com/voicegn/polymarket-bot/issues
- **Documentation:** See `docs/` folder
- **Logs:** Check `bot.log` first

---

*Last updated: 2026-01-29*
