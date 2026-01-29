//! Telegram notification module
//!
//! Sends trading signals, executions, and alerts to Telegram.

use crate::error::Result;
use crate::types::{Signal, Side, Trade};
use crate::monitor::PerformanceStats;
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Serialize;

/// Telegram notifier
pub struct Notifier {
    http: Client,
    bot_token: String,
    chat_id: String,
    enabled: bool,
}

#[derive(Debug, Serialize)]
struct TelegramMessage {
    chat_id: String,
    text: String,
    parse_mode: String,
}

impl Notifier {
    pub fn new(bot_token: String, chat_id: String) -> Self {
        Self {
            http: Client::new(),
            bot_token,
            chat_id,
            enabled: true,
        }
    }

    /// Create a disabled notifier (for when Telegram is not configured)
    pub fn disabled() -> Self {
        Self {
            http: Client::new(),
            bot_token: String::new(),
            chat_id: String::new(),
            enabled: false,
        }
    }

    /// Send a raw message
    pub async fn send(&self, text: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let url = format!(
            "https://api.telegram.org/bot{}/sendMessage",
            self.bot_token
        );

        let msg = TelegramMessage {
            chat_id: self.chat_id.clone(),
            text: text.to_string(),
            parse_mode: "HTML".to_string(),
        };

        let response = self.http.post(&url).json(&msg).send().await?;
        
        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            tracing::error!("Telegram send failed: {}", error_text);
        }

        Ok(())
    }

    /// Notify about a trading signal found
    pub async fn signal_found(&self, signal: &Signal, market_question: &str) -> Result<()> {
        let side_emoji = match signal.side {
            Side::Buy => "🟢",
            Side::Sell => "🔴",
        };

        let side_text = match signal.side {
            Side::Buy => "BUY",
            Side::Sell => "SELL",
        };

        let text = format!(
            "{} <b>Signal Found</b>\n\n\
            📊 <b>{}</b>\n\n\
            Direction: {} {}\n\
            Model: <code>{:.1}%</code>\n\
            Market: <code>{:.1}%</code>\n\
            Edge: <code>{:+.1}%</code>\n\
            Confidence: <code>{:.0}%</code>\n\
            Size: <code>{:.1}%</code> of portfolio",
            side_emoji,
            truncate(market_question, 100),
            side_emoji,
            side_text,
            signal.model_probability * Decimal::ONE_HUNDRED,
            signal.market_probability * Decimal::ONE_HUNDRED,
            signal.edge * Decimal::ONE_HUNDRED,
            signal.confidence * Decimal::ONE_HUNDRED,
            signal.suggested_size * Decimal::ONE_HUNDRED,
        );

        self.send(&text).await
    }

    /// Notify about trade execution
    pub async fn trade_executed(&self, trade: &Trade, market_question: &str) -> Result<()> {
        let side_emoji = match trade.side {
            Side::Buy => "🟢",
            Side::Sell => "🔴",
        };

        let text = format!(
            "✅ <b>Trade Executed</b>\n\n\
            📊 {}\n\n\
            {} {} @ <code>${:.4}</code>\n\
            Size: <code>${:.2}</code>\n\
            Fee: <code>${:.4}</code>\n\
            Order ID: <code>{}</code>",
            truncate(market_question, 80),
            side_emoji,
            match trade.side {
                Side::Buy => "BOUGHT",
                Side::Sell => "SOLD",
            },
            trade.price,
            trade.size,
            trade.fee,
            &trade.order_id[..8],
        );

        self.send(&text).await
    }

    /// Notify about an error
    pub async fn error(&self, context: &str, error: &str) -> Result<()> {
        let text = format!(
            "⚠️ <b>Error</b>\n\n\
            Context: {}\n\
            Error: <code>{}</code>",
            context,
            truncate(error, 200),
        );

        self.send(&text).await
    }

    /// Send daily performance report
    pub async fn daily_report(&self, stats: &PerformanceStats, balance: Decimal) -> Result<()> {
        let pnl_emoji = if stats.total_pnl >= Decimal::ZERO { "📈" } else { "📉" };

        let text = format!(
            "📊 <b>Daily Report</b>\n\n\
            💰 Balance: <code>${:.2}</code>\n\
            {} PnL: <code>{:+.2}</code>\n\n\
            Trades: {}\n\
            Win Rate: <code>{:.1}%</code>\n\
            Avg PnL/Trade: <code>{:+.2}</code>",
            balance,
            pnl_emoji,
            stats.total_pnl,
            stats.total_trades,
            stats.win_rate * Decimal::ONE_HUNDRED,
            stats.avg_pnl_per_trade,
        );

        self.send(&text).await
    }

    /// Notify bot startup
    pub async fn startup(&self, dry_run: bool) -> Result<()> {
        let mode = if dry_run { "DRY RUN 🧪" } else { "LIVE 🔥" };
        let text = format!(
            "🤖 <b>Polymarket Bot Started</b>\n\n\
            Mode: {}\n\
            Time: {}",
            mode,
            chrono::Utc::now().format("%Y-%m-%d %H:%M UTC"),
        );

        self.send(&text).await
    }

    /// Notify bot shutdown
    pub async fn shutdown(&self, reason: &str) -> Result<()> {
        let text = format!(
            "🛑 <b>Bot Stopped</b>\n\n\
            Reason: {}\n\
            Time: {}",
            reason,
            chrono::Utc::now().format("%Y-%m-%d %H:%M UTC"),
        );

        self.send(&text).await
    }

    /// Risk alert (e.g., daily loss limit hit)
    pub async fn risk_alert(&self, alert_type: &str, message: &str) -> Result<()> {
        let text = format!(
            "🚨 <b>Risk Alert: {}</b>\n\n\
            {}",
            alert_type,
            message,
        );

        self.send(&text).await
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
}
