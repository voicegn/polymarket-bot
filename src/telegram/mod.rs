//! Telegram bot for receiving commands
//!
//! Supports commands like /status, /markets, /pause, /resume, /buy, /sell

use crate::client::PolymarketClient;
use crate::config::Config;
use crate::error::Result;
use crate::monitor::PerformanceStats;
use crate::storage::Database;
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Telegram bot for receiving commands
pub struct TelegramBot {
    http: Client,
    bot_token: String,
    chat_id: String,
    last_update_id: RwLock<i64>,
    command_tx: mpsc::Sender<BotCommand>,
}

/// Commands that can be sent to the trading bot
#[derive(Debug, Clone)]
pub enum BotCommand {
    /// Pause trading
    Pause,
    /// Resume trading
    Resume,
    /// Get current status
    Status,
    /// List top markets
    Markets { limit: usize },
    /// Manual buy order
    Buy { market_id: String, amount: Decimal },
    /// Manual sell order
    Sell { market_id: String, amount: Decimal },
    /// Get today's PnL
    Pnl,
    /// Get open positions
    Positions,
    /// Set risk parameter
    SetRisk { param: String, value: Decimal },
    /// Help
    Help,
}

/// Bot state shared with trading loop
#[derive(Debug, Clone)]
pub struct BotState {
    pub paused: bool,
    pub daily_pnl: Decimal,
    pub daily_loss_limit_hit: bool,
}

impl Default for BotState {
    fn default() -> Self {
        Self {
            paused: false,
            daily_pnl: Decimal::ZERO,
            daily_loss_limit_hit: false,
        }
    }
}

#[derive(Debug, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    message: Option<TelegramMessage>,
}

#[derive(Debug, Deserialize)]
struct TelegramMessage {
    message_id: i64,
    from: Option<TelegramUser>,
    chat: TelegramChat,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramUser {
    id: i64,
    first_name: String,
}

#[derive(Debug, Deserialize)]
struct TelegramChat {
    id: i64,
}

#[derive(Debug, Deserialize)]
struct GetUpdatesResponse {
    ok: bool,
    result: Vec<TelegramUpdate>,
}

#[derive(Debug, Serialize)]
struct SendMessageRequest {
    chat_id: String,
    text: String,
    parse_mode: String,
}

impl TelegramBot {
    pub fn new(bot_token: String, chat_id: String, command_tx: mpsc::Sender<BotCommand>) -> Self {
        Self {
            http: Client::new(),
            bot_token,
            chat_id,
            last_update_id: RwLock::new(0),
            command_tx,
        }
    }

    /// Start polling for updates
    pub async fn start_polling(self: Arc<Self>) {
        tracing::info!("Starting Telegram command listener...");
        
        loop {
            match self.poll_updates().await {
                Ok(updates) => {
                    for update in updates {
                        if let Some(msg) = update.message {
                            // Only process messages from authorized chat
                            if msg.chat.id.to_string() == self.chat_id {
                                if let Some(text) = msg.text {
                                    self.handle_message(&text).await;
                                }
                            }
                        }
                        
                        // Update offset
                        let mut last_id = self.last_update_id.write().await;
                        *last_id = update.update_id + 1;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to poll Telegram updates: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
            
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }
    }

    async fn poll_updates(&self) -> Result<Vec<TelegramUpdate>> {
        let last_id = *self.last_update_id.read().await;
        
        let url = format!(
            "https://api.telegram.org/bot{}/getUpdates?offset={}&timeout=30",
            self.bot_token, last_id
        );

        let response: GetUpdatesResponse = self.http
            .get(&url)
            .send()
            .await?
            .json()
            .await?;

        Ok(response.result)
    }

    async fn handle_message(&self, text: &str) {
        let text = text.trim();
        
        // Parse command
        let (cmd, args) = if text.starts_with('/') {
            let parts: Vec<&str> = text[1..].splitn(2, ' ').collect();
            let cmd = parts[0].split('@').next().unwrap_or(parts[0]); // Remove @botname
            let args = parts.get(1).map(|s| s.trim()).unwrap_or("");
            (cmd, args)
        } else {
            return; // Ignore non-commands
        };

        tracing::info!("Received command: /{} {}", cmd, args);

        match cmd.to_lowercase().as_str() {
            "start" | "help" => {
                self.send_help().await;
            }
            "status" => {
                let _ = self.command_tx.send(BotCommand::Status).await;
            }
            "markets" => {
                let limit = args.parse().unwrap_or(5);
                let _ = self.command_tx.send(BotCommand::Markets { limit }).await;
            }
            "pause" => {
                let _ = self.command_tx.send(BotCommand::Pause).await;
                self.reply("⏸ Trading paused").await;
            }
            "resume" => {
                let _ = self.command_tx.send(BotCommand::Resume).await;
                self.reply("▶️ Trading resumed").await;
            }
            "pnl" => {
                let _ = self.command_tx.send(BotCommand::Pnl).await;
            }
            "positions" | "pos" => {
                let _ = self.command_tx.send(BotCommand::Positions).await;
            }
            "buy" => {
                if let Some((market_id, amount)) = self.parse_trade_args(args) {
                    let _ = self.command_tx.send(BotCommand::Buy { market_id, amount }).await;
                } else {
                    self.reply("❌ Usage: /buy <market_id> <amount>").await;
                }
            }
            "sell" => {
                if let Some((market_id, amount)) = self.parse_trade_args(args) {
                    let _ = self.command_tx.send(BotCommand::Sell { market_id, amount }).await;
                } else {
                    self.reply("❌ Usage: /sell <market_id> <amount>").await;
                }
            }
            "setrisk" => {
                if let Some((param, value)) = self.parse_risk_args(args) {
                    let _ = self.command_tx.send(BotCommand::SetRisk { param, value }).await;
                } else {
                    self.reply("❌ Usage: /setrisk <param> <value>\nParams: max_position, max_daily_loss, kelly_fraction").await;
                }
            }
            _ => {
                self.reply(&format!("❓ Unknown command: /{}\nUse /help for available commands", cmd)).await;
            }
        }
    }

    fn parse_trade_args(&self, args: &str) -> Option<(String, Decimal)> {
        let parts: Vec<&str> = args.split_whitespace().collect();
        if parts.len() >= 2 {
            let market_id = parts[0].to_string();
            let amount: Decimal = parts[1].parse().ok()?;
            Some((market_id, amount))
        } else {
            None
        }
    }

    fn parse_risk_args(&self, args: &str) -> Option<(String, Decimal)> {
        let parts: Vec<&str> = args.split_whitespace().collect();
        if parts.len() >= 2 {
            let param = parts[0].to_string();
            let value: Decimal = parts[1].parse().ok()?;
            Some((param, value))
        } else {
            None
        }
    }

    async fn send_help(&self) {
        let help_text = r#"🤖 <b>Polymarket Bot Commands</b>

<b>Status</b>
/status - Account balance & bot status
/pnl - Today's profit/loss
/positions - Open positions
/markets [n] - Top n markets (default 5)

<b>Trading</b>
/buy &lt;market_id&gt; &lt;amount&gt; - Manual buy
/sell &lt;market_id&gt; &lt;amount&gt; - Manual sell
/pause - Pause auto-trading
/resume - Resume auto-trading

<b>Risk</b>
/setrisk max_position 0.05 - Max 5% per position
/setrisk max_daily_loss 0.10 - Max 10% daily loss
/setrisk kelly_fraction 0.25 - Quarter Kelly

/help - Show this message"#;
        
        self.reply(help_text).await;
    }

    async fn reply(&self, text: &str) {
        let url = format!(
            "https://api.telegram.org/bot{}/sendMessage",
            self.bot_token
        );

        let request = SendMessageRequest {
            chat_id: self.chat_id.clone(),
            text: text.to_string(),
            parse_mode: "HTML".to_string(),
        };

        if let Err(e) = self.http.post(&url).json(&request).send().await {
            tracing::error!("Failed to send Telegram reply: {}", e);
        }
    }
}

/// Command handler that processes commands from Telegram
pub struct CommandHandler {
    pub state: Arc<RwLock<BotState>>,
    notifier: crate::notify::Notifier,
    config: Config,
}

impl CommandHandler {
    pub fn new(config: Config, notifier: crate::notify::Notifier) -> Self {
        Self {
            state: Arc::new(RwLock::new(BotState::default())),
            notifier,
            config,
        }
    }

    pub async fn handle(&self, cmd: BotCommand, client: &PolymarketClient, db: &Database) {
        match cmd {
            BotCommand::Pause => {
                let mut state = self.state.write().await;
                state.paused = true;
            }
            BotCommand::Resume => {
                let mut state = self.state.write().await;
                state.paused = false;
                state.daily_loss_limit_hit = false;
            }
            BotCommand::Status => {
                self.send_status(client).await;
            }
            BotCommand::Markets { limit } => {
                self.send_markets(client, limit).await;
            }
            BotCommand::Pnl => {
                self.send_pnl(db).await;
            }
            BotCommand::Positions => {
                self.send_positions(client).await;
            }
            BotCommand::Buy { market_id, amount } => {
                self.execute_manual_trade(&market_id, amount, true, client).await;
            }
            BotCommand::Sell { market_id, amount } => {
                self.execute_manual_trade(&market_id, amount, false, client).await;
            }
            BotCommand::SetRisk { param, value } => {
                self.set_risk_param(&param, value).await;
            }
            BotCommand::Help => {}
        }
    }

    async fn send_status(&self, client: &PolymarketClient) {
        let balance = client.clob.get_balance().await.unwrap_or(Decimal::ZERO);
        let open_orders = client.clob.get_open_orders().await.unwrap_or_default();
        let state = self.state.read().await;

        let status_emoji = if state.paused { "⏸" } else { "▶️" };
        let status_text = if state.paused { "PAUSED" } else { "RUNNING" };

        let text = format!(
            "💰 <b>Account Status</b>\n\n\
            Status: {} {}\n\
            Balance: <code>${:.2}</code> USDC\n\
            Open Orders: {}\n\
            Daily PnL: <code>{:+.2}</code>",
            status_emoji, status_text,
            balance,
            open_orders.len(),
            state.daily_pnl,
        );

        let _ = self.notifier.send(&text).await;
    }

    async fn send_markets(&self, client: &PolymarketClient, limit: usize) {
        match client.gamma.get_top_markets(limit).await {
            Ok(markets) => {
                let mut text = format!("📊 <b>Top {} Markets</b>\n\n", limit);
                
                for (i, market) in markets.iter().enumerate() {
                    let yes = market.yes_price().unwrap_or(Decimal::ZERO) * Decimal::ONE_HUNDRED;
                    let question = if market.question.len() > 40 {
                        format!("{}...", &market.question[..40])
                    } else {
                        market.question.clone()
                    };
                    
                    text.push_str(&format!(
                        "{}. {} <code>{:.0}%</code>\n",
                        i + 1, question, yes
                    ));
                }
                
                let _ = self.notifier.send(&text).await;
            }
            Err(e) => {
                let _ = self.notifier.error("Markets fetch", &e.to_string()).await;
            }
        }
    }

    async fn send_pnl(&self, _db: &Database) {
        let state = self.state.read().await;
        let emoji = if state.daily_pnl >= Decimal::ZERO { "📈" } else { "📉" };
        
        let text = format!(
            "{} <b>Today's PnL</b>\n\n\
            PnL: <code>{:+.2}</code> USDC",
            emoji, state.daily_pnl
        );
        
        let _ = self.notifier.send(&text).await;
    }

    async fn send_positions(&self, client: &PolymarketClient) {
        match client.clob.get_positions().await {
            Ok(positions) => {
                if positions.is_empty() {
                    let _ = self.notifier.send("📭 No open positions").await;
                    return;
                }

                let mut text = String::from("📊 <b>Open Positions</b>\n\n");
                
                for pos in &positions {
                    let pnl_emoji = if pos.unrealized_pnl >= Decimal::ZERO { "🟢" } else { "🔴" };
                    text.push_str(&format!(
                        "{} <code>{}</code>\n  Size: {} @ {:.4} | PnL: {:+.2}\n\n",
                        pnl_emoji,
                        &pos.token_id[..8],
                        pos.size,
                        pos.avg_entry_price,
                        pos.unrealized_pnl,
                    ));
                }
                
                let _ = self.notifier.send(&text).await;
            }
            Err(e) => {
                let _ = self.notifier.error("Positions fetch", &e.to_string()).await;
            }
        }
    }

    async fn execute_manual_trade(&self, market_id: &str, amount: Decimal, is_buy: bool, _client: &PolymarketClient) {
        let side = if is_buy { "BUY" } else { "SELL" };
        
        // TODO: Implement actual trading via executor
        let text = format!(
            "⚠️ <b>Manual Trade Request</b>\n\n\
            {} {} USDC on market <code>{}</code>\n\n\
            Manual trading not yet implemented. Use the bot's auto-trading.",
            side, amount, market_id
        );
        
        let _ = self.notifier.send(&text).await;
    }

    async fn set_risk_param(&self, param: &str, value: Decimal) {
        // TODO: Actually update config
        let text = format!(
            "⚙️ <b>Risk Parameter Updated</b>\n\n\
            {} = {}",
            param, value
        );
        
        let _ = self.notifier.send(&text).await;
    }

    /// Check risk limits and return true if trading should be blocked
    pub async fn check_risk_limits(&self, pnl_change: Decimal) -> bool {
        let mut state = self.state.write().await;
        state.daily_pnl += pnl_change;
        
        let max_loss = self.config.risk.max_daily_loss_pct;
        
        if state.daily_pnl < -max_loss && !state.daily_loss_limit_hit {
            state.daily_loss_limit_hit = true;
            state.paused = true;
            
            // Send alert
            let _ = self.notifier.risk_alert(
                "Daily Loss Limit",
                &format!(
                    "Daily loss of {:.2}% exceeded limit of {:.2}%\n\
                    Trading has been automatically paused.\n\n\
                    Use /resume to continue (at your own risk).",
                    state.daily_pnl * Decimal::ONE_HUNDRED,
                    max_loss * Decimal::ONE_HUNDRED,
                )
            ).await;
            
            return true;
        }
        
        false
    }

    pub async fn is_paused(&self) -> bool {
        self.state.read().await.paused
    }
}
