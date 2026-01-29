//! Polymarket Probability Trading Bot
//!
//! An automated trading system for Polymarket prediction markets.

use chrono::Timelike;
use clap::{Parser, Subcommand};
use polymarket_bot::{
    client::PolymarketClient,
    config::Config,
    executor::Executor,
    model::{EnsembleModel, LlmModel, ProbabilityModel},
    monitor::Monitor,
    notify::Notifier,
    storage::Database,
    strategy::SignalGenerator,
    telegram::{TelegramBot, CommandHandler, BotCommand},
};
use rust_decimal::Decimal;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser)]
#[command(name = "polymarket-bot")]
#[command(about = "Automated trading bot for Polymarket prediction markets")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Config file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the trading bot
    Run {
        /// Dry run mode (no actual trades)
        #[arg(long)]
        dry_run: bool,
    },
    /// Show market data
    Markets {
        /// Number of top markets to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    /// Analyze a specific market
    Analyze {
        /// Market ID to analyze
        market_id: String,
    },
    /// Show account status
    Status,
    /// Send status report to Telegram
    Report,
    /// Test Telegram notification
    TestNotify,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    // Load configuration
    let config = Config::load(&cli.config)?;

    match cli.command {
        Commands::Run { dry_run } => run_bot(config, dry_run).await,
        Commands::Markets { limit } => show_markets(config, limit).await,
        Commands::Analyze { market_id } => analyze_market(config, &market_id).await,
        Commands::Status => show_status(config).await,
        Commands::Report => send_report(config).await,
        Commands::TestNotify => test_notify(config).await,
    }
}

async fn run_bot(config: Config, dry_run: bool) -> anyhow::Result<()> {
    tracing::info!("Starting Polymarket trading bot");

    if dry_run {
        tracing::warn!("Running in DRY RUN mode - no actual trades will be executed");
    }

    // Initialize Telegram notifier
    let notifier = if let Some(tg) = &config.telegram {
        Notifier::new(tg.bot_token.clone(), tg.chat_id.clone())
    } else {
        tracing::warn!("Telegram not configured, notifications disabled");
        Notifier::disabled()
    };

    // Send startup notification
    if let Err(e) = notifier.startup(dry_run).await {
        tracing::warn!("Failed to send startup notification: {}", e);
    }

    // Initialize components
    let client = Arc::new(PolymarketClient::new(config.polymarket.clone()).await?);
    client.clob.initialize().await?;

    let db = Arc::new(Database::connect(&config.database.path).await?);
    let monitor = Monitor::new(1000);

    // Initialize command handler for Telegram
    let cmd_handler = Arc::new(CommandHandler::new(config.clone(), notifier.clone()));

    // Create command channel
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<BotCommand>(100);

    // Start Telegram command listener if configured
    if let Some(tg) = &config.telegram {
        let telegram_bot = Arc::new(TelegramBot::new(
            tg.bot_token.clone(),
            tg.chat_id.clone(),
            cmd_tx,
        ));
        
        let bot_clone = telegram_bot.clone();
        tokio::spawn(async move {
            bot_clone.start_polling().await;
        });
        
        tracing::info!("Telegram command listener started");
    }

    // Initialize model
    let mut model = EnsembleModel::new();
    if let Some(llm_config) = &config.llm {
        let llm = LlmModel::anthropic(llm_config.api_key.clone());
        model.add_model(Box::new(llm), Decimal::new(70, 2)); // 70% weight
    }

    // Initialize strategy
    let signal_gen = SignalGenerator::new(config.strategy.clone(), config.risk.clone());
    let executor = Arc::new(Executor::new(client.clob.clone(), config.risk.clone()));
    let tg_config = config.telegram.clone();
    let notifier = Arc::new(notifier);

    tracing::info!("Bot initialized. Starting main loop...");

    // Spawn daily report task
    if tg_config.as_ref().map(|c| c.notify_daily).unwrap_or(false) {
        let notifier_clone = notifier.clone();
        let db_clone = db.clone();
        let client_clone = client.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(24 * 60 * 60));
            // Wait for first tick (starts immediately)
            interval.tick().await;
            
            loop {
                interval.tick().await;
                
                // Send daily report at midnight UTC
                let now = chrono::Utc::now();
                if now.hour() == 0 && now.minute() < 5 {
                    let balance = client_clone.clob.get_balance().await.unwrap_or(Decimal::ZERO);
                    let stats = db_clone.get_daily_stats().await.unwrap_or_default();
                    let _ = notifier_clone.daily_report(&stats, balance).await;
                }
            }
        });
    }

    // Main trading loop
    loop {
        // Process any pending Telegram commands
        while let Ok(cmd) = cmd_rx.try_recv() {
            cmd_handler.handle(cmd, &client, &db).await;
        }

        // Check if trading is paused
        if cmd_handler.is_paused().await {
            tracing::info!("Trading paused, waiting...");
            tokio::time::sleep(Duration::from_secs(10)).await;
            continue;
        }

        // Get portfolio value
        let balance = match executor.clob.get_balance().await {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("Failed to get balance: {}", e);
                if tg_config.as_ref().map(|c| c.notify_errors).unwrap_or(false) {
                    let _ = notifier.error("Balance fetch", &e.to_string()).await;
                }
                tokio::time::sleep(Duration::from_secs(60)).await;
                continue;
            }
        };

        tracing::info!("Current balance: ${:.2}", balance);

        // Get top markets
        let markets = match client.gamma.get_top_markets(20).await {
            Ok(m) => m,
            Err(e) => {
                tracing::error!("Failed to get markets: {}", e);
                if tg_config.as_ref().map(|c| c.notify_errors).unwrap_or(false) {
                    let _ = notifier.error("Market fetch", &e.to_string()).await;
                }
                tokio::time::sleep(Duration::from_secs(60)).await;
                continue;
            }
        };

        tracing::info!("Scanning {} markets...", markets.len());

        // Analyze each market
        for market in &markets {
            // Skip low liquidity markets
            if market.liquidity < Decimal::new(10000, 0) {
                continue;
            }

            // Get model prediction
            let prediction = match model.predict(market).await {
                Ok(p) => p,
                Err(e) => {
                    tracing::debug!("Model failed for {}: {}", market.id, e);
                    continue;
                }
            };

            // Generate signal
            if let Some(signal) = signal_gen.generate(market, &prediction) {
                tracing::info!(
                    "Signal: {} {} | Model: {:.1}% vs Market: {:.1}% | Edge: {:.1}%",
                    match signal.side {
                        polymarket_bot::types::Side::Buy => "BUY",
                        polymarket_bot::types::Side::Sell => "SELL",
                    },
                    market.question,
                    signal.model_probability * Decimal::ONE_HUNDRED,
                    signal.market_probability * Decimal::ONE_HUNDRED,
                    signal.edge * Decimal::ONE_HUNDRED
                );

                // Send signal notification
                if tg_config.as_ref().map(|c| c.notify_signals).unwrap_or(false) {
                    let _ = notifier.signal_found(&signal, &market.question).await;
                }

                if !dry_run {
                    match executor.execute(&signal, balance).await {
                        Ok(Some(trade)) => {
                            tracing::info!("Trade executed: {}", trade.id);
                            db.save_trade(&trade).await?;

                            // Update PnL tracking for risk management
                            // (simplified - real PnL requires mark-to-market)
                            let _ = cmd_handler.check_risk_limits(Decimal::ZERO).await;

                            // Send trade notification
                            if tg_config.as_ref().map(|c| c.notify_trades).unwrap_or(false) {
                                let _ = notifier.trade_executed(&trade, &market.question).await;
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            tracing::error!("Execution failed: {}", e);
                            if tg_config.as_ref().map(|c| c.notify_errors).unwrap_or(false) {
                                let _ = notifier.error("Trade execution", &e.to_string()).await;
                            }
                        }
                    }
                }
            }
        }

        // Log stats periodically
        monitor.log_stats().await;

        // Wait before next scan
        tracing::info!(
            "Sleeping for {} seconds...",
            config.strategy.scan_interval_secs
        );
        tokio::time::sleep(Duration::from_secs(config.strategy.scan_interval_secs)).await;
    }
}

async fn show_markets(config: Config, limit: usize) -> anyhow::Result<()> {
    let client = PolymarketClient::new(config.polymarket).await?;
    let markets = client.gamma.get_top_markets(limit).await?;

    println!("\n📊 Top {} Polymarket Markets:\n", limit);
    println!("{:<50} {:>8} {:>8} {:>12}", "Question", "Yes", "No", "Volume");
    println!("{}", "-".repeat(80));

    for market in markets {
        let yes = market.yes_price().unwrap_or(Decimal::ZERO);
        let no = market.no_price().unwrap_or(Decimal::ZERO);

        let question = if market.question.len() > 47 {
            format!("{}...", &market.question[..47])
        } else {
            market.question.clone()
        };

        println!(
            "{:<50} {:>7.0}% {:>7.0}% ${:>10.0}",
            question,
            yes * Decimal::ONE_HUNDRED,
            no * Decimal::ONE_HUNDRED,
            market.volume
        );
    }

    Ok(())
}

async fn analyze_market(config: Config, market_id: &str) -> anyhow::Result<()> {
    let client = PolymarketClient::new(config.polymarket.clone()).await?;
    let market = client.gamma.get_market(market_id).await?;

    println!("\n📈 Market Analysis\n");
    println!("Question: {}", market.question);
    if let Some(desc) = &market.description {
        println!("Description: {}", desc);
    }
    println!("\nCurrent Prices:");
    for outcome in &market.outcomes {
        println!(
            "  {} = {:.1}%",
            outcome.outcome,
            outcome.price * Decimal::ONE_HUNDRED
        );
    }
    println!("\nVolume: ${:.0}", market.volume);
    println!("Liquidity: ${:.0}", market.liquidity);

    // Run model if configured
    if let Some(llm_config) = &config.llm {
        println!("\n🤖 Running LLM analysis...\n");
        let llm = LlmModel::anthropic(llm_config.api_key.clone());
        match llm.predict(&market).await {
            Ok(pred) => {
                println!("Model Probability: {:.1}%", pred.probability * Decimal::ONE_HUNDRED);
                println!("Confidence: {:.1}%", pred.confidence * Decimal::ONE_HUNDRED);
                println!("Reasoning: {}", pred.reasoning);

                let market_prob = market.yes_price().unwrap_or(Decimal::ZERO);
                let edge = pred.probability - market_prob;
                println!("\nEdge: {:.1}%", edge * Decimal::ONE_HUNDRED);
            }
            Err(e) => {
                println!("Model error: {}", e);
            }
        }
    }

    Ok(())
}

async fn show_status(config: Config) -> anyhow::Result<()> {
    let client = PolymarketClient::new(config.polymarket).await?;
    client.clob.initialize().await?;

    let balance = client.clob.get_balance().await?;
    let open_orders = client.clob.get_open_orders().await?;

    println!("\n💰 Account Status\n");
    println!("Balance: ${:.2} USDC", balance);
    println!("Open Orders: {}", open_orders.len());

    if !open_orders.is_empty() {
        println!("\nOpen Orders:");
        for order in &open_orders {
            println!(
                "  {} - Status: {}, Filled: {:.2}, Remaining: {:.2}",
                order.order_id, order.status, order.filled_size, order.remaining_size
            );
        }
    }

    Ok(())
}

async fn send_report(config: Config) -> anyhow::Result<()> {
    let tg_config = config.telegram.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Telegram not configured in config.toml"))?;
    
    let notifier = Notifier::new(tg_config.bot_token.clone(), tg_config.chat_id.clone());
    
    // Get account status
    let client = PolymarketClient::new(config.polymarket).await?;
    client.clob.initialize().await?;
    let balance = client.clob.get_balance().await?;
    
    // Get stats from database
    let db = Database::connect(&config.database.path).await?;
    let stats = db.get_daily_stats().await.unwrap_or_default();
    
    // Send report
    notifier.daily_report(&stats, balance).await?;
    
    println!("✅ Report sent to Telegram");
    Ok(())
}

async fn test_notify(config: Config) -> anyhow::Result<()> {
    let tg_config = config.telegram.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Telegram not configured in config.toml"))?;
    
    let notifier = Notifier::new(tg_config.bot_token.clone(), tg_config.chat_id.clone());
    
    notifier.send("🧪 <b>Test Notification</b>\n\nIf you see this, Telegram integration is working!").await?;
    
    println!("✅ Test notification sent!");
    Ok(())
}
