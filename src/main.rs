//! Polymarket Probability Trading Bot
//!
//! An automated trading system for Polymarket prediction markets.

use chrono::Timelike;
use clap::{Parser, Subcommand};
use polymarket_bot::{
    client::PolymarketClient,
    config::Config,
    executor::Executor,
    ingester::{
        processor::SignalProcessor,
        telegram::TelegramBotSource,
        twitter::{TwitterSource, TwitterRssSource},
        ParsedSignal, RawSignal, SignalSource,
    },
    model::{EnsembleModel, LlmModel, ProbabilityModel},
    monitor::Monitor,
    notify::Notifier,
    storage::Database,
    strategy::{
        SignalGenerator,
        copy_trade::{CopyTrader, TopTrader},
        crypto_hf::{CryptoHfStrategy, CryptoPriceTracker},
    },
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
    
    // Skip CLOB auth in dry-run mode (not needed for reading markets)
    if !dry_run {
        client.clob.initialize().await?;
    } else {
        tracing::info!("Skipping CLOB authentication in dry-run mode");
    }

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
        match LlmModel::from_config(llm_config) {
            Ok(llm) => {
                tracing::info!("LLM model initialized: {}", llm.name());
                model.add_model(Box::new(llm), Decimal::new(70, 2)); // 70% weight
            }
            Err(e) => {
                tracing::warn!("Failed to initialize LLM model: {}", e);
            }
        }
    }

    // Initialize strategy
    let signal_gen = SignalGenerator::new(config.strategy.clone(), config.risk.clone());
    let crypto_strategy = CryptoHfStrategy::default();
    let mut crypto_tracker = CryptoPriceTracker::new();
    
    // Initialize crypto price history from Binance klines
    if let Err(e) = crypto_tracker.init_history().await {
        tracing::warn!("Failed to initialize crypto price history: {}", e);
    }
    
    let executor = Arc::new(Executor::new(client.clob.clone(), config.risk.clone()));
    let tg_config = config.telegram.clone();
    let notifier = Arc::new(notifier);

    tracing::info!("Bot initialized. Starting main loop...");

    // ========== Signal Ingester Pipeline ==========
    // Spawn the external signal ingestion system if configured
    let (parsed_signal_tx, mut parsed_signal_rx) = mpsc::channel::<ParsedSignal>(100);
    
    if let Some(ingester_config) = &config.ingester {
        if ingester_config.enabled {
            tracing::info!("Starting signal ingester pipeline...");
            
            // Raw signal channel
            let (raw_tx, raw_rx) = mpsc::channel::<RawSignal>(500);
            
            // Start signal sources
            if let Some(tg_bot_config) = &ingester_config.telegram_bot {
                let source = TelegramBotSource::new(
                    tg_bot_config.bot_token.clone(),
                    tg_bot_config.channels.clone(),
                );
                let tx = raw_tx.clone();
                tokio::spawn(async move {
                    if let Err(e) = source.run(tx).await {
                        tracing::error!("Telegram bot source error: {}", e);
                    }
                });
                tracing::info!("Telegram bot source started");
            }
            
            if let Some(twitter_config) = &ingester_config.twitter {
                if twitter_config.bearer_token.is_some() {
                    let source = TwitterSource::new(
                        polymarket_bot::ingester::TwitterIngesterConfig {
                            bearer_token: twitter_config.bearer_token.clone(),
                            watch_users: twitter_config.user_ids.clone(),
                            keywords: twitter_config.keywords.clone(),
                        },
                        ingester_config.author_trust.clone(),
                    );
                    let tx = raw_tx.clone();
                    tokio::spawn(async move {
                        if let Err(e) = source.run(tx).await {
                            tracing::error!("Twitter source error: {}", e);
                        }
                    });
                    tracing::info!("Twitter API source started");
                } else if let Some(nitter) = &twitter_config.nitter_instance {
                    // Use RSS fallback
                    let source = TwitterRssSource::new(
                        nitter.clone(),
                        twitter_config.user_ids.clone(),
                        twitter_config.keywords.clone(),
                    );
                    let tx = raw_tx.clone();
                    tokio::spawn(async move {
                        if let Err(e) = source.run(tx).await {
                            tracing::error!("Twitter RSS source error: {}", e);
                        }
                    });
                    tracing::info!("Twitter RSS source started (via {})", nitter);
                }
            }
            
            // Start signal processor
            if let Some(llm_config) = &config.llm {
                let processor = SignalProcessor::new(llm_config.clone())
                    .with_thresholds(
                        ingester_config.processing.min_confidence,
                        ingester_config.processing.min_agg_score,
                    )
                    .with_window(ingester_config.processing.aggregation_window_secs);
                
                let parsed_tx = parsed_signal_tx.clone();
                tokio::spawn(async move {
                    if let Err(e) = processor.run(raw_rx, parsed_tx).await {
                        tracing::error!("Signal processor error: {}", e);
                    }
                });
                tracing::info!("Signal processor started");
            }
        }
    }
    
    // Spawn parsed signal handler (trades based on external signals)
    {
        let notifier_for_signals = notifier.clone();
        let _executor_for_signals = executor.clone();
        let _db_for_signals = db.clone();
        let _dry_run_mode = dry_run;
        
        tokio::spawn(async move {
            while let Some(signal) = parsed_signal_rx.recv().await {
                tracing::info!(
                    "📊 Received aggregated signal: {} {:?} (score: {:.2}, conf: {:.2})",
                    signal.token,
                    signal.direction,
                    signal.agg_score,
                    signal.confidence
                );
                
                // TODO: Map external signals to Polymarket markets
                // For now, just notify about high-confidence signals
                if signal.agg_score >= 0.7 {
                    let msg = format!(
                        "🎯 *High Confidence Signal*\n\n\
                        Token: `{}`\n\
                        Direction: {:?}\n\
                        Score: {:.0}%\n\
                        Confidence: {:.0}%\n\
                        Timeframe: {}\n\
                        Sources: {}\n\n\
                        Reasoning: {}",
                        signal.token,
                        signal.direction,
                        signal.agg_score * 100.0,
                        signal.confidence * 100.0,
                        signal.timeframe,
                        signal.sources.len(),
                        signal.reasoning
                    );
                    
                    let _ = notifier_for_signals.send_raw(&msg).await;
                    
                    // TODO: Execute trade when we can map signals to markets
                    // if !dry_run_mode {
                    //     if let Ok(Some(trade)) = executor_for_signals.execute_external_signal(&signal).await {
                    //         let _ = notifier_for_signals.trade_executed(&trade, &signal.token).await;
                    //     }
                    // }
                }
            }
        });
    }

    // ========== Copy Trading ==========
    // Follow top traders' positions
    if let Some(copy_config) = &config.copy_trade {
        if copy_config.enabled {
            tracing::info!("Starting copy trading module...");
            
            let mut copy_trader = CopyTrader::new()
                .with_copy_ratio(copy_config.copy_ratio);
            
            // Add traders to follow
            for username in &copy_config.follow_users {
                let trader = TopTrader {
                    username: username.clone(),
                    address: None,  // Will be resolved
                    win_rate: 0.7,  // Assume good until we analyze
                    total_profit: Decimal::ZERO,
                    weight: 1.0,
                    updated_at: chrono::Utc::now(),
                };
                copy_trader.add_trader(trader);
                tracing::info!("Following trader: @{}", username);
            }
            
            // Add addresses to follow
            for address in &copy_config.follow_addresses {
                let trader = TopTrader {
                    username: format!("{}...", &address[..8]),
                    address: Some(address.clone()),
                    win_rate: 0.6,
                    total_profit: Decimal::ZERO,
                    weight: 1.0,
                    updated_at: chrono::Utc::now(),
                };
                copy_trader.add_trader(trader);
                tracing::info!("Following address: {}", address);
            }
            
            let _executor_for_copy = executor.clone();
            let notifier_for_copy = notifier.clone();
            let delay_secs = copy_config.delay_secs;
            let _dry_run_copy = dry_run;
            
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(30));
                
                loop {
                    interval.tick().await;
                    
                    match copy_trader.check_for_signals().await {
                        Ok(signals) => {
                            for signal in signals {
                                tracing::info!(
                                    "📋 Copy signal from @{}: {} {}",
                                    signal.trader.username,
                                    match signal.side { 
                                        polymarket_bot::types::Side::Buy => "BUY",
                                        polymarket_bot::types::Side::Sell => "SELL",
                                    },
                                    signal.market_id
                                );
                                
                                // Delay before copying
                                if delay_secs > 0 {
                                    tokio::time::sleep(Duration::from_secs(delay_secs)).await;
                                }
                                
                                // Notify about copy signal
                                let msg = format!(
                                    "📋 *Copy Trade Signal*\n\n\
                                    Trader: @{}\n\
                                    Market: `{}`\n\
                                    Side: {}\n\
                                    Their Size: ${:.2}\n\
                                    Our Size: ${:.2}",
                                    signal.trader.username,
                                    signal.market_id,
                                    match signal.side {
                                        polymarket_bot::types::Side::Buy => "BUY",
                                        polymarket_bot::types::Side::Sell => "SELL",
                                    },
                                    signal.trader_size,
                                    signal.suggested_size
                                );
                                let _ = notifier_for_copy.send_raw(&msg).await;
                                
                                // TODO: Execute copy trade
                                // if !dry_run_copy {
                                //     let market_signal = signal.to_signal(Decimal::new(50, 2));
                                //     if let Ok(Some(trade)) = executor_for_copy.execute(&market_signal, balance).await {
                                //         let _ = notifier_for_copy.trade_executed(&trade, &signal.market_id).await;
                                //     }
                                // }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Copy trader error: {}", e);
                        }
                    }
                }
            });
            
            tracing::info!("Copy trading module started");
        }
    }

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

        // Get portfolio value (use simulated balance in dry-run mode)
        let balance = if dry_run {
            Decimal::new(1000, 0)  // $1000 simulated balance
        } else {
            match executor.clob.get_balance().await {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!("Failed to get balance: {}", e);
                    if tg_config.as_ref().map(|c| c.notify_errors).unwrap_or(false) {
                        let _ = notifier.error("Balance fetch", &e.to_string()).await;
                    }
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    continue;
                }
            }
        };

        tracing::info!("Current balance: ${:.2}", balance);

        // Get top markets + crypto markets
        let mut markets = match client.gamma.get_top_markets(20).await {
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

        // Also fetch crypto markets (BTC/ETH Up/Down)
        match client.gamma.get_crypto_markets().await {
            Ok(crypto_markets) => {
                tracing::info!("Found {} crypto markets", crypto_markets.len());
                markets.extend(crypto_markets);
            }
            Err(e) => {
                tracing::warn!("Failed to fetch crypto markets: {}", e);
            }
        }

        // Update crypto prices for HF strategy
        if let Err(e) = crypto_tracker.update_prices().await {
            tracing::debug!("Failed to update crypto prices: {}", e);
        }

        tracing::info!("Scanning {} markets...", markets.len());

        // Analyze each market
        for market in &markets {
            // Check if this is a crypto Up/Down market
            let is_crypto_market = CryptoHfStrategy::is_crypto_hf_market(market).is_some();
            
            // Skip low liquidity markets (lower threshold for crypto markets)
            let min_liquidity = if is_crypto_market {
                Decimal::new(1000, 0)  // $1,000 for crypto markets
            } else {
                Decimal::new(10000, 0) // $10,000 for regular markets
            };
            
            if market.liquidity < min_liquidity {
                continue;
            }

            // Generate signal: use crypto strategy for crypto markets, LLM for others
            let signal = if is_crypto_market {
                // Use Binance price momentum for crypto markets
                crypto_strategy.generate_signal(market, &crypto_tracker)
            } else {
                // Use LLM prediction for regular markets
                let prediction = match model.predict(market).await {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::debug!("Model failed for {}: {}", market.id, e);
                        continue;
                    }
                };
                signal_gen.generate(market, &prediction)
            };

            if let Some(signal) = signal {
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
        let llm = LlmModel::from_config(llm_config)?;
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
