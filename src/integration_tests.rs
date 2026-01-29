//! Integration-style tests for full workflow coverage

#[cfg(test)]
mod tests {
    use crate::config::{
        Config, PolymarketConfig, StrategyConfig, RiskConfig, DatabaseConfig,
        LlmConfig, TelegramConfig, CopyTradeConfig, IngesterConfig,
    };
    use crate::types::{Market, Outcome, Signal, Side, Order, OrderType, Trade, Position, OrderStatus};
    use crate::model::Prediction;
    use crate::strategy::SignalGenerator;
    use crate::analysis::{TradeAnalyzer, TradeRecord, TradeOutcome, TradingPattern};
    use crate::monitor::{Monitor, PerformanceStats};
    use crate::telegram::BotState;
    use chrono::Utc;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    // ========== Config Integration Tests ==========

    #[test]
    fn test_full_config_creation() {
        let config = Config {
            polymarket: PolymarketConfig {
                clob_url: "https://clob.polymarket.com".to_string(),
                gamma_url: "https://gamma-api.polymarket.com".to_string(),
                private_key: "test_key".to_string(),
                funder_address: None,
                chain_id: 137,
                signature_type: 0,
            },
            strategy: StrategyConfig::default(),
            risk: RiskConfig::default(),
            database: DatabaseConfig { path: "test.db".to_string() },
            llm: Some(LlmConfig {
                provider: "deepseek".to_string(),
                api_key: "sk-test".to_string(),
                model: None,
                base_url: None,
            }),
            telegram: Some(TelegramConfig {
                bot_token: "123:abc".to_string(),
                chat_id: "12345".to_string(),
                notify_signals: true,
                notify_trades: true,
                notify_errors: true,
                notify_daily: true,
            }),
            ingester: None,
            copy_trade: None,
        };
        
        assert_eq!(config.polymarket.chain_id, 137);
        assert!(config.llm.is_some());
        assert!(config.telegram.is_some());
    }

    #[test]
    fn test_config_with_copy_trade() {
        let copy_trade = CopyTradeConfig {
            enabled: true,
            follow_users: vec!["user1".to_string(), "user2".to_string()],
            follow_addresses: vec!["0x123".to_string()],
            copy_ratio: 0.5,
            delay_secs: 30,
        };
        
        assert!(copy_trade.enabled);
        assert_eq!(copy_trade.follow_users.len(), 2);
    }

    // ========== Trading Workflow Tests ==========

    #[test]
    fn test_market_to_signal_workflow() {
        let market = Market {
            id: "btc-up-down".to_string(),
            question: "Bitcoin Up or Down?".to_string(),
            description: Some("Hourly BTC direction".to_string()),
            end_date: Some(Utc::now()),
            volume: dec!(50000),
            liquidity: dec!(25000),
            outcomes: vec![
                Outcome {
                    token_id: "yes-up".to_string(),
                    outcome: "Yes".to_string(),
                    price: dec!(0.55),
                },
                Outcome {
                    token_id: "no-down".to_string(),
                    outcome: "No".to_string(),
                    price: dec!(0.45),
                },
            ],
            active: true,
            closed: false,
        };
        
        let prediction = Prediction {
            probability: dec!(0.70),
            confidence: dec!(0.80),
            reasoning: "Strong momentum".to_string(),
        };
        
        let generator = SignalGenerator::new(
            StrategyConfig::default(),
            RiskConfig::default(),
        );
        
        let signal = generator.generate(&market, &prediction);
        assert!(signal.is_some());
        
        let s = signal.unwrap();
        assert_eq!(s.side, Side::Buy);
        assert!(s.edge > dec!(0.10));
    }

    #[test]
    fn test_signal_to_order_workflow() {
        let signal = Signal {
            market_id: "test-market".to_string(),
            token_id: "test-token".to_string(),
            side: Side::Buy,
            model_probability: dec!(0.70),
            market_probability: dec!(0.55),
            edge: dec!(0.15),
            confidence: dec!(0.80),
            suggested_size: dec!(0.05),
            timestamp: Utc::now(),
        };
        
        // Convert signal to order
        let portfolio_value = dec!(1000);
        let size_usd = signal.suggested_size * portfolio_value;
        let size_shares = size_usd / signal.market_probability;
        
        let order = Order {
            token_id: signal.token_id.clone(),
            side: signal.side,
            price: signal.market_probability + dec!(0.01), // Limit price with buffer
            size: size_shares,
            order_type: OrderType::GTC,
        };
        
        assert_eq!(order.side, Side::Buy);
        assert!(order.size > dec!(0));
    }

    #[test]
    fn test_order_to_trade_workflow() {
        let order = Order {
            token_id: "token123".to_string(),
            side: Side::Buy,
            price: dec!(0.55),
            size: dec!(100),
            order_type: OrderType::GTC,
        };
        
        // Simulate order filled
        let order_status = OrderStatus {
            order_id: "order123".to_string(),
            status: "FILLED".to_string(),
            filled_size: dec!(100),
            remaining_size: dec!(0),
            avg_price: Some(dec!(0.55)),
        };
        
        // Create trade record
        let trade = Trade {
            id: "trade123".to_string(),
            order_id: order_status.order_id.clone(),
            token_id: order.token_id.clone(),
            market_id: "market123".to_string(),
            side: order.side,
            price: order_status.avg_price.unwrap(),
            size: order_status.filled_size,
            fee: dec!(0.25),
            timestamp: Utc::now(),
        };
        
        assert_eq!(trade.size, dec!(100));
        assert_eq!(trade.fee, dec!(0.25));
    }

    // ========== Analysis Workflow Tests ==========

    #[test]
    fn test_trade_analysis_workflow() {
        let mut analyzer = TradeAnalyzer::new();
        
        // Add trades
        let trades = vec![
            (dec!(50), TradeOutcome::Win),
            (dec!(30), TradeOutcome::Win),
            (dec!(-20), TradeOutcome::Loss),
            (dec!(40), TradeOutcome::Win),
            (dec!(-15), TradeOutcome::Loss),
        ];
        
        for (pnl, outcome) in trades {
            analyzer.add_trade(TradeRecord {
                trader: "test_trader".to_string(),
                market_id: "m1".to_string(),
                market_question: "Test?".to_string(),
                side: Side::Buy,
                entry_price: dec!(0.50),
                exit_price: Some(dec!(0.60)),
                size: dec!(100),
                entry_time: Utc::now(),
                exit_time: Some(Utc::now()),
                pnl: Some(pnl),
                outcome: Some(outcome),
            });
        }
        
        let insights = analyzer.analyze_trader("test_trader");
        assert_eq!(insights.total_trades, 5);
        assert_eq!(insights.total_pnl, dec!(85));
        assert!(insights.win_rate > 0.5);
    }

    // ========== Monitor Workflow Tests ==========

    #[tokio::test]
    async fn test_monitor_workflow() {
        let monitor = Monitor::new(100);
        
        // Record some trades
        for i in 0..10 {
            let pnl = if i % 3 == 0 { dec!(-20) } else { dec!(30) };
            monitor.record_trade(crate::monitor::TradeRecord {
                timestamp: Utc::now(),
                market_id: format!("market{}", i),
                side: "BUY".to_string(),
                size: dec!(100),
                price: dec!(0.55),
                pnl: Some(pnl),
            }).await;
        }
        
        let stats = monitor.get_stats().await;
        assert_eq!(stats.total_trades, 10);
        assert!(stats.win_rate > dec!(0.5));
    }

    // ========== Bot State Tests ==========

    #[test]
    fn test_bot_state_transitions() {
        let mut state = BotState::default();
        assert!(!state.paused);
        
        // Pause
        state.paused = true;
        assert!(state.paused);
        
        // Resume
        state.paused = false;
        assert!(!state.paused);
    }

    #[test]
    fn test_bot_state_pnl_tracking() {
        let mut state = BotState::default();
        
        // Add profit
        state.daily_pnl = dec!(500);
        assert!(state.daily_pnl > Decimal::ZERO);
        
        // Add loss
        state.daily_pnl = dec!(-300);
        assert!(state.daily_pnl < Decimal::ZERO);
        
        // Hit loss limit
        state.daily_loss_limit_hit = true;
        assert!(state.daily_loss_limit_hit);
    }

    // ========== Position Tests ==========

    #[test]
    fn test_position_pnl_calculation() {
        let position = Position {
            token_id: "token1".to_string(),
            market_id: "market1".to_string(),
            side: Side::Buy,
            size: dec!(100),
            avg_entry_price: dec!(0.50),
            current_price: dec!(0.60),
            unrealized_pnl: dec!(10), // (0.60 - 0.50) * 100
        };
        
        assert_eq!(position.unrealized_pnl, dec!(10));
        
        // Manual calculation
        let expected_pnl = (position.current_price - position.avg_entry_price) * position.size;
        assert_eq!(expected_pnl, dec!(10));
    }

    #[test]
    fn test_position_loss() {
        let position = Position {
            token_id: "token1".to_string(),
            market_id: "market1".to_string(),
            side: Side::Buy,
            size: dec!(100),
            avg_entry_price: dec!(0.60),
            current_price: dec!(0.50),
            unrealized_pnl: dec!(-10),
        };
        
        assert!(position.unrealized_pnl < Decimal::ZERO);
    }

    // ========== Pattern Recognition Tests ==========

    #[test]
    fn test_trading_pattern_identification() {
        let pattern = TradingPattern {
            name: "Contrarian".to_string(),
            description: "Buys at extreme lows, sells at extreme highs".to_string(),
            win_rate: 0.72,
            avg_win: dec!(150),
            avg_loss: dec!(80),
            expected_value: dec!(50),
            sample_count: 50,
            confidence: 0.85,
        };
        
        assert!(pattern.win_rate > 0.7);
        assert!(pattern.avg_win > pattern.avg_loss);
        assert!(pattern.expected_value > Decimal::ZERO);
    }
}
