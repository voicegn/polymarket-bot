//! Boundary Condition Tests
//!
//! Tests for extreme inputs, zero values, overflow scenarios, and edge cases.

use crate::config::{RiskConfig, StrategyConfig};
use crate::strategy::{
    DynamicKelly, DynamicKellyConfig, MarketContext,
    SignalGenerator,
};
use crate::model::Prediction;
use crate::types::{Market, Outcome, Side, Signal, Position};
use chrono::Utc;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ZERO VALUE TESTS ====================

    #[test]
    fn test_kelly_with_zero_probability() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let result = kelly.calculate_position_size(
            dec!(0),        // Zero model prob
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        assert_eq!(result.position_size, dec!(0));
    }

    #[test]
    fn test_kelly_with_zero_market_price() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0),        // Zero market price
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        // Should handle division by near-zero gracefully
        assert!(result.position_size >= dec!(0));
    }

    #[test]
    fn test_kelly_with_zero_confidence() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0),        // Zero confidence
            dec!(1.0),
            None,
        );
        
        // Zero confidence should lead to minimum position
        assert!(result.position_size <= dec!(0.01));
    }

    #[test]
    fn test_kelly_with_zero_budget() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(0),        // Zero budget
            None,
        );
        
        // Should still give some position (budget multiplier is 0.5 at worst)
        assert!(result.position_size > dec!(0));
    }

    #[test]
    fn test_kelly_with_zero_account_value() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(0));
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        assert!(result.position_size >= dec!(0));
    }

    #[test]
    fn test_market_with_zero_liquidity() {
        let market = Market {
            id: "zero-liq".to_string(),
            question: "Zero liquidity market?".to_string(),
            description: None,
            end_date: Some(Utc::now()),
            volume: dec!(0),
            liquidity: dec!(0),
            outcomes: vec![
                Outcome {
                    token_id: "yes".to_string(),
                    outcome: "Yes".to_string(),
                    price: dec!(0.50),
                },
                Outcome {
                    token_id: "no".to_string(),
                    outcome: "No".to_string(),
                    price: dec!(0.50),
                },
            ],
            active: true,
            closed: false,
        };
        
        assert_eq!(market.yes_price(), Some(dec!(0.50)));
        assert_eq!(market.no_price(), Some(dec!(0.50)));
        assert_eq!(market.arbitrage_opportunity(), None);
    }

    // ==================== EXTREME VALUE TESTS ====================

    #[test]
    fn test_kelly_with_extreme_edge() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        // Extreme edge: model says 99%, market says 1%
        let result = kelly.calculate_position_size(
            dec!(0.99),
            dec!(0.01),
            dec!(1.0),
            dec!(1.0),
            None,
        );
        
        // Should be capped at max fraction
        assert!(result.effective_fraction <= dec!(0.50));
    }

    #[test]
    fn test_kelly_with_maximum_probabilities() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let result = kelly.calculate_position_size(
            dec!(1.0),      // 100% probability
            dec!(0.99),
            dec!(1.0),
            dec!(1.0),
            None,
        );
        
        assert!(result.position_size > dec!(0));
        assert!(result.effective_fraction <= dec!(0.50));
    }

    #[test]
    fn test_kelly_with_market_price_near_one() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let result = kelly.calculate_position_size(
            dec!(0.999),
            dec!(0.99),     // Very high market price
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        // Very small potential profit, should still work
        println!("Near-one price result: {:?}", result);
    }

    #[test]
    fn test_market_with_extreme_prices() {
        let market = Market {
            id: "extreme".to_string(),
            question: "Extreme prices?".to_string(),
            description: None,
            end_date: None,
            volume: dec!(100000),
            liquidity: dec!(50000),
            outcomes: vec![
                Outcome {
                    token_id: "yes".to_string(),
                    outcome: "Yes".to_string(),
                    price: dec!(0.001),
                },
                Outcome {
                    token_id: "no".to_string(),
                    outcome: "No".to_string(),
                    price: dec!(0.999),
                },
            ],
            active: true,
            closed: false,
        };
        
        assert_eq!(market.yes_price(), Some(dec!(0.001)));
        assert_eq!(market.no_price(), Some(dec!(0.999)));
        assert_eq!(market.arbitrage_opportunity(), None);
    }

    #[test]
    fn test_large_decimal_values() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(1_000_000_000));
        
        kelly.update_account_value(dec!(1_000_000_000));
        kelly.update_account_value(dec!(999_000_000));
        
        let stats = kelly.get_stats();
        assert!(stats.current_drawdown > dec!(0));
        assert!(stats.current_drawdown < dec!(0.01));
    }

    #[test]
    fn test_very_small_decimal_values() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(0.0001));
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        assert!(result.position_size >= dec!(0));
    }

    // ==================== OVERFLOW PREVENTION TESTS ====================

    #[test]
    fn test_kelly_many_consecutive_wins() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        for i in 0..100 {
            kelly.record_trade(Decimal::from(100 + i));
        }
        
        let result = kelly.calculate_position_size(
            dec!(0.70),
            dec!(0.50),
            dec!(0.90),
            dec!(1.0),
            None,
        );
        
        assert!(result.adjustments.streak_multiplier <= dec!(1.30));
        assert!(result.effective_fraction <= dec!(0.50));
    }

    #[test]
    fn test_kelly_many_consecutive_losses() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        for i in 0..100 {
            kelly.record_trade(-Decimal::from(50 + i));
        }
        
        let result = kelly.calculate_position_size(
            dec!(0.70),
            dec!(0.50),
            dec!(0.90),
            dec!(1.0),
            None,
        );
        
        assert!(result.adjustments.streak_multiplier >= dec!(0.50));
        assert!(result.effective_fraction >= dec!(0.05));
    }

    #[test]
    fn test_extreme_drawdown_handling() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        kelly.update_account_value(dec!(10000));
        kelly.update_account_value(dec!(100));  // 99% drawdown
        
        let result = kelly.calculate_position_size(
            dec!(0.70),
            dec!(0.50),
            dec!(0.90),
            dec!(1.0),
            None,
        );
        
        assert!(result.effective_fraction >= dec!(0.05));
        assert!(result.adjustments.drawdown_multiplier >= dec!(0.5));
    }

    // ==================== SIGNAL GENERATOR BOUNDARY TESTS ====================

    #[test]
    fn test_signal_generator_tiny_edge() {
        let generator = SignalGenerator::new(
            StrategyConfig::default(),
            RiskConfig::default(),
        );
        
        let market = create_test_market(dec!(0.5001), dec!(0.4999));
        let prediction = Prediction {
            probability: dec!(0.501),  // Tiny edge
            confidence: dec!(0.90),
            reasoning: "Test".to_string(),
        };
        
        let signal = generator.generate(&market, &prediction);
        assert!(signal.is_none());
    }

    #[test]
    fn test_signal_generator_low_confidence() {
        let generator = SignalGenerator::new(
            StrategyConfig::default(),
            RiskConfig::default(),
        );
        
        let market = create_test_market(dec!(0.50), dec!(0.50));
        let prediction = Prediction {
            probability: dec!(0.70),
            confidence: dec!(0.30),   // Low confidence
            reasoning: "Test".to_string(),
        };
        
        let signal = generator.generate(&market, &prediction);
        assert!(signal.is_none());
    }

    #[test]
    fn test_signal_generator_perfect_confidence() {
        let generator = SignalGenerator::new(
            StrategyConfig::default(),
            RiskConfig::default(),
        );
        
        let market = create_test_market(dec!(0.50), dec!(0.50));
        let prediction = Prediction {
            probability: dec!(0.70),
            confidence: dec!(1.0),
            reasoning: "Test".to_string(),
        };
        
        let signal = generator.generate(&market, &prediction);
        assert!(signal.is_some());
    }

    #[test]
    fn test_signal_generator_empty_outcomes() {
        let generator = SignalGenerator::new(
            StrategyConfig::default(),
            RiskConfig::default(),
        );
        
        let market = Market {
            id: "empty".to_string(),
            question: "Empty outcomes?".to_string(),
            description: None,
            end_date: None,
            volume: dec!(10000),
            liquidity: dec!(5000),
            outcomes: vec![],  // Empty outcomes
            active: true,
            closed: false,
        };
        
        let prediction = Prediction {
            probability: dec!(0.70),
            confidence: dec!(0.90),
            reasoning: "Test".to_string(),
        };
        
        let signal = generator.generate(&market, &prediction);
        assert!(signal.is_none());  // Should handle gracefully
    }

    // ==================== POSITION BOUNDARY TESTS ====================

    #[test]
    fn test_position_zero_values() {
        let position = Position {
            token_id: "test".to_string(),
            market_id: "market".to_string(),
            side: Side::Buy,
            size: dec!(0),
            avg_entry_price: dec!(0),
            current_price: dec!(0),
            unrealized_pnl: dec!(0),
        };
        
        assert_eq!(position.size, dec!(0));
        assert_eq!(position.unrealized_pnl, dec!(0));
    }

    #[test]
    fn test_position_negative_pnl() {
        let position = Position {
            token_id: "test".to_string(),
            market_id: "market".to_string(),
            side: Side::Buy,
            size: dec!(100),
            avg_entry_price: dec!(0.70),
            current_price: dec!(0.30),
            unrealized_pnl: dec!(-40),
        };
        
        assert!(position.unrealized_pnl < dec!(0));
    }

    // ==================== ARBITRAGE BOUNDARY TESTS ====================

    #[test]
    fn test_arbitrage_prices_sum_to_one() {
        let market = Market {
            id: "no-arb".to_string(),
            question: "Prices sum to one?".to_string(),
            description: None,
            end_date: None,
            volume: dec!(10000),
            liquidity: dec!(5000),
            outcomes: vec![
                Outcome { token_id: "yes".to_string(), outcome: "Yes".to_string(), price: dec!(0.50) },
                Outcome { token_id: "no".to_string(), outcome: "No".to_string(), price: dec!(0.50) },
            ],
            active: true,
            closed: false,
        };
        
        assert!(market.arbitrage_opportunity().is_none());
    }

    #[test]
    fn test_arbitrage_prices_sum_over_one() {
        let market = Market {
            id: "overpriced".to_string(),
            question: "Overpriced?".to_string(),
            description: None,
            end_date: None,
            volume: dec!(10000),
            liquidity: dec!(5000),
            outcomes: vec![
                Outcome { token_id: "yes".to_string(), outcome: "Yes".to_string(), price: dec!(0.60) },
                Outcome { token_id: "no".to_string(), outcome: "No".to_string(), price: dec!(0.50) },
            ],
            active: true,
            closed: false,
        };
        
        assert!(market.arbitrage_opportunity().is_none());
    }

    #[test]
    fn test_arbitrage_prices_sum_under_one() {
        let market = Market {
            id: "arb".to_string(),
            question: "Arbitrage opportunity?".to_string(),
            description: None,
            end_date: None,
            volume: dec!(10000),
            liquidity: dec!(5000),
            outcomes: vec![
                Outcome { token_id: "yes".to_string(), outcome: "Yes".to_string(), price: dec!(0.40) },
                Outcome { token_id: "no".to_string(), outcome: "No".to_string(), price: dec!(0.50) },
            ],
            active: true,
            closed: false,
        };
        
        let arb = market.arbitrage_opportunity();
        assert!(arb.is_some());
        assert!((arb.unwrap() - dec!(0.10)).abs() < dec!(0.001));
    }

    // ==================== SIGNAL TRADEABLE BOUNDARY TESTS ====================

    #[test]
    fn test_signal_at_exact_threshold() {
        let signal = Signal {
            market_id: "test".to_string(),
            token_id: "token".to_string(),
            side: Side::Buy,
            model_probability: dec!(0.60),
            market_probability: dec!(0.50),
            edge: dec!(0.05),       // Exactly at threshold
            confidence: dec!(0.60), // Exactly at threshold
            suggested_size: dec!(0.05),
            timestamp: Utc::now(),
        };
        
        assert!(signal.is_tradeable(dec!(0.05), dec!(0.60)));
        assert!(!signal.is_tradeable(dec!(0.051), dec!(0.60)));
    }

    #[test]
    fn test_signal_negative_edge() {
        let signal = Signal {
            market_id: "test".to_string(),
            token_id: "token".to_string(),
            side: Side::Sell,
            model_probability: dec!(0.40),
            market_probability: dec!(0.50),
            edge: dec!(-0.10),
            confidence: dec!(0.80),
            suggested_size: dec!(0.05),
            timestamp: Utc::now(),
        };
        
        // Negative edge should use abs() in is_tradeable
        assert!(signal.is_tradeable(dec!(0.05), dec!(0.60)));
    }

    // ==================== VOLATILITY CONTEXT TESTS ====================

    #[test]
    fn test_kelly_with_extreme_volatility() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let ctx = MarketContext {
            volatility: dec!(0.50),  // 50% volatility (very high)
            liquidity_score: dec!(1.0),
            time_pressure: dec!(0),
        };
        
        let result = kelly.calculate_position_size(
            dec!(0.70),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            Some(&ctx),
        );
        
        // High volatility should reduce position
        assert!(result.adjustments.volatility_multiplier < dec!(1));
    }

    #[test]
    fn test_kelly_with_low_liquidity_score() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let ctx = MarketContext {
            volatility: dec!(0.05),
            liquidity_score: dec!(0.1),  // Very low liquidity
            time_pressure: dec!(0),
        };
        
        let result = kelly.calculate_position_size(
            dec!(0.70),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            Some(&ctx),
        );
        
        // Low liquidity should cap multiplier
        assert!(result.adjustments.liquidity_multiplier >= dec!(0.3));
    }

    #[test]
    fn test_kelly_with_high_time_pressure() {
        let kelly = DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000));
        
        let ctx = MarketContext {
            volatility: dec!(0.05),
            liquidity_score: dec!(1.0),
            time_pressure: dec!(1.0),  // Maximum time pressure
        };
        
        let result = kelly.calculate_position_size(
            dec!(0.70),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            Some(&ctx),
        );
        
        // Time pressure should reduce position
        assert!(result.adjustments.time_multiplier < dec!(1));
    }

    // ==================== HELPERS ====================

    fn create_test_market(yes_price: Decimal, no_price: Decimal) -> Market {
        Market {
            id: "test".to_string(),
            question: "Test market?".to_string(),
            description: None,
            end_date: Some(Utc::now() + chrono::Duration::days(1)),
            volume: dec!(10000),
            liquidity: dec!(5000),
            outcomes: vec![
                Outcome { token_id: "yes".to_string(), outcome: "Yes".to_string(), price: yes_price },
                Outcome { token_id: "no".to_string(), outcome: "No".to_string(), price: no_price },
            ],
            active: true,
            closed: false,
        }
    }
}
