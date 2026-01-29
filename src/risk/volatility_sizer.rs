//! Volatility-Adaptive Position Sizing
//!
//! Adjusts position sizes based on market volatility:
//! - High volatility → smaller positions
//! - Low volatility → larger positions (up to max)

use crate::utils::sqrt_decimal;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};

/// Configuration for volatility-based sizing
#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    /// Number of price points to calculate volatility
    pub window_size: usize,
    /// Target volatility (annualized, e.g., 0.20 = 20%)
    pub target_volatility: Decimal,
    /// Minimum size multiplier (e.g., 0.25 = 25% of base)
    pub min_multiplier: Decimal,
    /// Maximum size multiplier (e.g., 1.5 = 150% of base)
    pub max_multiplier: Decimal,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            target_volatility: dec!(0.30), // 30% annualized
            min_multiplier: dec!(0.25),
            max_multiplier: dec!(1.5),
        }
    }
}

/// Tracks volatility and provides position size multipliers
pub struct VolatilityPositionSizer {
    config: VolatilityConfig,
    /// Price history per market
    price_history: HashMap<String, VecDeque<Decimal>>,
    /// Cached volatility calculations
    volatility_cache: HashMap<String, VolatilityData>,
}

#[derive(Debug, Clone)]
struct VolatilityData {
    /// Calculated volatility (standard deviation of returns)
    volatility: Decimal,
    /// Timestamp of last update
    last_update: i64,
}

impl VolatilityPositionSizer {
    pub fn new(config: VolatilityConfig) -> Self {
        Self {
            config,
            price_history: HashMap::new(),
            volatility_cache: HashMap::new(),
        }
    }

    /// Add a new price point for a market
    pub fn add_price_point(&mut self, market_id: &str, price: Decimal) {
        let history = self.price_history
            .entry(market_id.to_string())
            .or_insert_with(VecDeque::new);
        
        history.push_back(price);
        
        // Keep only window_size points
        while history.len() > self.config.window_size {
            history.pop_front();
        }

        // Recalculate volatility if we have enough data
        if history.len() >= 2 {
            self.recalculate_volatility(market_id);
        }
    }

    /// Get the position size multiplier for a market
    pub fn get_size_multiplier(&self, market_id: &str) -> Decimal {
        let Some(vol_data) = self.volatility_cache.get(market_id) else {
            return Decimal::ONE; // No data, use base size
        };

        if vol_data.volatility == Decimal::ZERO {
            return self.config.max_multiplier;
        }

        // Inverse volatility scaling: high vol = small multiplier
        let multiplier = self.config.target_volatility / vol_data.volatility;
        
        // Clamp to configured range
        multiplier
            .max(self.config.min_multiplier)
            .min(self.config.max_multiplier)
    }

    /// Get the current volatility for a market
    pub fn get_volatility(&self, market_id: &str) -> Option<Decimal> {
        self.volatility_cache.get(market_id).map(|d| d.volatility)
    }

    /// Check if a market is in high volatility regime
    pub fn is_high_volatility(&self, market_id: &str) -> bool {
        self.get_volatility(market_id)
            .map(|v| v > self.config.target_volatility * dec!(1.5))
            .unwrap_or(false)
    }

    /// Recalculate volatility for a market
    fn recalculate_volatility(&mut self, market_id: &str) {
        let Some(history) = self.price_history.get(market_id) else {
            return;
        };

        if history.len() < 2 {
            return;
        }

        // Calculate returns
        let prices: Vec<Decimal> = history.iter().copied().collect();
        let returns: Vec<Decimal> = prices
            .windows(2)
            .filter_map(|w| {
                if w[0] == Decimal::ZERO {
                    None
                } else {
                    Some((w[1] - w[0]) / w[0])
                }
            })
            .collect();

        if returns.is_empty() {
            return;
        }

        // Calculate standard deviation
        let n = Decimal::from(returns.len() as i64);
        let mean = returns.iter().sum::<Decimal>() / n;
        
        let variance: Decimal = returns
            .iter()
            .map(|r| {
                let diff = *r - mean;
                diff * diff
            })
            .sum::<Decimal>() / n;

        // Approximate square root using Newton's method
        let std_dev = sqrt_decimal(variance);

        // Annualize (assuming ~250 trading days, but we use raw for now)
        let volatility = std_dev;

        self.volatility_cache.insert(
            market_id.to_string(),
            VolatilityData {
                volatility,
                last_update: chrono::Utc::now().timestamp(),
            },
        );
    }

    /// Clear data for a specific market
    pub fn clear_market(&mut self, market_id: &str) {
        self.price_history.remove(market_id);
        self.volatility_cache.remove(market_id);
    }

    /// Get all tracked markets
    pub fn tracked_markets(&self) -> Vec<String> {
        self.price_history.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = VolatilityConfig::default();
        assert_eq!(config.window_size, 20);
        assert_eq!(config.target_volatility, dec!(0.30));
    }

    #[test]
    fn test_new_sizer() {
        let sizer = VolatilityPositionSizer::new(VolatilityConfig::default());
        assert!(sizer.price_history.is_empty());
    }

    #[test]
    fn test_add_price_points() {
        let mut sizer = VolatilityPositionSizer::new(VolatilityConfig::default());
        
        sizer.add_price_point("market1", dec!(0.50));
        sizer.add_price_point("market1", dec!(0.52));
        sizer.add_price_point("market1", dec!(0.48));
        
        assert_eq!(sizer.price_history.get("market1").unwrap().len(), 3);
    }

    #[test]
    fn test_default_multiplier_no_data() {
        let sizer = VolatilityPositionSizer::new(VolatilityConfig::default());
        let multiplier = sizer.get_size_multiplier("unknown");
        assert_eq!(multiplier, Decimal::ONE);
    }

    #[test]
    fn test_volatility_calculation() {
        let mut sizer = VolatilityPositionSizer::new(VolatilityConfig::default());
        
        // Add some price data with known volatility
        let prices = vec![
            dec!(0.50), dec!(0.52), dec!(0.48), dec!(0.51), dec!(0.49),
            dec!(0.53), dec!(0.47), dec!(0.50), dec!(0.52), dec!(0.48),
        ];
        
        for price in prices {
            sizer.add_price_point("market1", price);
        }
        
        let vol = sizer.get_volatility("market1");
        assert!(vol.is_some());
        // Volatility should be positive
        assert!(vol.unwrap() > Decimal::ZERO);
    }

    #[test]
    fn test_high_volatility_small_multiplier() {
        let config = VolatilityConfig {
            window_size: 5,
            target_volatility: dec!(0.10),
            min_multiplier: dec!(0.25),
            max_multiplier: dec!(1.5),
        };
        let mut sizer = VolatilityPositionSizer::new(config);
        
        // High volatility prices (large swings)
        let prices = vec![
            dec!(0.50), dec!(0.70), dec!(0.40), dec!(0.65), dec!(0.35),
        ];
        
        for price in prices {
            sizer.add_price_point("volatile_market", price);
        }
        
        let multiplier = sizer.get_size_multiplier("volatile_market");
        // Should be clamped to min due to high volatility
        assert!(multiplier <= Decimal::ONE);
    }

    #[test]
    fn test_low_volatility_large_multiplier() {
        let config = VolatilityConfig {
            window_size: 5,
            target_volatility: dec!(0.30),
            min_multiplier: dec!(0.25),
            max_multiplier: dec!(1.5),
        };
        let mut sizer = VolatilityPositionSizer::new(config);
        
        // Low volatility prices (small swings)
        let prices = vec![
            dec!(0.500), dec!(0.502), dec!(0.501), dec!(0.503), dec!(0.502),
        ];
        
        for price in prices {
            sizer.add_price_point("stable_market", price);
        }
        
        let multiplier = sizer.get_size_multiplier("stable_market");
        // Should be clamped to max due to low volatility
        assert_eq!(multiplier, dec!(1.5));
    }

    #[test]
    fn test_window_size_limit() {
        let config = VolatilityConfig {
            window_size: 5,
            ..VolatilityConfig::default()
        };
        let mut sizer = VolatilityPositionSizer::new(config);
        
        // Add more than window size
        for i in 0..10 {
            sizer.add_price_point("market1", Decimal::from(i));
        }
        
        assert_eq!(sizer.price_history.get("market1").unwrap().len(), 5);
    }

    #[test]
    fn test_clear_market() {
        let mut sizer = VolatilityPositionSizer::new(VolatilityConfig::default());
        
        sizer.add_price_point("market1", dec!(0.50));
        sizer.add_price_point("market1", dec!(0.52));
        
        sizer.clear_market("market1");
        
        assert!(sizer.price_history.get("market1").is_none());
        assert!(sizer.volatility_cache.get("market1").is_none());
    }

    #[test]
    fn test_tracked_markets() {
        let mut sizer = VolatilityPositionSizer::new(VolatilityConfig::default());
        
        sizer.add_price_point("market1", dec!(0.50));
        sizer.add_price_point("market2", dec!(0.60));
        
        let markets = sizer.tracked_markets();
        assert_eq!(markets.len(), 2);
        assert!(markets.contains(&"market1".to_string()));
        assert!(markets.contains(&"market2".to_string()));
    }

    #[test]
    fn test_is_high_volatility() {
        let config = VolatilityConfig {
            window_size: 5,
            target_volatility: dec!(0.10),
            min_multiplier: dec!(0.25),
            max_multiplier: dec!(1.5),
        };
        let mut sizer = VolatilityPositionSizer::new(config);
        
        // No data - not high vol
        assert!(!sizer.is_high_volatility("unknown"));
        
        // High volatility prices
        let prices = vec![
            dec!(0.50), dec!(0.80), dec!(0.30), dec!(0.75), dec!(0.25),
        ];
        
        for price in prices {
            sizer.add_price_point("wild_market", price);
        }
        
        assert!(sizer.is_high_volatility("wild_market"));
    }
}
