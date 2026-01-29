//! Volatility-adaptive position management
//!
//! Dynamically adjusts:
//! - Take-profit levels based on recent volatility
//! - Stop-loss levels based on ATR (Average True Range)
//! - Position sizing based on volatility regime

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};

/// Rolling volatility calculator
#[derive(Debug, Clone)]
pub struct VolatilityTracker {
    /// Price history per market
    price_history: HashMap<String, VecDeque<PricePoint>>,
    /// Window size for volatility calculation
    window_size: usize,
    /// Cached volatility values
    volatility_cache: HashMap<String, VolatilityMetrics>,
}

#[derive(Debug, Clone)]
struct PricePoint {
    price: Decimal,
    timestamp: DateTime<Utc>,
}

/// Volatility metrics for a market
#[derive(Debug, Clone)]
pub struct VolatilityMetrics {
    /// Standard deviation of returns
    pub std_dev: Decimal,
    /// Average True Range (simplified)
    pub atr: Decimal,
    /// Volatility regime
    pub regime: VolatilityRegime,
    /// 1-hour price change
    pub change_1h: Option<Decimal>,
    /// High-low range in window
    pub range: Decimal,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityRegime {
    /// Very low volatility (<1%)
    VeryLow,
    /// Low volatility (1-3%)
    Low,
    /// Normal volatility (3-6%)
    Normal,
    /// High volatility (6-10%)
    High,
    /// Extreme volatility (>10%)
    Extreme,
}

impl VolatilityRegime {
    /// Multiplier for take-profit target
    pub fn take_profit_multiplier(&self) -> Decimal {
        match self {
            Self::VeryLow => dec!(0.6),  // Lower target in calm markets
            Self::Low => dec!(0.8),
            Self::Normal => dec!(1.0),
            Self::High => dec!(1.3),     // Higher target when volatile
            Self::Extreme => dec!(1.5),
        }
    }

    /// Multiplier for stop-loss
    pub fn stop_loss_multiplier(&self) -> Decimal {
        match self {
            Self::VeryLow => dec!(0.7),  // Tighter stop in calm markets
            Self::Low => dec!(0.85),
            Self::Normal => dec!(1.0),
            Self::High => dec!(1.3),     // Wider stop when volatile
            Self::Extreme => dec!(1.5),
        }
    }

    /// Position size multiplier
    pub fn position_size_multiplier(&self) -> Decimal {
        match self {
            Self::VeryLow => dec!(1.2),  // Larger positions in calm markets
            Self::Low => dec!(1.1),
            Self::Normal => dec!(1.0),
            Self::High => dec!(0.7),     // Smaller positions when volatile
            Self::Extreme => dec!(0.5),
        }
    }

    fn from_std_dev(std_dev: Decimal) -> Self {
        if std_dev < dec!(0.01) {
            Self::VeryLow
        } else if std_dev < dec!(0.03) {
            Self::Low
        } else if std_dev < dec!(0.06) {
            Self::Normal
        } else if std_dev < dec!(0.10) {
            Self::High
        } else {
            Self::Extreme
        }
    }
}

impl VolatilityTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            price_history: HashMap::new(),
            window_size,
            volatility_cache: HashMap::new(),
        }
    }

    /// Record a new price point
    pub fn record_price(&mut self, market_id: &str, price: Decimal) {
        let history = self.price_history
            .entry(market_id.to_string())
            .or_insert_with(|| VecDeque::with_capacity(self.window_size + 1));

        history.push_back(PricePoint {
            price,
            timestamp: Utc::now(),
        });

        // Keep only window_size points
        while history.len() > self.window_size {
            history.pop_front();
        }

        // Update cached volatility
        self.update_volatility(market_id);
    }

    /// Update volatility metrics for a market
    fn update_volatility(&mut self, market_id: &str) {
        let Some(history) = self.price_history.get(market_id) else {
            return;
        };

        if history.len() < 3 {
            return;
        }

        // Calculate returns
        let returns: Vec<Decimal> = history
            .iter()
            .zip(history.iter().skip(1))
            .filter_map(|(prev, curr)| {
                if prev.price > Decimal::ZERO {
                    Some((curr.price - prev.price) / prev.price)
                } else {
                    None
                }
            })
            .collect();

        if returns.is_empty() {
            return;
        }

        // Calculate standard deviation
        let n = Decimal::from(returns.len() as u32);
        let mean = returns.iter().copied().sum::<Decimal>() / n;
        let variance = returns
            .iter()
            .map(|r| (*r - mean) * (*r - mean))
            .sum::<Decimal>()
            / n;
        
        let std_dev = variance.sqrt().unwrap_or(Decimal::ZERO);

        // Calculate ATR (simplified: average absolute return)
        let atr = returns
            .iter()
            .map(|r| r.abs())
            .sum::<Decimal>()
            / n;

        // Calculate range
        let prices: Vec<Decimal> = history.iter().map(|p| p.price).collect();
        let high = prices.iter().copied().max().unwrap_or(Decimal::ZERO);
        let low = prices.iter().copied().min().unwrap_or(Decimal::ZERO);
        let range = if low > Decimal::ZERO {
            (high - low) / low
        } else {
            Decimal::ZERO
        };

        // Calculate 1h change
        let change_1h = history.front().map(|first| {
            let now = Utc::now();
            if now - first.timestamp <= Duration::hours(2) {
                if let Some(last) = history.back() {
                    if first.price > Decimal::ZERO {
                        return Some((last.price - first.price) / first.price);
                    }
                }
            }
            None
        }).flatten();

        let regime = VolatilityRegime::from_std_dev(std_dev);

        self.volatility_cache.insert(
            market_id.to_string(),
            VolatilityMetrics {
                std_dev,
                atr,
                regime,
                change_1h,
                range,
                updated_at: Utc::now(),
            },
        );
    }

    /// Get volatility metrics for a market
    pub fn get_volatility(&self, market_id: &str) -> Option<&VolatilityMetrics> {
        self.volatility_cache.get(market_id)
    }

    /// Get volatility regime (with default)
    pub fn get_regime(&self, market_id: &str) -> VolatilityRegime {
        self.volatility_cache
            .get(market_id)
            .map(|m| m.regime)
            .unwrap_or(VolatilityRegime::Normal)
    }
}

/// Volatility-adaptive exit manager
pub struct VolatilityAdaptiveExits {
    tracker: VolatilityTracker,
    /// Base take-profit level
    base_take_profit: Decimal,
    /// Base stop-loss level
    base_stop_loss: Decimal,
    /// Minimum take-profit (never go below)
    min_take_profit: Decimal,
    /// Maximum stop-loss (never exceed)
    max_stop_loss: Decimal,
}

impl VolatilityAdaptiveExits {
    pub fn new(base_take_profit: Decimal, base_stop_loss: Decimal) -> Self {
        Self {
            tracker: VolatilityTracker::new(30), // 30 data points
            base_take_profit,
            base_stop_loss,
            min_take_profit: dec!(0.02), // Never below 2%
            max_stop_loss: dec!(0.15),   // Never exceed 15%
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(dec!(0.06), dec!(0.08))
    }

    /// Record price for volatility tracking
    pub fn record_price(&mut self, market_id: &str, price: Decimal) {
        self.tracker.record_price(market_id, price);
    }

    /// Get adaptive take-profit level
    pub fn get_take_profit(&self, market_id: &str) -> Decimal {
        let regime = self.tracker.get_regime(market_id);
        let adjusted = self.base_take_profit * regime.take_profit_multiplier();
        adjusted.max(self.min_take_profit)
    }

    /// Get adaptive stop-loss level
    pub fn get_stop_loss(&self, market_id: &str) -> Decimal {
        let regime = self.tracker.get_regime(market_id);
        let adjusted = self.base_stop_loss * regime.stop_loss_multiplier();
        adjusted.min(self.max_stop_loss)
    }

    /// Get position size multiplier
    pub fn get_size_multiplier(&self, market_id: &str) -> Decimal {
        self.tracker.get_regime(market_id).position_size_multiplier()
    }

    /// Get full adaptive parameters
    pub fn get_adaptive_params(&self, market_id: &str) -> AdaptiveParams {
        let regime = self.tracker.get_regime(market_id);
        let volatility = self.tracker.get_volatility(market_id);

        AdaptiveParams {
            take_profit: self.get_take_profit(market_id),
            stop_loss: self.get_stop_loss(market_id),
            size_multiplier: self.get_size_multiplier(market_id),
            regime,
            std_dev: volatility.map(|v| v.std_dev),
            atr: volatility.map(|v| v.atr),
        }
    }

    /// Get volatility metrics
    pub fn get_volatility(&self, market_id: &str) -> Option<&VolatilityMetrics> {
        self.tracker.get_volatility(market_id)
    }
}

/// Adaptive parameters for a market
#[derive(Debug, Clone)]
pub struct AdaptiveParams {
    pub take_profit: Decimal,
    pub stop_loss: Decimal,
    pub size_multiplier: Decimal,
    pub regime: VolatilityRegime,
    pub std_dev: Option<Decimal>,
    pub atr: Option<Decimal>,
}

/// ATR-based trailing stop calculator
pub struct AtrTrailingStop {
    /// ATR multiplier for stop distance
    atr_multiplier: Decimal,
    /// Minimum stop distance
    min_stop_distance: Decimal,
    /// Current high water marks
    high_water_marks: HashMap<String, Decimal>,
    /// Entry prices for reference
    entry_prices: HashMap<String, Decimal>,
}

impl AtrTrailingStop {
    pub fn new(atr_multiplier: Decimal) -> Self {
        Self {
            atr_multiplier,
            min_stop_distance: dec!(0.02), // 2% minimum
            high_water_marks: HashMap::new(),
            entry_prices: HashMap::new(),
        }
    }

    /// Record a new position entry
    pub fn record_entry(&mut self, market_id: &str, entry_price: Decimal) {
        self.entry_prices.insert(market_id.to_string(), entry_price);
        self.high_water_marks.insert(market_id.to_string(), entry_price);
    }

    /// Update with new price
    pub fn update_price(&mut self, market_id: &str, price: Decimal) {
        let high = self.high_water_marks
            .entry(market_id.to_string())
            .or_insert(price);
        if price > *high {
            *high = price;
        }
    }

    /// Calculate trailing stop level
    pub fn get_stop_level(
        &self,
        market_id: &str,
        current_price: Decimal,
        atr: Option<Decimal>,
    ) -> Option<Decimal> {
        let high = self.high_water_marks.get(market_id)?;
        let entry = self.entry_prices.get(market_id)?;

        // Calculate stop distance
        let stop_distance = atr
            .map(|a| a * self.atr_multiplier)
            .unwrap_or(self.min_stop_distance)
            .max(self.min_stop_distance);

        // Trailing stop from high water mark
        let trailing_stop = *high * (Decimal::ONE - stop_distance);

        // Never below entry (unless we're already below entry)
        let floor = if current_price > *entry {
            *entry * (Decimal::ONE - dec!(0.02)) // 2% below entry minimum
        } else {
            Decimal::ZERO
        };

        Some(trailing_stop.max(floor))
    }

    /// Check if stop is triggered
    pub fn is_stopped(&self, market_id: &str, current_price: Decimal, atr: Option<Decimal>) -> bool {
        self.get_stop_level(market_id, current_price, atr)
            .map(|stop| current_price <= stop)
            .unwrap_or(false)
    }

    /// Close position (cleanup)
    pub fn close_position(&mut self, market_id: &str) {
        self.high_water_marks.remove(market_id);
        self.entry_prices.remove(market_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_regime_multipliers() {
        assert!(VolatilityRegime::VeryLow.take_profit_multiplier() < dec!(1));
        assert!(VolatilityRegime::High.take_profit_multiplier() > dec!(1));
        
        assert!(VolatilityRegime::VeryLow.stop_loss_multiplier() < dec!(1));
        assert!(VolatilityRegime::High.stop_loss_multiplier() > dec!(1));
        
        assert!(VolatilityRegime::VeryLow.position_size_multiplier() > dec!(1));
        assert!(VolatilityRegime::High.position_size_multiplier() < dec!(1));
    }

    #[test]
    fn test_volatility_tracker_basic() {
        let mut tracker = VolatilityTracker::new(10);
        
        // Record stable prices
        for i in 0..10 {
            tracker.record_price("stable", dec!(0.50) + Decimal::from(i) * dec!(0.001));
        }
        
        let vol = tracker.get_volatility("stable");
        assert!(vol.is_some());
        
        // Should be low volatility
        assert!(matches!(
            tracker.get_regime("stable"),
            VolatilityRegime::VeryLow | VolatilityRegime::Low
        ));
    }

    #[test]
    fn test_volatility_tracker_high_volatility() {
        let mut tracker = VolatilityTracker::new(10);
        
        // Record wild price swings
        let prices = [0.30, 0.50, 0.35, 0.60, 0.40, 0.55, 0.45, 0.65, 0.35, 0.55];
        for p in prices {
            tracker.record_price("wild", Decimal::from_f64_retain(p).unwrap());
        }
        
        // Should be high or extreme volatility
        assert!(matches!(
            tracker.get_regime("wild"),
            VolatilityRegime::High | VolatilityRegime::Extreme
        ));
    }

    #[test]
    fn test_adaptive_exits() {
        let mut exits = VolatilityAdaptiveExits::with_defaults();
        
        // Simulate stable market
        for i in 0..20 {
            exits.record_price("stable", dec!(0.50) + Decimal::from(i) * dec!(0.0005));
        }
        
        // Simulate volatile market
        let volatile_prices = [0.40, 0.55, 0.45, 0.60, 0.42, 0.58, 0.44, 0.56];
        for p in volatile_prices {
            exits.record_price("volatile", Decimal::from_f64_retain(p).unwrap());
        }
        
        // Stable should have tighter targets
        let stable_tp = exits.get_take_profit("stable");
        let volatile_tp = exits.get_take_profit("volatile");
        assert!(stable_tp < volatile_tp || stable_tp == exits.min_take_profit);
        
        // Volatile should have wider stops
        let stable_sl = exits.get_stop_loss("stable");
        let volatile_sl = exits.get_stop_loss("volatile");
        assert!(volatile_sl >= stable_sl);
    }

    #[test]
    fn test_adaptive_params() {
        let exits = VolatilityAdaptiveExits::with_defaults();
        
        let params = exits.get_adaptive_params("unknown");
        
        // Unknown market should use defaults (normal regime)
        assert_eq!(params.regime, VolatilityRegime::Normal);
        assert!(params.take_profit > Decimal::ZERO);
        assert!(params.stop_loss > Decimal::ZERO);
    }

    #[test]
    fn test_atr_trailing_stop() {
        let mut stop = AtrTrailingStop::new(dec!(2.0)); // 2x ATR
        
        // Enter at 0.50
        stop.record_entry("test", dec!(0.50));
        
        // Price rises to 0.55
        stop.update_price("test", dec!(0.55));
        
        // Price now at 0.52, ATR = 0.02 (2%)
        let stop_level = stop.get_stop_level("test", dec!(0.52), Some(dec!(0.02)));
        assert!(stop_level.is_some());
        
        // Stop should be below high water mark (0.55) by 2x ATR (4%)
        // 0.55 * 0.96 = 0.528
        let level = stop_level.unwrap();
        assert!(level < dec!(0.55));
        assert!(level > dec!(0.50)); // But above entry
    }

    #[test]
    fn test_trailing_stop_triggered() {
        let mut stop = AtrTrailingStop::new(dec!(1.5));
        
        stop.record_entry("test", dec!(0.50));
        stop.update_price("test", dec!(0.60)); // High water mark
        
        // Price drops - with 1.5x ATR of 2% = 3% stop
        // Stop level = 0.60 * 0.97 = 0.582
        
        // Not triggered at 0.58
        assert!(!stop.is_stopped("test", dec!(0.59), Some(dec!(0.02))));
        
        // Triggered at 0.57
        assert!(stop.is_stopped("test", dec!(0.57), Some(dec!(0.02))));
    }

    #[test]
    fn test_position_cleanup() {
        let mut stop = AtrTrailingStop::new(dec!(2.0));
        
        stop.record_entry("test", dec!(0.50));
        assert!(stop.get_stop_level("test", dec!(0.50), None).is_some());
        
        stop.close_position("test");
        assert!(stop.get_stop_level("test", dec!(0.50), None).is_none());
    }
}
