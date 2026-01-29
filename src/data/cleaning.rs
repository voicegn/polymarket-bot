//! Data cleaning and validation
//!
//! Features:
//! - Anomaly detection (price spikes, invalid values)
//! - Data validation (schema, bounds, consistency)
//! - Outlier filtering

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Cleaning configuration
#[derive(Debug, Clone)]
pub struct CleaningConfig {
    /// Maximum allowed price change percentage per update
    pub max_price_change_pct: Decimal,
    /// Minimum valid price (Polymarket: 0.00)
    pub min_price: Decimal,
    /// Maximum valid price (Polymarket: 1.00)
    pub max_price: Decimal,
    /// Maximum spread percentage
    pub max_spread_pct: Decimal,
    /// Window size for moving average outlier detection
    pub ma_window_size: usize,
    /// Standard deviations for outlier detection
    pub outlier_std_devs: Decimal,
    /// Maximum age for stale data (seconds)
    pub max_data_age_secs: i64,
}

impl Default for CleaningConfig {
    fn default() -> Self {
        Self {
            max_price_change_pct: dec!(20.0),  // 20% max change
            min_price: dec!(0.001),             // Minimum valid price
            max_price: dec!(0.999),             // Maximum valid price
            max_spread_pct: dec!(50.0),         // 50% max spread
            ma_window_size: 20,
            outlier_std_devs: dec!(3.0),        // 3 sigma
            max_data_age_secs: 300,             // 5 minutes
        }
    }
}

/// Result of data validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the data is valid
    pub is_valid: bool,
    /// List of anomalies found
    pub anomalies: Vec<Anomaly>,
    /// Cleaned/adjusted value (if applicable)
    pub cleaned_value: Option<Decimal>,
}

impl ValidationResult {
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            anomalies: vec![],
            cleaned_value: None,
        }
    }

    pub fn invalid(anomalies: Vec<Anomaly>) -> Self {
        Self {
            is_valid: false,
            anomalies,
            cleaned_value: None,
        }
    }

    pub fn with_cleaned(mut self, value: Decimal) -> Self {
        self.cleaned_value = Some(value);
        self
    }
}

/// Type of anomaly detected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Anomaly {
    /// Price outside valid bounds
    OutOfBounds { value: Decimal, min: Decimal, max: Decimal },
    /// Price spike exceeds threshold
    PriceSpike { old: Decimal, new: Decimal, change_pct: Decimal },
    /// Spread too wide
    WideSpread { spread_pct: Decimal, threshold: Decimal },
    /// Statistical outlier
    StatisticalOutlier { value: Decimal, mean: Decimal, std_dev: Decimal },
    /// Data too old/stale
    StaleData { age_secs: i64, max_age: i64 },
    /// Invalid bid/ask relationship
    InvalidBidAsk { bid: Decimal, ask: Decimal },
    /// Zero or negative value
    InvalidValue { value: Decimal, reason: String },
    /// Missing required field
    MissingField { field: String },
}

/// Price data point for cleaning
#[derive(Debug, Clone)]
pub struct PricePoint {
    pub price: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Data cleaner with stateful tracking
pub struct DataCleaner {
    config: CleaningConfig,
    /// Historical prices for MA calculation (per token)
    price_history: VecDeque<PricePoint>,
    /// Last valid price
    last_valid_price: Option<Decimal>,
}

impl DataCleaner {
    /// Create a new data cleaner
    pub fn new(config: CleaningConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::with_capacity(100),
            last_valid_price: None,
        }
    }

    /// Create with default config
    pub fn default_cleaner() -> Self {
        Self::new(CleaningConfig::default())
    }

    /// Validate a single price
    pub fn validate_price(&mut self, price: Decimal, timestamp: DateTime<Utc>) -> ValidationResult {
        let mut anomalies = Vec::new();

        // Check bounds
        if price < self.config.min_price || price > self.config.max_price {
            anomalies.push(Anomaly::OutOfBounds {
                value: price,
                min: self.config.min_price,
                max: self.config.max_price,
            });
        }

        // Check for price spike
        if let Some(last_price) = self.last_valid_price {
            if last_price > Decimal::ZERO {
                let change_pct = ((price - last_price).abs() / last_price) * dec!(100);
                if change_pct > self.config.max_price_change_pct {
                    anomalies.push(Anomaly::PriceSpike {
                        old: last_price,
                        new: price,
                        change_pct,
                    });
                }
            }
        }

        // Statistical outlier detection
        if let Some(outlier) = self.check_statistical_outlier(price) {
            anomalies.push(outlier);
        }

        // Check data freshness
        let age = (Utc::now() - timestamp).num_seconds();
        if age > self.config.max_data_age_secs {
            anomalies.push(Anomaly::StaleData {
                age_secs: age,
                max_age: self.config.max_data_age_secs,
            });
        }

        // Update history if valid
        let is_valid = anomalies.is_empty();
        if is_valid {
            self.update_history(price, timestamp);
        }

        let mut result = if is_valid {
            ValidationResult::valid()
        } else {
            ValidationResult::invalid(anomalies)
        };

        // Try to provide cleaned value
        if !is_valid {
            if let Some(cleaned) = self.suggest_cleaned_value(price) {
                result = result.with_cleaned(cleaned);
            }
        }

        result
    }

    /// Validate bid/ask pair
    pub fn validate_bid_ask(
        &self,
        bid: Decimal,
        ask: Decimal,
        timestamp: DateTime<Utc>,
    ) -> ValidationResult {
        let mut anomalies = Vec::new();

        // Basic validation
        if bid <= Decimal::ZERO {
            anomalies.push(Anomaly::InvalidValue {
                value: bid,
                reason: "Bid must be positive".to_string(),
            });
        }
        if ask <= Decimal::ZERO {
            anomalies.push(Anomaly::InvalidValue {
                value: ask,
                reason: "Ask must be positive".to_string(),
            });
        }

        // Bid should be less than ask
        if bid >= ask {
            anomalies.push(Anomaly::InvalidBidAsk { bid, ask });
        }

        // Check spread
        if ask > Decimal::ZERO {
            let spread_pct = ((ask - bid) / ask) * dec!(100);
            if spread_pct > self.config.max_spread_pct {
                anomalies.push(Anomaly::WideSpread {
                    spread_pct,
                    threshold: self.config.max_spread_pct,
                });
            }
        }

        // Check bounds
        if bid < self.config.min_price || bid > self.config.max_price {
            anomalies.push(Anomaly::OutOfBounds {
                value: bid,
                min: self.config.min_price,
                max: self.config.max_price,
            });
        }
        if ask < self.config.min_price || ask > self.config.max_price {
            anomalies.push(Anomaly::OutOfBounds {
                value: ask,
                min: self.config.min_price,
                max: self.config.max_price,
            });
        }

        // Check freshness
        let age = (Utc::now() - timestamp).num_seconds();
        if age > self.config.max_data_age_secs {
            anomalies.push(Anomaly::StaleData {
                age_secs: age,
                max_age: self.config.max_data_age_secs,
            });
        }

        if anomalies.is_empty() {
            ValidationResult::valid()
        } else {
            ValidationResult::invalid(anomalies)
        }
    }

    /// Check for statistical outlier using moving average
    fn check_statistical_outlier(&self, price: Decimal) -> Option<Anomaly> {
        if self.price_history.len() < self.config.ma_window_size {
            return None;
        }

        let (mean, std_dev) = self.calculate_stats();
        
        if std_dev > Decimal::ZERO {
            let z_score = (price - mean).abs() / std_dev;
            if z_score > self.config.outlier_std_devs {
                return Some(Anomaly::StatisticalOutlier {
                    value: price,
                    mean,
                    std_dev,
                });
            }
        }

        None
    }

    /// Calculate mean and standard deviation
    fn calculate_stats(&self) -> (Decimal, Decimal) {
        if self.price_history.is_empty() {
            return (Decimal::ZERO, Decimal::ZERO);
        }

        let n = Decimal::from(self.price_history.len() as u32);
        let sum: Decimal = self.price_history.iter().map(|p| p.price).sum();
        let mean = sum / n;

        let variance: Decimal = self.price_history
            .iter()
            .map(|p| (p.price - mean) * (p.price - mean))
            .sum::<Decimal>() / n;

        // Simple square root approximation for Decimal
        let std_dev = decimal_sqrt(variance);

        (mean, std_dev)
    }

    /// Update price history
    fn update_history(&mut self, price: Decimal, timestamp: DateTime<Utc>) {
        self.price_history.push_back(PricePoint { price, timestamp });
        
        // Keep only recent history
        while self.price_history.len() > self.config.ma_window_size * 2 {
            self.price_history.pop_front();
        }

        self.last_valid_price = Some(price);
    }

    /// Suggest a cleaned value based on context
    fn suggest_cleaned_value(&self, price: Decimal) -> Option<Decimal> {
        // Clamp to bounds
        let clamped = price.max(self.config.min_price).min(self.config.max_price);
        
        // If we have history, use EMA
        if !self.price_history.is_empty() {
            let (mean, _) = self.calculate_stats();
            // Weight towards mean if anomalous
            if (price - mean).abs() > (clamped - mean).abs() {
                return Some(clamped);
            }
            // Return mean if price is way off
            if self.price_history.len() >= self.config.ma_window_size {
                return Some(mean);
            }
        }

        // Return clamped value
        Some(clamped)
    }

    /// Clear history (e.g., for new token)
    pub fn reset(&mut self) {
        self.price_history.clear();
        self.last_valid_price = None;
    }

    /// Get current statistics
    pub fn stats(&self) -> CleanerStats {
        let (mean, std_dev) = self.calculate_stats();
        CleanerStats {
            history_size: self.price_history.len(),
            mean,
            std_dev,
            last_price: self.last_valid_price,
        }
    }
}

/// Statistics from the cleaner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanerStats {
    pub history_size: usize,
    pub mean: Decimal,
    pub std_dev: Decimal,
    pub last_price: Option<Decimal>,
}

/// Newton's method square root for Decimal
fn decimal_sqrt(n: Decimal) -> Decimal {
    if n <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    
    let mut x = n;
    let two = dec!(2);
    
    // Newton's method iterations
    for _ in 0..20 {
        let x_next = (x + n / x) / two;
        if (x_next - x).abs() < dec!(0.0000001) {
            return x_next;
        }
        x = x_next;
    }
    
    x
}

/// Validate multiple prices and filter outliers
pub fn filter_outliers(prices: &[Decimal], config: &CleaningConfig) -> Vec<Decimal> {
    if prices.is_empty() {
        return vec![];
    }

    // Calculate median
    let mut sorted = prices.to_vec();
    sorted.sort();
    let median = sorted[sorted.len() / 2];

    // Calculate MAD (median absolute deviation)
    let mut deviations: Vec<Decimal> = prices
        .iter()
        .map(|p| (*p - median).abs())
        .collect();
    deviations.sort();
    let mad = deviations[deviations.len() / 2];

    // Filter using MAD
    let threshold = mad * config.outlier_std_devs;
    prices
        .iter()
        .filter(|p| (**p - median).abs() <= threshold)
        .copied()
        .collect()
}

/// Validate required fields in a data structure
pub fn validate_required_fields<T: std::fmt::Debug>(
    _data: &T,
    checks: &[(&str, bool)],
) -> ValidationResult {
    let mut anomalies = Vec::new();

    for (field_name, is_present) in checks {
        if !is_present {
            anomalies.push(Anomaly::MissingField {
                field: field_name.to_string(),
            });
        }
    }

    if anomalies.is_empty() {
        ValidationResult::valid()
    } else {
        ValidationResult::invalid(anomalies)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_cleaning_config_default() {
        let config = CleaningConfig::default();
        assert_eq!(config.min_price, dec!(0.001));
        assert_eq!(config.max_price, dec!(0.999));
        assert_eq!(config.max_price_change_pct, dec!(20.0));
    }

    #[test]
    fn test_validate_price_valid() {
        let mut cleaner = DataCleaner::default_cleaner();
        let result = cleaner.validate_price(dec!(0.5), Utc::now());
        assert!(result.is_valid);
        assert!(result.anomalies.is_empty());
    }

    #[test]
    fn test_validate_price_out_of_bounds() {
        let mut cleaner = DataCleaner::default_cleaner();
        
        // Too low
        let result = cleaner.validate_price(dec!(0.0001), Utc::now());
        assert!(!result.is_valid);
        assert!(matches!(result.anomalies[0], Anomaly::OutOfBounds { .. }));
        
        // Too high
        cleaner.reset();
        let result = cleaner.validate_price(dec!(1.5), Utc::now());
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_price_spike() {
        let mut cleaner = DataCleaner::default_cleaner();
        
        // First price
        cleaner.validate_price(dec!(0.5), Utc::now());
        
        // Large spike (>20%)
        let result = cleaner.validate_price(dec!(0.8), Utc::now());
        assert!(!result.is_valid);
        assert!(result.anomalies.iter().any(|a| matches!(a, Anomaly::PriceSpike { .. })));
    }

    #[test]
    fn test_validate_bid_ask_valid() {
        let cleaner = DataCleaner::default_cleaner();
        let result = cleaner.validate_bid_ask(dec!(0.45), dec!(0.55), Utc::now());
        assert!(result.is_valid);
    }

    #[test]
    fn test_validate_bid_ask_invalid() {
        let cleaner = DataCleaner::default_cleaner();
        
        // Bid >= Ask
        let result = cleaner.validate_bid_ask(dec!(0.55), dec!(0.45), Utc::now());
        assert!(!result.is_valid);
        assert!(result.anomalies.iter().any(|a| matches!(a, Anomaly::InvalidBidAsk { .. })));
    }

    #[test]
    fn test_validate_bid_ask_wide_spread() {
        let cleaner = DataCleaner::default_cleaner();
        let result = cleaner.validate_bid_ask(dec!(0.1), dec!(0.9), Utc::now());
        assert!(!result.is_valid);
        assert!(result.anomalies.iter().any(|a| matches!(a, Anomaly::WideSpread { .. })));
    }

    #[test]
    fn test_stale_data() {
        let cleaner = DataCleaner::default_cleaner();
        let old_time = Utc::now() - Duration::seconds(600); // 10 minutes ago
        let result = cleaner.validate_bid_ask(dec!(0.45), dec!(0.55), old_time);
        assert!(!result.is_valid);
        assert!(result.anomalies.iter().any(|a| matches!(a, Anomaly::StaleData { .. })));
    }

    #[test]
    fn test_filter_outliers() {
        let config = CleaningConfig::default();
        let prices = vec![
            dec!(0.50), dec!(0.51), dec!(0.49), dec!(0.50),
            dec!(0.99), // outlier
            dec!(0.48), dec!(0.52),
        ];
        
        let filtered = filter_outliers(&prices, &config);
        assert!(!filtered.contains(&dec!(0.99)));
        assert!(filtered.contains(&dec!(0.50)));
    }

    #[test]
    fn test_decimal_sqrt() {
        // sqrt(4) ≈ 2 (with floating point precision)
        assert!((decimal_sqrt(dec!(4)) - dec!(2)).abs() < dec!(0.0001));
        assert!((decimal_sqrt(dec!(2)) - dec!(1.4142135)).abs() < dec!(0.0001));
        assert_eq!(decimal_sqrt(dec!(0)), dec!(0));
    }

    #[test]
    fn test_cleaner_stats() {
        let mut cleaner = DataCleaner::default_cleaner();
        
        for i in 0..10 {
            cleaner.validate_price(dec!(0.5) + Decimal::from(i) * dec!(0.01), Utc::now());
        }
        
        let stats = cleaner.stats();
        assert_eq!(stats.history_size, 10);
        assert!(stats.mean > dec!(0.5));
        assert!(stats.last_price.is_some());
    }

    #[test]
    fn test_validate_required_fields() {
        let data = "test";
        
        // All present
        let result = validate_required_fields(&data, &[("field1", true), ("field2", true)]);
        assert!(result.is_valid);
        
        // Missing field
        let result = validate_required_fields(&data, &[("field1", true), ("field2", false)]);
        assert!(!result.is_valid);
        assert!(matches!(result.anomalies[0], Anomaly::MissingField { .. }));
    }

    #[test]
    fn test_cleaned_value_suggestion() {
        let mut cleaner = DataCleaner::default_cleaner();
        
        // Out of bounds - should suggest clamped value
        let result = cleaner.validate_price(dec!(1.5), Utc::now());
        assert!(!result.is_valid);
        assert!(result.cleaned_value.is_some());
        assert!(result.cleaned_value.unwrap() <= dec!(0.999));
    }

    #[test]
    fn test_cleaner_reset() {
        let mut cleaner = DataCleaner::default_cleaner();
        cleaner.validate_price(dec!(0.5), Utc::now());
        
        cleaner.reset();
        
        let stats = cleaner.stats();
        assert_eq!(stats.history_size, 0);
        assert!(stats.last_price.is_none());
    }
}
