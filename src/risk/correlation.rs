//! Market Correlation Detection
//!
//! Detects correlation between markets to avoid overexposure
//! to correlated positions.

use crate::utils::sqrt_decimal;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};

/// Market correlation detection and management
pub struct CorrelationDetector {
    /// Price history per market (timestamp -> price)
    price_history: HashMap<String, VecDeque<(i64, Decimal)>>,
    /// Cached correlation matrix
    correlation_cache: CorrelationMatrix,
    /// Minimum correlation to consider markets correlated
    correlation_threshold: Decimal,
    /// Maximum data points to keep
    max_history: usize,
}

/// Correlation matrix between markets
#[derive(Debug, Clone, Default)]
pub struct CorrelationMatrix {
    /// Correlation coefficients: (market_a, market_b) -> correlation
    correlations: HashMap<(String, String), MarketCorrelation>,
}

/// Correlation data between two markets
#[derive(Debug, Clone)]
pub struct MarketCorrelation {
    pub market_a: String,
    pub market_b: String,
    pub correlation: Decimal,
    pub sample_count: usize,
    pub last_update: i64,
}

impl CorrelationDetector {
    pub fn new(correlation_threshold: f64) -> Self {
        Self {
            price_history: HashMap::new(),
            correlation_cache: CorrelationMatrix::default(),
            correlation_threshold: Decimal::try_from(correlation_threshold).unwrap_or(dec!(0.7)),
            max_history: 100,
        }
    }

    /// Add a price point for a market
    pub fn add_price_point(&mut self, market_id: &str, price: Decimal, timestamp: i64) {
        let history = self.price_history
            .entry(market_id.to_string())
            .or_insert_with(VecDeque::new);
        
        history.push_back((timestamp, price));
        
        while history.len() > self.max_history {
            history.pop_front();
        }

        // Update correlations with other markets
        self.update_correlations(market_id);
    }

    /// Get correlation between two markets
    pub fn get_correlation(&self, market_a: &str, market_b: &str) -> Option<Decimal> {
        let key = self.make_key(market_a, market_b);
        self.correlation_cache.correlations.get(&key).map(|c| c.correlation)
    }

    /// Get a penalty multiplier for position sizing based on correlation with existing positions
    /// Returns a value between 0.5 and 1.0
    pub fn get_correlation_penalty(&self, market_id: &str, existing_markets: &[String]) -> Decimal {
        if existing_markets.is_empty() {
            return Decimal::ONE;
        }

        let mut max_correlation = Decimal::ZERO;

        for existing_market in existing_markets {
            if let Some(corr) = self.get_correlation(market_id, existing_market) {
                if corr.abs() > max_correlation {
                    max_correlation = corr.abs();
                }
            }
        }

        // Higher correlation = higher penalty (smaller position)
        // correlation of 0 -> penalty of 1.0 (no reduction)
        // correlation of 1 -> penalty of 0.5 (50% reduction)
        if max_correlation > self.correlation_threshold {
            Decimal::ONE - (max_correlation * dec!(0.5))
        } else {
            Decimal::ONE
        }
    }

    /// Check if two markets are highly correlated
    pub fn are_correlated(&self, market_a: &str, market_b: &str) -> bool {
        self.get_correlation(market_a, market_b)
            .map(|c| c.abs() >= self.correlation_threshold)
            .unwrap_or(false)
    }

    /// Get all correlation data
    pub fn get_matrix(&self) -> &CorrelationMatrix {
        &self.correlation_cache
    }

    /// Get all markets correlated with the given market
    pub fn get_correlated_markets(&self, market_id: &str) -> Vec<MarketCorrelation> {
        self.correlation_cache
            .correlations
            .values()
            .filter(|c| {
                (c.market_a == market_id || c.market_b == market_id)
                    && c.correlation.abs() >= self.correlation_threshold
            })
            .cloned()
            .collect()
    }

    /// Update correlations for a market against all other tracked markets
    fn update_correlations(&mut self, market_id: &str) {
        let other_markets: Vec<String> = self.price_history
            .keys()
            .filter(|k| *k != market_id)
            .cloned()
            .collect();

        for other in other_markets {
            self.calculate_correlation(market_id, &other);
        }
    }

    /// Calculate Pearson correlation coefficient between two markets
    fn calculate_correlation(&mut self, market_a: &str, market_b: &str) {
        let Some(history_a) = self.price_history.get(market_a) else {
            return;
        };
        let Some(history_b) = self.price_history.get(market_b) else {
            return;
        };

        // Find overlapping timestamps
        let times_a: HashMap<i64, Decimal> = history_a.iter().copied().collect();
        let times_b: HashMap<i64, Decimal> = history_b.iter().copied().collect();

        let mut pairs: Vec<(Decimal, Decimal)> = Vec::new();
        for (ts, price_a) in &times_a {
            if let Some(&price_b) = times_b.get(ts) {
                pairs.push((*price_a, price_b));
            }
        }

        // Need at least 5 pairs for meaningful correlation
        if pairs.len() < 5 {
            return;
        }

        let correlation = self.pearson_correlation(&pairs);
        let key = self.make_key(market_a, market_b);

        self.correlation_cache.correlations.insert(
            key,
            MarketCorrelation {
                market_a: market_a.to_string(),
                market_b: market_b.to_string(),
                correlation,
                sample_count: pairs.len(),
                last_update: chrono::Utc::now().timestamp(),
            },
        );
    }

    /// Calculate Pearson correlation coefficient
    fn pearson_correlation(&self, pairs: &[(Decimal, Decimal)]) -> Decimal {
        let n = Decimal::from(pairs.len() as i64);
        
        let sum_x: Decimal = pairs.iter().map(|(x, _)| *x).sum();
        let sum_y: Decimal = pairs.iter().map(|(_, y)| *y).sum();
        let sum_xy: Decimal = pairs.iter().map(|(x, y)| *x * *y).sum();
        let sum_x2: Decimal = pairs.iter().map(|(x, _)| *x * *x).sum();
        let sum_y2: Decimal = pairs.iter().map(|(_, y)| *y * *y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator_sq = (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y);

        if denominator_sq <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        // Approximate square root
        let denominator = sqrt_decimal(denominator_sq);
        if denominator == Decimal::ZERO {
            return Decimal::ZERO;
        }

        numerator / denominator
    }

    /// Create a canonical key for the correlation cache
    fn make_key(&self, market_a: &str, market_b: &str) -> (String, String) {
        if market_a < market_b {
            (market_a.to_string(), market_b.to_string())
        } else {
            (market_b.to_string(), market_a.to_string())
        }
    }
}

impl CorrelationMatrix {
    /// Get the number of correlation entries
    pub fn len(&self) -> usize {
        self.correlations.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.correlations.is_empty()
    }

    /// Iterate over all correlations
    pub fn iter(&self) -> impl Iterator<Item = &MarketCorrelation> {
        self.correlations.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_detector() {
        let detector = CorrelationDetector::new(0.7);
        assert!(detector.price_history.is_empty());
        assert_eq!(detector.correlation_threshold, dec!(0.7));
    }

    #[test]
    fn test_add_price_points() {
        let mut detector = CorrelationDetector::new(0.7);
        
        detector.add_price_point("market1", dec!(0.50), 1000);
        detector.add_price_point("market1", dec!(0.52), 2000);
        
        assert_eq!(detector.price_history.get("market1").unwrap().len(), 2);
    }

    #[test]
    fn test_perfect_positive_correlation() {
        let mut detector = CorrelationDetector::new(0.7);
        
        // Two markets moving together
        for i in 1..=10 {
            let price = Decimal::from(i) / dec!(10);
            detector.add_price_point("market1", price, i as i64);
            detector.add_price_point("market2", price, i as i64);
        }
        
        let corr = detector.get_correlation("market1", "market2");
        assert!(corr.is_some());
        let corr = corr.unwrap();
        assert!(corr > dec!(0.9)); // Should be close to 1.0
    }

    #[test]
    fn test_perfect_negative_correlation() {
        let mut detector = CorrelationDetector::new(0.7);
        
        // Two markets moving opposite
        for i in 1..=10 {
            let price1 = Decimal::from(i) / dec!(10);
            let price2 = Decimal::ONE - price1;
            detector.add_price_point("market1", price1, i as i64);
            detector.add_price_point("market2", price2, i as i64);
        }
        
        let corr = detector.get_correlation("market1", "market2");
        assert!(corr.is_some());
        let corr = corr.unwrap();
        assert!(corr < dec!(-0.9)); // Should be close to -1.0
    }

    #[test]
    fn test_no_correlation_insufficient_data() {
        let mut detector = CorrelationDetector::new(0.7);
        
        // Only 3 data points (need 5 minimum)
        for i in 1..=3 {
            detector.add_price_point("market1", Decimal::from(i), i as i64);
            detector.add_price_point("market2", Decimal::from(i * 2), i as i64);
        }
        
        let corr = detector.get_correlation("market1", "market2");
        assert!(corr.is_none());
    }

    #[test]
    fn test_are_correlated() {
        let mut detector = CorrelationDetector::new(0.7);
        
        // Two markets moving together
        for i in 1..=10 {
            let price = Decimal::from(i) / dec!(10);
            detector.add_price_point("market1", price, i as i64);
            detector.add_price_point("market2", price, i as i64);
        }
        
        assert!(detector.are_correlated("market1", "market2"));
        assert!(!detector.are_correlated("market1", "unknown"));
    }

    #[test]
    fn test_correlation_penalty_no_existing() {
        let detector = CorrelationDetector::new(0.7);
        let penalty = detector.get_correlation_penalty("market1", &[]);
        assert_eq!(penalty, Decimal::ONE);
    }

    #[test]
    fn test_correlation_penalty_with_correlated() {
        let mut detector = CorrelationDetector::new(0.7);
        
        // Two highly correlated markets
        for i in 1..=10 {
            let price = Decimal::from(i) / dec!(10);
            detector.add_price_point("market1", price, i as i64);
            detector.add_price_point("market2", price, i as i64);
        }
        
        let existing = vec!["market1".to_string()];
        let penalty = detector.get_correlation_penalty("market2", &existing);
        
        // Should be reduced due to high correlation
        assert!(penalty < Decimal::ONE);
        assert!(penalty >= dec!(0.5));
    }

    #[test]
    fn test_get_correlated_markets() {
        let mut detector = CorrelationDetector::new(0.7);
        
        // market1 and market2 correlated
        for i in 1..=10 {
            let price = Decimal::from(i) / dec!(10);
            detector.add_price_point("market1", price, i as i64);
            detector.add_price_point("market2", price, i as i64);
        }
        
        // market3 uncorrelated
        for i in 1..=10 {
            let price = dec!(0.5);
            detector.add_price_point("market3", price, i as i64);
        }
        
        let correlated = detector.get_correlated_markets("market1");
        assert_eq!(correlated.len(), 1);
        assert!(correlated[0].market_a == "market1" || correlated[0].market_b == "market1");
    }

    #[test]
    fn test_canonical_key() {
        let detector = CorrelationDetector::new(0.7);
        
        let key1 = detector.make_key("a", "b");
        let key2 = detector.make_key("b", "a");
        
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_max_history_limit() {
        let mut detector = CorrelationDetector::new(0.7);
        detector.max_history = 10;
        
        for i in 0..20 {
            detector.add_price_point("market1", Decimal::from(i), i as i64);
        }
        
        assert_eq!(detector.price_history.get("market1").unwrap().len(), 10);
    }

    #[test]
    fn test_correlation_matrix() {
        let matrix = CorrelationMatrix::default();
        assert!(matrix.is_empty());
        assert_eq!(matrix.len(), 0);
    }
}
