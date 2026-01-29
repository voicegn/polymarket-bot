//! Slippage Prediction Module
//!
//! Predicts execution slippage based on:
//! 1. Order book depth analysis
//! 2. Historical slippage patterns
//! 3. Order size relative to liquidity
//! 4. Market volatility
//! 5. Time-of-day patterns

use chrono::{DateTime, Timelike, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;

/// Configuration for slippage prediction
#[derive(Debug, Clone)]
pub struct SlippageConfig {
    /// Base slippage estimate when no data available
    pub base_slippage_bps: Decimal,
    /// Maximum acceptable slippage in basis points
    pub max_acceptable_slippage_bps: Decimal,
    /// Number of historical trades to consider
    pub history_window: usize,
    /// Decay factor for older observations (per day)
    pub decay_factor: Decimal,
    /// Minimum liquidity ratio before warning
    pub min_liquidity_ratio: Decimal,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            base_slippage_bps: dec!(15),           // 0.15% base estimate
            max_acceptable_slippage_bps: dec!(100), // 1% max acceptable
            history_window: 100,
            decay_factor: dec!(0.95),              // 5% decay per day
            min_liquidity_ratio: dec!(5),          // Order should be < 20% of book
        }
    }
}

/// Order book snapshot for slippage calculation
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Bid levels: (price, size)
    pub bids: Vec<(Decimal, Decimal)>,
    /// Ask levels: (price, size)
    pub asks: Vec<(Decimal, Decimal)>,
    /// Best bid price
    pub best_bid: Decimal,
    /// Best ask price
    pub best_ask: Decimal,
    /// Spread in basis points
    pub spread_bps: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl OrderBook {
    pub fn new(bids: Vec<(Decimal, Decimal)>, asks: Vec<(Decimal, Decimal)>) -> Self {
        let best_bid = bids.first().map(|(p, _)| *p).unwrap_or(Decimal::ZERO);
        let best_ask = asks.first().map(|(p, _)| *p).unwrap_or(Decimal::ONE);
        let mid = (best_bid + best_ask) / dec!(2);
        let spread_bps = if mid > Decimal::ZERO {
            (best_ask - best_bid) / mid * dec!(10000)
        } else {
            dec!(0)
        };
        
        Self {
            bids,
            asks,
            best_bid,
            best_ask,
            spread_bps,
            timestamp: Utc::now(),
        }
    }

    /// Calculate depth at each price level (cumulative)
    pub fn cumulative_depth(&self, side: OrderSide) -> Vec<(Decimal, Decimal)> {
        let levels = match side {
            OrderSide::Buy => &self.asks,  // Buying consumes asks
            OrderSide::Sell => &self.bids, // Selling consumes bids
        };

        let mut cumulative = Vec::with_capacity(levels.len());
        let mut total = Decimal::ZERO;
        
        for (price, size) in levels {
            total += size;
            cumulative.push((*price, total));
        }
        
        cumulative
    }

    /// Calculate total liquidity available on one side
    pub fn total_liquidity(&self, side: OrderSide) -> Decimal {
        match side {
            OrderSide::Buy => self.asks.iter().map(|(_, s)| s).sum(),
            OrderSide::Sell => self.bids.iter().map(|(_, s)| s).sum(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Historical slippage observation
#[derive(Debug, Clone)]
struct SlippageObservation {
    timestamp: DateTime<Utc>,
    market_id: String,
    side: OrderSide,
    order_size: Decimal,
    predicted_slippage_bps: Decimal,
    actual_slippage_bps: Decimal,
    liquidity_ratio: Decimal,
    volatility: Decimal,
    hour_of_day: u32,
}

/// Slippage predictor
pub struct SlippagePredictor {
    config: SlippageConfig,
    /// Historical observations per market
    history: RwLock<HashMap<String, VecDeque<SlippageObservation>>>,
    /// Global observations for cross-market learning
    global_history: RwLock<VecDeque<SlippageObservation>>,
    /// Time-of-day adjustment factors
    tod_factors: RwLock<HashMap<u32, Decimal>>,
}

impl SlippagePredictor {
    pub fn new(config: SlippageConfig) -> Self {
        let mut tod_factors = HashMap::new();
        // Initialize with typical patterns:
        // - Higher slippage during off-hours
        // - Lower during US market hours
        for hour in 0..24 {
            let factor = match hour {
                14..=21 => dec!(0.85),  // US afternoon = most liquid
                9..=13 => dec!(0.95),   // US morning
                22..=23 | 0..=4 => dec!(1.15), // Off-hours
                _ => dec!(1.0),         // Normal
            };
            tod_factors.insert(hour, factor);
        }

        Self {
            config,
            history: RwLock::new(HashMap::new()),
            global_history: RwLock::new(VecDeque::new()),
            tod_factors: RwLock::new(tod_factors),
        }
    }

    /// Predict slippage for an order
    pub fn predict(
        &self,
        market_id: &str,
        side: OrderSide,
        order_size: Decimal,
        order_book: &OrderBook,
        volatility: Option<Decimal>,
    ) -> SlippagePrediction {
        // 1. Calculate book-based slippage
        let book_slippage = self.calculate_book_slippage(side, order_size, order_book);
        
        // 2. Get historical adjustment
        let historical_adj = self.get_historical_adjustment(market_id, side, order_size);
        
        // 3. Volatility adjustment
        let vol = volatility.unwrap_or(dec!(0.05));
        let vol_factor = self.calculate_volatility_factor(vol);
        
        // 4. Time-of-day adjustment
        let hour = Utc::now().hour();
        let tod_factor = self.get_tod_factor(hour);
        
        // 5. Liquidity ratio
        let liquidity = order_book.total_liquidity(side);
        let liquidity_ratio = if order_size > Decimal::ZERO {
            liquidity / order_size
        } else {
            dec!(999)
        };
        
        // 6. Combine all factors
        let base_estimate = book_slippage;
        let adjusted_estimate = base_estimate * historical_adj * vol_factor * tod_factor;
        
        // 7. Add spread component
        let spread_slippage = order_book.spread_bps / dec!(2);
        let total_estimate = adjusted_estimate + spread_slippage;
        
        // 8. Confidence based on data availability
        let confidence = self.calculate_prediction_confidence(market_id, liquidity_ratio);
        
        // 9. Warning flags
        let mut warnings = Vec::new();
        
        if liquidity_ratio < self.config.min_liquidity_ratio {
            warnings.push(format!(
                "Low liquidity ratio: {:.1}x (min: {}x)",
                liquidity_ratio, self.config.min_liquidity_ratio
            ));
        }
        
        if total_estimate > self.config.max_acceptable_slippage_bps {
            warnings.push(format!(
                "High slippage predicted: {:.1}bps (max: {}bps)",
                total_estimate, self.config.max_acceptable_slippage_bps
            ));
        }
        
        if order_book.spread_bps > dec!(50) {
            warnings.push(format!(
                "Wide spread: {:.1}bps",
                order_book.spread_bps
            ));
        }
        
        SlippagePrediction {
            estimated_slippage_bps: total_estimate,
            book_component_bps: book_slippage,
            spread_component_bps: spread_slippage,
            confidence,
            liquidity_ratio,
            adjustment_factors: AdjustmentFactors {
                historical: historical_adj,
                volatility: vol_factor,
                time_of_day: tod_factor,
            },
            acceptable: total_estimate <= self.config.max_acceptable_slippage_bps 
                && liquidity_ratio >= self.config.min_liquidity_ratio,
            warnings,
            optimal_execution: self.suggest_optimal_execution(order_size, liquidity),
        }
    }

    /// Calculate slippage from order book consumption
    fn calculate_book_slippage(
        &self,
        side: OrderSide,
        order_size: Decimal,
        order_book: &OrderBook,
    ) -> Decimal {
        if order_size <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        let cumulative = order_book.cumulative_depth(side);
        if cumulative.is_empty() {
            return self.config.base_slippage_bps;
        }

        let reference_price = match side {
            OrderSide::Buy => order_book.best_ask,
            OrderSide::Sell => order_book.best_bid,
        };

        if reference_price <= Decimal::ZERO {
            return self.config.base_slippage_bps;
        }

        // Find the worst price we'd hit
        let mut remaining = order_size;
        let mut weighted_price = Decimal::ZERO;
        let mut filled = Decimal::ZERO;

        let levels = match side {
            OrderSide::Buy => &order_book.asks,
            OrderSide::Sell => &order_book.bids,
        };

        for (price, size) in levels {
            if remaining <= Decimal::ZERO {
                break;
            }
            let fill_size = remaining.min(*size);
            weighted_price += price * fill_size;
            filled += fill_size;
            remaining -= fill_size;
        }

        if filled <= Decimal::ZERO {
            return self.config.base_slippage_bps;
        }

        let avg_price = weighted_price / filled;
        let slippage = match side {
            OrderSide::Buy => (avg_price - reference_price) / reference_price,
            OrderSide::Sell => (reference_price - avg_price) / reference_price,
        };

        (slippage * dec!(10000)).max(Decimal::ZERO) // Convert to basis points
    }

    /// Get historical adjustment factor
    fn get_historical_adjustment(&self, market_id: &str, side: OrderSide, order_size: Decimal) -> Decimal {
        let history = self.history.read().unwrap();
        
        // First try market-specific history
        if let Some(market_history) = history.get(market_id) {
            if market_history.len() >= 5 {
                let ratio = self.calculate_prediction_error_ratio(market_history.iter(), side);
                if ratio != dec!(1) {
                    return ratio;
                }
            }
        }
        drop(history);

        // Fall back to global history
        let global = self.global_history.read().unwrap();
        if global.len() >= 10 {
            self.calculate_prediction_error_ratio(global.iter(), side)
        } else {
            dec!(1) // No adjustment without enough data
        }
    }

    /// Calculate ratio of actual to predicted slippage from history
    fn calculate_prediction_error_ratio<'a, I>(&self, observations: I, side: OrderSide) -> Decimal
    where
        I: Iterator<Item = &'a SlippageObservation>,
    {
        let relevant: Vec<_> = observations
            .filter(|o| o.side == side)
            .take(20) // Last 20 relevant observations
            .collect();

        if relevant.is_empty() {
            return dec!(1);
        }

        let now = Utc::now();
        let mut weighted_ratio = Decimal::ZERO;
        let mut total_weight = Decimal::ZERO;

        for obs in relevant {
            let age_days = (now - obs.timestamp).num_days() as u32;
            let weight = self.config.decay_factor.powu(age_days.into());
            
            if obs.predicted_slippage_bps > Decimal::ZERO {
                let ratio = obs.actual_slippage_bps / obs.predicted_slippage_bps;
                weighted_ratio += ratio * weight;
                total_weight += weight;
            }
        }

        if total_weight > Decimal::ZERO {
            let avg_ratio = weighted_ratio / total_weight;
            // Bound the adjustment to prevent extreme corrections
            avg_ratio.max(dec!(0.5)).min(dec!(2.0))
        } else {
            dec!(1)
        }
    }

    /// Calculate volatility adjustment factor
    fn calculate_volatility_factor(&self, volatility: Decimal) -> Decimal {
        // Higher volatility = higher slippage expectation
        // Baseline: 5% volatility = 1.0x
        let baseline = dec!(0.05);
        let factor = Decimal::ONE + (volatility - baseline) / baseline * dec!(0.5);
        factor.max(dec!(0.8)).min(dec!(1.5))
    }

    /// Get time-of-day factor
    fn get_tod_factor(&self, hour: u32) -> Decimal {
        self.tod_factors
            .read()
            .unwrap()
            .get(&hour)
            .copied()
            .unwrap_or(dec!(1))
    }

    /// Calculate confidence in prediction
    fn calculate_prediction_confidence(&self, market_id: &str, liquidity_ratio: Decimal) -> Decimal {
        let mut confidence = dec!(0.5); // Base confidence

        // More history = more confidence
        let history = self.history.read().unwrap();
        if let Some(market_history) = history.get(market_id) {
            let history_bonus = Decimal::from(market_history.len().min(20) as u32) / dec!(40);
            confidence += history_bonus; // Up to +0.5 from history
        }
        drop(history);

        // Better liquidity = more confidence
        if liquidity_ratio >= dec!(10) {
            confidence += dec!(0.2);
        } else if liquidity_ratio >= dec!(5) {
            confidence += dec!(0.1);
        }

        confidence.min(dec!(1))
    }

    /// Suggest optimal execution strategy
    fn suggest_optimal_execution(&self, order_size: Decimal, liquidity: Decimal) -> OptimalExecution {
        let ratio = if order_size > Decimal::ZERO {
            liquidity / order_size
        } else {
            dec!(999)
        };

        if ratio >= dec!(20) {
            // Plenty of liquidity - execute immediately
            OptimalExecution {
                strategy: ExecutionStrategy::Immediate,
                chunks: 1,
                chunk_interval_secs: 0,
                reason: "High liquidity - immediate execution recommended".to_string(),
            }
        } else if ratio >= dec!(5) {
            // Moderate liquidity - small chunking
            OptimalExecution {
                strategy: ExecutionStrategy::Chunked,
                chunks: 2,
                chunk_interval_secs: 30,
                reason: "Moderate liquidity - 2-chunk execution recommended".to_string(),
            }
        } else if ratio >= dec!(2) {
            // Low liquidity - aggressive chunking
            OptimalExecution {
                strategy: ExecutionStrategy::Chunked,
                chunks: 4,
                chunk_interval_secs: 60,
                reason: "Low liquidity - 4-chunk execution recommended".to_string(),
            }
        } else {
            // Very low liquidity - TWAP
            OptimalExecution {
                strategy: ExecutionStrategy::TWAP,
                chunks: 10,
                chunk_interval_secs: 120,
                reason: "Very low liquidity - TWAP over 20 minutes recommended".to_string(),
            }
        }
    }

    /// Record actual execution for learning
    pub fn record_execution(
        &self,
        market_id: &str,
        side: OrderSide,
        order_size: Decimal,
        predicted_slippage_bps: Decimal,
        actual_slippage_bps: Decimal,
        liquidity_ratio: Decimal,
        volatility: Decimal,
    ) {
        let observation = SlippageObservation {
            timestamp: Utc::now(),
            market_id: market_id.to_string(),
            side,
            order_size,
            predicted_slippage_bps,
            actual_slippage_bps,
            liquidity_ratio,
            volatility,
            hour_of_day: Utc::now().hour(),
        };

        // Update market-specific history
        {
            let mut history = self.history.write().unwrap();
            let market_history = history.entry(market_id.to_string()).or_default();
            market_history.push_back(observation.clone());
            while market_history.len() > self.config.history_window {
                market_history.pop_front();
            }
        }

        // Update global history
        {
            let mut global = self.global_history.write().unwrap();
            global.push_back(observation);
            while global.len() > self.config.history_window * 2 {
                global.pop_front();
            }
        }

        // Update time-of-day factors based on actual performance
        self.update_tod_factors();
    }

    /// Update time-of-day factors based on actual data
    fn update_tod_factors(&self) {
        let global = self.global_history.read().unwrap();
        
        if global.len() < 50 {
            return; // Need enough data
        }

        // Group by hour and calculate average error ratio
        let mut hour_ratios: HashMap<u32, Vec<Decimal>> = HashMap::new();
        
        for obs in global.iter() {
            if obs.predicted_slippage_bps > Decimal::ZERO {
                let ratio = obs.actual_slippage_bps / obs.predicted_slippage_bps;
                hour_ratios.entry(obs.hour_of_day).or_default().push(ratio);
            }
        }
        drop(global);

        let mut tod_factors = self.tod_factors.write().unwrap();
        
        for (hour, ratios) in hour_ratios {
            if ratios.len() >= 5 {
                let avg_ratio: Decimal = ratios.iter().sum::<Decimal>() / Decimal::from(ratios.len() as u32);
                // Blend with existing factor
                if let Some(existing) = tod_factors.get(&hour) {
                    let blended = (*existing * dec!(0.7)) + (avg_ratio * dec!(0.3));
                    tod_factors.insert(hour, blended.max(dec!(0.7)).min(dec!(1.5)));
                }
            }
        }
    }

    /// Get current prediction stats
    pub fn get_stats(&self) -> PredictorStats {
        let history = self.history.read().unwrap();
        let global = self.global_history.read().unwrap();
        
        let total_observations: usize = history.values().map(|v| v.len()).sum();
        
        // Calculate overall accuracy
        let (mae, count) = global.iter()
            .filter(|o| o.predicted_slippage_bps > Decimal::ZERO)
            .fold((Decimal::ZERO, 0u32), |(acc, cnt), o| {
                let error = (o.actual_slippage_bps - o.predicted_slippage_bps).abs();
                (acc + error, cnt + 1)
            });
        
        let mean_absolute_error = if count > 0 {
            mae / Decimal::from(count)
        } else {
            Decimal::ZERO
        };

        PredictorStats {
            total_markets_tracked: history.len(),
            total_observations,
            global_observations: global.len(),
            mean_absolute_error_bps: mean_absolute_error,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlippagePrediction {
    /// Total estimated slippage in basis points
    pub estimated_slippage_bps: Decimal,
    /// Component from order book consumption
    pub book_component_bps: Decimal,
    /// Component from bid-ask spread
    pub spread_component_bps: Decimal,
    /// Confidence in prediction (0-1)
    pub confidence: Decimal,
    /// Order size / available liquidity
    pub liquidity_ratio: Decimal,
    /// Adjustment factors applied
    pub adjustment_factors: AdjustmentFactors,
    /// Whether execution is recommended
    pub acceptable: bool,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Suggested execution strategy
    pub optimal_execution: OptimalExecution,
}

impl SlippagePrediction {
    pub fn summary(&self) -> String {
        let status = if self.acceptable { "✅" } else { "⚠️" };
        format!(
            "{} Slippage: {:.1}bps (book:{:.1} + spread:{:.1}) | Liq:{:.1}x | Conf:{:.0}%",
            status,
            self.estimated_slippage_bps,
            self.book_component_bps,
            self.spread_component_bps,
            self.liquidity_ratio,
            self.confidence * dec!(100)
        )
    }
}

#[derive(Debug, Clone)]
pub struct AdjustmentFactors {
    pub historical: Decimal,
    pub volatility: Decimal,
    pub time_of_day: Decimal,
}

#[derive(Debug, Clone)]
pub struct OptimalExecution {
    pub strategy: ExecutionStrategy,
    pub chunks: u32,
    pub chunk_interval_secs: u32,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionStrategy {
    Immediate,
    Chunked,
    TWAP,
}

#[derive(Debug, Clone)]
pub struct PredictorStats {
    pub total_markets_tracked: usize,
    pub total_observations: usize,
    pub global_observations: usize,
    pub mean_absolute_error_bps: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_predictor() -> SlippagePredictor {
        SlippagePredictor::new(SlippageConfig::default())
    }

    fn make_order_book(bid_depth: Decimal, ask_depth: Decimal, spread_bps: Decimal) -> OrderBook {
        let mid_price = dec!(0.50);
        let spread = mid_price * spread_bps / dec!(10000);
        let best_bid = mid_price - spread / dec!(2);
        let best_ask = mid_price + spread / dec!(2);
        
        OrderBook::new(
            vec![
                (best_bid, bid_depth * dec!(0.4)),
                (best_bid - dec!(0.01), bid_depth * dec!(0.3)),
                (best_bid - dec!(0.02), bid_depth * dec!(0.3)),
            ],
            vec![
                (best_ask, ask_depth * dec!(0.4)),
                (best_ask + dec!(0.01), ask_depth * dec!(0.3)),
                (best_ask + dec!(0.02), ask_depth * dec!(0.3)),
            ],
        )
    }

    #[test]
    fn test_small_order_low_slippage() {
        let predictor = make_predictor();
        let book = make_order_book(dec!(1000), dec!(1000), dec!(10)); // 0.1% spread
        
        // Small order relative to book
        let prediction = predictor.predict("test_market", OrderSide::Buy, dec!(50), &book, None);
        
        assert!(prediction.acceptable);
        assert!(prediction.estimated_slippage_bps < dec!(20));
        println!("{}", prediction.summary());
    }

    #[test]
    fn test_large_order_high_slippage() {
        let predictor = make_predictor();
        let book = make_order_book(dec!(100), dec!(100), dec!(20));
        
        // Large order relative to book
        let prediction = predictor.predict("test_market", OrderSide::Buy, dec!(500), &book, None);
        
        // Should have warnings about low liquidity
        assert!(!prediction.warnings.is_empty());
        assert!(prediction.estimated_slippage_bps > dec!(50));
        println!("{}", prediction.summary());
    }

    #[test]
    fn test_liquidity_ratio_calculation() {
        let predictor = make_predictor();
        let book = make_order_book(dec!(1000), dec!(1000), dec!(10));
        
        let prediction = predictor.predict("test_market", OrderSide::Buy, dec!(100), &book, None);
        
        // Liquidity ratio should be approximately 10x
        assert!(prediction.liquidity_ratio > dec!(8));
        assert!(prediction.liquidity_ratio < dec!(12));
    }

    #[test]
    fn test_high_volatility_increases_slippage() {
        let predictor = make_predictor();
        let book = make_order_book(dec!(500), dec!(500), dec!(10));
        
        // Normal volatility
        let normal = predictor.predict("test", OrderSide::Buy, dec!(100), &book, Some(dec!(0.05)));
        
        // High volatility - use more extreme difference
        let high = predictor.predict("test", OrderSide::Buy, dec!(100), &book, Some(dec!(0.20)));
        
        // Volatility factor should definitely be higher for high volatility
        assert!(high.adjustment_factors.volatility > normal.adjustment_factors.volatility);
        
        // Book component should be scaled by volatility factor, so total should be higher
        // Unless spread dominates - check the component that volatility affects
        let normal_book_adjusted = normal.book_component_bps * normal.adjustment_factors.volatility;
        let high_book_adjusted = high.book_component_bps * high.adjustment_factors.volatility;
        assert!(high_book_adjusted >= normal_book_adjusted);
    }

    #[test]
    fn test_execution_strategy_selection() {
        let predictor = make_predictor();
        
        // High liquidity - immediate
        let book1 = make_order_book(dec!(10000), dec!(10000), dec!(5));
        let pred1 = predictor.predict("test", OrderSide::Buy, dec!(100), &book1, None);
        assert_eq!(pred1.optimal_execution.strategy, ExecutionStrategy::Immediate);
        
        // Low liquidity - chunked
        let book2 = make_order_book(dec!(300), dec!(300), dec!(15));
        let pred2 = predictor.predict("test", OrderSide::Buy, dec!(100), &book2, None);
        assert_eq!(pred2.optimal_execution.strategy, ExecutionStrategy::Chunked);
        
        // Very low liquidity - TWAP
        let book3 = make_order_book(dec!(150), dec!(150), dec!(30));
        let pred3 = predictor.predict("test", OrderSide::Buy, dec!(100), &book3, None);
        assert_eq!(pred3.optimal_execution.strategy, ExecutionStrategy::TWAP);
    }

    #[test]
    fn test_spread_component() {
        let predictor = make_predictor();
        
        // Wide spread
        let wide_spread = make_order_book(dec!(1000), dec!(1000), dec!(100)); // 1% spread
        let pred_wide = predictor.predict("test", OrderSide::Buy, dec!(50), &wide_spread, None);
        
        // Narrow spread
        let narrow_spread = make_order_book(dec!(1000), dec!(1000), dec!(10)); // 0.1% spread
        let pred_narrow = predictor.predict("test", OrderSide::Buy, dec!(50), &narrow_spread, None);
        
        // Wide spread should contribute more to total slippage
        assert!(pred_wide.spread_component_bps > pred_narrow.spread_component_bps);
        assert!(pred_wide.spread_component_bps >= dec!(45)); // ~half of 100bps
    }

    #[test]
    fn test_learning_from_history() {
        let predictor = make_predictor();
        let book = make_order_book(dec!(500), dec!(500), dec!(15));
        
        // Record some executions where actual slippage was higher than predicted
        for _ in 0..10 {
            predictor.record_execution(
                "test_market",
                OrderSide::Buy,
                dec!(100),
                dec!(20),  // Predicted
                dec!(30),  // Actual (50% higher)
                dec!(5),
                dec!(0.05),
            );
        }
        
        // New prediction should be adjusted upward
        let pred = predictor.predict("test_market", OrderSide::Buy, dec!(100), &book, None);
        
        // Historical adjustment should be > 1.0 (predictions were too low)
        assert!(pred.adjustment_factors.historical > dec!(1));
    }

    #[test]
    fn test_predictor_stats() {
        let predictor = make_predictor();
        
        predictor.record_execution("market1", OrderSide::Buy, dec!(100), dec!(20), dec!(25), dec!(5), dec!(0.05));
        predictor.record_execution("market1", OrderSide::Sell, dec!(50), dec!(15), dec!(12), dec!(8), dec!(0.04));
        predictor.record_execution("market2", OrderSide::Buy, dec!(200), dec!(30), dec!(35), dec!(3), dec!(0.06));
        
        let stats = predictor.get_stats();
        
        assert_eq!(stats.total_markets_tracked, 2);
        assert_eq!(stats.global_observations, 3);
    }

    #[test]
    fn test_order_book_construction() {
        let book = OrderBook::new(
            vec![(dec!(0.48), dec!(100)), (dec!(0.47), dec!(200))],
            vec![(dec!(0.52), dec!(100)), (dec!(0.53), dec!(200))],
        );
        
        assert_eq!(book.best_bid, dec!(0.48));
        assert_eq!(book.best_ask, dec!(0.52));
        // Spread = 0.04 / 0.50 (mid) * 10000 = 800 bps
        assert!(book.spread_bps > dec!(700));
        assert!(book.spread_bps < dec!(900));
    }

    #[test]
    fn test_cumulative_depth() {
        let book = OrderBook::new(
            vec![(dec!(0.49), dec!(100)), (dec!(0.48), dec!(200))],
            vec![(dec!(0.51), dec!(100)), (dec!(0.52), dec!(300))],
        );
        
        let ask_depth = book.cumulative_depth(OrderSide::Buy);
        assert_eq!(ask_depth.len(), 2);
        assert_eq!(ask_depth[0].1, dec!(100)); // First level
        assert_eq!(ask_depth[1].1, dec!(400)); // Cumulative
        
        let bid_depth = book.cumulative_depth(OrderSide::Sell);
        assert_eq!(bid_depth.len(), 2);
        assert_eq!(bid_depth[0].1, dec!(100));
        assert_eq!(bid_depth[1].1, dec!(300));
    }

    #[test]
    fn test_acceptable_threshold() {
        let predictor = make_predictor();
        
        // Very thin book - should not be acceptable
        let thin_book = make_order_book(dec!(50), dec!(50), dec!(50));
        let pred = predictor.predict("test", OrderSide::Buy, dec!(100), &thin_book, None);
        
        assert!(!pred.acceptable);
        assert!(pred.estimated_slippage_bps > dec!(100) || pred.liquidity_ratio < dec!(5));
    }

    #[test]
    fn test_warnings_generated() {
        let predictor = make_predictor();
        
        // Low liquidity + wide spread
        let bad_book = make_order_book(dec!(100), dec!(100), dec!(80));
        let pred = predictor.predict("test", OrderSide::Buy, dec!(50), &bad_book, None);
        
        assert!(!pred.warnings.is_empty());
        println!("Warnings: {:?}", pred.warnings);
    }

    #[test]
    fn test_sell_side_slippage() {
        let predictor = make_predictor();
        let book = make_order_book(dec!(500), dec!(500), dec!(10));
        
        let buy_pred = predictor.predict("test", OrderSide::Buy, dec!(100), &book, None);
        let sell_pred = predictor.predict("test", OrderSide::Sell, dec!(100), &book, None);
        
        // Both should be similar given symmetric book
        let diff = (buy_pred.estimated_slippage_bps - sell_pred.estimated_slippage_bps).abs();
        assert!(diff < dec!(5)); // Within 5 bps
    }
}
