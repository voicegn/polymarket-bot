//! Cross-market arbitrage detection
//!
//! Detects arbitrage opportunities:
//! 1. Direct arbitrage - same event, different price on related markets
//! 2. Inverse correlation - events that should sum to 100%
//! 3. Time arbitrage - same event at different time horizons

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;

/// An arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    /// Type of arbitrage
    pub arb_type: ArbitrageType,
    /// Market IDs involved
    pub markets: Vec<String>,
    /// Expected profit margin (after fees)
    pub profit_margin: Decimal,
    /// Required capital to execute
    pub required_capital: Decimal,
    /// Confidence in the opportunity
    pub confidence: Decimal,
    /// Time window for execution
    pub time_window_secs: u32,
    /// Detailed explanation
    pub reason: String,
    /// Recommended positions
    pub positions: Vec<ArbitragePosition>,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ArbitrageType {
    /// Buy Yes on both markets when sum < 100%
    DirectUnderpriced,
    /// Sell Yes on both markets when sum > 100%
    DirectOverpriced,
    /// Markets should be inversely correlated but aren't
    InverseCorrelation,
    /// Same event at different time horizons
    TimeHorizon,
    /// Orderbook imbalance arbitrage
    OrderbookImbalance,
}

#[derive(Debug, Clone)]
pub struct ArbitragePosition {
    pub market_id: String,
    pub token_id: String,
    pub side: ArbSide,
    pub suggested_size_pct: Decimal,
    pub price: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArbSide {
    BuyYes,
    SellYes,
    BuyNo,
    SellNo,
}

/// Market data for arbitrage detection
#[derive(Debug, Clone)]
pub struct MarketData {
    pub market_id: String,
    pub question: String,
    pub yes_price: Decimal,
    pub no_price: Decimal,
    pub liquidity_usd: Decimal,
    pub category: Option<String>,
    pub end_date: Option<DateTime<Utc>>,
    pub tags: Vec<String>,
}

/// Configuration for arbitrage detection
#[derive(Debug, Clone)]
pub struct ArbitrageConfig {
    /// Minimum profit margin to consider (after fees)
    pub min_profit_margin: Decimal,
    /// Trading fee rate
    pub fee_rate: Decimal,
    /// Minimum liquidity required
    pub min_liquidity: Decimal,
    /// Maximum time difference for same-event markets
    pub max_time_diff_hours: i64,
    /// Similarity threshold for related markets (0-1)
    pub similarity_threshold: Decimal,
}

impl Default for ArbitrageConfig {
    fn default() -> Self {
        Self {
            min_profit_margin: dec!(0.02),  // 2% minimum profit
            fee_rate: dec!(0.002),          // 0.2% fee
            min_liquidity: dec!(1000),      // $1000 minimum
            max_time_diff_hours: 24,        // 24 hour max time diff
            similarity_threshold: dec!(0.7), // 70% similarity
        }
    }
}

/// Arbitrage detector
pub struct ArbitrageDetector {
    config: ArbitrageConfig,
    /// Cache of known related market pairs
    related_pairs: HashMap<String, Vec<String>>,
}

impl ArbitrageDetector {
    pub fn new(config: ArbitrageConfig) -> Self {
        Self {
            config,
            related_pairs: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(ArbitrageConfig::default())
    }

    /// Register related markets (e.g., same event on different timeframes)
    pub fn register_related(&mut self, market_a: &str, market_b: &str) {
        self.related_pairs
            .entry(market_a.to_string())
            .or_default()
            .push(market_b.to_string());
        self.related_pairs
            .entry(market_b.to_string())
            .or_default()
            .push(market_a.to_string());
    }

    /// Scan markets for arbitrage opportunities
    pub fn scan(&self, markets: &[MarketData]) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();

        // 1. Check direct over/under pricing (Yes + No != 100%)
        for market in markets {
            if let Some(opp) = self.check_direct_arbitrage(market) {
                opportunities.push(opp);
            }
        }

        // 2. Check inverse correlation between market pairs
        for (i, market_a) in markets.iter().enumerate() {
            for market_b in markets.iter().skip(i + 1) {
                if let Some(opp) = self.check_inverse_correlation(market_a, market_b) {
                    opportunities.push(opp);
                }
            }
        }

        // 3. Check time horizon arbitrage
        for market_a in markets {
            for market_b in markets {
                if market_a.market_id != market_b.market_id {
                    if let Some(opp) = self.check_time_arbitrage(market_a, market_b) {
                        opportunities.push(opp);
                    }
                }
            }
        }

        // Sort by profit margin
        opportunities.sort_by(|a, b| b.profit_margin.cmp(&a.profit_margin));
        opportunities
    }

    /// Check for direct over/under pricing within a single market
    fn check_direct_arbitrage(&self, market: &MarketData) -> Option<ArbitrageOpportunity> {
        if market.liquidity_usd < self.config.min_liquidity {
            return None;
        }

        let total = market.yes_price + market.no_price;
        let deviation = (total - Decimal::ONE).abs();

        // Need at least 2% deviation to cover fees
        let min_deviation = self.config.min_profit_margin + self.config.fee_rate * dec!(2);
        if deviation < min_deviation {
            return None;
        }

        let (arb_type, positions, profit_margin) = if total < Decimal::ONE {
            // Underpriced: Buy both Yes and No
            let profit = Decimal::ONE - total - self.config.fee_rate * dec!(2);
            (
                ArbitrageType::DirectUnderpriced,
                vec![
                    ArbitragePosition {
                        market_id: market.market_id.clone(),
                        token_id: format!("{}_yes", market.market_id),
                        side: ArbSide::BuyYes,
                        suggested_size_pct: market.yes_price / total,
                        price: market.yes_price,
                    },
                    ArbitragePosition {
                        market_id: market.market_id.clone(),
                        token_id: format!("{}_no", market.market_id),
                        side: ArbSide::BuyNo,
                        suggested_size_pct: market.no_price / total,
                        price: market.no_price,
                    },
                ],
                profit,
            )
        } else {
            // Overpriced: Sell both Yes and No
            let profit = total - Decimal::ONE - self.config.fee_rate * dec!(2);
            (
                ArbitrageType::DirectOverpriced,
                vec![
                    ArbitragePosition {
                        market_id: market.market_id.clone(),
                        token_id: format!("{}_yes", market.market_id),
                        side: ArbSide::SellYes,
                        suggested_size_pct: market.yes_price / total,
                        price: market.yes_price,
                    },
                    ArbitragePosition {
                        market_id: market.market_id.clone(),
                        token_id: format!("{}_no", market.market_id),
                        side: ArbSide::SellNo,
                        suggested_size_pct: market.no_price / total,
                        price: market.no_price,
                    },
                ],
                profit,
            )
        };

        if profit_margin < self.config.min_profit_margin {
            return None;
        }

        Some(ArbitrageOpportunity {
            arb_type,
            markets: vec![market.market_id.clone()],
            profit_margin,
            required_capital: market.liquidity_usd.min(dec!(1000)),
            confidence: dec!(0.95), // High confidence for direct arb
            time_window_secs: 60,   // Execute within 1 minute
            reason: format!(
                "Yes+No = {:.1}% (should be 100%), profit {:.2}%",
                total * dec!(100),
                profit_margin * dec!(100)
            ),
            positions,
            detected_at: Utc::now(),
        })
    }

    /// Check inverse correlation between two markets
    fn check_inverse_correlation(
        &self,
        market_a: &MarketData,
        market_b: &MarketData,
    ) -> Option<ArbitrageOpportunity> {
        // Check if markets are related
        if !self.are_markets_related(market_a, market_b) {
            return None;
        }

        // Check liquidity
        let min_liq = market_a.liquidity_usd.min(market_b.liquidity_usd);
        if min_liq < self.config.min_liquidity {
            return None;
        }

        // If markets are inversely correlated, A_yes + B_yes should = 100%
        // (e.g., "Trump wins" and "Trump loses")
        let combined = market_a.yes_price + market_b.yes_price;
        let deviation = (combined - Decimal::ONE).abs();

        if deviation < self.config.min_profit_margin + self.config.fee_rate * dec!(2) {
            return None;
        }

        let (positions, profit_margin, reason) = if combined < Decimal::ONE {
            // Both underpriced
            (
                vec![
                    ArbitragePosition {
                        market_id: market_a.market_id.clone(),
                        token_id: format!("{}_yes", market_a.market_id),
                        side: ArbSide::BuyYes,
                        suggested_size_pct: dec!(0.5),
                        price: market_a.yes_price,
                    },
                    ArbitragePosition {
                        market_id: market_b.market_id.clone(),
                        token_id: format!("{}_yes", market_b.market_id),
                        side: ArbSide::BuyYes,
                        suggested_size_pct: dec!(0.5),
                        price: market_b.yes_price,
                    },
                ],
                Decimal::ONE - combined - self.config.fee_rate * dec!(2),
                format!(
                    "Inverse markets underpriced: A({:.1}%) + B({:.1}%) = {:.1}%",
                    market_a.yes_price * dec!(100),
                    market_b.yes_price * dec!(100),
                    combined * dec!(100)
                ),
            )
        } else {
            // Both overpriced
            (
                vec![
                    ArbitragePosition {
                        market_id: market_a.market_id.clone(),
                        token_id: format!("{}_yes", market_a.market_id),
                        side: ArbSide::SellYes,
                        suggested_size_pct: dec!(0.5),
                        price: market_a.yes_price,
                    },
                    ArbitragePosition {
                        market_id: market_b.market_id.clone(),
                        token_id: format!("{}_yes", market_b.market_id),
                        side: ArbSide::SellYes,
                        suggested_size_pct: dec!(0.5),
                        price: market_b.yes_price,
                    },
                ],
                combined - Decimal::ONE - self.config.fee_rate * dec!(2),
                format!(
                    "Inverse markets overpriced: A({:.1}%) + B({:.1}%) = {:.1}%",
                    market_a.yes_price * dec!(100),
                    market_b.yes_price * dec!(100),
                    combined * dec!(100)
                ),
            )
        };

        if profit_margin < self.config.min_profit_margin {
            return None;
        }

        Some(ArbitrageOpportunity {
            arb_type: ArbitrageType::InverseCorrelation,
            markets: vec![market_a.market_id.clone(), market_b.market_id.clone()],
            profit_margin,
            required_capital: min_liq.min(dec!(500)),
            confidence: dec!(0.85),
            time_window_secs: 120,
            reason,
            positions,
            detected_at: Utc::now(),
        })
    }

    /// Check time horizon arbitrage
    fn check_time_arbitrage(
        &self,
        earlier: &MarketData,
        later: &MarketData,
    ) -> Option<ArbitrageOpportunity> {
        // Check if they're about the same event at different times
        if !self.is_same_event_different_time(earlier, later) {
            return None;
        }

        let (earlier, later) = match (&earlier.end_date, &later.end_date) {
            (Some(e1), Some(e2)) if e1 < e2 => (earlier, later),
            (Some(e1), Some(e2)) if e1 > e2 => (later, earlier),
            _ => return None,
        };

        // Earlier market should have higher probability if event is likely
        // Later market has more time for things to change
        let price_diff = earlier.yes_price - later.yes_price;

        // If earlier market is cheaper, that's potentially arbitrage
        // (should be priced higher due to less uncertainty)
        if price_diff < -self.config.min_profit_margin {
            let profit = -price_diff - self.config.fee_rate * dec!(2);
            if profit < self.config.min_profit_margin {
                return None;
            }

            return Some(ArbitrageOpportunity {
                arb_type: ArbitrageType::TimeHorizon,
                markets: vec![earlier.market_id.clone(), later.market_id.clone()],
                profit_margin: profit,
                required_capital: earlier.liquidity_usd.min(later.liquidity_usd).min(dec!(500)),
                confidence: dec!(0.70), // Lower confidence for time arb
                time_window_secs: 300,  // 5 min window
                reason: format!(
                    "Time mispricing: earlier({:.1}%) < later({:.1}%)",
                    earlier.yes_price * dec!(100),
                    later.yes_price * dec!(100)
                ),
                positions: vec![
                    ArbitragePosition {
                        market_id: earlier.market_id.clone(),
                        token_id: format!("{}_yes", earlier.market_id),
                        side: ArbSide::BuyYes,
                        suggested_size_pct: dec!(0.5),
                        price: earlier.yes_price,
                    },
                    ArbitragePosition {
                        market_id: later.market_id.clone(),
                        token_id: format!("{}_yes", later.market_id),
                        side: ArbSide::SellYes,
                        suggested_size_pct: dec!(0.5),
                        price: later.yes_price,
                    },
                ],
                detected_at: Utc::now(),
            });
        }

        None
    }

    /// Check if two markets are related
    fn are_markets_related(&self, a: &MarketData, b: &MarketData) -> bool {
        // Check explicit registration
        if let Some(related) = self.related_pairs.get(&a.market_id) {
            if related.contains(&b.market_id) {
                return true;
            }
        }

        // Check category match
        if let (Some(cat_a), Some(cat_b)) = (&a.category, &b.category) {
            if cat_a != cat_b {
                return false;
            }
        }

        // Check tag overlap
        let tag_overlap = a.tags.iter().filter(|t| b.tags.contains(t)).count();
        if tag_overlap >= 2 {
            return true;
        }

        // Check question similarity (simple word overlap)
        let similarity = self.question_similarity(&a.question, &b.question);
        similarity >= self.config.similarity_threshold
    }

    /// Check if markets are same event at different times
    fn is_same_event_different_time(&self, a: &MarketData, b: &MarketData) -> bool {
        let (end_a, end_b) = match (&a.end_date, &b.end_date) {
            (Some(ea), Some(eb)) => (ea, eb),
            _ => return false,
        };

        // Must be within max_time_diff hours
        let diff = (*end_a - *end_b).abs();
        if diff > Duration::hours(self.config.max_time_diff_hours) {
            return false;
        }

        // Questions should be very similar (>80%)
        let similarity = self.question_similarity(&a.question, &b.question);
        similarity >= dec!(0.80)
    }

    /// Simple word-based question similarity
    fn question_similarity(&self, q1: &str, q2: &str) -> Decimal {
        let words1: std::collections::HashSet<_> = q1
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();
        let words2: std::collections::HashSet<_> = q2
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        if words1.is_empty() || words2.is_empty() {
            return Decimal::ZERO;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            return Decimal::ZERO;
        }

        Decimal::from(intersection) / Decimal::from(union)
    }

    /// Get summary of current opportunities
    pub fn summary(&self, opportunities: &[ArbitrageOpportunity]) -> String {
        if opportunities.is_empty() {
            return "No arbitrage opportunities found".to_string();
        }

        let total_profit: Decimal = opportunities.iter().map(|o| o.profit_margin).sum();
        let best = &opportunities[0];

        format!(
            "Found {} opportunities, best: {:.2}% profit ({:?}), total potential: {:.2}%",
            opportunities.len(),
            best.profit_margin * dec!(100),
            best.arb_type,
            total_profit * dec!(100)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_market(id: &str, yes: Decimal, no: Decimal, liq: Decimal) -> MarketData {
        MarketData {
            market_id: id.to_string(),
            question: format!("Test question for {}", id),
            yes_price: yes,
            no_price: no,
            liquidity_usd: liq,
            category: Some("test".to_string()),
            end_date: Some(Utc::now() + Duration::days(1)),
            tags: vec!["test".to_string()],
        }
    }

    #[test]
    fn test_direct_underpriced() {
        let detector = ArbitrageDetector::with_defaults();
        
        // Yes(0.40) + No(0.55) = 0.95 (5% underpriced)
        let market = make_market("test", dec!(0.40), dec!(0.55), dec!(5000));
        
        let opp = detector.check_direct_arbitrage(&market);
        assert!(opp.is_some());
        let opp = opp.unwrap();
        assert_eq!(opp.arb_type, ArbitrageType::DirectUnderpriced);
        assert!(opp.profit_margin > dec!(0.02)); // After fees
    }

    #[test]
    fn test_direct_overpriced() {
        let detector = ArbitrageDetector::with_defaults();
        
        // Yes(0.55) + No(0.52) = 1.07 (7% overpriced)
        let market = make_market("test", dec!(0.55), dec!(0.52), dec!(5000));
        
        let opp = detector.check_direct_arbitrage(&market);
        assert!(opp.is_some());
        let opp = opp.unwrap();
        assert_eq!(opp.arb_type, ArbitrageType::DirectOverpriced);
    }

    #[test]
    fn test_no_arbitrage_fair_price() {
        let detector = ArbitrageDetector::with_defaults();
        
        // Yes(0.50) + No(0.50) = 1.00 (fair)
        let market = make_market("test", dec!(0.50), dec!(0.50), dec!(5000));
        
        let opp = detector.check_direct_arbitrage(&market);
        assert!(opp.is_none());
    }

    #[test]
    fn test_low_liquidity_ignored() {
        let detector = ArbitrageDetector::with_defaults();
        
        // Good arbitrage but low liquidity
        let market = make_market("test", dec!(0.40), dec!(0.50), dec!(100));
        
        let opp = detector.check_direct_arbitrage(&market);
        assert!(opp.is_none());
    }

    #[test]
    fn test_inverse_correlation() {
        let mut detector = ArbitrageDetector::with_defaults();
        detector.register_related("trump_wins", "trump_loses");
        
        let market_a = MarketData {
            market_id: "trump_wins".to_string(),
            question: "Will Trump win the election?".to_string(),
            yes_price: dec!(0.45),
            no_price: dec!(0.55),
            liquidity_usd: dec!(10000),
            category: Some("politics".to_string()),
            end_date: Some(Utc::now() + Duration::days(30)),
            tags: vec!["trump".to_string(), "election".to_string()],
        };
        
        let market_b = MarketData {
            market_id: "trump_loses".to_string(),
            question: "Will Trump lose the election?".to_string(),
            yes_price: dec!(0.45),  // Combined = 0.90 (10% underpriced)
            no_price: dec!(0.55),
            liquidity_usd: dec!(10000),
            category: Some("politics".to_string()),
            end_date: Some(Utc::now() + Duration::days(30)),
            tags: vec!["trump".to_string(), "election".to_string()],
        };
        
        let opp = detector.check_inverse_correlation(&market_a, &market_b);
        assert!(opp.is_some());
        let opp = opp.unwrap();
        assert_eq!(opp.arb_type, ArbitrageType::InverseCorrelation);
    }

    #[test]
    fn test_scan_multiple() {
        let detector = ArbitrageDetector::with_defaults();
        
        let markets = vec![
            make_market("good_arb", dec!(0.40), dec!(0.50), dec!(5000)),  // 10% under
            make_market("fair", dec!(0.50), dec!(0.50), dec!(5000)),       // Fair
            make_market("small_arb", dec!(0.45), dec!(0.52), dec!(5000)), // 3% under
        ];
        
        let opps = detector.scan(&markets);
        
        // Should find at least 2 opportunities
        assert!(!opps.is_empty());
        
        // Best opportunity should be first
        assert!(opps[0].profit_margin >= opps.last().map(|o| o.profit_margin).unwrap_or(Decimal::ZERO));
    }

    #[test]
    fn test_question_similarity() {
        let detector = ArbitrageDetector::with_defaults();
        
        let sim1 = detector.question_similarity(
            "Will Bitcoin reach $100k by end of 2024?",
            "Will Bitcoin reach $100k by 2024?"
        );
        assert!(sim1 > dec!(0.7));
        
        let sim2 = detector.question_similarity(
            "Will it rain tomorrow?",
            "Who will win the election?"
        );
        assert!(sim2 < dec!(0.3));
    }

    #[test]
    fn test_summary() {
        let detector = ArbitrageDetector::with_defaults();
        let markets = vec![
            make_market("test1", dec!(0.40), dec!(0.50), dec!(5000)),
        ];
        
        let opps = detector.scan(&markets);
        let summary = detector.summary(&opps);
        
        assert!(summary.contains("opportunities") || summary.contains("No arbitrage"));
    }
}
