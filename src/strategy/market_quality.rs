//! Market quality scoring for trade filtering
//!
//! Evaluates markets based on:
//! 1. Liquidity depth - higher is better for execution
//! 2. Spread - tighter spreads mean less slippage  
//! 3. Market maturity - older markets have more stable prices
//! 4. Volume activity - recent volume indicates active market
//! 5. Price stability - volatile prices increase risk

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Comprehensive market quality assessment
#[derive(Debug, Clone)]
pub struct MarketQuality {
    /// Overall quality score (0.0 - 1.0)
    pub score: Decimal,
    /// Liquidity score component
    pub liquidity_score: Decimal,
    /// Spread score component (tighter = higher)
    pub spread_score: Decimal,
    /// Market maturity score
    pub maturity_score: Decimal,
    /// Volume activity score
    pub volume_score: Decimal,
    /// Price stability score
    pub stability_score: Decimal,
    /// Human-readable assessment
    pub assessment: QualityAssessment,
    /// Reasons for the score
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityAssessment {
    /// Excellent - ideal for trading
    Excellent,
    /// Good - safe to trade
    Good,
    /// Marginal - trade with caution
    Marginal,
    /// Poor - avoid trading
    Poor,
    /// Untradeable - do not trade
    Untradeable,
}

impl QualityAssessment {
    pub fn from_score(score: Decimal) -> Self {
        if score >= dec!(0.80) {
            Self::Excellent
        } else if score >= dec!(0.65) {
            Self::Good
        } else if score >= dec!(0.50) {
            Self::Marginal
        } else if score >= dec!(0.30) {
            Self::Poor
        } else {
            Self::Untradeable
        }
    }

    /// Minimum edge required for this quality level
    pub fn min_edge_multiplier(&self) -> Decimal {
        match self {
            Self::Excellent => dec!(1.0),   // Normal edge requirement
            Self::Good => dec!(1.2),        // 20% higher edge needed
            Self::Marginal => dec!(1.5),    // 50% higher edge needed
            Self::Poor => dec!(2.0),        // Double edge needed
            Self::Untradeable => dec!(999), // Effectively blocks trading
        }
    }

    /// Position size multiplier for this quality level
    pub fn position_size_multiplier(&self) -> Decimal {
        match self {
            Self::Excellent => dec!(1.0),
            Self::Good => dec!(0.8),
            Self::Marginal => dec!(0.5),
            Self::Poor => dec!(0.25),
            Self::Untradeable => dec!(0),
        }
    }
}

/// Configuration for quality scoring
#[derive(Debug, Clone)]
pub struct QualityScorerConfig {
    /// Minimum liquidity in USD for full score
    pub min_liquidity_for_full_score: Decimal,
    /// Liquidity below this is untradeable
    pub min_liquidity_threshold: Decimal,
    /// Maximum acceptable spread (%)
    pub max_acceptable_spread: Decimal,
    /// Ideal spread for full score
    pub ideal_spread: Decimal,
    /// Minimum market age in hours
    pub min_market_age_hours: i64,
    /// Ideal market age in hours for full score
    pub ideal_market_age_hours: i64,
    /// Minimum 24h volume for full score
    pub min_volume_24h_for_full_score: Decimal,
    /// Price change threshold for stability penalty
    pub max_price_volatility_1h: Decimal,
    /// Weight for each component
    pub weights: QualityWeights,
}

#[derive(Debug, Clone)]
pub struct QualityWeights {
    pub liquidity: Decimal,
    pub spread: Decimal,
    pub maturity: Decimal,
    pub volume: Decimal,
    pub stability: Decimal,
}

impl Default for QualityScorerConfig {
    fn default() -> Self {
        Self {
            min_liquidity_for_full_score: dec!(50000),  // $50k for full score
            min_liquidity_threshold: dec!(5000),        // $5k minimum
            max_acceptable_spread: dec!(0.05),          // 5% max spread
            ideal_spread: dec!(0.01),                   // 1% ideal
            min_market_age_hours: 2,                    // 2 hour minimum
            ideal_market_age_hours: 24,                 // 24h for full score
            min_volume_24h_for_full_score: dec!(10000), // $10k volume
            max_price_volatility_1h: dec!(0.10),        // 10% 1h volatility max
            weights: QualityWeights::default(),
        }
    }
}

impl Default for QualityWeights {
    fn default() -> Self {
        Self {
            liquidity: dec!(0.30),  // 30% weight - most important
            spread: dec!(0.25),     // 25% weight - execution quality
            maturity: dec!(0.15),   // 15% weight
            volume: dec!(0.15),     // 15% weight
            stability: dec!(0.15),  // 15% weight
        }
    }
}

/// Market data needed for quality scoring
#[derive(Debug, Clone)]
pub struct MarketMetrics {
    /// Total liquidity in USD
    pub liquidity_usd: Decimal,
    /// Bid-ask spread as decimal (0.02 = 2%)
    pub spread: Decimal,
    /// Market creation time
    pub created_at: DateTime<Utc>,
    /// 24-hour trading volume in USD
    pub volume_24h: Decimal,
    /// Price 1 hour ago (for volatility)
    pub price_1h_ago: Option<Decimal>,
    /// Current price
    pub current_price: Decimal,
    /// Number of unique traders (if available)
    pub unique_traders: Option<u32>,
}

/// Scores market quality for trade decisions
pub struct MarketQualityScorer {
    config: QualityScorerConfig,
}

impl MarketQualityScorer {
    pub fn new(config: QualityScorerConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(QualityScorerConfig::default())
    }

    /// Score a market's quality
    pub fn score(&self, metrics: &MarketMetrics) -> MarketQuality {
        let mut reasons = Vec::new();

        // 1. Liquidity score
        let liquidity_score = self.score_liquidity(metrics.liquidity_usd, &mut reasons);

        // 2. Spread score
        let spread_score = self.score_spread(metrics.spread, &mut reasons);

        // 3. Maturity score
        let maturity_score = self.score_maturity(metrics.created_at, &mut reasons);

        // 4. Volume score
        let volume_score = self.score_volume(metrics.volume_24h, &mut reasons);

        // 5. Stability score
        let stability_score = self.score_stability(
            metrics.current_price,
            metrics.price_1h_ago,
            &mut reasons,
        );

        // Calculate weighted average
        let w = &self.config.weights;
        let weighted_sum = liquidity_score * w.liquidity
            + spread_score * w.spread
            + maturity_score * w.maturity
            + volume_score * w.volume
            + stability_score * w.stability;

        let total_weight = w.liquidity + w.spread + w.maturity + w.volume + w.stability;
        let score = weighted_sum / total_weight;

        // Apply hard cutoffs
        let final_score = if metrics.liquidity_usd < self.config.min_liquidity_threshold {
            dec!(0) // Untradeable if below minimum liquidity
        } else if metrics.spread > self.config.max_acceptable_spread {
            score.min(dec!(0.40)) // Cap at Poor if spread too wide
        } else {
            score
        };

        let assessment = QualityAssessment::from_score(final_score);

        MarketQuality {
            score: final_score,
            liquidity_score,
            spread_score,
            maturity_score,
            volume_score,
            stability_score,
            assessment,
            reasons,
        }
    }

    fn score_liquidity(&self, liquidity: Decimal, reasons: &mut Vec<String>) -> Decimal {
        if liquidity < self.config.min_liquidity_threshold {
            reasons.push(format!(
                "Liquidity too low: ${:.0} < ${:.0} minimum",
                liquidity, self.config.min_liquidity_threshold
            ));
            return dec!(0);
        }

        let ratio = liquidity / self.config.min_liquidity_for_full_score;
        let score = ratio.min(dec!(1));

        if score < dec!(0.5) {
            reasons.push(format!("Low liquidity: ${:.0}", liquidity));
        } else if score >= dec!(1) {
            reasons.push(format!("Excellent liquidity: ${:.0}", liquidity));
        }

        score
    }

    fn score_spread(&self, spread: Decimal, reasons: &mut Vec<String>) -> Decimal {
        if spread <= Decimal::ZERO {
            return dec!(1); // No spread data = assume good
        }

        if spread > self.config.max_acceptable_spread {
            reasons.push(format!(
                "Spread too wide: {:.2}% > {:.2}% max",
                spread * dec!(100),
                self.config.max_acceptable_spread * dec!(100)
            ));
            return dec!(0);
        }

        if spread <= self.config.ideal_spread {
            reasons.push(format!("Tight spread: {:.2}%", spread * dec!(100)));
            return dec!(1);
        }

        // Linear interpolation between ideal and max
        let range = self.config.max_acceptable_spread - self.config.ideal_spread;
        let excess = spread - self.config.ideal_spread;
        let score = dec!(1) - (excess / range);

        if score < dec!(0.5) {
            reasons.push(format!("Wide spread: {:.2}%", spread * dec!(100)));
        }

        score.max(dec!(0))
    }

    fn score_maturity(&self, created_at: DateTime<Utc>, reasons: &mut Vec<String>) -> Decimal {
        let age = Utc::now() - created_at;
        let age_hours = age.num_hours();

        if age_hours < self.config.min_market_age_hours {
            reasons.push(format!(
                "Market too new: {}h < {}h minimum",
                age_hours, self.config.min_market_age_hours
            ));
            return dec!(0.2); // Not zero, but penalized
        }

        let ratio = Decimal::from(age_hours) / Decimal::from(self.config.ideal_market_age_hours);
        let score = ratio.min(dec!(1));

        if age_hours < 6 {
            reasons.push(format!("Young market: {}h old", age_hours));
        }

        score
    }

    fn score_volume(&self, volume_24h: Decimal, reasons: &mut Vec<String>) -> Decimal {
        if volume_24h <= Decimal::ZERO {
            reasons.push("No recent trading volume".to_string());
            return dec!(0.3); // Some penalty for no volume
        }

        let ratio = volume_24h / self.config.min_volume_24h_for_full_score;
        let score = ratio.min(dec!(1));

        if score < dec!(0.5) {
            reasons.push(format!("Low 24h volume: ${:.0}", volume_24h));
        } else if score >= dec!(1) {
            reasons.push(format!("Active trading: ${:.0} 24h volume", volume_24h));
        }

        score
    }

    fn score_stability(
        &self,
        current_price: Decimal,
        price_1h_ago: Option<Decimal>,
        reasons: &mut Vec<String>,
    ) -> Decimal {
        let Some(old_price) = price_1h_ago else {
            return dec!(0.7); // No data = neutral score
        };

        if old_price <= Decimal::ZERO {
            return dec!(0.7);
        }

        let change = ((current_price - old_price) / old_price).abs();

        if change > self.config.max_price_volatility_1h {
            reasons.push(format!(
                "High volatility: {:.1}% change in 1h",
                change * dec!(100)
            ));
            return dec!(0.2);
        }

        // Lower volatility = higher score
        let ratio = change / self.config.max_price_volatility_1h;
        let score = dec!(1) - ratio;

        if change > dec!(0.05) {
            reasons.push(format!("Moderate volatility: {:.1}% 1h", change * dec!(100)));
        }

        score.max(dec!(0))
    }

    /// Quick check if market meets minimum quality bar
    pub fn is_tradeable(&self, metrics: &MarketMetrics) -> bool {
        let quality = self.score(metrics);
        quality.assessment != QualityAssessment::Untradeable
    }

    /// Get adjusted edge requirement for this market
    pub fn adjusted_min_edge(&self, base_min_edge: Decimal, metrics: &MarketMetrics) -> Decimal {
        let quality = self.score(metrics);
        base_min_edge * quality.assessment.min_edge_multiplier()
    }

    /// Get adjusted position size for this market
    pub fn adjusted_position_size(&self, base_size: Decimal, metrics: &MarketMetrics) -> Decimal {
        let quality = self.score(metrics);
        base_size * quality.assessment.position_size_multiplier()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_metrics(
        liquidity: Decimal,
        spread: Decimal,
        age_hours: i64,
        volume: Decimal,
    ) -> MarketMetrics {
        MarketMetrics {
            liquidity_usd: liquidity,
            spread,
            created_at: Utc::now() - Duration::hours(age_hours),
            volume_24h: volume,
            price_1h_ago: Some(dec!(0.50)),
            current_price: dec!(0.50),
            unique_traders: None,
        }
    }

    #[test]
    fn test_excellent_market() {
        let scorer = MarketQualityScorer::with_defaults();
        let metrics = make_metrics(
            dec!(100000), // $100k liquidity
            dec!(0.005),  // 0.5% spread
            48,           // 2 days old
            dec!(50000),  // $50k volume
        );

        let quality = scorer.score(&metrics);
        assert_eq!(quality.assessment, QualityAssessment::Excellent);
        assert!(quality.score >= dec!(0.80));
    }

    #[test]
    fn test_good_market() {
        let scorer = MarketQualityScorer::with_defaults();
        let metrics = make_metrics(
            dec!(30000), // $30k liquidity
            dec!(0.02),  // 2% spread
            24,          // 1 day old
            dec!(8000),  // $8k volume
        );

        let quality = scorer.score(&metrics);
        assert!(matches!(
            quality.assessment,
            QualityAssessment::Good | QualityAssessment::Excellent
        ));
    }

    #[test]
    fn test_poor_liquidity() {
        let scorer = MarketQualityScorer::with_defaults();
        let metrics = make_metrics(
            dec!(3000), // Only $3k liquidity - below minimum
            dec!(0.01),
            24,
            dec!(1000),
        );

        let quality = scorer.score(&metrics);
        assert_eq!(quality.assessment, QualityAssessment::Untradeable);
        assert!(!scorer.is_tradeable(&metrics));
    }

    #[test]
    fn test_wide_spread() {
        let scorer = MarketQualityScorer::with_defaults();
        let metrics = make_metrics(
            dec!(50000),
            dec!(0.06), // 6% spread - too wide
            24,
            dec!(10000),
        );

        let quality = scorer.score(&metrics);
        // Wide spread caps score at Poor
        assert!(matches!(
            quality.assessment,
            QualityAssessment::Poor | QualityAssessment::Marginal
        ));
    }

    #[test]
    fn test_new_market() {
        let scorer = MarketQualityScorer::with_defaults();
        let metrics = make_metrics(
            dec!(50000),
            dec!(0.01),
            1, // Only 1 hour old
            dec!(1000),
        );

        let quality = scorer.score(&metrics);
        // New market gets penalty
        assert!(quality.maturity_score < dec!(0.5));
    }

    #[test]
    fn test_high_volatility() {
        let scorer = MarketQualityScorer::with_defaults();
        let mut metrics = make_metrics(dec!(50000), dec!(0.01), 24, dec!(10000));
        metrics.price_1h_ago = Some(dec!(0.40));
        metrics.current_price = dec!(0.55); // 37.5% change

        let quality = scorer.score(&metrics);
        assert!(quality.stability_score < dec!(0.5));
    }

    #[test]
    fn test_edge_multiplier() {
        let scorer = MarketQualityScorer::with_defaults();

        // Excellent market - normal edge
        let excellent = make_metrics(dec!(100000), dec!(0.005), 48, dec!(50000));
        let edge = scorer.adjusted_min_edge(dec!(0.05), &excellent);
        assert_eq!(edge, dec!(0.05));

        // Poor market - higher edge required
        let poor = make_metrics(dec!(6000), dec!(0.04), 3, dec!(500));
        let edge = scorer.adjusted_min_edge(dec!(0.05), &poor);
        assert!(edge > dec!(0.05));
    }

    #[test]
    fn test_position_size_adjustment() {
        let scorer = MarketQualityScorer::with_defaults();

        // Excellent market - full size
        let excellent = make_metrics(dec!(100000), dec!(0.005), 48, dec!(50000));
        let size = scorer.adjusted_position_size(dec!(100), &excellent);
        assert_eq!(size, dec!(100));

        // Marginal market - reduced size
        let marginal = make_metrics(dec!(10000), dec!(0.03), 6, dec!(2000));
        let quality = scorer.score(&marginal);
        let size = scorer.adjusted_position_size(dec!(100), &marginal);
        assert!(size < dec!(100));
        assert_eq!(size, dec!(100) * quality.assessment.position_size_multiplier());
    }
}
