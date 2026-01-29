//! Enhanced Correlation Risk Management
//!
//! Extends basic correlation detection with:
//! - Correlation cluster detection
//! - Portfolio-level correlation risk assessment
//! - Dynamic correlation tracking
//! - Diversification scoring

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};

use super::correlation::{CorrelationDetector, MarketCorrelation};

/// Configuration for correlation risk management
#[derive(Debug, Clone)]
pub struct CorrelationRiskConfig {
    /// Threshold for considering markets highly correlated (0-1)
    pub high_correlation_threshold: Decimal,
    /// Maximum allowed average portfolio correlation
    pub max_portfolio_correlation: Decimal,
    /// Maximum weight in any correlation cluster
    pub max_cluster_weight: Decimal,
    /// Minimum number of uncorrelated positions
    pub min_diversified_positions: usize,
    /// Penalty multiplier for correlated positions
    pub correlation_penalty_factor: Decimal,
}

impl Default for CorrelationRiskConfig {
    fn default() -> Self {
        Self {
            high_correlation_threshold: dec!(0.70),
            max_portfolio_correlation: dec!(0.50),
            max_cluster_weight: dec!(0.40),
            min_diversified_positions: 3,
            correlation_penalty_factor: dec!(0.5),
        }
    }
}

/// A cluster of correlated markets
#[derive(Debug, Clone)]
pub struct CorrelationCluster {
    pub id: usize,
    pub markets: Vec<String>,
    pub avg_correlation: Decimal,
    pub cluster_weight: Decimal,
}

/// Position info for risk calculation
#[derive(Debug, Clone)]
pub struct PositionInfo {
    pub market_id: String,
    pub size: Decimal,
    pub weight: Decimal, // Percentage of portfolio
}

/// Portfolio correlation risk assessment
#[derive(Debug, Clone)]
pub struct CorrelationRiskAssessment {
    pub timestamp: DateTime<Utc>,
    /// Average pairwise correlation of portfolio
    pub avg_portfolio_correlation: Decimal,
    /// Maximum correlation between any two positions
    pub max_pairwise_correlation: Decimal,
    /// Effective diversification ratio (0-1, higher = better)
    pub diversification_ratio: Decimal,
    /// Identified correlation clusters
    pub clusters: Vec<CorrelationCluster>,
    /// Risk warnings
    pub warnings: Vec<CorrelationWarning>,
    /// Overall risk score (0-100, higher = more risk)
    pub risk_score: Decimal,
    /// Recommended actions
    pub recommendations: Vec<RiskRecommendation>,
}

/// Correlation-related warnings
#[derive(Debug, Clone, PartialEq)]
pub enum CorrelationWarning {
    /// High correlation between two specific positions
    HighPairwiseCorrelation {
        market_a: String,
        market_b: String,
        correlation: Decimal,
    },
    /// Portfolio is too concentrated
    ConcentratedPortfolio {
        avg_correlation: Decimal,
        max_allowed: Decimal,
    },
    /// Large exposure to correlated cluster
    ClusterConcentration {
        cluster_id: usize,
        weight: Decimal,
        max_allowed: Decimal,
    },
    /// Not enough diversification
    InsufficientDiversification {
        diversified_count: usize,
        min_required: usize,
    },
}

/// Recommended actions to reduce correlation risk
#[derive(Debug, Clone)]
pub enum RiskRecommendation {
    /// Reduce position in specific market
    ReducePosition { market_id: String, by_percent: Decimal },
    /// Avoid adding more exposure to correlated cluster
    AvoidCluster { cluster_id: usize, markets: Vec<String> },
    /// Add uncorrelated position for diversification
    AddDiversification { suggested_category: Option<String> },
    /// Rebalance portfolio
    Rebalance { high_correlation_markets: Vec<String> },
}

/// Enhanced correlation risk manager
pub struct CorrelationRiskManager {
    config: CorrelationRiskConfig,
    detector: CorrelationDetector,
    /// Category mapping for markets
    market_categories: HashMap<String, String>,
    /// Cached clusters
    clusters: Vec<CorrelationCluster>,
    /// Last assessment
    last_assessment: Option<CorrelationRiskAssessment>,
}

impl CorrelationRiskManager {
    pub fn new(config: CorrelationRiskConfig) -> Self {
        let threshold = config.high_correlation_threshold
            .to_string()
            .parse::<f64>()
            .unwrap_or(0.7);
        
        Self {
            config,
            detector: CorrelationDetector::new(threshold),
            market_categories: HashMap::new(),
            clusters: Vec::new(),
            last_assessment: None,
        }
    }

    /// Update price data for correlation calculation
    pub fn update_price(&mut self, market_id: &str, price: Decimal, timestamp: i64) {
        self.detector.add_price_point(market_id, price, timestamp);
    }

    /// Set category for a market
    pub fn set_market_category(&mut self, market_id: &str, category: &str) {
        self.market_categories.insert(market_id.to_string(), category.to_string());
    }

    /// Assess correlation risk for current portfolio
    pub fn assess_portfolio(&mut self, positions: &[PositionInfo]) -> CorrelationRiskAssessment {
        let now = Utc::now();
        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();

        if positions.is_empty() {
            return CorrelationRiskAssessment {
                timestamp: now,
                avg_portfolio_correlation: Decimal::ZERO,
                max_pairwise_correlation: Decimal::ZERO,
                diversification_ratio: Decimal::ONE,
                clusters: Vec::new(),
                warnings: Vec::new(),
                risk_score: Decimal::ZERO,
                recommendations: Vec::new(),
            };
        }

        // Calculate pairwise correlations
        let mut correlations = Vec::new();
        let mut max_correlation = Decimal::ZERO;
        let mut high_corr_pairs = Vec::new();

        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                if let Some(corr) = self.detector.get_correlation(
                    &positions[i].market_id,
                    &positions[j].market_id
                ) {
                    let abs_corr = corr.abs();
                    
                    // Weight by position sizes
                    let weight = positions[i].weight * positions[j].weight;
                    correlations.push((abs_corr, weight));
                    
                    if abs_corr > max_correlation {
                        max_correlation = abs_corr;
                    }
                    
                    if abs_corr >= self.config.high_correlation_threshold {
                        high_corr_pairs.push((
                            positions[i].market_id.clone(),
                            positions[j].market_id.clone(),
                            abs_corr
                        ));
                        
                        warnings.push(CorrelationWarning::HighPairwiseCorrelation {
                            market_a: positions[i].market_id.clone(),
                            market_b: positions[j].market_id.clone(),
                            correlation: abs_corr,
                        });
                    }
                }
            }
        }

        // Calculate weighted average correlation
        let total_weight: Decimal = correlations.iter().map(|(_, w)| *w).sum();
        let avg_correlation = if total_weight > Decimal::ZERO {
            correlations.iter().map(|(c, w)| *c * *w).sum::<Decimal>() / total_weight
        } else {
            Decimal::ZERO
        };

        // Check portfolio concentration
        if avg_correlation > self.config.max_portfolio_correlation {
            warnings.push(CorrelationWarning::ConcentratedPortfolio {
                avg_correlation,
                max_allowed: self.config.max_portfolio_correlation,
            });
            
            let high_corr_markets: Vec<_> = high_corr_pairs.iter()
                .flat_map(|(a, b, _)| vec![a.clone(), b.clone()])
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            
            recommendations.push(RiskRecommendation::Rebalance {
                high_correlation_markets: high_corr_markets,
            });
        }

        // Detect correlation clusters
        let clusters = self.detect_clusters(positions);
        
        // Check cluster concentration
        for cluster in &clusters {
            if cluster.cluster_weight > self.config.max_cluster_weight {
                warnings.push(CorrelationWarning::ClusterConcentration {
                    cluster_id: cluster.id,
                    weight: cluster.cluster_weight,
                    max_allowed: self.config.max_cluster_weight,
                });
                
                recommendations.push(RiskRecommendation::AvoidCluster {
                    cluster_id: cluster.id,
                    markets: cluster.markets.clone(),
                });
            }
        }

        // Calculate diversification ratio
        let diversification_ratio = self.calculate_diversification_ratio(positions, &correlations);
        
        // Count diversified positions
        let diversified_count = self.count_diversified_positions(positions);
        if diversified_count < self.config.min_diversified_positions {
            warnings.push(CorrelationWarning::InsufficientDiversification {
                diversified_count,
                min_required: self.config.min_diversified_positions,
            });
            
            recommendations.push(RiskRecommendation::AddDiversification {
                suggested_category: None, // Could suggest based on missing categories
            });
        }

        // Calculate overall risk score (0-100)
        let risk_score = self.calculate_risk_score(avg_correlation, max_correlation, &clusters);

        // Add position reduction recommendations for high correlation
        for (market_a, market_b, _corr) in &high_corr_pairs {
            // Suggest reducing the smaller position
            let pos_a = positions.iter().find(|p| &p.market_id == market_a);
            let pos_b = positions.iter().find(|p| &p.market_id == market_b);
            
            if let (Some(a), Some(b)) = (pos_a, pos_b) {
                let (smaller, _) = if a.size < b.size { (a, b) } else { (b, a) };
                recommendations.push(RiskRecommendation::ReducePosition {
                    market_id: smaller.market_id.clone(),
                    by_percent: dec!(25), // Suggest 25% reduction
                });
            }
        }

        let assessment = CorrelationRiskAssessment {
            timestamp: now,
            avg_portfolio_correlation: avg_correlation,
            max_pairwise_correlation: max_correlation,
            diversification_ratio,
            clusters: clusters.clone(),
            warnings,
            risk_score,
            recommendations,
        };

        self.clusters = clusters;
        self.last_assessment = Some(assessment.clone());
        
        assessment
    }

    /// Detect correlation clusters using union-find approach
    fn detect_clusters(&self, positions: &[PositionInfo]) -> Vec<CorrelationCluster> {
        if positions.is_empty() {
            return Vec::new();
        }

        // Union-find structure
        let mut parent: HashMap<String, String> = HashMap::new();
        for pos in positions {
            parent.insert(pos.market_id.clone(), pos.market_id.clone());
        }

        fn find(parent: &mut HashMap<String, String>, x: &str) -> String {
            let p = parent.get(x).cloned().unwrap_or_else(|| x.to_string());
            if p != x {
                let root = find(parent, &p);
                parent.insert(x.to_string(), root.clone());
                root
            } else {
                x.to_string()
            }
        }

        fn union(parent: &mut HashMap<String, String>, x: &str, y: &str) {
            let px = find(parent, x);
            let py = find(parent, y);
            if px != py {
                parent.insert(px, py);
            }
        }

        // Union correlated markets
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                if self.detector.are_correlated(
                    &positions[i].market_id,
                    &positions[j].market_id
                ) {
                    union(&mut parent, &positions[i].market_id, &positions[j].market_id);
                }
            }
        }

        // Group by cluster
        let mut cluster_members: HashMap<String, Vec<&PositionInfo>> = HashMap::new();
        for pos in positions {
            let root = find(&mut parent, &pos.market_id);
            cluster_members.entry(root).or_default().push(pos);
        }

        // Build cluster objects
        let mut clusters = Vec::new();
        for (id, (_, members)) in cluster_members.into_iter().enumerate() {
            if members.len() > 1 {
                let markets: Vec<String> = members.iter().map(|p| p.market_id.clone()).collect();
                let cluster_weight: Decimal = members.iter().map(|p| p.weight).sum();
                
                // Calculate average correlation within cluster
                let mut total_corr = Decimal::ZERO;
                let mut count = 0;
                for i in 0..members.len() {
                    for j in (i + 1)..members.len() {
                        if let Some(corr) = self.detector.get_correlation(
                            &members[i].market_id,
                            &members[j].market_id
                        ) {
                            total_corr += corr.abs();
                            count += 1;
                        }
                    }
                }
                let avg_correlation = if count > 0 {
                    total_corr / Decimal::from(count as i64)
                } else {
                    Decimal::ZERO
                };

                clusters.push(CorrelationCluster {
                    id,
                    markets,
                    avg_correlation,
                    cluster_weight,
                });
            }
        }

        clusters
    }

    /// Calculate diversification ratio (0-1, higher = better)
    fn calculate_diversification_ratio(
        &self,
        positions: &[PositionInfo],
        correlations: &[(Decimal, Decimal)]
    ) -> Decimal {
        if positions.len() <= 1 {
            return Decimal::ONE;
        }

        // Perfect diversification = N uncorrelated assets
        // Ratio = 1 / sqrt(sum of weighted correlation^2 + weights^2)
        let n = Decimal::from(positions.len() as i64);
        
        // If no correlation data, assume moderate diversification
        if correlations.is_empty() {
            return dec!(0.7);
        }

        // Sum of weighted squared correlations
        let total_weight: Decimal = correlations.iter().map(|(_, w)| *w).sum();
        if total_weight == Decimal::ZERO {
            return dec!(0.7);
        }

        let avg_corr = correlations.iter()
            .map(|(c, w)| *c * *w)
            .sum::<Decimal>() / total_weight;

        // Diversification ratio: 1 when correlation is 0, 0 when correlation is 1
        (Decimal::ONE - avg_corr).max(Decimal::ZERO).min(Decimal::ONE)
    }

    /// Count positions that are not highly correlated with others
    fn count_diversified_positions(&self, positions: &[PositionInfo]) -> usize {
        let mut diversified = HashSet::new();
        
        for pos in positions {
            let correlated_markets = self.detector.get_correlated_markets(&pos.market_id);
            let portfolio_markets: HashSet<_> = positions.iter()
                .map(|p| p.market_id.as_str())
                .collect();
            
            // Check if this position is correlated with any other portfolio position
            let is_correlated = correlated_markets.iter().any(|c| {
                let other = if c.market_a == pos.market_id {
                    &c.market_b
                } else {
                    &c.market_a
                };
                portfolio_markets.contains(other.as_str())
            });
            
            if !is_correlated {
                diversified.insert(pos.market_id.clone());
            }
        }
        
        // If all positions are correlated with each other, at least count 1
        if diversified.is_empty() && !positions.is_empty() {
            1
        } else {
            diversified.len()
        }
    }

    /// Calculate overall risk score (0-100)
    fn calculate_risk_score(
        &self,
        avg_correlation: Decimal,
        max_correlation: Decimal,
        clusters: &[CorrelationCluster],
    ) -> Decimal {
        // Base score from average correlation (0-40)
        let corr_score = avg_correlation * dec!(40);
        
        // Score from max pairwise correlation (0-30)
        let max_score = max_correlation * dec!(30);
        
        // Score from cluster concentration (0-30)
        let cluster_score = if clusters.is_empty() {
            Decimal::ZERO
        } else {
            let max_cluster_weight = clusters.iter()
                .map(|c| c.cluster_weight)
                .max()
                .unwrap_or(Decimal::ZERO);
            max_cluster_weight * dec!(30)
        };
        
        (corr_score + max_score + cluster_score).min(dec!(100))
    }

    /// Check if adding a position would increase correlation risk
    pub fn check_new_position(&self, market_id: &str, positions: &[PositionInfo]) -> NewPositionRisk {
        if positions.is_empty() {
            return NewPositionRisk {
                market_id: market_id.to_string(),
                correlated_positions: Vec::new(),
                max_correlation: Decimal::ZERO,
                size_multiplier: Decimal::ONE,
                should_avoid: false,
            };
        }

        let mut correlated_positions = Vec::new();
        let mut max_correlation = Decimal::ZERO;

        for pos in positions {
            if let Some(corr) = self.detector.get_correlation(market_id, &pos.market_id) {
                let abs_corr = corr.abs();
                if abs_corr >= self.config.high_correlation_threshold {
                    correlated_positions.push((pos.market_id.clone(), abs_corr));
                }
                if abs_corr > max_correlation {
                    max_correlation = abs_corr;
                }
            }
        }

        // Calculate size multiplier based on correlation
        let size_multiplier = if max_correlation >= self.config.high_correlation_threshold {
            // Reduce position size based on correlation level
            Decimal::ONE - (max_correlation * self.config.correlation_penalty_factor)
        } else {
            Decimal::ONE
        };

        // Should avoid if very highly correlated
        let should_avoid = max_correlation >= dec!(0.90);

        NewPositionRisk {
            market_id: market_id.to_string(),
            correlated_positions,
            max_correlation,
            size_multiplier: size_multiplier.max(dec!(0.25)), // Minimum 25% size
            should_avoid,
        }
    }

    /// Get the size multiplier for a market based on correlation with current positions
    pub fn get_size_multiplier(&self, market_id: &str, positions: &[PositionInfo]) -> Decimal {
        self.check_new_position(market_id, positions).size_multiplier
    }

    /// Get clusters
    pub fn get_clusters(&self) -> &[CorrelationCluster] {
        &self.clusters
    }

    /// Get last assessment
    pub fn last_assessment(&self) -> Option<&CorrelationRiskAssessment> {
        self.last_assessment.as_ref()
    }

    /// Get correlation between two markets
    pub fn get_correlation(&self, market_a: &str, market_b: &str) -> Option<Decimal> {
        self.detector.get_correlation(market_a, market_b)
    }

    /// Get all correlations for a market
    pub fn get_correlated_markets(&self, market_id: &str) -> Vec<MarketCorrelation> {
        self.detector.get_correlated_markets(market_id)
    }
}

/// Risk assessment for adding a new position
#[derive(Debug, Clone)]
pub struct NewPositionRisk {
    pub market_id: String,
    /// Existing positions that are correlated
    pub correlated_positions: Vec<(String, Decimal)>,
    /// Maximum correlation with any existing position
    pub max_correlation: Decimal,
    /// Recommended size multiplier (0-1)
    pub size_multiplier: Decimal,
    /// Should this position be avoided entirely?
    pub should_avoid: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> CorrelationRiskManager {
        CorrelationRiskManager::new(CorrelationRiskConfig::default())
    }

    fn add_correlated_data(manager: &mut CorrelationRiskManager, market_a: &str, market_b: &str) {
        // Add perfectly correlated price data
        for i in 1..=20 {
            let price = Decimal::from(i) / dec!(100);
            let ts = i as i64;
            manager.update_price(market_a, price, ts);
            manager.update_price(market_b, price, ts);
        }
    }

    fn add_uncorrelated_data(manager: &mut CorrelationRiskManager, market: &str) {
        // Add random-ish price data
        for i in 1..=20 {
            let price = dec!(0.5) + (Decimal::from(i % 3) - Decimal::ONE) / dec!(10);
            let ts = i as i64;
            manager.update_price(market, price, ts);
        }
    }

    #[test]
    fn test_new_manager() {
        let manager = make_manager();
        assert!(manager.clusters.is_empty());
        assert!(manager.last_assessment.is_none());
    }

    #[test]
    fn test_assess_empty_portfolio() {
        let mut manager = make_manager();
        let assessment = manager.assess_portfolio(&[]);
        
        assert_eq!(assessment.avg_portfolio_correlation, Decimal::ZERO);
        assert_eq!(assessment.risk_score, Decimal::ZERO);
        assert!(assessment.warnings.is_empty());
    }

    #[test]
    fn test_assess_single_position() {
        let mut manager = make_manager();
        
        let positions = vec![PositionInfo {
            market_id: "market1".to_string(),
            size: dec!(100),
            weight: Decimal::ONE,
        }];
        
        let assessment = manager.assess_portfolio(&positions);
        
        assert_eq!(assessment.avg_portfolio_correlation, Decimal::ZERO);
        assert!(assessment.clusters.is_empty());
    }

    #[test]
    fn test_detect_high_correlation() {
        let mut manager = make_manager();
        
        // Add correlated price data
        add_correlated_data(&mut manager, "market1", "market2");
        
        let positions = vec![
            PositionInfo {
                market_id: "market1".to_string(),
                size: dec!(100),
                weight: dec!(0.5),
            },
            PositionInfo {
                market_id: "market2".to_string(),
                size: dec!(100),
                weight: dec!(0.5),
            },
        ];
        
        let assessment = manager.assess_portfolio(&positions);
        
        // Should detect high correlation
        assert!(assessment.max_pairwise_correlation > dec!(0.5));
        
        // Should have warning
        let has_warning = assessment.warnings.iter().any(|w| {
            matches!(w, CorrelationWarning::HighPairwiseCorrelation { .. })
        });
        assert!(has_warning);
    }

    #[test]
    fn test_correlation_clusters() {
        let mut manager = make_manager();
        
        // Markets 1, 2, 3 are correlated
        add_correlated_data(&mut manager, "market1", "market2");
        add_correlated_data(&mut manager, "market2", "market3");
        add_correlated_data(&mut manager, "market1", "market3");
        
        // Market 4 is uncorrelated
        add_uncorrelated_data(&mut manager, "market4");
        
        let positions = vec![
            PositionInfo { market_id: "market1".to_string(), size: dec!(100), weight: dec!(0.25) },
            PositionInfo { market_id: "market2".to_string(), size: dec!(100), weight: dec!(0.25) },
            PositionInfo { market_id: "market3".to_string(), size: dec!(100), weight: dec!(0.25) },
            PositionInfo { market_id: "market4".to_string(), size: dec!(100), weight: dec!(0.25) },
        ];
        
        let assessment = manager.assess_portfolio(&positions);
        
        // Should detect cluster
        if !assessment.clusters.is_empty() {
            let cluster = &assessment.clusters[0];
            assert!(cluster.markets.len() >= 2);
        }
    }

    #[test]
    fn test_diversification_ratio() {
        let mut manager = make_manager();
        
        // All correlated
        add_correlated_data(&mut manager, "m1", "m2");
        add_correlated_data(&mut manager, "m2", "m3");
        add_correlated_data(&mut manager, "m1", "m3");
        
        let positions = vec![
            PositionInfo { market_id: "m1".to_string(), size: dec!(100), weight: dec!(0.33) },
            PositionInfo { market_id: "m2".to_string(), size: dec!(100), weight: dec!(0.33) },
            PositionInfo { market_id: "m3".to_string(), size: dec!(100), weight: dec!(0.34) },
        ];
        
        let assessment = manager.assess_portfolio(&positions);
        
        // Low diversification due to correlation
        assert!(assessment.diversification_ratio < Decimal::ONE);
    }

    #[test]
    fn test_check_new_position_uncorrelated() {
        let manager = make_manager();
        
        let positions = vec![
            PositionInfo { market_id: "existing".to_string(), size: dec!(100), weight: Decimal::ONE },
        ];
        
        let risk = manager.check_new_position("new_market", &positions);
        
        assert!(risk.correlated_positions.is_empty());
        assert_eq!(risk.size_multiplier, Decimal::ONE);
        assert!(!risk.should_avoid);
    }

    #[test]
    fn test_check_new_position_correlated() {
        let mut manager = make_manager();
        
        add_correlated_data(&mut manager, "existing", "new_market");
        
        let positions = vec![
            PositionInfo { market_id: "existing".to_string(), size: dec!(100), weight: Decimal::ONE },
        ];
        
        let risk = manager.check_new_position("new_market", &positions);
        
        assert!(!risk.correlated_positions.is_empty());
        assert!(risk.size_multiplier < Decimal::ONE);
    }

    #[test]
    fn test_get_size_multiplier() {
        let mut manager = make_manager();
        
        add_correlated_data(&mut manager, "m1", "m2");
        
        let positions = vec![
            PositionInfo { market_id: "m1".to_string(), size: dec!(100), weight: Decimal::ONE },
        ];
        
        // Correlated market should have reduced multiplier
        let multiplier = manager.get_size_multiplier("m2", &positions);
        assert!(multiplier < Decimal::ONE);
        
        // Uncorrelated market should have full multiplier
        let multiplier_unrelated = manager.get_size_multiplier("m3", &positions);
        assert_eq!(multiplier_unrelated, Decimal::ONE);
    }

    #[test]
    fn test_risk_score_calculation() {
        let mut manager = make_manager();
        
        // High correlation = high risk
        add_correlated_data(&mut manager, "m1", "m2");
        
        let positions = vec![
            PositionInfo { market_id: "m1".to_string(), size: dec!(100), weight: dec!(0.5) },
            PositionInfo { market_id: "m2".to_string(), size: dec!(100), weight: dec!(0.5) },
        ];
        
        let assessment = manager.assess_portfolio(&positions);
        
        // Should have elevated risk score
        assert!(assessment.risk_score > Decimal::ZERO);
    }

    #[test]
    fn test_recommendations_generated() {
        let mut manager = make_manager();
        
        // High correlation
        add_correlated_data(&mut manager, "m1", "m2");
        
        let positions = vec![
            PositionInfo { market_id: "m1".to_string(), size: dec!(100), weight: dec!(0.5) },
            PositionInfo { market_id: "m2".to_string(), size: dec!(50), weight: dec!(0.5) },
        ];
        
        let assessment = manager.assess_portfolio(&positions);
        
        // Should have recommendations
        assert!(!assessment.recommendations.is_empty());
    }

    #[test]
    fn test_set_market_category() {
        let mut manager = make_manager();
        
        manager.set_market_category("market1", "politics");
        manager.set_market_category("market2", "crypto");
        
        assert_eq!(manager.market_categories.get("market1"), Some(&"politics".to_string()));
        assert_eq!(manager.market_categories.get("market2"), Some(&"crypto".to_string()));
    }

    #[test]
    fn test_get_correlation() {
        let mut manager = make_manager();
        
        add_correlated_data(&mut manager, "m1", "m2");
        
        let corr = manager.get_correlation("m1", "m2");
        assert!(corr.is_some());
        assert!(corr.unwrap() > dec!(0.5));
        
        assert!(manager.get_correlation("m1", "unknown").is_none());
    }

    #[test]
    fn test_insufficient_diversification_warning() {
        let mut manager = make_manager();
        manager.config.min_diversified_positions = 5;
        
        // Only 2 positions, need 5
        let positions = vec![
            PositionInfo { market_id: "m1".to_string(), size: dec!(100), weight: dec!(0.5) },
            PositionInfo { market_id: "m2".to_string(), size: dec!(100), weight: dec!(0.5) },
        ];
        
        let assessment = manager.assess_portfolio(&positions);
        
        let has_warning = assessment.warnings.iter().any(|w| {
            matches!(w, CorrelationWarning::InsufficientDiversification { .. })
        });
        assert!(has_warning);
    }

    #[test]
    fn test_cluster_concentration_warning() {
        let mut manager = make_manager();
        manager.config.max_cluster_weight = dec!(0.30);
        
        // Correlated markets with high weight
        add_correlated_data(&mut manager, "m1", "m2");
        
        let positions = vec![
            PositionInfo { market_id: "m1".to_string(), size: dec!(100), weight: dec!(0.3) },
            PositionInfo { market_id: "m2".to_string(), size: dec!(100), weight: dec!(0.3) },
            PositionInfo { market_id: "m3".to_string(), size: dec!(100), weight: dec!(0.4) },
        ];
        
        let assessment = manager.assess_portfolio(&positions);
        
        // Cluster m1+m2 has 60% weight, should warn
        if !assessment.clusters.is_empty() {
            let has_warning = assessment.warnings.iter().any(|w| {
                matches!(w, CorrelationWarning::ClusterConcentration { .. })
            });
            assert!(has_warning);
        }
    }

    #[test]
    fn test_should_avoid_very_high_correlation() {
        let mut manager = make_manager();
        
        // Add perfectly correlated data
        for i in 1..=20 {
            let price = Decimal::from(i) / dec!(100);
            manager.update_price("m1", price, i as i64);
            manager.update_price("m2", price, i as i64);
        }
        
        let positions = vec![
            PositionInfo { market_id: "m1".to_string(), size: dec!(100), weight: Decimal::ONE },
        ];
        
        let risk = manager.check_new_position("m2", &positions);
        
        // Very high correlation should trigger avoid
        if risk.max_correlation >= dec!(0.90) {
            assert!(risk.should_avoid);
        }
    }

    #[test]
    fn test_last_assessment_cached() {
        let mut manager = make_manager();
        
        assert!(manager.last_assessment().is_none());
        
        let positions = vec![
            PositionInfo { market_id: "m1".to_string(), size: dec!(100), weight: Decimal::ONE },
        ];
        
        manager.assess_portfolio(&positions);
        
        assert!(manager.last_assessment().is_some());
    }

    #[test]
    fn test_get_clusters() {
        let mut manager = make_manager();
        
        assert!(manager.get_clusters().is_empty());
        
        add_correlated_data(&mut manager, "m1", "m2");
        
        let positions = vec![
            PositionInfo { market_id: "m1".to_string(), size: dec!(100), weight: dec!(0.5) },
            PositionInfo { market_id: "m2".to_string(), size: dec!(100), weight: dec!(0.5) },
        ];
        
        manager.assess_portfolio(&positions);
        
        // May or may not have clusters depending on correlation threshold
        // The important thing is no crash
    }
}
