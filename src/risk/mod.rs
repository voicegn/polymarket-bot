//! Advanced Risk Management Module
//!
//! Provides comprehensive risk controls:
//! - Daily P&L tracking with loss limits
//! - Volatility-adaptive position sizing
//! - Market correlation detection
//! - Dynamic position management
//! - Black swan protection
//! - Liquidity monitoring
//! - Enhanced correlation risk analysis

mod daily_pnl;
mod volatility_sizer;
mod correlation;
mod position_manager;
mod black_swan;
mod liquidity_monitor;
mod correlation_risk;

#[cfg(test)]
mod tests;

pub use daily_pnl::{DailyPnlTracker, DailyPnlState};
pub use volatility_sizer::{VolatilityPositionSizer, VolatilityConfig};
pub use correlation::{CorrelationDetector, CorrelationMatrix, MarketCorrelation};
pub use position_manager::{DynamicPositionManager, PositionSizeRequest, PositionSizeResult};
pub use black_swan::{
    BlackSwanProtector, BlackSwanConfig, BlackSwanEvent, 
    ProtectionAction, ProtectionState
};
pub use liquidity_monitor::{
    LiquidityMonitor, LiquidityConfig, LiquidityAssessment, 
    LiquidityAlert, OrderBookSnapshot, OrderBookLevel
};
pub use correlation_risk::{
    CorrelationRiskManager, CorrelationRiskConfig, CorrelationRiskAssessment,
    CorrelationCluster, CorrelationWarning, RiskRecommendation,
    PositionInfo, NewPositionRisk
};

use crate::config::RiskConfig;
use crate::types::{Market, Position, Signal};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Integrated risk manager combining all risk controls
pub struct RiskManager {
    pub config: RiskConfig,
    pub pnl_tracker: DailyPnlTracker,
    pub volatility_sizer: VolatilityPositionSizer,
    pub correlation_detector: CorrelationDetector,
    pub position_manager: DynamicPositionManager,
    pub black_swan_protector: BlackSwanProtector,
    pub liquidity_monitor: LiquidityMonitor,
    pub correlation_risk: CorrelationRiskManager,
}

impl RiskManager {
    /// Create a new risk manager with the given configuration
    pub fn new(config: RiskConfig) -> Self {
        let volatility_config = VolatilityConfig::default();
        let black_swan_config = BlackSwanConfig::default();
        let liquidity_config = LiquidityConfig::default();
        let correlation_risk_config = CorrelationRiskConfig::default();
        
        Self {
            pnl_tracker: DailyPnlTracker::new(config.max_daily_loss_pct),
            volatility_sizer: VolatilityPositionSizer::new(volatility_config),
            correlation_detector: CorrelationDetector::new(0.7), // 70% correlation threshold
            position_manager: DynamicPositionManager::new(config.clone()),
            black_swan_protector: BlackSwanProtector::new(black_swan_config),
            liquidity_monitor: LiquidityMonitor::new(liquidity_config),
            correlation_risk: CorrelationRiskManager::new(correlation_risk_config),
            config,
        }
    }

    /// Check if trading is allowed based on all risk constraints
    pub fn can_trade(&self) -> RiskCheckResult {
        // Check daily loss limit
        if self.pnl_tracker.is_limit_reached() {
            return RiskCheckResult::Blocked {
                reason: "Daily loss limit reached".to_string(),
            };
        }

        // Check black swan protection
        if !self.black_swan_protector.can_trade() {
            return RiskCheckResult::Blocked {
                reason: "Black swan protection active".to_string(),
            };
        }

        RiskCheckResult::Allowed
    }

    /// Calculate the maximum position size for a signal
    pub fn calculate_position_size(
        &mut self,
        signal: &Signal,
        market: &Market,
        balance: Decimal,
        current_positions: &[Position],
    ) -> Option<Decimal> {
        // First check if we can trade at all
        if let RiskCheckResult::Blocked { .. } = self.can_trade() {
            return None;
        }

        // Check position count limit
        if current_positions.len() >= self.config.max_open_positions {
            return None;
        }

        // Check if market should be avoided (black swan)
        if self.black_swan_protector.should_avoid_market(&market.id) {
            return None;
        }

        // Check liquidity
        if !self.liquidity_monitor.is_tradeable(&market.id) {
            return None;
        }

        // Build position size request
        let request = PositionSizeRequest {
            signal_confidence: signal.confidence,
            signal_edge: signal.edge,
            market_id: market.id.clone(),
            balance,
            current_exposure: self.calculate_exposure(current_positions),
        };

        // Get base position size from dynamic manager
        let result = self.position_manager.calculate_size(&request);

        // Apply volatility multiplier
        let volatility_multiplier = self.volatility_sizer.get_size_multiplier(&market.id);
        let size_after_vol = result.size * volatility_multiplier;

        // Apply black swan protection multiplier
        let black_swan_multiplier = self.black_swan_protector.get_size_multiplier(&market.id);
        let size_after_bs = size_after_vol * black_swan_multiplier;

        // Apply liquidity multiplier
        let liquidity_multiplier = self.liquidity_monitor.get_size_multiplier(&market.id);
        let size_after_liq = size_after_bs * liquidity_multiplier;

        // Check correlation - reduce size if highly correlated with existing positions
        let position_infos: Vec<PositionInfo> = current_positions
            .iter()
            .map(|p| PositionInfo {
                market_id: p.market_id.clone(),
                size: p.size * p.current_price,
                weight: if balance > Decimal::ZERO {
                    (p.size * p.current_price) / balance
                } else {
                    Decimal::ZERO
                },
            })
            .collect();
        
        let correlation_multiplier = self.correlation_risk
            .get_size_multiplier(&market.id, &position_infos);
        let final_size = size_after_liq * correlation_multiplier;

        // Ensure minimum viable size
        if final_size < Decimal::ONE {
            return None;
        }

        Some(final_size)
    }

    /// Record a trade execution for P&L tracking
    pub fn record_trade(&mut self, pnl: Decimal) {
        self.pnl_tracker.record_pnl(pnl);
    }

    /// Update volatility data for a market
    pub fn update_volatility(&mut self, market_id: &str, price: Decimal) {
        self.volatility_sizer.add_price_point(market_id, price);
    }

    /// Update correlation data
    pub fn update_correlation(&mut self, market_id: &str, price: Decimal, timestamp: i64) {
        self.correlation_detector.add_price_point(market_id, price, timestamp);
        self.correlation_risk.update_price(market_id, price, timestamp);
    }

    /// Update black swan monitoring
    pub fn update_black_swan(&mut self, market_id: &str, price: Decimal, liquidity: Option<Decimal>) -> Option<BlackSwanEvent> {
        self.black_swan_protector.update(market_id, price, liquidity)
    }

    /// Update liquidity monitoring
    pub fn update_liquidity(&mut self, snapshot: OrderBookSnapshot) -> LiquidityAssessment {
        self.liquidity_monitor.update_order_book(snapshot)
    }

    /// Get current daily P&L
    pub fn daily_pnl(&self) -> Decimal {
        self.pnl_tracker.current_pnl()
    }

    /// Reset daily trackers (call at start of new day)
    pub fn reset_daily(&mut self) {
        self.pnl_tracker.reset();
        self.black_swan_protector.clear_protection();
    }

    /// Calculate total exposure from positions
    fn calculate_exposure(&self, positions: &[Position]) -> Decimal {
        positions.iter()
            .map(|p| p.size * p.current_price)
            .sum()
    }

    /// Get comprehensive risk state
    pub fn get_risk_state(&self, current_positions: &[Position], balance: Decimal) -> RiskState {
        let position_infos: Vec<PositionInfo> = current_positions
            .iter()
            .map(|p| PositionInfo {
                market_id: p.market_id.clone(),
                size: p.size * p.current_price,
                weight: if balance > Decimal::ZERO {
                    (p.size * p.current_price) / balance
                } else {
                    Decimal::ZERO
                },
            })
            .collect();

        RiskState {
            can_trade: matches!(self.can_trade(), RiskCheckResult::Allowed),
            daily_pnl: self.daily_pnl(),
            daily_pnl_pct: if balance > Decimal::ZERO {
                self.daily_pnl() / balance * dec!(100)
            } else {
                Decimal::ZERO
            },
            black_swan_active: self.black_swan_protector.protection_state().is_active,
            black_swan_event: self.black_swan_protector.protection_state().event.clone(),
            low_liquidity_markets: self.liquidity_monitor
                .low_liquidity_markets()
                .iter()
                .map(|a| a.market_id.clone())
                .collect(),
            high_risk_markets: current_positions
                .iter()
                .filter(|p| self.black_swan_protector.should_avoid_market(&p.market_id))
                .map(|p| p.market_id.clone())
                .collect(),
            position_count: current_positions.len(),
            max_positions: self.config.max_open_positions,
        }
    }

    /// Assess correlation risk for current portfolio
    pub fn assess_correlation_risk(&mut self, positions: &[Position], balance: Decimal) -> CorrelationRiskAssessment {
        let position_infos: Vec<PositionInfo> = positions
            .iter()
            .map(|p| PositionInfo {
                market_id: p.market_id.clone(),
                size: p.size * p.current_price,
                weight: if balance > Decimal::ZERO {
                    (p.size * p.current_price) / balance
                } else {
                    Decimal::ZERO
                },
            })
            .collect();

        self.correlation_risk.assess_portfolio(&position_infos)
    }

    /// Check for correlated crash across portfolio
    pub fn check_correlated_crash(&mut self, positions: &[Position]) -> Option<BlackSwanEvent> {
        let market_ids: Vec<String> = positions.iter()
            .map(|p| p.market_id.clone())
            .collect();
        
        self.black_swan_protector.check_correlated_crash(&market_ids)
    }

    /// Get recommended protection action
    pub fn get_protection_action(&self) -> ProtectionAction {
        self.black_swan_protector.get_recommended_action()
    }
}

/// Result of a risk check
#[derive(Debug, Clone, PartialEq)]
pub enum RiskCheckResult {
    Allowed,
    Blocked { reason: String },
}

/// Comprehensive risk state
#[derive(Debug, Clone)]
pub struct RiskState {
    pub can_trade: bool,
    pub daily_pnl: Decimal,
    pub daily_pnl_pct: Decimal,
    pub black_swan_active: bool,
    pub black_swan_event: Option<BlackSwanEvent>,
    pub low_liquidity_markets: Vec<String>,
    pub high_risk_markets: Vec<String>,
    pub position_count: usize,
    pub max_positions: usize,
}
