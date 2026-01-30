//! Trade execution and risk management

pub mod arbitrage_executor;
pub mod smart_executor;
pub mod gradual_exit;
pub mod slippage_predictor;
pub mod price_optimizer;
pub mod twap_vwap;
pub mod latency_optimizer;

pub use arbitrage_executor::{ArbitrageExecutor, ArbitrageExecutorConfig, ArbitrageResult, ArbitrageMetrics};
pub use slippage_predictor::{SlippagePredictor, SlippageConfig, SlippagePrediction, OrderBook, OrderSide};
pub use price_optimizer::{PriceOptimizer, PriceOptimizerConfig, PriceRecommendation, ExecutionUrgency, RecommendedOrderType};
pub use twap_vwap::{ExecutionEngine, ExecutionConfig, ExecutionAlgorithm, ExecutionSummary, default_crypto_volume_profile};
pub use latency_optimizer::{
    LatencyOptimizer, LatencyConfig, LatencyStats, LatencySummary,
    LocalOrderBook, OrderBookDelta, PriceLevel,
    BatchSubmitter, BatchConfig, BatchableOrder, BatchOrderResult,
    ConnectionPool, TemplateCache, PreSignedTemplate,
};

#[cfg(test)]
mod tests;
#[cfg(test)]
mod smart_executor_tests;
#[cfg(test)]
mod gradual_exit_tests;

use crate::client::ClobClient;
use crate::config::RiskConfig;
use crate::error::{BotError, Result};
use crate::types::{Order, OrderType, Signal, Trade};
use rust_decimal::Decimal;
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Trade executor with risk management
pub struct Executor {
    pub clob: ClobClient,
    risk_config: RiskConfig,
    positions: RwLock<HashMap<String, Decimal>>, // token_id -> size
    daily_pnl: RwLock<Decimal>,
}

impl Executor {
    pub fn new(clob: ClobClient, risk_config: RiskConfig) -> Self {
        Self {
            clob,
            risk_config,
            positions: RwLock::new(HashMap::new()),
            daily_pnl: RwLock::new(Decimal::ZERO),
        }
    }

    /// Execute a trading signal
    pub async fn execute(&self, signal: &Signal, portfolio_value: Decimal) -> Result<Option<Trade>> {
        // Pre-trade risk checks
        self.check_risk_limits(signal, portfolio_value).await?;

        // Calculate actual order size
        let size_usd = signal.suggested_size * portfolio_value;
        let size_shares = size_usd / signal.market_probability;

        // Get current market price for limit order
        let book = self.clob.get_order_book(&signal.token_id).await?;
        let limit_price = match signal.side {
            crate::types::Side::Buy => book
                .best_ask()
                .ok_or_else(|| BotError::Execution("No asks available".into()))?,
            crate::types::Side::Sell => book
                .best_bid()
                .ok_or_else(|| BotError::Execution("No bids available".into()))?,
        };

        // Create and place order
        let order = Order {
            token_id: signal.token_id.clone(),
            side: signal.side,
            price: limit_price,
            size: size_shares,
            order_type: OrderType::GTC,
        };

        tracing::info!(
            "Placing order: {} {:.2} shares of {} @ {:.4}",
            match signal.side {
                crate::types::Side::Buy => "BUY",
                crate::types::Side::Sell => "SELL",
            },
            size_shares,
            signal.token_id,
            limit_price
        );

        let order_status = self.clob.place_order(&order).await?;

        // Update positions
        self.update_position(&signal.token_id, signal.side, size_shares)
            .await;

        Ok(Some(Trade {
            id: uuid::Uuid::new_v4().to_string(),
            order_id: order_status.order_id,
            token_id: signal.token_id.clone(),
            market_id: signal.market_id.clone(),
            side: signal.side,
            price: limit_price,
            size: size_shares,
            fee: Decimal::ZERO, // TODO: Calculate fee
            timestamp: chrono::Utc::now(),
        }))
    }

    /// Check all risk limits before trading
    async fn check_risk_limits(&self, signal: &Signal, portfolio_value: Decimal) -> Result<()> {
        // Check daily loss limit
        let daily_pnl = *self.daily_pnl.read().await;
        let max_loss = self.risk_config.max_daily_loss_pct * portfolio_value;
        if daily_pnl < -max_loss {
            return Err(BotError::RiskLimit(format!(
                "Daily loss limit exceeded: {:.2}",
                daily_pnl
            )));
        }

        // Check position limit
        let positions = self.positions.read().await;
        if positions.len() >= self.risk_config.max_open_positions {
            if !positions.contains_key(&signal.token_id) {
                return Err(BotError::RiskLimit(format!(
                    "Max open positions ({}) reached",
                    self.risk_config.max_open_positions
                )));
            }
        }

        // Check total exposure
        let total_exposure: Decimal = positions.values().sum();
        let new_exposure = total_exposure + signal.suggested_size * portfolio_value;
        let max_exposure = self.risk_config.max_exposure_pct * portfolio_value;
        if new_exposure > max_exposure {
            return Err(BotError::RiskLimit(format!(
                "Max exposure exceeded: {:.2} > {:.2}",
                new_exposure, max_exposure
            )));
        }

        Ok(())
    }

    /// Update position tracking
    async fn update_position(
        &self,
        token_id: &str,
        side: crate::types::Side,
        size: Decimal,
    ) {
        let mut positions = self.positions.write().await;
        let current = positions.get(token_id).copied().unwrap_or(Decimal::ZERO);
        let new_size = match side {
            crate::types::Side::Buy => current + size,
            crate::types::Side::Sell => current - size,
        };

        if new_size == Decimal::ZERO {
            positions.remove(token_id);
        } else {
            positions.insert(token_id.to_string(), new_size);
        }
    }

    /// Update daily P&L
    pub async fn update_pnl(&self, pnl_change: Decimal) {
        let mut daily_pnl = self.daily_pnl.write().await;
        *daily_pnl += pnl_change;
    }

    /// Reset daily P&L (call at start of new day)
    pub async fn reset_daily_pnl(&self) {
        let mut daily_pnl = self.daily_pnl.write().await;
        *daily_pnl = Decimal::ZERO;
    }

    /// Get current positions
    pub async fn get_positions(&self) -> HashMap<String, Decimal> {
        self.positions.read().await.clone()
    }
}
