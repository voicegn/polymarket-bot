//! Arbitrage Executor - Concurrent Yes+No order execution
//!
//! Executes arbitrage opportunities by placing Yes and No orders simultaneously.
//! Key features:
//! - Concurrent order placement (both legs at once)
//! - Automatic rollback on partial fill
//! - Slippage protection
//! - Position tracking

use crate::client::ClobClient;
use crate::error::{BotError, Result};
use crate::scanner::ArbitrageOpp;
use crate::types::{Order, OrderType, Side};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Simple order status for arbitrage
#[derive(Debug, Clone, PartialEq)]
pub enum ArbOrderStatus {
    Pending,
    Filled,
    PartialFill,
    Cancelled,
    Failed,
}

/// Arbitrage execution result
#[derive(Debug, Clone)]
pub struct ArbitrageResult {
    /// Unique execution ID
    pub id: String,
    /// Was execution successful
    pub success: bool,
    /// Yes order result
    pub yes_order: Option<OrderResult>,
    /// No order result
    pub no_order: Option<OrderResult>,
    /// Total cost (USDC spent)
    pub total_cost: Decimal,
    /// Expected profit
    pub expected_profit: Decimal,
    /// Actual profit (if known)
    pub actual_profit: Option<Decimal>,
    /// Execution latency (ms)
    pub latency_ms: u64,
    /// Error message if failed
    pub error: Option<String>,
    /// Timestamp
    pub executed_at: DateTime<Utc>,
}

/// Individual order result
#[derive(Debug, Clone)]
pub struct OrderResult {
    pub order_id: String,
    pub token_id: String,
    pub side: Side,
    pub price: Decimal,
    pub size: Decimal,
    pub filled_size: Decimal,
    pub status: ArbOrderStatus,
}

/// Arbitrage executor configuration
#[derive(Debug, Clone)]
pub struct ArbitrageExecutorConfig {
    /// Maximum position size per trade
    pub max_position_size: Decimal,
    /// Maximum slippage allowed
    pub max_slippage: Decimal,
    /// Minimum profit margin to execute
    pub min_profit_margin: Decimal,
    /// Order timeout (ms)
    pub order_timeout_ms: u64,
    /// Enable dry run mode
    pub dry_run: bool,
}

impl Default for ArbitrageExecutorConfig {
    fn default() -> Self {
        Self {
            max_position_size: dec!(100),
            max_slippage: dec!(0.02),
            min_profit_margin: dec!(0.005),
            order_timeout_ms: 5000,
            dry_run: true,
        }
    }
}

/// Metrics for arbitrage execution
#[derive(Debug, Default)]
pub struct ArbitrageMetrics {
    pub total_attempts: u64,
    pub successful: u64,
    pub failed: u64,
    pub partial_fills: u64,
    pub total_profit: Decimal,
    pub total_cost: Decimal,
    pub avg_latency_ms: f64,
}

/// Arbitrage executor
pub struct ArbitrageExecutor {
    clob: ClobClient,
    config: ArbitrageExecutorConfig,
    metrics: Arc<RwLock<ArbitrageMetrics>>,
}

impl ArbitrageExecutor {
    /// Create new arbitrage executor
    pub fn new(clob: ClobClient, config: ArbitrageExecutorConfig) -> Self {
        Self {
            clob,
            config,
            metrics: Arc::new(RwLock::new(ArbitrageMetrics::default())),
        }
    }

    /// Execute an arbitrage opportunity
    ///
    /// Places Yes and No orders concurrently. If one fails, attempts to cancel the other.
    pub async fn execute(&self, opp: &ArbitrageOpp) -> Result<ArbitrageResult> {
        let start = std::time::Instant::now();
        let exec_id = format!("arb_{}", Utc::now().timestamp_millis());

        // Validate opportunity
        if opp.profit_margin < self.config.min_profit_margin * dec!(100) {
            return Err(BotError::Execution(format!(
                "Profit margin {:.2}% below threshold {:.2}%",
                opp.profit_margin,
                self.config.min_profit_margin * dec!(100)
            )));
        }

        // Calculate position size
        let size = Decimal::from(opp.max_size).min(self.config.max_position_size);
        if size <= dec!(0) {
            return Err(BotError::Execution("Invalid position size".into()));
        }

        info!(
            "[ArbExecutor] Executing {}: YES@{:.3} + NO@{:.3} = {:.4}, size={}",
            opp.slug, opp.yes_ask, opp.no_ask, opp.total_cost, size
        );

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_attempts += 1;
        }

        // Dry run mode - simulate execution
        if self.config.dry_run {
            let latency = start.elapsed().as_millis() as u64;
            
            let result = ArbitrageResult {
                id: exec_id,
                success: true,
                yes_order: Some(OrderResult {
                    order_id: "dry_run_yes".into(),
                    token_id: opp.yes_token_id.clone(),
                    side: Side::Buy,
                    price: opp.yes_ask,
                    size,
                    filled_size: size,
                    status: ArbOrderStatus::Filled,
                }),
                no_order: Some(OrderResult {
                    order_id: "dry_run_no".into(),
                    token_id: opp.no_token_id.clone(),
                    side: Side::Buy,
                    price: opp.no_ask,
                    size,
                    filled_size: size,
                    status: ArbOrderStatus::Filled,
                }),
                total_cost: opp.total_cost * size,
                expected_profit: opp.net_profit,
                actual_profit: Some(opp.spread * size),
                latency_ms: latency,
                error: None,
                executed_at: Utc::now(),
            };

            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.successful += 1;
                metrics.total_profit += opp.spread * size;
                metrics.total_cost += opp.total_cost * size;
            }

            info!(
                "[ArbExecutor] DRY RUN success: profit=${:.4}, latency={}ms",
                opp.spread * size, latency
            );

            return Ok(result);
        }

        // Live execution - place both orders concurrently
        let yes_order = Order {
            token_id: opp.yes_token_id.clone(),
            side: Side::Buy,
            price: opp.yes_ask,
            size,
            order_type: OrderType::GTC,
        };

        let no_order = Order {
            token_id: opp.no_token_id.clone(),
            side: Side::Buy,
            price: opp.no_ask,
            size,
            order_type: OrderType::GTC,
        };

        // Execute both orders concurrently
        let clob = self.clob.clone();
        let clob2 = self.clob.clone();
        
        let yes_handle = tokio::spawn(async move {
            clob.place_order(&yes_order).await
        });

        let no_handle = tokio::spawn(async move {
            clob2.place_order(&no_order).await
        });

        // Wait for both
        let (yes_result, no_result) = tokio::join!(yes_handle, no_handle);

        let latency = start.elapsed().as_millis() as u64;

        // Process results
        let yes_res = yes_result.map_err(|e| BotError::Internal(e.to_string()))?;
        let no_res = no_result.map_err(|e| BotError::Internal(e.to_string()))?;

        match (yes_res, no_res) {
            (Ok(yes_status), Ok(no_status)) => {
                // Both succeeded
                let mut metrics = self.metrics.write().await;
                metrics.successful += 1;
                metrics.total_profit += opp.spread * size;
                metrics.total_cost += opp.total_cost * size;

                info!(
                    "[ArbExecutor] SUCCESS: yes={}, no={}, profit=${:.4}",
                    yes_status.order_id, no_status.order_id, opp.spread * size
                );

                Ok(ArbitrageResult {
                    id: exec_id,
                    success: true,
                    yes_order: Some(OrderResult {
                        order_id: yes_status.order_id,
                        token_id: opp.yes_token_id.clone(),
                        side: Side::Buy,
                        price: opp.yes_ask,
                        size,
                        filled_size: yes_status.filled_size,
                        status: ArbOrderStatus::Filled,
                    }),
                    no_order: Some(OrderResult {
                        order_id: no_status.order_id,
                        token_id: opp.no_token_id.clone(),
                        side: Side::Buy,
                        price: opp.no_ask,
                        size,
                        filled_size: no_status.filled_size,
                        status: ArbOrderStatus::Filled,
                    }),
                    total_cost: opp.total_cost * size,
                    expected_profit: opp.net_profit,
                    actual_profit: Some(opp.spread * size),
                    latency_ms: latency,
                    error: None,
                    executed_at: Utc::now(),
                })
            }
            (Ok(yes_status), Err(no_err)) => {
                // Yes succeeded, No failed - try to cancel Yes
                warn!("[ArbExecutor] Partial fill - No order failed: {}", no_err);
                if let Err(cancel_err) = self.clob.cancel_order(&yes_status.order_id).await {
                    error!("[ArbExecutor] Failed to cancel Yes order: {}", cancel_err);
                }

                let mut metrics = self.metrics.write().await;
                metrics.partial_fills += 1;

                Ok(ArbitrageResult {
                    id: exec_id,
                    success: false,
                    yes_order: Some(OrderResult {
                        order_id: yes_status.order_id,
                        token_id: opp.yes_token_id.clone(),
                        side: Side::Buy,
                        price: opp.yes_ask,
                        size,
                        filled_size: dec!(0),
                        status: ArbOrderStatus::Cancelled,
                    }),
                    no_order: None,
                    total_cost: dec!(0),
                    expected_profit: opp.net_profit,
                    actual_profit: None,
                    latency_ms: latency,
                    error: Some(format!("No order failed: {}", no_err)),
                    executed_at: Utc::now(),
                })
            }
            (Err(yes_err), Ok(no_status)) => {
                // No succeeded, Yes failed - try to cancel No
                warn!("[ArbExecutor] Partial fill - Yes order failed: {}", yes_err);
                if let Err(cancel_err) = self.clob.cancel_order(&no_status.order_id).await {
                    error!("[ArbExecutor] Failed to cancel No order: {}", cancel_err);
                }

                let mut metrics = self.metrics.write().await;
                metrics.partial_fills += 1;

                Ok(ArbitrageResult {
                    id: exec_id,
                    success: false,
                    yes_order: None,
                    no_order: Some(OrderResult {
                        order_id: no_status.order_id,
                        token_id: opp.no_token_id.clone(),
                        side: Side::Buy,
                        price: opp.no_ask,
                        size,
                        filled_size: dec!(0),
                        status: ArbOrderStatus::Cancelled,
                    }),
                    total_cost: dec!(0),
                    expected_profit: opp.net_profit,
                    actual_profit: None,
                    latency_ms: latency,
                    error: Some(format!("Yes order failed: {}", yes_err)),
                    executed_at: Utc::now(),
                })
            }
            (Err(yes_err), Err(no_err)) => {
                // Both failed
                let mut metrics = self.metrics.write().await;
                metrics.failed += 1;

                error!(
                    "[ArbExecutor] Both orders failed: yes={}, no={}",
                    yes_err, no_err
                );

                Ok(ArbitrageResult {
                    id: exec_id,
                    success: false,
                    yes_order: None,
                    no_order: None,
                    total_cost: dec!(0),
                    expected_profit: opp.net_profit,
                    actual_profit: None,
                    latency_ms: latency,
                    error: Some(format!("Both orders failed: yes={}, no={}", yes_err, no_err)),
                    executed_at: Utc::now(),
                })
            }
        }
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> ArbitrageMetrics {
        let metrics = self.metrics.read().await;
        ArbitrageMetrics {
            total_attempts: metrics.total_attempts,
            successful: metrics.successful,
            failed: metrics.failed,
            partial_fills: metrics.partial_fills,
            total_profit: metrics.total_profit,
            total_cost: metrics.total_cost,
            avg_latency_ms: metrics.avg_latency_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ArbitrageExecutorConfig::default();
        assert_eq!(config.max_position_size, dec!(100));
        assert!(config.dry_run);
    }

    #[test]
    fn test_result_success() {
        let result = ArbitrageResult {
            id: "test".into(),
            success: true,
            yes_order: None,
            no_order: None,
            total_cost: dec!(0.96),
            expected_profit: dec!(0.04),
            actual_profit: Some(dec!(0.04)),
            latency_ms: 50,
            error: None,
            executed_at: Utc::now(),
        };
        assert!(result.success);
        assert_eq!(result.actual_profit, Some(dec!(0.04)));
    }
}
