//! Monitoring and alerting

use rust_decimal::Decimal;
use std::collections::VecDeque;
use tokio::sync::RwLock;

/// Performance monitor
pub struct Monitor {
    trades: RwLock<VecDeque<TradeRecord>>,
    max_history: usize,
}

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub market_id: String,
    pub side: String,
    pub size: Decimal,
    pub price: Decimal,
    pub pnl: Option<Decimal>,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: Decimal,
    pub total_pnl: Decimal,
    pub avg_pnl_per_trade: Decimal,
    pub sharpe_ratio: Option<Decimal>,
}

impl Monitor {
    pub fn new(max_history: usize) -> Self {
        Self {
            trades: RwLock::new(VecDeque::with_capacity(max_history)),
            max_history,
        }
    }

    pub async fn record_trade(&self, record: TradeRecord) {
        let mut trades = self.trades.write().await;
        if trades.len() >= self.max_history {
            trades.pop_front();
        }
        trades.push_back(record);
    }

    pub async fn get_stats(&self) -> PerformanceStats {
        let trades = self.trades.read().await;
        
        let total_trades = trades.len();
        let mut winning = 0;
        let mut losing = 0;
        let mut total_pnl = Decimal::ZERO;

        for trade in trades.iter() {
            if let Some(pnl) = trade.pnl {
                total_pnl += pnl;
                if pnl > Decimal::ZERO {
                    winning += 1;
                } else if pnl < Decimal::ZERO {
                    losing += 1;
                }
            }
        }

        let win_rate = if total_trades > 0 {
            Decimal::from(winning) / Decimal::from(total_trades)
        } else {
            Decimal::ZERO
        };

        let avg_pnl = if total_trades > 0 {
            total_pnl / Decimal::from(total_trades)
        } else {
            Decimal::ZERO
        };

        PerformanceStats {
            total_trades,
            winning_trades: winning,
            losing_trades: losing,
            win_rate,
            total_pnl,
            avg_pnl_per_trade: avg_pnl,
            sharpe_ratio: None, // TODO: Calculate Sharpe ratio
        }
    }

    pub async fn log_stats(&self) {
        let stats = self.get_stats().await;
        tracing::info!(
            "Performance: {} trades, {:.1}% win rate, {:.2} total PnL",
            stats.total_trades,
            stats.win_rate * Decimal::ONE_HUNDRED,
            stats.total_pnl
        );
    }
}
