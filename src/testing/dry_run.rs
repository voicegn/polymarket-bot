//! Dry Run Simulation Module
//!
//! Simulates trading without executing real orders.
//! Collects performance metrics for analysis.

use crate::client::mock::{MockClobClient, MockGammaClient, ClobClientTrait, GammaClientTrait};
use crate::strategy::SignalGenerator;
use crate::config::{StrategyConfig, RiskConfig};
use crate::types::{Market, Signal, Side};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::MathematicalOps;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Result of a dry run simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_secs: i64,
    pub initial_balance: Decimal,
    pub final_balance: Decimal,
    pub total_pnl: Decimal,
    pub pnl_pct: Decimal,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub win_rate: Decimal,
    pub avg_win: Decimal,
    pub avg_loss: Decimal,
    pub max_drawdown: Decimal,
    pub sharpe_ratio: Decimal,
    pub trades: Vec<SimulatedTrade>,
    pub market_stats: HashMap<String, MarketStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedTrade {
    pub id: u32,
    pub timestamp: DateTime<Utc>,
    pub market_id: String,
    pub market_question: String,
    pub token_id: String,
    pub side: Side,
    pub entry_price: Decimal,
    pub exit_price: Option<Decimal>,
    pub size: Decimal,
    pub pnl: Decimal,
    pub pnl_pct: Decimal,
    pub edge: Decimal,
    pub confidence: Decimal,
    pub is_closed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MarketStats {
    pub trades: u32,
    pub wins: u32,
    pub losses: u32,
    pub total_pnl: Decimal,
    pub volume: Decimal,
}

/// Dry run simulator for testing strategies
pub struct DryRunSimulator {
    clob: MockClobClient,
    gamma: MockGammaClient,
    signal_gen: SignalGenerator,
    initial_balance: Decimal,
    trades: Vec<SimulatedTrade>,
    trade_counter: u32,
    market_prices: HashMap<String, Decimal>,
    start_time: DateTime<Utc>,
    balance_history: Vec<(DateTime<Utc>, Decimal)>,
    max_balance: Decimal,
    min_balance: Decimal,
}

impl DryRunSimulator {
    pub fn new(initial_balance: Decimal) -> Self {
        // Use relaxed config for simulation to generate more trades
        let strategy_config = StrategyConfig {
            min_edge: dec!(0.02),        // 2% edge (lower for testing)
            min_confidence: dec!(0.50),  // 50% confidence
            kelly_fraction: dec!(0.35),  // 35% Kelly
            scan_interval_secs: 180,
            model_update_interval_secs: 900,
            compound_enabled: true,
            compound_sqrt_scaling: true,
        };
        let risk_config = RiskConfig::default();
        
        Self {
            clob: MockClobClient::new().with_balance(initial_balance),
            gamma: MockGammaClient::new(),
            signal_gen: SignalGenerator::new(strategy_config, risk_config),
            initial_balance,
            trades: Vec::new(),
            trade_counter: 0,
            market_prices: HashMap::new(),
            start_time: Utc::now(),
            balance_history: vec![(Utc::now(), initial_balance)],
            max_balance: initial_balance,
            min_balance: initial_balance,
        }
    }

    pub fn with_markets(mut self, markets: Vec<Market>) -> Self {
        self.gamma = MockGammaClient::new().with_markets(markets);
        self
    }

    /// Run a single simulation step
    pub async fn step(&mut self) -> anyhow::Result<Vec<SimulatedTrade>> {
        let mut new_trades = Vec::new();
        
        // Get current markets
        let markets = self.gamma.get_top_markets(20).await?;
        let balance = self.clob.get_balance().await?;
        
        if markets.is_empty() {
            tracing::debug!("No markets available");
        }
        
        for market in &markets {
            // Skip low liquidity
            if market.liquidity < dec!(1000) {
                tracing::debug!("Skipping {} - low liquidity: {}", market.id, market.liquidity);
                continue;
            }

            // Generate mock prediction (simplified)
            let market_prob = market.yes_price().unwrap_or(dec!(0.5));
            let model_prob = self.generate_mock_prediction(&market);
            let edge = (model_prob - market_prob).abs();
            
            tracing::debug!(
                "Market {}: market_prob={}, model_prob={}, edge={}",
                market.id, market_prob, model_prob, edge
            );
            
            // Create mock prediction
            let prediction = crate::model::Prediction {
                probability: model_prob,
                confidence: dec!(0.80),  // High confidence
                reasoning: "Dry run simulation".to_string(),
            };

            // Generate signal
            if let Some(signal) = self.signal_gen.generate(&market, &prediction) {
                tracing::debug!("Signal generated for {}: {:?}", market.id, signal.side);
                if let Some(trade) = self.execute_simulated_trade(&signal, &market, balance).await? {
                    new_trades.push(trade);
                }
            } else {
                tracing::debug!("No signal for {}", market.id);
            }
        }

        // Update balance tracking
        let current_balance = self.clob.get_balance().await?;
        self.balance_history.push((Utc::now(), current_balance));
        self.max_balance = self.max_balance.max(current_balance);
        self.min_balance = self.min_balance.min(current_balance);

        Ok(new_trades)
    }

    /// Execute a simulated trade
    async fn execute_simulated_trade(
        &mut self,
        signal: &Signal,
        market: &Market,
        balance: Decimal,
    ) -> anyhow::Result<Option<SimulatedTrade>> {
        let size_usd = signal.suggested_size * balance;
        
        // Minimum trade size
        if size_usd < dec!(1) {
            return Ok(None);
        }

        // Place order via mock client
        let order = crate::types::Order {
            token_id: signal.token_id.clone(),
            side: signal.side,
            price: signal.market_probability,
            size: size_usd / signal.market_probability,
            order_type: crate::types::OrderType::GTC,
        };

        match self.clob.place_order(&order).await {
            Ok(_status) => {
                self.trade_counter += 1;
                
                // Simulate exit (random profit/loss based on edge)
                use rand::Rng;
                let mut rng = rand::rng();
                let edge_factor = signal.edge;
                let random_val: f64 = rng.random_range(-0.05..0.05);
                let random_factor = Decimal::from_f64_retain(random_val).unwrap_or(dec!(0));
                let exit_prob = signal.market_probability + edge_factor + random_factor;
                let exit_prob = exit_prob.max(dec!(0.01)).min(dec!(0.99));
                
                let pnl_pct = match signal.side {
                    Side::Buy => (exit_prob - signal.market_probability) / signal.market_probability,
                    Side::Sell => (signal.market_probability - exit_prob) / signal.market_probability,
                };
                let pnl = size_usd * pnl_pct;

                let trade = SimulatedTrade {
                    id: self.trade_counter,
                    timestamp: Utc::now(),
                    market_id: market.id.clone(),
                    market_question: market.question.clone(),
                    token_id: signal.token_id.clone(),
                    side: signal.side,
                    entry_price: signal.market_probability,
                    exit_price: Some(exit_prob),
                    size: size_usd,
                    pnl,
                    pnl_pct,
                    edge: signal.edge,
                    confidence: signal.confidence,
                    is_closed: true,
                };

                self.trades.push(trade.clone());
                self.market_prices.insert(signal.token_id.clone(), exit_prob);

                Ok(Some(trade))
            }
            Err(e) => {
                tracing::warn!("Simulated order failed: {}", e);
                Ok(None)
            }
        }
    }

    /// Generate mock model prediction with real randomness
    fn generate_mock_prediction(&self, market: &Market) -> Decimal {
        use rand::Rng;
        let base = market.yes_price().unwrap_or(dec!(0.5));
        // Add variance to simulate model disagreement (±15% range)
        let mut rng = rand::rng();
        let random_val: f64 = rng.random_range(-0.15..0.15);
        let variance = Decimal::from_f64_retain(random_val).unwrap_or(dec!(0));
        (base + variance).max(dec!(0.05)).min(dec!(0.95))
    }

    /// Run simulation for specified duration (simulated time)
    pub async fn run_for(&mut self, steps: u32, step_delay_ms: u64) -> anyhow::Result<()> {
        for _ in 0..steps {
            self.step().await?;
            if step_delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(step_delay_ms)).await;
            }
        }
        Ok(())
    }

    /// Get simulation results
    pub async fn get_results(&self) -> anyhow::Result<SimulationResult> {
        let final_balance = self.clob.get_balance().await?;
        let total_pnl = final_balance - self.initial_balance;
        let pnl_pct = if self.initial_balance > Decimal::ZERO {
            total_pnl / self.initial_balance * dec!(100)
        } else {
            Decimal::ZERO
        };

        let winning: Vec<_> = self.trades.iter().filter(|t| t.pnl > Decimal::ZERO).collect();
        let losing: Vec<_> = self.trades.iter().filter(|t| t.pnl < Decimal::ZERO).collect();

        let win_rate = if !self.trades.is_empty() {
            Decimal::from(winning.len() as i64) / Decimal::from(self.trades.len() as i64) * dec!(100)
        } else {
            Decimal::ZERO
        };

        let avg_win = if !winning.is_empty() {
            winning.iter().map(|t| t.pnl).sum::<Decimal>() / Decimal::from(winning.len() as i64)
        } else {
            Decimal::ZERO
        };

        let avg_loss = if !losing.is_empty() {
            losing.iter().map(|t| t.pnl.abs()).sum::<Decimal>() / Decimal::from(losing.len() as i64)
        } else {
            Decimal::ZERO
        };

        // Calculate max drawdown
        let max_drawdown = if self.max_balance > Decimal::ZERO {
            (self.max_balance - self.min_balance) / self.max_balance * dec!(100)
        } else {
            Decimal::ZERO
        };

        // Calculate Sharpe (simplified)
        let sharpe_ratio = self.calculate_sharpe();

        // Market stats
        let mut market_stats: HashMap<String, MarketStats> = HashMap::new();
        for trade in &self.trades {
            let stats = market_stats.entry(trade.market_id.clone()).or_default();
            stats.trades += 1;
            if trade.pnl > Decimal::ZERO {
                stats.wins += 1;
            } else if trade.pnl < Decimal::ZERO {
                stats.losses += 1;
            }
            stats.total_pnl += trade.pnl;
            stats.volume += trade.size;
        }

        Ok(SimulationResult {
            start_time: self.start_time,
            end_time: Utc::now(),
            duration_secs: (Utc::now() - self.start_time).num_seconds(),
            initial_balance: self.initial_balance,
            final_balance,
            total_pnl,
            pnl_pct,
            total_trades: self.trades.len() as u32,
            winning_trades: winning.len() as u32,
            losing_trades: losing.len() as u32,
            win_rate,
            avg_win,
            avg_loss,
            max_drawdown,
            sharpe_ratio,
            trades: self.trades.clone(),
            market_stats,
        })
    }

    fn calculate_sharpe(&self) -> Decimal {
        if self.balance_history.len() < 2 {
            return Decimal::ZERO;
        }

        let returns: Vec<Decimal> = self.balance_history
            .windows(2)
            .map(|w| {
                if w[0].1 > Decimal::ZERO {
                    (w[1].1 - w[0].1) / w[0].1
                } else {
                    Decimal::ZERO
                }
            })
            .collect();

        if returns.is_empty() {
            return Decimal::ZERO;
        }

        let avg_return = returns.iter().sum::<Decimal>() / Decimal::from(returns.len() as i64);
        let variance: Decimal = returns
            .iter()
            .map(|r| (*r - avg_return) * (*r - avg_return))
            .sum::<Decimal>()
            / Decimal::from(returns.len() as i64);

        let std_dev = variance.sqrt().unwrap_or(dec!(0.0001));
        if std_dev > Decimal::ZERO {
            avg_return / std_dev * dec!(15.87) // Annualized (sqrt(252))
        } else {
            Decimal::ZERO
        }
    }

    /// Generate report in Markdown format
    pub fn generate_report(&self, result: &SimulationResult) -> String {
        let mut report = String::new();
        
        report.push_str("# 🧪 Dry Run Simulation Report\n\n");
        report.push_str(&format!("**Generated**: {}\n\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        report.push_str("## 📊 Summary\n\n");
        report.push_str("| Metric | Value |\n");
        report.push_str("|--------|-------|\n");
        report.push_str(&format!("| Duration | {} seconds |\n", result.duration_secs));
        report.push_str(&format!("| Initial Balance | ${:.2} |\n", result.initial_balance));
        report.push_str(&format!("| Final Balance | ${:.2} |\n", result.final_balance));
        report.push_str(&format!("| Total P&L | ${:.2} ({:.2}%) |\n", result.total_pnl, result.pnl_pct));
        report.push_str(&format!("| Total Trades | {} |\n", result.total_trades));
        report.push_str(&format!("| Win Rate | {:.1}% |\n", result.win_rate));
        report.push_str(&format!("| Avg Win | ${:.2} |\n", result.avg_win));
        report.push_str(&format!("| Avg Loss | ${:.2} |\n", result.avg_loss));
        report.push_str(&format!("| Max Drawdown | {:.2}% |\n", result.max_drawdown));
        report.push_str(&format!("| Sharpe Ratio | {:.2} |\n", result.sharpe_ratio));
        
        report.push_str("\n## 🏆 Performance by Market\n\n");
        report.push_str("| Market | Trades | Wins | Losses | P&L | Volume |\n");
        report.push_str("|--------|--------|------|--------|-----|--------|\n");
        
        for (market_id, stats) in &result.market_stats {
            let short_id = if market_id.len() > 20 {
                format!("{}...", &market_id[..17])
            } else {
                market_id.clone()
            };
            report.push_str(&format!(
                "| {} | {} | {} | {} | ${:.2} | ${:.2} |\n",
                short_id, stats.trades, stats.wins, stats.losses, stats.total_pnl, stats.volume
            ));
        }
        
        if !result.trades.is_empty() {
            report.push_str("\n## 📝 Trade Log (Last 20)\n\n");
            report.push_str("| # | Time | Market | Side | Size | P&L | Edge |\n");
            report.push_str("|---|------|--------|------|------|-----|------|\n");
            
            for trade in result.trades.iter().rev().take(20) {
                let short_q = if trade.market_question.len() > 30 {
                    format!("{}...", &trade.market_question[..27])
                } else {
                    trade.market_question.clone()
                };
                report.push_str(&format!(
                    "| {} | {} | {} | {:?} | ${:.2} | ${:.2} | {:.1}% |\n",
                    trade.id,
                    trade.timestamp.format("%H:%M:%S"),
                    short_q,
                    trade.side,
                    trade.size,
                    trade.pnl,
                    trade.edge * dec!(100)
                ));
            }
        }

        report.push_str("\n---\n*Generated by Polymarket Bot Dry Run Simulator*\n");
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dry_run_simulator_creation() {
        let sim = DryRunSimulator::new(dec!(1000));
        let results = sim.get_results().await.unwrap();
        assert_eq!(results.initial_balance, dec!(1000));
        assert_eq!(results.total_trades, 0);
    }

    #[tokio::test]
    async fn test_dry_run_step() {
        let mut sim = DryRunSimulator::new(dec!(1000));
        let trades = sim.step().await.unwrap();
        // May or may not generate trades depending on signals
        assert!(trades.len() <= 20); // Max markets scanned
    }

    #[tokio::test]
    async fn test_dry_run_multiple_steps() {
        let mut sim = DryRunSimulator::new(dec!(1000));
        sim.run_for(5, 0).await.unwrap();
        let results = sim.get_results().await.unwrap();
        assert!(results.duration_secs >= 0);
    }

    #[tokio::test]
    async fn test_report_generation() {
        let mut sim = DryRunSimulator::new(dec!(1000));
        sim.run_for(3, 0).await.unwrap();
        let results = sim.get_results().await.unwrap();
        let report = sim.generate_report(&results);
        assert!(report.contains("Dry Run Simulation Report"));
        assert!(report.contains("Summary"));
    }
}
