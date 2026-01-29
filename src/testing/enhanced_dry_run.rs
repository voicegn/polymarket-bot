//! Enhanced Dry Run Simulation
//!
//! Comprehensive simulation with:
//! - Full lifecycle trading simulation
//! - Multiple strategy testing
//! - Performance attribution
//! - Edge case handling

use crate::client::mock::{MockClobClient, MockGammaClient, ClobClientTrait, GammaClientTrait};
use crate::config::{RiskConfig, StrategyConfig};
use crate::strategy::{SignalGenerator, DynamicKelly, DynamicKellyConfig};
use crate::types::{Market, Side};
use crate::model::Prediction;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Comprehensive dry run simulation configuration
#[derive(Debug, Clone)]
pub struct EnhancedDryRunConfig {
    pub initial_balance: Decimal,
    pub steps: u32,
    pub use_dynamic_kelly: bool,
    pub market_volatility: Decimal,
    pub simulate_slippage: bool,
    pub slippage_factor: Decimal,
    pub simulate_partial_fills: bool,
    pub partial_fill_prob: Decimal,
    pub simulate_failures: bool,
    pub failure_prob: Decimal,
}

impl Default for EnhancedDryRunConfig {
    fn default() -> Self {
        Self {
            initial_balance: dec!(10000),
            steps: 100,
            use_dynamic_kelly: true,
            market_volatility: dec!(0.05),
            simulate_slippage: true,
            slippage_factor: dec!(0.005),
            simulate_partial_fills: true,
            partial_fill_prob: dec!(0.20),
            simulate_failures: true,
            failure_prob: dec!(0.05),
        }
    }
}

/// Enhanced simulation result with detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSimResult {
    pub initial_balance: Decimal,
    pub final_balance: Decimal,
    pub total_pnl: Decimal,
    pub pnl_pct: Decimal,
    pub total_trades: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub win_rate: Decimal,
    pub max_drawdown: Decimal,
    pub sharpe_ratio: Decimal,
    pub sortino_ratio: Decimal,
    pub profit_factor: Decimal,
    pub total_slippage: Decimal,
    pub avg_slippage: Decimal,
    pub failed_orders: u32,
    pub partial_fills: u32,
    pub signals_generated: u32,
    pub signals_filtered: u32,
    pub trades: Vec<EnhancedSimTrade>,
    pub equity_curve: Vec<(u32, Decimal)>,
    pub pnl_by_market: HashMap<String, Decimal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedSimTrade {
    pub id: u32,
    pub step: u32,
    pub market_id: String,
    pub market_question: String,
    pub side: Side,
    pub intended_size: Decimal,
    pub executed_size: Decimal,
    pub entry_price: Decimal,
    pub exit_price: Option<Decimal>,
    pub slippage: Decimal,
    pub pnl: Decimal,
    pub edge: Decimal,
    pub confidence: Decimal,
    pub kelly_fraction: Decimal,
    pub partial_fill: bool,
    pub status: TradeStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TradeStatus {
    Completed,
    PartialFill,
    Failed,
    Open,
}

#[derive(Debug, Clone)]
struct OpenPosition {
    market_id: String,
    side: Side,
    size: Decimal,
    entry_price: Decimal,
    entry_step: u32,
    edge: Decimal,
}

/// Enhanced dry run simulator
pub struct EnhancedDryRun {
    config: EnhancedDryRunConfig,
    clob: MockClobClient,
    gamma: MockGammaClient,
    signal_gen: SignalGenerator,
    kelly: Option<DynamicKelly>,
    current_balance: Decimal,
    current_step: u32,
    trades: Vec<EnhancedSimTrade>,
    trade_counter: u32,
    open_positions: HashMap<String, OpenPosition>,
    equity_curve: Vec<(u32, Decimal)>,
    peak_balance: Decimal,
    max_drawdown: Decimal,
    signals_generated: u32,
    signals_filtered: u32,
    failed_orders: u32,
    partial_fills: u32,
    total_slippage: Decimal,
    random_seed: u64,
}

impl EnhancedDryRun {
    pub fn new(config: EnhancedDryRunConfig) -> Self {
        let strategy_config = StrategyConfig::default();
        let risk_config = RiskConfig::default();
        
        let kelly = if config.use_dynamic_kelly {
            Some(DynamicKelly::new(DynamicKellyConfig::default(), config.initial_balance))
        } else {
            None
        };
        
        Self {
            config: config.clone(),
            clob: MockClobClient::new().with_balance(config.initial_balance),
            gamma: MockGammaClient::new(),
            signal_gen: SignalGenerator::new(strategy_config, risk_config),
            kelly,
            current_balance: config.initial_balance,
            current_step: 0,
            trades: Vec::new(),
            trade_counter: 0,
            open_positions: HashMap::new(),
            equity_curve: vec![(0, config.initial_balance)],
            peak_balance: config.initial_balance,
            max_drawdown: dec!(0),
            signals_generated: 0,
            signals_filtered: 0,
            failed_orders: 0,
            partial_fills: 0,
            total_slippage: dec!(0),
            random_seed: 42,
        }
    }

    pub fn with_markets(mut self, markets: Vec<Market>) -> Self {
        self.gamma = MockGammaClient::new().with_markets(markets);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    pub async fn run(&mut self) -> anyhow::Result<EnhancedSimResult> {
        for _ in 0..self.config.steps {
            self.step().await?;
        }
        self.close_all_positions().await?;
        Ok(self.compile_results())
    }

    pub async fn step(&mut self) -> anyhow::Result<()> {
        self.current_step += 1;
        self.update_position_pnl().await?;
        
        let markets = self.gamma.get_top_markets(20).await?;
        
        for market in &markets {
            if market.liquidity < dec!(1000) {
                continue;
            }
            
            let prediction = self.generate_prediction(market);
            
            if let Some(signal) = self.signal_gen.generate(market, &prediction) {
                self.signals_generated += 1;
                self.process_signal(&signal, market).await?;
            }
        }
        
        let total_equity = self.calculate_total_equity();
        self.equity_curve.push((self.current_step, total_equity));
        self.update_drawdown(total_equity);
        self.consider_exits().await?;
        
        Ok(())
    }

    async fn process_signal(&mut self, signal: &crate::types::Signal, market: &Market) -> anyhow::Result<()> {
        let kelly_fraction = if let Some(ref kelly) = self.kelly {
            let result = kelly.calculate_position_size(
                signal.model_probability,
                signal.market_probability,
                signal.confidence,
                dec!(1.0),
                None,
            );
            result.effective_fraction
        } else {
            signal.suggested_size
        };
        
        let intended_size = kelly_fraction * self.current_balance;
        
        if intended_size < dec!(1) {
            self.signals_filtered += 1;
            return Ok(());
        }
        
        let (executed_size, slippage, status) = self.simulate_execution(intended_size).await;
        
        if status == TradeStatus::Failed {
            self.failed_orders += 1;
            self.add_failed_trade(signal, market, intended_size);
            return Ok(());
        }
        
        if status == TradeStatus::PartialFill {
            self.partial_fills += 1;
        }
        
        self.total_slippage += slippage;
        let cost = executed_size + slippage;
        self.current_balance -= cost;
        
        self.trade_counter += 1;
        let position = OpenPosition {
            market_id: market.id.clone(),
            side: signal.side,
            size: executed_size,
            entry_price: signal.market_probability + slippage / executed_size.max(dec!(0.001)),
            entry_step: self.current_step,
            edge: signal.edge,
        };
        
        self.open_positions.insert(signal.token_id.clone(), position);
        
        self.trades.push(EnhancedSimTrade {
            id: self.trade_counter,
            step: self.current_step,
            market_id: market.id.clone(),
            market_question: market.question.clone(),
            side: signal.side,
            intended_size,
            executed_size,
            entry_price: signal.market_probability,
            exit_price: None,
            slippage,
            pnl: dec!(0),
            edge: signal.edge,
            confidence: signal.confidence,
            kelly_fraction,
            partial_fill: status == TradeStatus::PartialFill,
            status: TradeStatus::Open,
        });
        
        if let Some(ref kelly) = self.kelly {
            kelly.update_account_value(self.calculate_total_equity());
        }
        
        Ok(())
    }

    async fn simulate_execution(&mut self, intended_size: Decimal) -> (Decimal, Decimal, TradeStatus) {
        if self.config.simulate_failures && self.random() < self.config.failure_prob {
            return (dec!(0), dec!(0), TradeStatus::Failed);
        }
        
        let slippage = if self.config.simulate_slippage {
            intended_size * self.config.slippage_factor * (dec!(0.5) + self.random())
        } else {
            dec!(0)
        };
        
        let executed_size = if self.config.simulate_partial_fills && self.random() < self.config.partial_fill_prob {
            intended_size * (dec!(0.5) + self.random() * dec!(0.5))
        } else {
            intended_size
        };
        
        let status = if executed_size < intended_size {
            TradeStatus::PartialFill
        } else {
            TradeStatus::Completed
        };
        
        (executed_size, slippage, status)
    }

    async fn update_position_pnl(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn consider_exits(&mut self) -> anyhow::Result<()> {
        // Collect positions to potentially close
        let positions_info: Vec<(String, u32)> = self.open_positions
            .iter()
            .map(|(token_id, pos)| (token_id.clone(), pos.entry_step))
            .collect();
        
        let mut to_close = Vec::new();
        for (token_id, entry_step) in positions_info {
            let steps_held = self.current_step - entry_step;
            let should_close = steps_held > 10 || self.random() < dec!(0.1);
            if should_close {
                to_close.push(token_id);
            }
        }
        
        for token_id in to_close {
            self.close_position(&token_id).await?;
        }
        
        Ok(())
    }

    async fn close_position(&mut self, token_id: &str) -> anyhow::Result<()> {
        if let Some(position) = self.open_positions.remove(token_id) {
            let noise = (self.random() - dec!(0.5)) * dec!(0.1);
            let exit_price = (position.entry_price + position.edge + noise).max(dec!(0.01)).min(dec!(0.99));
            
            let pnl = match position.side {
                Side::Buy => (exit_price - position.entry_price) * position.size,
                Side::Sell => (position.entry_price - exit_price) * position.size,
            };
            
            self.current_balance += position.size + pnl;
            
            for trade in self.trades.iter_mut().rev() {
                if trade.market_id == position.market_id && trade.status == TradeStatus::Open {
                    trade.exit_price = Some(exit_price);
                    trade.pnl = pnl;
                    trade.status = TradeStatus::Completed;
                    break;
                }
            }
            
            if let Some(ref kelly) = self.kelly {
                kelly.record_trade(pnl);
                kelly.update_account_value(self.calculate_total_equity());
            }
        }
        
        Ok(())
    }

    async fn close_all_positions(&mut self) -> anyhow::Result<()> {
        let tokens: Vec<String> = self.open_positions.keys().cloned().collect();
        for token in tokens {
            self.close_position(&token).await?;
        }
        Ok(())
    }

    fn calculate_total_equity(&self) -> Decimal {
        let position_value: Decimal = self.open_positions.values().map(|p| p.size).sum();
        self.current_balance + position_value
    }

    fn update_drawdown(&mut self, equity: Decimal) {
        if equity > self.peak_balance {
            self.peak_balance = equity;
        } else {
            let drawdown = (self.peak_balance - equity) / self.peak_balance;
            self.max_drawdown = self.max_drawdown.max(drawdown);
        }
    }

    fn generate_prediction(&mut self, market: &Market) -> Prediction {
        let base = market.yes_price().unwrap_or(dec!(0.5));
        let variance = (self.random() - dec!(0.5)) * dec!(0.20);
        let prob = (base + variance).max(dec!(0.05)).min(dec!(0.95));
        
        Prediction {
            probability: prob,
            confidence: dec!(0.5) + self.random() * dec!(0.5),
            reasoning: "Dry run simulation".to_string(),
        }
    }

    fn add_failed_trade(&mut self, signal: &crate::types::Signal, market: &Market, intended_size: Decimal) {
        self.trade_counter += 1;
        self.trades.push(EnhancedSimTrade {
            id: self.trade_counter,
            step: self.current_step,
            market_id: market.id.clone(),
            market_question: market.question.clone(),
            side: signal.side,
            intended_size,
            executed_size: dec!(0),
            entry_price: signal.market_probability,
            exit_price: None,
            slippage: dec!(0),
            pnl: dec!(0),
            edge: signal.edge,
            confidence: signal.confidence,
            kelly_fraction: dec!(0),
            partial_fill: false,
            status: TradeStatus::Failed,
        });
    }

    fn random(&mut self) -> Decimal {
        self.random_seed = self.random_seed.wrapping_mul(1103515245).wrapping_add(12345);
        Decimal::from(self.random_seed % 1000) / dec!(1000)
    }

    fn compile_results(&self) -> EnhancedSimResult {
        let completed_trades: Vec<_> = self.trades.iter()
            .filter(|t| t.status == TradeStatus::Completed)
            .collect();
        
        let winning_trades = completed_trades.iter().filter(|t| t.pnl > dec!(0)).count() as u32;
        let losing_trades = completed_trades.iter().filter(|t| t.pnl < dec!(0)).count() as u32;
        
        let win_rate = if !completed_trades.is_empty() {
            Decimal::from(winning_trades) / Decimal::from(completed_trades.len() as u32)
        } else {
            dec!(0)
        };
        
        let total_pnl = self.current_balance - self.config.initial_balance;
        let pnl_pct = total_pnl / self.config.initial_balance * dec!(100);
        
        let gross_profit: Decimal = completed_trades.iter()
            .filter(|t| t.pnl > dec!(0)).map(|t| t.pnl).sum();
        let gross_loss: Decimal = completed_trades.iter()
            .filter(|t| t.pnl < dec!(0)).map(|t| t.pnl.abs()).sum();
        
        let profit_factor = if gross_loss > dec!(0) {
            gross_profit / gross_loss
        } else if gross_profit > dec!(0) {
            dec!(999)
        } else {
            dec!(0)
        };
        
        let sharpe_ratio = self.calculate_sharpe();
        let sortino_ratio = self.calculate_sortino();
        
        let mut pnl_by_market: HashMap<String, Decimal> = HashMap::new();
        for trade in &self.trades {
            if trade.status == TradeStatus::Completed {
                *pnl_by_market.entry(trade.market_id.clone()).or_default() += trade.pnl;
            }
        }
        
        let avg_slippage = if !self.trades.is_empty() {
            self.total_slippage / Decimal::from(self.trades.len() as u32)
        } else {
            dec!(0)
        };
        
        EnhancedSimResult {
            initial_balance: self.config.initial_balance,
            final_balance: self.current_balance,
            total_pnl,
            pnl_pct,
            total_trades: self.trades.len() as u32,
            winning_trades,
            losing_trades,
            win_rate,
            max_drawdown: self.max_drawdown * dec!(100),
            sharpe_ratio,
            sortino_ratio,
            profit_factor,
            total_slippage: self.total_slippage,
            avg_slippage,
            failed_orders: self.failed_orders,
            partial_fills: self.partial_fills,
            signals_generated: self.signals_generated,
            signals_filtered: self.signals_filtered,
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
            pnl_by_market,
        }
    }

    fn calculate_sharpe(&self) -> Decimal {
        if self.equity_curve.len() < 2 {
            return dec!(0);
        }

        let returns: Vec<Decimal> = self.equity_curve
            .windows(2)
            .filter_map(|w| {
                if w[0].1 > Decimal::ZERO {
                    Some((w[1].1 - w[0].1) / w[0].1)
                } else {
                    None
                }
            })
            .collect();

        if returns.is_empty() {
            return dec!(0);
        }

        let avg_return = returns.iter().sum::<Decimal>() / Decimal::from(returns.len() as u32);
        let variance: Decimal = returns.iter()
            .map(|r| (*r - avg_return) * (*r - avg_return))
            .sum::<Decimal>() / Decimal::from(returns.len() as u32);

        let std_dev = crate::utils::sqrt_decimal(variance);
        if std_dev > Decimal::ZERO {
            avg_return / std_dev * dec!(15.87)
        } else {
            dec!(0)
        }
    }

    fn calculate_sortino(&self) -> Decimal {
        if self.equity_curve.len() < 2 {
            return dec!(0);
        }

        let returns: Vec<Decimal> = self.equity_curve
            .windows(2)
            .filter_map(|w| {
                if w[0].1 > Decimal::ZERO {
                    Some((w[1].1 - w[0].1) / w[0].1)
                } else {
                    None
                }
            })
            .collect();

        if returns.is_empty() {
            return dec!(0);
        }

        let avg_return = returns.iter().sum::<Decimal>() / Decimal::from(returns.len() as u32);
        let downside_variance: Decimal = returns.iter()
            .filter(|r| **r < Decimal::ZERO)
            .map(|r| r * r)
            .sum::<Decimal>() / Decimal::from(returns.len() as u32);

        let downside_dev = crate::utils::sqrt_decimal(downside_variance);
        if downside_dev > Decimal::ZERO {
            avg_return / downside_dev * dec!(15.87)
        } else if avg_return > Decimal::ZERO {
            dec!(999)
        } else {
            dec!(0)
        }
    }

    pub fn generate_report(&self, result: &EnhancedSimResult) -> String {
        let mut report = String::new();
        
        report.push_str("# 🧪 Enhanced Dry Run Report\n\n");
        report.push_str(&format!("**Generated**: {}\n\n", Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        
        report.push_str("## 📊 Performance Summary\n\n");
        report.push_str("| Metric | Value |\n|--------|-------|\n");
        report.push_str(&format!("| Initial Balance | ${:.2} |\n", result.initial_balance));
        report.push_str(&format!("| Final Balance | ${:.2} |\n", result.final_balance));
        report.push_str(&format!("| Total P&L | ${:.2} ({:+.2}%) |\n", result.total_pnl, result.pnl_pct));
        report.push_str(&format!("| Sharpe Ratio | {:.2} |\n", result.sharpe_ratio));
        report.push_str(&format!("| Sortino Ratio | {:.2} |\n", result.sortino_ratio));
        report.push_str(&format!("| Max Drawdown | {:.2}% |\n", result.max_drawdown));
        report.push_str(&format!("| Profit Factor | {:.2} |\n", result.profit_factor));
        
        report.push_str("\n## 📈 Trade Statistics\n\n");
        report.push_str("| Metric | Value |\n|--------|-------|\n");
        report.push_str(&format!("| Total Trades | {} |\n", result.total_trades));
        report.push_str(&format!("| Winning Trades | {} |\n", result.winning_trades));
        report.push_str(&format!("| Losing Trades | {} |\n", result.losing_trades));
        report.push_str(&format!("| Win Rate | {:.1}% |\n", result.win_rate * dec!(100)));
        
        report.push_str("\n## ⚡ Execution Quality\n\n");
        report.push_str("| Metric | Value |\n|--------|-------|\n");
        report.push_str(&format!("| Total Slippage | ${:.2} |\n", result.total_slippage));
        report.push_str(&format!("| Failed Orders | {} |\n", result.failed_orders));
        report.push_str(&format!("| Partial Fills | {} |\n", result.partial_fills));
        
        report.push_str("\n## 📡 Signal Analysis\n\n");
        report.push_str("| Metric | Value |\n|--------|-------|\n");
        report.push_str(&format!("| Signals Generated | {} |\n", result.signals_generated));
        report.push_str(&format!("| Signals Filtered | {} |\n", result.signals_filtered));
        
        report.push_str("\n---\n*Enhanced Dry Run Simulator - Polymarket Bot*\n");
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_enhanced_dry_run_basic() {
        let config = EnhancedDryRunConfig {
            initial_balance: dec!(10000),
            steps: 10,
            simulate_failures: false,
            simulate_slippage: false,
            simulate_partial_fills: false,
            ..Default::default()
        };
        
        let mut sim = EnhancedDryRun::new(config);
        let result = sim.run().await.unwrap();
        
        assert_eq!(result.initial_balance, dec!(10000));
        assert!(!result.equity_curve.is_empty());
    }

    #[tokio::test]
    async fn test_enhanced_dry_run_with_failures() {
        let config = EnhancedDryRunConfig {
            initial_balance: dec!(10000),
            steps: 20,
            simulate_failures: true,
            failure_prob: dec!(0.30),
            ..Default::default()
        };
        
        let mut sim = EnhancedDryRun::new(config);
        let result = sim.run().await.unwrap();
        
        println!("Failed orders: {}", result.failed_orders);
    }

    #[tokio::test]
    async fn test_enhanced_dry_run_deterministic() {
        let config = EnhancedDryRunConfig {
            initial_balance: dec!(10000),
            steps: 50,
            simulate_failures: false,  // Disable random failures for determinism
            simulate_partial_fills: false,  // Disable random partial fills
            ..Default::default()
        };
        
        let mut sim1 = EnhancedDryRun::new(config.clone()).with_seed(12345);
        let result1 = sim1.run().await.unwrap();
        
        let mut sim2 = EnhancedDryRun::new(config).with_seed(12345);
        let result2 = sim2.run().await.unwrap();
        
        // Trade counts should match with same seed and no randomness
        assert_eq!(result1.total_trades, result2.total_trades);
        // Check equity curve has same length
        assert_eq!(result1.equity_curve.len(), result2.equity_curve.len());
    }

    #[tokio::test]
    async fn test_report_generation() {
        let config = EnhancedDryRunConfig {
            initial_balance: dec!(10000),
            steps: 20,
            ..Default::default()
        };
        
        let mut sim = EnhancedDryRun::new(config);
        let result = sim.run().await.unwrap();
        let report = sim.generate_report(&result);
        
        assert!(report.contains("Enhanced Dry Run Report"));
        assert!(report.contains("Performance Summary"));
        assert!(report.contains("Trade Statistics"));
    }

    #[tokio::test]
    async fn test_equity_curve_tracking() {
        let config = EnhancedDryRunConfig {
            initial_balance: dec!(10000),
            steps: 50,
            ..Default::default()
        };
        
        let mut sim = EnhancedDryRun::new(config);
        let result = sim.run().await.unwrap();
        
        assert_eq!(result.equity_curve.len(), 51);
        assert_eq!(result.equity_curve[0], (0, dec!(10000)));
    }

    #[tokio::test]
    async fn test_slippage_simulation() {
        let config = EnhancedDryRunConfig {
            initial_balance: dec!(10000),
            steps: 30,
            simulate_slippage: true,
            slippage_factor: dec!(0.02),
            simulate_failures: false,
            ..Default::default()
        };
        
        let mut sim = EnhancedDryRun::new(config);
        let result = sim.run().await.unwrap();
        
        if result.total_trades > 0 {
            println!("Total slippage: ${:.4}", result.total_slippage);
        }
    }
}
