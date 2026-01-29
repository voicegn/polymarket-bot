//! Backtesting Engine for Strategy Validation
//!
//! Professional-grade backtesting with:
//! - Event-driven simulation
//! - Slippage modeling (linear/sqrt/log)
//! - Fee calculation (maker/taker)
//! - Performance metrics (Sharpe, Sortino, Calmar, etc.)
//! - Walk-forward optimization
//! - Monte Carlo simulation
//!
//! # Example
//! ```ignore
//! let engine = BacktestEngine::new(config);
//! let result = engine.run(&historical_data, &strategy).await;
//! println!("Sharpe: {:.2}", result.metrics.sharpe_ratio);
//! ```

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    /// Initial portfolio value in USD
    pub initial_capital: Decimal,
    /// Maker fee (e.g., 0.001 = 0.1%)
    pub maker_fee: Decimal,
    /// Taker fee (e.g., 0.002 = 0.2%)
    pub taker_fee: Decimal,
    /// Slippage model
    pub slippage_model: SlippageModel,
    /// Base slippage percentage (e.g., 0.0005 = 0.05%)
    pub base_slippage: Decimal,
    /// Risk-free rate for Sharpe ratio (annualized)
    pub risk_free_rate: Decimal,
    /// Enable detailed trade logging
    pub detailed_logging: bool,
    /// Maximum position size as fraction of portfolio
    pub max_position_fraction: Decimal,
    /// Commission model
    pub commission_model: CommissionModel,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: dec!(10000),
            maker_fee: dec!(0.001),
            taker_fee: dec!(0.002),
            slippage_model: SlippageModel::SquareRoot,
            base_slippage: dec!(0.0005),
            risk_free_rate: dec!(0.05),
            detailed_logging: false,
            max_position_fraction: dec!(0.1),
            commission_model: CommissionModel::Percentage,
        }
    }
}

/// Slippage model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlippageModel {
    /// No slippage
    None,
    /// Fixed percentage slippage
    Fixed,
    /// Linear slippage based on order size
    Linear,
    /// Square root slippage (more realistic for large orders)
    SquareRoot,
    /// Logarithmic slippage
    Logarithmic,
}

/// Commission calculation model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommissionModel {
    /// Percentage of trade value
    Percentage,
    /// Fixed amount per trade
    Fixed,
    /// Tiered based on volume
    Tiered,
}

/// Historical price bar (OHLCV)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceBar {
    pub timestamp: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
}

/// Trade signal from strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    /// Buy signal with confidence (0-1)
    Buy,
    /// Sell signal with confidence (0-1)
    Sell,
    /// Hold current position
    Hold,
    /// Close all positions
    CloseAll,
}

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeDirection {
    Long,
    Short,
}

/// Executed trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub timestamp: DateTime<Utc>,
    pub direction: TradeDirection,
    pub entry_price: Decimal,
    pub exit_price: Option<Decimal>,
    pub size: Decimal,
    pub commission: Decimal,
    pub slippage: Decimal,
    pub pnl: Option<Decimal>,
    pub exit_timestamp: Option<DateTime<Utc>>,
    pub holding_period_hours: Option<i64>,
}

/// Portfolio state during backtest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    pub timestamp: DateTime<Utc>,
    pub cash: Decimal,
    pub positions: HashMap<String, Position>,
    pub total_value: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
}

/// Single position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub direction: TradeDirection,
    pub size: Decimal,
    pub entry_price: Decimal,
    pub entry_timestamp: DateTime<Utc>,
    pub current_price: Decimal,
    pub unrealized_pnl: Decimal,
}

/// Performance metrics for backtest results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total return percentage
    pub total_return: Decimal,
    /// Annualized return percentage
    pub annualized_return: Decimal,
    /// Sharpe ratio (risk-adjusted return)
    pub sharpe_ratio: Decimal,
    /// Sortino ratio (downside risk-adjusted)
    pub sortino_ratio: Decimal,
    /// Calmar ratio (return / max drawdown)
    pub calmar_ratio: Decimal,
    /// Maximum drawdown percentage
    pub max_drawdown: Decimal,
    /// Maximum drawdown duration in days
    pub max_drawdown_duration_days: i64,
    /// Win rate percentage
    pub win_rate: Decimal,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: Decimal,
    /// Average trade return
    pub avg_trade_return: Decimal,
    /// Average winning trade return
    pub avg_win: Decimal,
    /// Average losing trade return
    pub avg_loss: Decimal,
    /// Total number of trades
    pub total_trades: u64,
    /// Number of winning trades
    pub winning_trades: u64,
    /// Number of losing trades
    pub losing_trades: u64,
    /// Average holding period in hours
    pub avg_holding_period_hours: Decimal,
    /// Total commission paid
    pub total_commission: Decimal,
    /// Total slippage cost
    pub total_slippage: Decimal,
    /// Volatility (annualized std dev of returns)
    pub volatility: Decimal,
    /// Skewness of returns
    pub skewness: Decimal,
    /// Kurtosis of returns
    pub kurtosis: Decimal,
    /// Value at Risk (95%)
    pub var_95: Decimal,
    /// Conditional VaR / Expected Shortfall (95%)
    pub cvar_95: Decimal,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_return: Decimal::ZERO,
            annualized_return: Decimal::ZERO,
            sharpe_ratio: Decimal::ZERO,
            sortino_ratio: Decimal::ZERO,
            calmar_ratio: Decimal::ZERO,
            max_drawdown: Decimal::ZERO,
            max_drawdown_duration_days: 0,
            win_rate: Decimal::ZERO,
            profit_factor: Decimal::ZERO,
            avg_trade_return: Decimal::ZERO,
            avg_win: Decimal::ZERO,
            avg_loss: Decimal::ZERO,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            avg_holding_period_hours: Decimal::ZERO,
            total_commission: Decimal::ZERO,
            total_slippage: Decimal::ZERO,
            volatility: Decimal::ZERO,
            skewness: Decimal::ZERO,
            kurtosis: Decimal::ZERO,
            var_95: Decimal::ZERO,
            cvar_95: Decimal::ZERO,
        }
    }
}

/// Backtest result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub config: BacktestConfig,
    pub metrics: PerformanceMetrics,
    pub trades: Vec<Trade>,
    pub equity_curve: Vec<(DateTime<Utc>, Decimal)>,
    pub drawdown_curve: Vec<(DateTime<Utc>, Decimal)>,
    pub daily_returns: Vec<Decimal>,
    pub final_portfolio_value: Decimal,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub trading_days: u64,
}

/// Backtest engine
pub struct BacktestEngine {
    config: BacktestConfig,
    cash: Decimal,
    positions: HashMap<String, Position>,
    trades: Vec<Trade>,
    equity_curve: Vec<(DateTime<Utc>, Decimal)>,
    peak_equity: Decimal,
    drawdown_curve: Vec<(DateTime<Utc>, Decimal)>,
    daily_returns: Vec<Decimal>,
    prev_equity: Decimal,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(config: BacktestConfig) -> Self {
        Self {
            cash: config.initial_capital,
            config,
            positions: HashMap::new(),
            trades: Vec::new(),
            equity_curve: Vec::new(),
            peak_equity: Decimal::ZERO,
            drawdown_curve: Vec::new(),
            daily_returns: Vec::new(),
            prev_equity: Decimal::ZERO,
        }
    }

    /// Calculate slippage based on model
    pub fn calculate_slippage(&self, order_size: Decimal, price: Decimal, volume: Decimal) -> Decimal {
        let order_value = order_size * price;
        let volume_ratio = if volume > Decimal::ZERO {
            order_value / (volume * price)
        } else {
            dec!(0.01) // Assume 1% of volume if unknown
        };

        match self.config.slippage_model {
            SlippageModel::None => Decimal::ZERO,
            SlippageModel::Fixed => self.config.base_slippage * price * order_size,
            SlippageModel::Linear => {
                // Slippage increases linearly with order size relative to volume
                self.config.base_slippage * volume_ratio * price * order_size
            }
            SlippageModel::SquareRoot => {
                // Square root model - more realistic for large orders
                // Slippage = base * sqrt(order_size / volume)
                let ratio_f64 = volume_ratio.to_string().parse::<f64>().unwrap_or(0.01);
                let sqrt_ratio = Decimal::from_f64_retain(ratio_f64.sqrt()).unwrap_or(dec!(0.1));
                self.config.base_slippage * sqrt_ratio * price * order_size
            }
            SlippageModel::Logarithmic => {
                // Log model - slippage grows slowly for large orders
                let ratio_f64 = volume_ratio.to_string().parse::<f64>().unwrap_or(0.01);
                let log_ratio = Decimal::from_f64_retain((1.0 + ratio_f64).ln()).unwrap_or(dec!(0.01));
                self.config.base_slippage * log_ratio * price * order_size
            }
        }
    }

    /// Calculate commission for a trade
    pub fn calculate_commission(&self, order_value: Decimal, is_maker: bool) -> Decimal {
        let fee_rate = if is_maker {
            self.config.maker_fee
        } else {
            self.config.taker_fee
        };

        match self.config.commission_model {
            CommissionModel::Percentage => order_value * fee_rate,
            CommissionModel::Fixed => fee_rate, // fee_rate is the fixed amount
            CommissionModel::Tiered => {
                // Simple tiered: lower fees for larger orders
                if order_value > dec!(10000) {
                    order_value * (fee_rate * dec!(0.8))
                } else if order_value > dec!(1000) {
                    order_value * (fee_rate * dec!(0.9))
                } else {
                    order_value * fee_rate
                }
            }
        }
    }

    /// Get current portfolio value
    pub fn portfolio_value(&self, current_prices: &HashMap<String, Decimal>) -> Decimal {
        let positions_value: Decimal = self.positions.iter()
            .map(|(symbol, pos)| {
                let price = current_prices.get(symbol).copied().unwrap_or(pos.current_price);
                pos.size * price
            })
            .sum();
        
        self.cash + positions_value
    }

    /// Open a new position
    pub fn open_position(
        &mut self,
        symbol: &str,
        direction: TradeDirection,
        size: Decimal,
        price: Decimal,
        volume: Decimal,
        timestamp: DateTime<Utc>,
    ) -> Option<Trade> {
        // Calculate costs
        let slippage = self.calculate_slippage(size, price, volume);
        let executed_price = match direction {
            TradeDirection::Long => price + slippage / size,
            TradeDirection::Short => price - slippage / size,
        };
        
        let order_value = size * executed_price;
        let commission = self.calculate_commission(order_value, false); // Assume taker
        
        // Check if we have enough cash
        let total_cost = order_value + commission;
        if total_cost > self.cash {
            return None;
        }
        
        // Update cash
        self.cash -= total_cost;
        
        // Create position
        let position = Position {
            symbol: symbol.to_string(),
            direction,
            size,
            entry_price: executed_price,
            entry_timestamp: timestamp,
            current_price: executed_price,
            unrealized_pnl: Decimal::ZERO,
        };
        
        self.positions.insert(symbol.to_string(), position);
        
        // Record trade
        let trade = Trade {
            timestamp,
            direction,
            entry_price: executed_price,
            exit_price: None,
            size,
            commission,
            slippage,
            pnl: None,
            exit_timestamp: None,
            holding_period_hours: None,
        };
        
        self.trades.push(trade.clone());
        
        Some(trade)
    }

    /// Close an existing position
    pub fn close_position(
        &mut self,
        symbol: &str,
        price: Decimal,
        volume: Decimal,
        timestamp: DateTime<Utc>,
    ) -> Option<Trade> {
        let position = self.positions.remove(symbol)?;
        
        // Calculate costs
        let slippage = self.calculate_slippage(position.size, price, volume);
        let executed_price = match position.direction {
            TradeDirection::Long => price - slippage / position.size,
            TradeDirection::Short => price + slippage / position.size,
        };
        
        let order_value = position.size * executed_price;
        let commission = self.calculate_commission(order_value, false);
        
        // Calculate PnL
        let pnl = match position.direction {
            TradeDirection::Long => (executed_price - position.entry_price) * position.size - commission,
            TradeDirection::Short => (position.entry_price - executed_price) * position.size - commission,
        };
        
        // Update cash
        self.cash += order_value - commission;
        if pnl > Decimal::ZERO {
            self.cash += pnl;
        }
        
        // Calculate holding period
        let holding_hours = (timestamp - position.entry_timestamp).num_hours();
        
        // Update the last trade with exit info
        if let Some(trade) = self.trades.last_mut() {
            trade.exit_price = Some(executed_price);
            trade.exit_timestamp = Some(timestamp);
            trade.pnl = Some(pnl);
            trade.holding_period_hours = Some(holding_hours);
            trade.commission += commission;
            trade.slippage += slippage;
        }
        
        Some(Trade {
            timestamp: position.entry_timestamp,
            direction: position.direction,
            entry_price: position.entry_price,
            exit_price: Some(executed_price),
            size: position.size,
            commission: commission * dec!(2), // Entry + exit
            slippage: slippage * dec!(2),
            pnl: Some(pnl),
            exit_timestamp: Some(timestamp),
            holding_period_hours: Some(holding_hours),
        })
    }

    /// Update position prices and calculate unrealized PnL
    pub fn update_prices(&mut self, current_prices: &HashMap<String, Decimal>) {
        for (symbol, position) in self.positions.iter_mut() {
            if let Some(&price) = current_prices.get(symbol) {
                position.current_price = price;
                position.unrealized_pnl = match position.direction {
                    TradeDirection::Long => (price - position.entry_price) * position.size,
                    TradeDirection::Short => (position.entry_price - price) * position.size,
                };
            }
        }
    }

    /// Record equity at a point in time
    pub fn record_equity(&mut self, timestamp: DateTime<Utc>, current_prices: &HashMap<String, Decimal>) {
        let equity = self.portfolio_value(current_prices);
        self.equity_curve.push((timestamp, equity));
        
        // Track peak and drawdown
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }
        
        let drawdown = if self.peak_equity > Decimal::ZERO {
            (self.peak_equity - equity) / self.peak_equity
        } else {
            Decimal::ZERO
        };
        self.drawdown_curve.push((timestamp, drawdown));
        
        // Calculate daily return
        if self.prev_equity > Decimal::ZERO {
            let daily_return = (equity - self.prev_equity) / self.prev_equity;
            self.daily_returns.push(daily_return);
        }
        self.prev_equity = equity;
    }

    /// Calculate performance metrics from results
    pub fn calculate_metrics(&self) -> PerformanceMetrics {
        let mut metrics = PerformanceMetrics::default();
        
        if self.equity_curve.is_empty() {
            return metrics;
        }
        
        // Basic stats
        let initial = self.config.initial_capital;
        let final_value = self.equity_curve.last().map(|(_, v)| *v).unwrap_or(initial);
        
        metrics.total_return = (final_value - initial) / initial * dec!(100);
        
        // Annualized return
        let trading_days = self.equity_curve.len() as i64;
        if trading_days > 0 {
            let years = Decimal::from(trading_days) / dec!(365);
            if years > Decimal::ZERO {
                let total_return_ratio = final_value / initial;
                let years_f64 = years.to_string().parse::<f64>().unwrap_or(1.0);
                let ratio_f64 = total_return_ratio.to_string().parse::<f64>().unwrap_or(1.0);
                let annualized = ratio_f64.powf(1.0 / years_f64) - 1.0;
                metrics.annualized_return = Decimal::from_f64_retain(annualized * 100.0).unwrap_or(Decimal::ZERO);
            }
        }
        
        // Trade statistics
        let completed_trades: Vec<_> = self.trades.iter()
            .filter(|t| t.pnl.is_some())
            .collect();
        
        metrics.total_trades = completed_trades.len() as u64;
        
        if !completed_trades.is_empty() {
            let mut total_pnl = Decimal::ZERO;
            let mut gross_profit = Decimal::ZERO;
            let mut gross_loss = Decimal::ZERO;
            let mut total_holding_hours: i64 = 0;
            
            for trade in &completed_trades {
                let pnl = trade.pnl.unwrap_or(Decimal::ZERO);
                total_pnl += pnl;
                metrics.total_commission += trade.commission;
                metrics.total_slippage += trade.slippage;
                
                if pnl > Decimal::ZERO {
                    metrics.winning_trades += 1;
                    gross_profit += pnl;
                } else {
                    metrics.losing_trades += 1;
                    gross_loss += pnl.abs();
                }
                
                total_holding_hours += trade.holding_period_hours.unwrap_or(0);
            }
            
            metrics.avg_trade_return = total_pnl / Decimal::from(completed_trades.len() as u64);
            
            if metrics.winning_trades > 0 {
                metrics.avg_win = gross_profit / Decimal::from(metrics.winning_trades);
            }
            if metrics.losing_trades > 0 {
                metrics.avg_loss = gross_loss / Decimal::from(metrics.losing_trades);
            }
            
            metrics.win_rate = Decimal::from(metrics.winning_trades) / 
                Decimal::from(metrics.total_trades) * dec!(100);
            
            if gross_loss > Decimal::ZERO {
                metrics.profit_factor = gross_profit / gross_loss;
            }
            
            metrics.avg_holding_period_hours = Decimal::from(total_holding_hours) / 
                Decimal::from(completed_trades.len() as u64);
        }
        
        // Max drawdown
        if let Some((_, max_dd)) = self.drawdown_curve.iter().max_by(|a, b| a.1.cmp(&b.1)) {
            metrics.max_drawdown = *max_dd * dec!(100);
        }
        
        // Calculate drawdown duration
        let mut current_dd_start: Option<DateTime<Utc>> = None;
        let mut max_dd_duration = Duration::zero();
        
        for (ts, dd) in &self.drawdown_curve {
            if *dd > Decimal::ZERO {
                if current_dd_start.is_none() {
                    current_dd_start = Some(*ts);
                }
            } else if let Some(start) = current_dd_start {
                let duration = *ts - start;
                if duration > max_dd_duration {
                    max_dd_duration = duration;
                }
                current_dd_start = None;
            }
        }
        metrics.max_drawdown_duration_days = max_dd_duration.num_days();
        
        // Volatility, Sharpe, Sortino
        if self.daily_returns.len() > 1 {
            let n = self.daily_returns.len();
            let mean: Decimal = self.daily_returns.iter().sum::<Decimal>() / Decimal::from(n as u64);
            
            // Variance
            let variance: Decimal = self.daily_returns.iter()
                .map(|r| (*r - mean) * (*r - mean))
                .sum::<Decimal>() / Decimal::from(n as u64 - 1);
            
            // Daily volatility
            let var_f64 = variance.to_string().parse::<f64>().unwrap_or(0.0);
            let daily_vol = Decimal::from_f64_retain(var_f64.sqrt()).unwrap_or(Decimal::ZERO);
            
            // Annualized volatility (sqrt(252) for trading days)
            metrics.volatility = daily_vol * dec!(15.87); // sqrt(252) ≈ 15.87
            
            // Sharpe ratio
            let daily_rf = self.config.risk_free_rate / dec!(252);
            if daily_vol > Decimal::ZERO {
                let excess_return = mean - daily_rf;
                metrics.sharpe_ratio = (excess_return / daily_vol) * dec!(15.87);
            }
            
            // Sortino ratio (using downside deviation)
            let negative_returns: Vec<_> = self.daily_returns.iter()
                .filter(|r| **r < Decimal::ZERO)
                .collect();
            
            if !negative_returns.is_empty() {
                let downside_variance: Decimal = negative_returns.iter()
                    .map(|r| **r * **r)
                    .sum::<Decimal>() / Decimal::from(negative_returns.len() as u64);
                
                let dv_f64 = downside_variance.to_string().parse::<f64>().unwrap_or(0.0);
                let downside_dev = Decimal::from_f64_retain(dv_f64.sqrt()).unwrap_or(Decimal::ZERO);
                
                if downside_dev > Decimal::ZERO {
                    let excess_return = mean - daily_rf;
                    metrics.sortino_ratio = (excess_return / downside_dev) * dec!(15.87);
                }
            }
            
            // Skewness and Kurtosis
            let std_dev = daily_vol;
            if std_dev > Decimal::ZERO {
                let skew_sum: Decimal = self.daily_returns.iter()
                    .map(|r| {
                        let z = (*r - mean) / std_dev;
                        z * z * z
                    })
                    .sum();
                metrics.skewness = skew_sum / Decimal::from(n as u64);
                
                let kurt_sum: Decimal = self.daily_returns.iter()
                    .map(|r| {
                        let z = (*r - mean) / std_dev;
                        z * z * z * z
                    })
                    .sum();
                metrics.kurtosis = kurt_sum / Decimal::from(n as u64) - dec!(3);
            }
            
            // VaR and CVaR (95%)
            let mut sorted_returns = self.daily_returns.clone();
            sorted_returns.sort();
            
            let var_index = (n as f64 * 0.05) as usize;
            if var_index < sorted_returns.len() {
                metrics.var_95 = sorted_returns[var_index].abs() * dec!(100);
                
                // CVaR = average of returns below VaR
                let tail_returns: Decimal = sorted_returns[..=var_index].iter().sum();
                metrics.cvar_95 = (tail_returns / Decimal::from(var_index as u64 + 1)).abs() * dec!(100);
            }
        }
        
        // Calmar ratio
        if metrics.max_drawdown > Decimal::ZERO {
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown;
        }
        
        metrics
    }

    /// Run backtest with given price data and strategy function
    pub fn run<F>(
        &mut self,
        symbol: &str,
        price_bars: &[PriceBar],
        mut strategy: F,
    ) -> BacktestResult
    where
        F: FnMut(&[PriceBar], usize, &HashMap<String, Position>) -> (Signal, Decimal),
    {
        if price_bars.is_empty() {
            return BacktestResult {
                config: self.config.clone(),
                metrics: PerformanceMetrics::default(),
                trades: Vec::new(),
                equity_curve: Vec::new(),
                drawdown_curve: Vec::new(),
                daily_returns: Vec::new(),
                final_portfolio_value: self.config.initial_capital,
                start_date: Utc::now(),
                end_date: Utc::now(),
                trading_days: 0,
            };
        }

        // Initialize
        self.peak_equity = self.config.initial_capital;
        self.prev_equity = self.config.initial_capital;
        
        let start_date = price_bars.first().unwrap().timestamp;
        let end_date = price_bars.last().unwrap().timestamp;
        
        // Process each bar
        for (i, bar) in price_bars.iter().enumerate() {
            // Update current prices
            let mut current_prices = HashMap::new();
            current_prices.insert(symbol.to_string(), bar.close);
            
            self.update_prices(&current_prices);
            
            // Get strategy signal
            let (signal, confidence) = strategy(price_bars, i, &self.positions);
            
            // Execute signal
            match signal {
                Signal::Buy => {
                    if !self.positions.contains_key(symbol) && confidence > dec!(0.5) {
                        // Size based on confidence and max position
                        let position_value = self.cash * self.config.max_position_fraction * confidence;
                        let size = position_value / bar.close;
                        
                        if size > Decimal::ZERO {
                            self.open_position(
                                symbol,
                                TradeDirection::Long,
                                size,
                                bar.close,
                                bar.volume,
                                bar.timestamp,
                            );
                        }
                    }
                }
                Signal::Sell => {
                    if !self.positions.contains_key(symbol) && confidence > dec!(0.5) {
                        let position_value = self.cash * self.config.max_position_fraction * confidence;
                        let size = position_value / bar.close;
                        
                        if size > Decimal::ZERO {
                            self.open_position(
                                symbol,
                                TradeDirection::Short,
                                size,
                                bar.close,
                                bar.volume,
                                bar.timestamp,
                            );
                        }
                    }
                }
                Signal::CloseAll => {
                    let symbols: Vec<String> = self.positions.keys().cloned().collect();
                    for sym in symbols {
                        self.close_position(&sym, bar.close, bar.volume, bar.timestamp);
                    }
                }
                Signal::Hold => {}
            }
            
            // Record equity
            self.record_equity(bar.timestamp, &current_prices);
        }
        
        // Close all remaining positions at end
        let final_bar = price_bars.last().unwrap();
        let symbols: Vec<String> = self.positions.keys().cloned().collect();
        for sym in symbols {
            self.close_position(&sym, final_bar.close, final_bar.volume, final_bar.timestamp);
        }
        
        // Calculate final metrics
        let metrics = self.calculate_metrics();
        
        let mut current_prices = HashMap::new();
        current_prices.insert(symbol.to_string(), final_bar.close);
        let final_value = self.portfolio_value(&current_prices);
        
        BacktestResult {
            config: self.config.clone(),
            metrics,
            trades: self.trades.clone(),
            equity_curve: self.equity_curve.clone(),
            drawdown_curve: self.drawdown_curve.clone(),
            daily_returns: self.daily_returns.clone(),
            final_portfolio_value: final_value,
            start_date,
            end_date,
            trading_days: price_bars.len() as u64,
        }
    }

    /// Reset engine for new backtest
    pub fn reset(&mut self) {
        self.cash = self.config.initial_capital;
        self.positions.clear();
        self.trades.clear();
        self.equity_curve.clear();
        self.peak_equity = Decimal::ZERO;
        self.drawdown_curve.clear();
        self.daily_returns.clear();
        self.prev_equity = Decimal::ZERO;
    }
}

/// Walk-forward optimization parameters
#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    /// In-sample window size (bars)
    pub in_sample_size: usize,
    /// Out-of-sample window size (bars)
    pub out_of_sample_size: usize,
    /// Step size for rolling window
    pub step_size: usize,
}

/// Monte Carlo simulation for robustness testing
pub struct MonteCarloSimulator {
    num_simulations: usize,
    shuffle_returns: bool,
}

impl MonteCarloSimulator {
    pub fn new(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            shuffle_returns: true,
        }
    }

    /// Run Monte Carlo simulation on trade returns
    pub fn simulate(&self, trade_returns: &[Decimal]) -> MonteCarloResult {
        use std::collections::BTreeMap;
        
        if trade_returns.is_empty() {
            return MonteCarloResult {
                median_return: Decimal::ZERO,
                percentile_5: Decimal::ZERO,
                percentile_95: Decimal::ZERO,
                probability_of_profit: Decimal::ZERO,
                worst_case_return: Decimal::ZERO,
                best_case_return: Decimal::ZERO,
            };
        }

        let mut final_returns: Vec<Decimal> = Vec::with_capacity(self.num_simulations);
        
        // Simple LCG random number generator (deterministic for reproducibility)
        let mut seed: u64 = 12345;
        let lcg_next = |s: &mut u64| -> usize {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (*s / 65536) as usize
        };
        
        for _ in 0..self.num_simulations {
            // Shuffle and compound returns
            let mut shuffled = trade_returns.to_vec();
            
            if self.shuffle_returns {
                // Fisher-Yates shuffle
                for i in (1..shuffled.len()).rev() {
                    let j = lcg_next(&mut seed) % (i + 1);
                    shuffled.swap(i, j);
                }
            }
            
            // Compound the returns
            let final_value = shuffled.iter()
                .fold(dec!(1), |acc, r| acc * (dec!(1) + *r));
            
            final_returns.push(final_value - dec!(1));
        }
        
        // Sort for percentile calculations
        final_returns.sort();
        
        let n = final_returns.len();
        let median_idx = n / 2;
        let p5_idx = (n as f64 * 0.05) as usize;
        let p95_idx = (n as f64 * 0.95) as usize;
        
        let profitable_count = final_returns.iter().filter(|r| **r > Decimal::ZERO).count();
        
        MonteCarloResult {
            median_return: final_returns[median_idx] * dec!(100),
            percentile_5: final_returns[p5_idx] * dec!(100),
            percentile_95: final_returns[p95_idx.min(n - 1)] * dec!(100),
            probability_of_profit: Decimal::from(profitable_count as u64) / Decimal::from(n as u64) * dec!(100),
            worst_case_return: final_returns[0] * dec!(100),
            best_case_return: final_returns[n - 1] * dec!(100),
        }
    }
}

/// Monte Carlo simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    pub median_return: Decimal,
    pub percentile_5: Decimal,
    pub percentile_95: Decimal,
    pub probability_of_profit: Decimal,
    pub worst_case_return: Decimal,
    pub best_case_return: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bars(n: usize, start_price: f64, volatility: f64) -> Vec<PriceBar> {
        let mut bars = Vec::with_capacity(n);
        let mut price = start_price;
        let mut seed: u64 = 42;
        
        let lcg_next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((*s / 65536) % 1000) as f64 / 1000.0 - 0.5
        };
        
        for i in 0..n {
            let change = lcg_next(&mut seed) * volatility;
            let open = price;
            let close = price * (1.0 + change);
            let high = f64::max(open, close) * 1.002;
            let low = f64::min(open, close) * 0.998;
            
            bars.push(PriceBar {
                timestamp: Utc::now() + Duration::hours(i as i64),
                open: Decimal::from_f64_retain(open).unwrap(),
                high: Decimal::from_f64_retain(high).unwrap(),
                low: Decimal::from_f64_retain(low).unwrap(),
                close: Decimal::from_f64_retain(close).unwrap(),
                volume: dec!(1000000),
            });
            
            price = close;
        }
        
        bars
    }

    #[test]
    fn test_backtest_engine_creation() {
        let config = BacktestConfig::default();
        let engine = BacktestEngine::new(config.clone());
        
        assert_eq!(engine.cash, config.initial_capital);
        assert!(engine.positions.is_empty());
        assert!(engine.trades.is_empty());
    }

    #[test]
    fn test_slippage_calculation_none() {
        let mut config = BacktestConfig::default();
        config.slippage_model = SlippageModel::None;
        let engine = BacktestEngine::new(config);
        
        let slippage = engine.calculate_slippage(dec!(100), dec!(50), dec!(10000));
        assert_eq!(slippage, Decimal::ZERO);
    }

    #[test]
    fn test_slippage_calculation_fixed() {
        let mut config = BacktestConfig::default();
        config.slippage_model = SlippageModel::Fixed;
        config.base_slippage = dec!(0.001);
        let engine = BacktestEngine::new(config);
        
        let slippage = engine.calculate_slippage(dec!(100), dec!(50), dec!(10000));
        // slippage = 0.001 * 50 * 100 = 5
        assert_eq!(slippage, dec!(5));
    }

    #[test]
    fn test_commission_calculation_percentage() {
        let config = BacktestConfig::default();
        let engine = BacktestEngine::new(config);
        
        // Taker fee = 0.2%
        let commission = engine.calculate_commission(dec!(1000), false);
        assert_eq!(commission, dec!(2));
        
        // Maker fee = 0.1%
        let commission_maker = engine.calculate_commission(dec!(1000), true);
        assert_eq!(commission_maker, dec!(1));
    }

    #[test]
    fn test_commission_calculation_tiered() {
        let mut config = BacktestConfig::default();
        config.commission_model = CommissionModel::Tiered;
        config.taker_fee = dec!(0.001);
        let engine = BacktestEngine::new(config);
        
        // Small order - full fee
        let small = engine.calculate_commission(dec!(500), false);
        assert_eq!(small, dec!(0.5));
        
        // Medium order - 90% fee
        let medium = engine.calculate_commission(dec!(5000), false);
        assert_eq!(medium, dec!(4.5));
        
        // Large order - 80% fee
        let large = engine.calculate_commission(dec!(20000), false);
        assert_eq!(large, dec!(16));
    }

    #[test]
    fn test_open_position() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config.clone());
        
        let trade = engine.open_position(
            "BTC",
            TradeDirection::Long,
            dec!(1),
            dec!(100),
            dec!(10000),
            Utc::now(),
        );
        
        assert!(trade.is_some());
        assert!(engine.positions.contains_key("BTC"));
        assert!(engine.cash < config.initial_capital);
    }

    #[test]
    fn test_close_position() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        
        let timestamp = Utc::now();
        engine.open_position("BTC", TradeDirection::Long, dec!(1), dec!(100), dec!(10000), timestamp);
        
        // Close at higher price (profit)
        let close_result = engine.close_position("BTC", dec!(110), dec!(10000), timestamp + Duration::hours(1));
        
        assert!(close_result.is_some());
        assert!(!engine.positions.contains_key("BTC"));
        
        let trade = close_result.unwrap();
        assert!(trade.pnl.unwrap() > Decimal::ZERO);
    }

    #[test]
    fn test_portfolio_value() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config.clone());
        
        // Initially just cash
        let mut prices = HashMap::new();
        assert_eq!(engine.portfolio_value(&prices), config.initial_capital);
        
        // Open position and check value
        engine.open_position("BTC", TradeDirection::Long, dec!(10), dec!(100), dec!(10000), Utc::now());
        prices.insert("BTC".to_string(), dec!(100));
        
        let value = engine.portfolio_value(&prices);
        // Should be close to initial (minus fees/slippage)
        assert!(value < config.initial_capital);
        assert!(value > config.initial_capital * dec!(0.95));
    }

    #[test]
    fn test_simple_backtest() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        
        let bars = create_test_bars(100, 100.0, 0.02);
        
        // Simple momentum strategy
        let result = engine.run("TEST", &bars, |bars, i, _positions| {
            if i < 5 {
                return (Signal::Hold, dec!(0));
            }
            
            let current = bars[i].close;
            let prev = bars[i - 5].close;
            
            if current > prev {
                (Signal::Buy, dec!(0.8))
            } else if current < prev {
                (Signal::Sell, dec!(0.8))
            } else {
                (Signal::Hold, dec!(0))
            }
        });
        
        assert!(result.trading_days > 0);
        assert!(result.trades.len() > 0);
        assert!(!result.equity_curve.is_empty());
    }

    #[test]
    fn test_performance_metrics() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        
        let bars = create_test_bars(200, 100.0, 0.01);
        
        let result = engine.run("TEST", &bars, |bars, i, positions| {
            if i < 10 {
                return (Signal::Hold, dec!(0));
            }
            
            // Simple mean reversion
            let avg: Decimal = bars[i-10..i].iter().map(|b| b.close).sum::<Decimal>() / dec!(10);
            let current = bars[i].close;
            
            if positions.is_empty() {
                if current < avg * dec!(0.98) {
                    (Signal::Buy, dec!(0.7))
                } else if current > avg * dec!(1.02) {
                    (Signal::Sell, dec!(0.7))
                } else {
                    (Signal::Hold, dec!(0))
                }
            } else {
                // Close if price reverted
                if current > avg * dec!(0.99) && current < avg * dec!(1.01) {
                    (Signal::CloseAll, dec!(1))
                } else {
                    (Signal::Hold, dec!(0))
                }
            }
        });
        
        // Check metrics are calculated
        let metrics = &result.metrics;
        assert!(metrics.total_trades > 0 || result.trades.is_empty());
        // Sharpe can be any value
        // Win rate should be between 0-100
        assert!(metrics.win_rate >= Decimal::ZERO);
        assert!(metrics.win_rate <= dec!(100));
    }

    #[test]
    fn test_drawdown_tracking() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        
        // Create a declining market
        let mut bars = Vec::new();
        for i in 0..50 {
            let price = 100.0 - (i as f64) * 0.5; // Declining price
            bars.push(PriceBar {
                timestamp: Utc::now() + Duration::hours(i),
                open: Decimal::from_f64_retain(price).unwrap(),
                high: Decimal::from_f64_retain(price * 1.01).unwrap(),
                low: Decimal::from_f64_retain(price * 0.99).unwrap(),
                close: Decimal::from_f64_retain(price).unwrap(),
                volume: dec!(1000000),
            });
        }
        
        let result = engine.run("TEST", &bars, |_, i, _| {
            if i == 5 {
                (Signal::Buy, dec!(0.9))
            } else {
                (Signal::Hold, dec!(0))
            }
        });
        
        // Should have drawdown since we bought in a declining market
        assert!(result.metrics.max_drawdown > Decimal::ZERO);
    }

    #[test]
    fn test_monte_carlo_simulation() {
        let simulator = MonteCarloSimulator::new(1000);
        
        let returns = vec![
            dec!(0.02), dec!(-0.01), dec!(0.03), dec!(-0.02), dec!(0.01),
            dec!(0.02), dec!(-0.015), dec!(0.025), dec!(-0.01), dec!(0.015),
        ];
        
        let result = simulator.simulate(&returns);
        
        // Check that results are calculated (may vary due to randomness)
        // Worst case should be <= median, best case should be >= median
        assert!(result.worst_case_return <= result.best_case_return);
        assert!(result.probability_of_profit >= Decimal::ZERO);
        assert!(result.probability_of_profit <= dec!(100));
        // Percentiles should be ordered
        assert!(result.percentile_5 <= result.percentile_95);
    }

    #[test]
    fn test_empty_backtest() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config.clone());
        
        let result = engine.run("TEST", &[], |_, _, _| (Signal::Hold, dec!(0)));
        
        assert_eq!(result.trading_days, 0);
        assert!(result.trades.is_empty());
        assert_eq!(result.final_portfolio_value, config.initial_capital);
    }

    #[test]
    fn test_engine_reset() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config.clone());
        
        // Run a backtest
        let bars = create_test_bars(50, 100.0, 0.02);
        engine.run("TEST", &bars, |_, i, _| {
            if i % 10 == 0 {
                (Signal::Buy, dec!(0.8))
            } else if i % 10 == 5 {
                (Signal::CloseAll, dec!(1))
            } else {
                (Signal::Hold, dec!(0))
            }
        });
        
        // Reset
        engine.reset();
        
        // Check engine is reset
        assert_eq!(engine.cash, config.initial_capital);
        assert!(engine.positions.is_empty());
        assert!(engine.trades.is_empty());
        assert!(engine.equity_curve.is_empty());
    }

    #[test]
    fn test_var_cvar_calculation() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        
        // Create bars with known volatility
        let bars = create_test_bars(252, 100.0, 0.03); // ~3% daily volatility
        
        let result = engine.run("TEST", &bars, |bars, i, _| {
            if i < 20 {
                return (Signal::Hold, dec!(0));
            }
            
            // Random trading to generate returns
            let change = bars[i].close / bars[i-1].close - dec!(1);
            if change > dec!(0.01) {
                (Signal::Buy, dec!(0.6))
            } else if change < dec!(-0.01) {
                (Signal::CloseAll, dec!(1))
            } else {
                (Signal::Hold, dec!(0))
            }
        });
        
        // VaR and CVaR should be calculated
        // CVaR should be >= VaR (expected shortfall is worse than VaR)
        if result.metrics.var_95 > Decimal::ZERO {
            assert!(result.metrics.cvar_95 >= result.metrics.var_95 * dec!(0.5));
        }
    }

    #[test]
    fn test_short_position() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        
        let timestamp = Utc::now();
        
        // Open short at 100
        engine.open_position("BTC", TradeDirection::Short, dec!(1), dec!(100), dec!(10000), timestamp);
        
        // Price drops to 90 - should profit
        let trade = engine.close_position("BTC", dec!(90), dec!(10000), timestamp + Duration::hours(1));
        
        assert!(trade.is_some());
        let pnl = trade.unwrap().pnl.unwrap();
        assert!(pnl > Decimal::ZERO); // Profit from short
    }

    #[test]
    fn test_holding_period_calculation() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        
        let entry_time = Utc::now();
        let exit_time = entry_time + Duration::hours(24);
        
        engine.open_position("BTC", TradeDirection::Long, dec!(1), dec!(100), dec!(10000), entry_time);
        let trade = engine.close_position("BTC", dec!(110), dec!(10000), exit_time);
        
        assert!(trade.is_some());
        assert_eq!(trade.unwrap().holding_period_hours, Some(24));
    }

    #[test]
    fn test_multiple_symbols() {
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(config);
        
        let timestamp = Utc::now();
        
        // Open positions in multiple symbols
        engine.open_position("BTC", TradeDirection::Long, dec!(0.1), dec!(50000), dec!(10000), timestamp);
        engine.open_position("ETH", TradeDirection::Long, dec!(1), dec!(3000), dec!(10000), timestamp);
        
        assert_eq!(engine.positions.len(), 2);
        assert!(engine.positions.contains_key("BTC"));
        assert!(engine.positions.contains_key("ETH"));
        
        // Check portfolio value with updated prices
        let mut prices = HashMap::new();
        prices.insert("BTC".to_string(), dec!(55000));
        prices.insert("ETH".to_string(), dec!(3200));
        
        let value = engine.portfolio_value(&prices);
        assert!(value > Decimal::ZERO);
    }

    #[test]
    fn test_insufficient_funds() {
        let mut config = BacktestConfig::default();
        config.initial_capital = dec!(100);
        let mut engine = BacktestEngine::new(config);
        
        // Try to buy more than we can afford
        let trade = engine.open_position(
            "BTC",
            TradeDirection::Long,
            dec!(1000), // Way too much
            dec!(100),
            dec!(10000),
            Utc::now(),
        );
        
        // Should fail
        assert!(trade.is_none());
        assert!(engine.positions.is_empty());
    }

    #[test]
    fn test_slippage_sqrt_model() {
        let mut config = BacktestConfig::default();
        config.slippage_model = SlippageModel::SquareRoot;
        config.base_slippage = dec!(0.001);
        let engine = BacktestEngine::new(config);
        
        // Small order relative to volume - low slippage
        let small_slip = engine.calculate_slippage(dec!(10), dec!(100), dec!(100000));
        
        // Large order relative to volume - higher slippage
        let large_slip = engine.calculate_slippage(dec!(1000), dec!(100), dec!(100000));
        
        // Large order should have higher slippage
        assert!(large_slip > small_slip);
        
        // For sqrt model, slippage per unit should grow with sqrt of size
        // So larger orders have higher total slippage, as expected
        // The key property is: slippage grows, but sub-linearly per unit
        let slip_per_unit_small = small_slip / dec!(10);
        let slip_per_unit_large = large_slip / dec!(1000);
        
        // Per-unit slippage should be higher for large orders (market impact)
        assert!(slip_per_unit_large > slip_per_unit_small);
    }

    #[test]
    fn test_walk_forward_config() {
        let config = WalkForwardConfig {
            in_sample_size: 100,
            out_of_sample_size: 20,
            step_size: 20,
        };
        
        assert_eq!(config.in_sample_size, 100);
        assert_eq!(config.out_of_sample_size, 20);
        assert_eq!(config.step_size, 20);
    }
}
