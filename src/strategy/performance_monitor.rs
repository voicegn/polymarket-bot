//! Real-time performance monitoring and adaptive strategy tuning
//!
//! Features:
//! 1. Rolling performance metrics (win rate, PnL, Sharpe)
//! 2. Regime detection (trending/ranging/volatile)
//! 3. Automatic strategy parameter adjustment
//! 4. Anomaly detection and alerts
//! 5. Performance attribution by signal type

use chrono::{DateTime, Duration, Utc};
use rust_decimal::{Decimal, MathematicalOps};
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};

/// A completed trade for performance tracking
#[derive(Debug, Clone)]
pub struct CompletedTrade {
    pub trade_id: String,
    pub market_id: String,
    pub signal_type: String,
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub pnl: Decimal,
    pub pnl_pct: Decimal,
    pub hold_duration_mins: i64,
    pub entry_price: Decimal,
    pub exit_price: Decimal,
    pub size: Decimal,
}

/// Rolling performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Win rate (percentage)
    pub win_rate: Decimal,
    /// Average win size
    pub avg_win: Decimal,
    /// Average loss size
    pub avg_loss: Decimal,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: Option<Decimal>,
    /// Total PnL
    pub total_pnl: Decimal,
    /// Return percentage
    pub return_pct: Decimal,
    /// Number of trades
    pub trade_count: usize,
    /// Average hold time (minutes)
    pub avg_hold_mins: i64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: Option<Decimal>,
    /// Max drawdown percentage
    pub max_drawdown_pct: Decimal,
    /// Current drawdown percentage
    pub current_drawdown_pct: Decimal,
    /// Win streak
    pub win_streak: u32,
    /// Loss streak
    pub loss_streak: u32,
    /// Calculated at
    pub calculated_at: DateTime<Utc>,
}

/// Market regime
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketRegime {
    /// Strong trending up
    TrendingUp,
    /// Strong trending down
    TrendingDown,
    /// Ranging/sideways
    Ranging,
    /// High volatility
    Volatile,
    /// Low activity
    Quiet,
    /// Unknown/insufficient data
    Unknown,
}

impl MarketRegime {
    /// Recommended strategy adjustments for this regime
    pub fn strategy_adjustments(&self) -> RegimeAdjustments {
        match self {
            Self::TrendingUp | Self::TrendingDown => RegimeAdjustments {
                position_size_mult: dec!(1.2),   // Larger positions in trends
                take_profit_mult: dec!(1.3),     // Let winners run
                stop_loss_mult: dec!(1.0),       // Normal stops
                min_edge_mult: dec!(0.9),        // Slightly lower edge OK
                hold_time_mult: dec!(1.5),       // Hold longer
            },
            Self::Ranging => RegimeAdjustments {
                position_size_mult: dec!(1.0),
                take_profit_mult: dec!(0.8),     // Take profits quickly
                stop_loss_mult: dec!(0.9),       // Tighter stops
                min_edge_mult: dec!(1.1),        // Require more edge
                hold_time_mult: dec!(0.7),       // Shorter holds
            },
            Self::Volatile => RegimeAdjustments {
                position_size_mult: dec!(0.6),   // Smaller positions
                take_profit_mult: dec!(1.2),     // Wider targets
                stop_loss_mult: dec!(1.3),       // Wider stops
                min_edge_mult: dec!(1.3),        // Need more edge
                hold_time_mult: dec!(0.5),       // Quick exits
            },
            Self::Quiet => RegimeAdjustments {
                position_size_mult: dec!(0.8),
                take_profit_mult: dec!(0.7),
                stop_loss_mult: dec!(0.8),
                min_edge_mult: dec!(1.2),
                hold_time_mult: dec!(1.0),
            },
            Self::Unknown => RegimeAdjustments::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RegimeAdjustments {
    pub position_size_mult: Decimal,
    pub take_profit_mult: Decimal,
    pub stop_loss_mult: Decimal,
    pub min_edge_mult: Decimal,
    pub hold_time_mult: Decimal,
}

impl Default for RegimeAdjustments {
    fn default() -> Self {
        Self {
            position_size_mult: dec!(1.0),
            take_profit_mult: dec!(1.0),
            stop_loss_mult: dec!(1.0),
            min_edge_mult: dec!(1.0),
            hold_time_mult: dec!(1.0),
        }
    }
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub suggested_action: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlertType {
    /// Losing streak detected
    LosingStreak,
    /// Drawdown threshold breached
    DrawdownBreach,
    /// Win rate declining
    WinRateDecline,
    /// Performance anomaly
    Anomaly,
    /// Strategy performing well
    PositiveStreak,
    /// Regime change detected
    RegimeChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Configuration for performance monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Window size for rolling metrics (number of trades)
    pub window_size: usize,
    /// Losing streak threshold for alert
    pub losing_streak_threshold: u32,
    /// Drawdown percentage threshold for alert
    pub drawdown_threshold: Decimal,
    /// Win rate decline threshold (percentage points)
    pub win_rate_decline_threshold: Decimal,
    /// Minimum trades for regime detection
    pub min_trades_for_regime: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            losing_streak_threshold: 5,
            drawdown_threshold: dec!(0.10), // 10%
            win_rate_decline_threshold: dec!(0.15), // 15 percentage points
            min_trades_for_regime: 20,
        }
    }
}

/// Real-time performance monitor
pub struct PerformanceMonitor {
    config: MonitorConfig,
    /// All completed trades
    trades: VecDeque<CompletedTrade>,
    /// Performance by signal type
    performance_by_signal: HashMap<String, VecDeque<CompletedTrade>>,
    /// Equity curve
    equity_curve: Vec<(DateTime<Utc>, Decimal)>,
    /// Peak equity
    peak_equity: Decimal,
    /// Initial capital
    initial_capital: Decimal,
    /// Current streaks
    current_win_streak: u32,
    current_loss_streak: u32,
    /// Previous metrics (for comparison)
    previous_metrics: Option<PerformanceMetrics>,
    /// Active alerts
    alerts: Vec<PerformanceAlert>,
    /// Detected regime
    current_regime: MarketRegime,
}

impl PerformanceMonitor {
    pub fn new(config: MonitorConfig, initial_capital: Decimal) -> Self {
        Self {
            config,
            trades: VecDeque::new(),
            performance_by_signal: HashMap::new(),
            equity_curve: vec![(Utc::now(), initial_capital)],
            peak_equity: initial_capital,
            initial_capital,
            current_win_streak: 0,
            current_loss_streak: 0,
            previous_metrics: None,
            alerts: Vec::new(),
            current_regime: MarketRegime::Unknown,
        }
    }

    pub fn with_defaults(initial_capital: Decimal) -> Self {
        Self::new(MonitorConfig::default(), initial_capital)
    }

    /// Record a completed trade
    pub fn record_trade(&mut self, trade: CompletedTrade) {
        // Update streaks
        if trade.pnl > Decimal::ZERO {
            self.current_win_streak += 1;
            self.current_loss_streak = 0;
        } else {
            self.current_loss_streak += 1;
            self.current_win_streak = 0;
        }

        // Update equity curve
        let current_equity = self.equity_curve.last()
            .map(|(_, e)| *e)
            .unwrap_or(self.initial_capital);
        let new_equity = current_equity + trade.pnl;
        self.equity_curve.push((trade.exit_time, new_equity));

        // Update peak
        if new_equity > self.peak_equity {
            self.peak_equity = new_equity;
        }

        // Store by signal type
        self.performance_by_signal
            .entry(trade.signal_type.clone())
            .or_default()
            .push_back(trade.clone());

        // Store in main queue
        self.trades.push_back(trade);

        // Keep only window_size trades
        while self.trades.len() > self.config.window_size {
            self.trades.pop_front();
        }

        // Check for alerts
        self.check_alerts();

        // Update regime
        self.detect_regime();
    }

    /// Calculate current performance metrics
    pub fn calculate_metrics(&self) -> PerformanceMetrics {
        let trades: Vec<_> = self.trades.iter().collect();
        
        if trades.is_empty() {
            return PerformanceMetrics {
                win_rate: Decimal::ZERO,
                avg_win: Decimal::ZERO,
                avg_loss: Decimal::ZERO,
                profit_factor: None,
                total_pnl: Decimal::ZERO,
                return_pct: Decimal::ZERO,
                trade_count: 0,
                avg_hold_mins: 0,
                sharpe_ratio: None,
                max_drawdown_pct: Decimal::ZERO,
                current_drawdown_pct: Decimal::ZERO,
                win_streak: self.current_win_streak,
                loss_streak: self.current_loss_streak,
                calculated_at: Utc::now(),
            };
        }

        let total_trades = trades.len();
        let wins: Vec<_> = trades.iter().filter(|t| t.pnl > Decimal::ZERO).collect();
        let losses: Vec<_> = trades.iter().filter(|t| t.pnl <= Decimal::ZERO).collect();

        let win_rate = Decimal::from(wins.len() as u32) / Decimal::from(total_trades as u32);

        let avg_win = if !wins.is_empty() {
            wins.iter().map(|t| t.pnl).sum::<Decimal>() / Decimal::from(wins.len() as u32)
        } else {
            Decimal::ZERO
        };

        let avg_loss = if !losses.is_empty() {
            losses.iter().map(|t| t.pnl.abs()).sum::<Decimal>() / Decimal::from(losses.len() as u32)
        } else {
            Decimal::ZERO
        };

        let gross_profit: Decimal = wins.iter().map(|t| t.pnl).sum();
        let gross_loss: Decimal = losses.iter().map(|t| t.pnl.abs()).sum();
        let profit_factor = if gross_loss > Decimal::ZERO {
            Some(gross_profit / gross_loss)
        } else if gross_profit > Decimal::ZERO {
            Some(dec!(999)) // Effectively infinite
        } else {
            None
        };

        let total_pnl: Decimal = trades.iter().map(|t| t.pnl).sum();
        let return_pct = if self.initial_capital > Decimal::ZERO {
            total_pnl / self.initial_capital * dec!(100)
        } else {
            Decimal::ZERO
        };

        let avg_hold_mins = trades.iter().map(|t| t.hold_duration_mins).sum::<i64>() / total_trades as i64;

        // Calculate Sharpe ratio
        let sharpe_ratio = self.calculate_sharpe(&trades);

        // Calculate drawdown
        let (max_drawdown_pct, current_drawdown_pct) = self.calculate_drawdowns();

        PerformanceMetrics {
            win_rate,
            avg_win,
            avg_loss,
            profit_factor,
            total_pnl,
            return_pct,
            trade_count: total_trades,
            avg_hold_mins,
            sharpe_ratio,
            max_drawdown_pct,
            current_drawdown_pct,
            win_streak: self.current_win_streak,
            loss_streak: self.current_loss_streak,
            calculated_at: Utc::now(),
        }
    }

    fn calculate_sharpe(&self, trades: &[&CompletedTrade]) -> Option<Decimal> {
        if trades.len() < 10 {
            return None;
        }

        let returns: Vec<Decimal> = trades.iter().map(|t| t.pnl_pct).collect();
        let n = Decimal::from(returns.len() as u32);
        let mean = returns.iter().copied().sum::<Decimal>() / n;

        let variance: Decimal = returns
            .iter()
            .map(|r| (*r - mean) * (*r - mean))
            .sum::<Decimal>() / n;

        let std_dev = variance.sqrt()?;
        if std_dev == Decimal::ZERO {
            return None;
        }

        // Annualize assuming ~4 trades per day
        let annual_factor = dec!(365) * dec!(4);
        let annual_return = mean * annual_factor;
        let annual_std = std_dev * annual_factor.sqrt()?;

        Some(annual_return / annual_std)
    }

    fn calculate_drawdowns(&self) -> (Decimal, Decimal) {
        if self.equity_curve.len() < 2 {
            return (Decimal::ZERO, Decimal::ZERO);
        }

        let mut max_drawdown = Decimal::ZERO;
        let mut peak = self.initial_capital;

        for (_, equity) in &self.equity_curve {
            if *equity > peak {
                peak = *equity;
            }
            if peak > Decimal::ZERO {
                let drawdown = (peak - *equity) / peak;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }
        }

        let current_equity = self.equity_curve.last().map(|(_, e)| *e).unwrap_or(self.initial_capital);
        let current_drawdown = if self.peak_equity > Decimal::ZERO {
            (self.peak_equity - current_equity) / self.peak_equity
        } else {
            Decimal::ZERO
        };

        (max_drawdown * dec!(100), current_drawdown * dec!(100))
    }

    fn check_alerts(&mut self) {
        let metrics = self.calculate_metrics();

        // Check losing streak
        if self.current_loss_streak >= self.config.losing_streak_threshold {
            self.add_alert(PerformanceAlert {
                alert_type: AlertType::LosingStreak,
                severity: if self.current_loss_streak >= self.config.losing_streak_threshold + 2 {
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                },
                message: format!("{} consecutive losses", self.current_loss_streak),
                timestamp: Utc::now(),
                suggested_action: "Consider reducing position sizes or pausing trading".to_string(),
            });
        }

        // Check drawdown
        if metrics.current_drawdown_pct >= self.config.drawdown_threshold * dec!(100) {
            self.add_alert(PerformanceAlert {
                alert_type: AlertType::DrawdownBreach,
                severity: AlertSeverity::Critical,
                message: format!("Drawdown at {:.1}%", metrics.current_drawdown_pct),
                timestamp: Utc::now(),
                suggested_action: "Stop trading and review strategy".to_string(),
            });
        }

        // Check win rate decline
        if let Some(ref prev) = self.previous_metrics {
            let decline = prev.win_rate - metrics.win_rate;
            if decline >= self.config.win_rate_decline_threshold {
                self.add_alert(PerformanceAlert {
                    alert_type: AlertType::WinRateDecline,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "Win rate declined from {:.1}% to {:.1}%",
                        prev.win_rate * dec!(100),
                        metrics.win_rate * dec!(100)
                    ),
                    timestamp: Utc::now(),
                    suggested_action: "Review recent losing trades for patterns".to_string(),
                });
            }
        }

        // Check positive streak
        if self.current_win_streak >= 5 {
            self.add_alert(PerformanceAlert {
                alert_type: AlertType::PositiveStreak,
                severity: AlertSeverity::Info,
                message: format!("{} consecutive wins!", self.current_win_streak),
                timestamp: Utc::now(),
                suggested_action: "Consider slightly increasing position sizes".to_string(),
            });
        }

        // Store metrics for next comparison
        self.previous_metrics = Some(metrics);
    }

    fn add_alert(&mut self, alert: PerformanceAlert) {
        // Dedupe: don't add same alert type within 1 hour
        let dominated = self.alerts.iter().any(|a| {
            a.alert_type == alert.alert_type
                && (Utc::now() - a.timestamp) < Duration::hours(1)
        });

        if !dominated {
            self.alerts.push(alert);
        }

        // Keep only last 100 alerts
        while self.alerts.len() > 100 {
            self.alerts.remove(0);
        }
    }

    fn detect_regime(&mut self) {
        if self.trades.len() < self.config.min_trades_for_regime {
            self.current_regime = MarketRegime::Unknown;
            return;
        }

        // Calculate metrics for regime detection
        let recent_pnls: Vec<Decimal> = self.trades.iter().map(|t| t.pnl_pct).collect();
        let n = recent_pnls.len() as i64;

        // Mean return
        let mean = recent_pnls.iter().copied().sum::<Decimal>() / Decimal::from(n as u32);

        // Volatility (std dev of returns)
        let variance: Decimal = recent_pnls
            .iter()
            .map(|r| (*r - mean) * (*r - mean))
            .sum::<Decimal>() / Decimal::from(n as u32);
        let volatility = variance.sqrt().unwrap_or(Decimal::ZERO);

        // Calculate trend (simple linear regression slope)
        let trend = self.calculate_trend(&recent_pnls);

        // Classify regime
        self.current_regime = if volatility > dec!(0.05) {
            MarketRegime::Volatile
        } else if trend > dec!(0.001) {
            MarketRegime::TrendingUp
        } else if trend < dec!(-0.001) {
            MarketRegime::TrendingDown
        } else if volatility < dec!(0.01) {
            MarketRegime::Quiet
        } else {
            MarketRegime::Ranging
        };
    }

    fn calculate_trend(&self, values: &[Decimal]) -> Decimal {
        let n = values.len();
        if n < 3 {
            return Decimal::ZERO;
        }

        // Simple: compare first half avg to second half avg
        let first_half: Decimal = values.iter().take(n / 2).copied().sum::<Decimal>()
            / Decimal::from((n / 2) as u32);
        let second_half: Decimal = values.iter().skip(n / 2).copied().sum::<Decimal>()
            / Decimal::from((n - n / 2) as u32);

        second_half - first_half
    }

    /// Get current market regime
    pub fn get_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get recommended strategy adjustments
    pub fn get_adjustments(&self) -> RegimeAdjustments {
        self.current_regime.strategy_adjustments()
    }

    /// Get recent alerts
    pub fn get_alerts(&self, since: Option<DateTime<Utc>>) -> Vec<&PerformanceAlert> {
        match since {
            Some(t) => self.alerts.iter().filter(|a| a.timestamp >= t).collect(),
            None => self.alerts.iter().collect(),
        }
    }

    /// Get critical alerts
    pub fn get_critical_alerts(&self) -> Vec<&PerformanceAlert> {
        self.alerts
            .iter()
            .filter(|a| a.severity == AlertSeverity::Critical)
            .collect()
    }

    /// Get performance by signal type
    pub fn get_performance_by_signal(&self) -> HashMap<String, PerformanceMetrics> {
        self.performance_by_signal
            .iter()
            .map(|(signal_type, trades)| {
                let trades_vec: Vec<_> = trades.iter().collect();
                let metrics = self.calculate_metrics_for_trades(&trades_vec);
                (signal_type.clone(), metrics)
            })
            .collect()
    }

    fn calculate_metrics_for_trades(&self, trades: &[&CompletedTrade]) -> PerformanceMetrics {
        if trades.is_empty() {
            return PerformanceMetrics {
                win_rate: Decimal::ZERO,
                avg_win: Decimal::ZERO,
                avg_loss: Decimal::ZERO,
                profit_factor: None,
                total_pnl: Decimal::ZERO,
                return_pct: Decimal::ZERO,
                trade_count: 0,
                avg_hold_mins: 0,
                sharpe_ratio: None,
                max_drawdown_pct: Decimal::ZERO,
                current_drawdown_pct: Decimal::ZERO,
                win_streak: 0,
                loss_streak: 0,
                calculated_at: Utc::now(),
            };
        }

        let total = trades.len();
        let wins: Vec<_> = trades.iter().filter(|t| t.pnl > Decimal::ZERO).collect();
        let losses: Vec<_> = trades.iter().filter(|t| t.pnl <= Decimal::ZERO).collect();

        let win_rate = Decimal::from(wins.len() as u32) / Decimal::from(total as u32);
        let total_pnl: Decimal = trades.iter().map(|t| t.pnl).sum();

        let avg_win = if !wins.is_empty() {
            wins.iter().map(|t| t.pnl).sum::<Decimal>() / Decimal::from(wins.len() as u32)
        } else {
            Decimal::ZERO
        };

        let avg_loss = if !losses.is_empty() {
            losses.iter().map(|t| t.pnl.abs()).sum::<Decimal>() / Decimal::from(losses.len() as u32)
        } else {
            Decimal::ZERO
        };

        PerformanceMetrics {
            win_rate,
            avg_win,
            avg_loss,
            profit_factor: None,
            total_pnl,
            return_pct: Decimal::ZERO,
            trade_count: total,
            avg_hold_mins: 0,
            sharpe_ratio: None,
            max_drawdown_pct: Decimal::ZERO,
            current_drawdown_pct: Decimal::ZERO,
            win_streak: 0,
            loss_streak: 0,
            calculated_at: Utc::now(),
        }
    }

    /// Check if we should pause trading
    pub fn should_pause_trading(&self) -> (bool, Option<String>) {
        let critical = self.get_critical_alerts();
        if !critical.is_empty() {
            let reason = critical
                .first()
                .map(|a| a.message.clone())
                .unwrap_or_default();
            return (true, Some(reason));
        }

        if self.current_loss_streak >= self.config.losing_streak_threshold + 2 {
            return (true, Some(format!("{} consecutive losses", self.current_loss_streak)));
        }

        (false, None)
    }

    /// Get summary report
    pub fn get_summary(&self) -> String {
        let metrics = self.calculate_metrics();
        format!(
            "Trades: {} | Win Rate: {:.1}% | PnL: ${:.2} ({:+.1}%) | Sharpe: {} | DD: {:.1}% | Regime: {:?}",
            metrics.trade_count,
            metrics.win_rate * dec!(100),
            metrics.total_pnl,
            metrics.return_pct,
            metrics.sharpe_ratio.map(|s| format!("{:.2}", s)).unwrap_or("N/A".to_string()),
            metrics.max_drawdown_pct,
            self.current_regime
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(pnl: Decimal, signal: &str) -> CompletedTrade {
        let now = Utc::now();
        CompletedTrade {
            trade_id: uuid::Uuid::new_v4().to_string(),
            market_id: "test_market".to_string(),
            signal_type: signal.to_string(),
            entry_time: now - Duration::minutes(30),
            exit_time: now,
            pnl,
            pnl_pct: pnl / dec!(100), // Assume $100 position
            hold_duration_mins: 30,
            entry_price: dec!(0.50),
            exit_price: dec!(0.55),
            size: dec!(100),
        }
    }

    #[test]
    fn test_monitor_basic() {
        let mut monitor = PerformanceMonitor::with_defaults(dec!(1000));
        
        monitor.record_trade(make_trade(dec!(10), "llm"));
        monitor.record_trade(make_trade(dec!(-5), "llm"));
        monitor.record_trade(make_trade(dec!(15), "tech"));
        
        let metrics = monitor.calculate_metrics();
        assert_eq!(metrics.trade_count, 3);
        assert_eq!(metrics.total_pnl, dec!(20));
    }

    #[test]
    fn test_win_rate() {
        let mut monitor = PerformanceMonitor::with_defaults(dec!(1000));
        
        // 2 wins, 1 loss = 66.7%
        monitor.record_trade(make_trade(dec!(10), "llm"));
        monitor.record_trade(make_trade(dec!(10), "llm"));
        monitor.record_trade(make_trade(dec!(-5), "llm"));
        
        let metrics = monitor.calculate_metrics();
        assert!(metrics.win_rate > dec!(0.6));
        assert!(metrics.win_rate < dec!(0.7));
    }

    #[test]
    fn test_losing_streak_alert() {
        let mut monitor = PerformanceMonitor::with_defaults(dec!(1000));
        
        // Create losing streak
        for _ in 0..6 {
            monitor.record_trade(make_trade(dec!(-10), "llm"));
        }
        
        assert!(monitor.current_loss_streak >= 5);
        let alerts = monitor.get_alerts(None);
        assert!(alerts.iter().any(|a| a.alert_type == AlertType::LosingStreak));
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut monitor = PerformanceMonitor::with_defaults(dec!(1000));
        
        // Make some profit
        monitor.record_trade(make_trade(dec!(100), "llm"));
        // Then lose
        monitor.record_trade(make_trade(dec!(-150), "llm"));
        
        let metrics = monitor.calculate_metrics();
        assert!(metrics.current_drawdown_pct > Decimal::ZERO);
    }

    #[test]
    fn test_regime_detection() {
        let mut monitor = PerformanceMonitor::new(
            MonitorConfig {
                min_trades_for_regime: 5,
                ..Default::default()
            },
            dec!(1000),
        );
        
        // Add consistent winning trades (trending up)
        for i in 0..10 {
            monitor.record_trade(make_trade(dec!(5) + Decimal::from(i), "llm"));
        }
        
        assert!(matches!(
            monitor.get_regime(),
            MarketRegime::TrendingUp | MarketRegime::Ranging | MarketRegime::Quiet
        ));
    }

    #[test]
    fn test_performance_by_signal() {
        let mut monitor = PerformanceMonitor::with_defaults(dec!(1000));
        
        monitor.record_trade(make_trade(dec!(10), "llm"));
        monitor.record_trade(make_trade(dec!(20), "llm"));
        monitor.record_trade(make_trade(dec!(-5), "tech"));
        
        let by_signal = monitor.get_performance_by_signal();
        
        assert!(by_signal.contains_key("llm"));
        assert!(by_signal.contains_key("tech"));
        
        let llm_perf = &by_signal["llm"];
        assert_eq!(llm_perf.trade_count, 2);
        assert_eq!(llm_perf.total_pnl, dec!(30));
    }

    #[test]
    fn test_regime_adjustments() {
        let adj = MarketRegime::Volatile.strategy_adjustments();
        assert!(adj.position_size_mult < dec!(1));  // Smaller positions
        assert!(adj.min_edge_mult > dec!(1));       // Higher edge required
    }

    #[test]
    fn test_should_pause() {
        let mut monitor = PerformanceMonitor::new(
            MonitorConfig {
                losing_streak_threshold: 3,
                ..Default::default()
            },
            dec!(1000),
        );
        
        // Not pausing initially
        assert!(!monitor.should_pause_trading().0);
        
        // Create severe losing streak
        for _ in 0..6 {
            monitor.record_trade(make_trade(dec!(-10), "llm"));
        }
        
        let (should_pause, reason) = monitor.should_pause_trading();
        assert!(should_pause);
        assert!(reason.is_some());
    }

    #[test]
    fn test_summary() {
        let mut monitor = PerformanceMonitor::with_defaults(dec!(1000));
        
        monitor.record_trade(make_trade(dec!(10), "llm"));
        
        let summary = monitor.get_summary();
        assert!(summary.contains("Trades: 1"));
        assert!(summary.contains("Win Rate"));
    }
}
