//! Execution Quality Metrics Module
//!
//! Professional-grade execution quality analysis for quantitative trading.
//! Measures how well orders are executed compared to benchmarks.
//!
//! # Features
//! - Slippage analysis (expected vs actual price)
//! - Implementation shortfall (decision price vs execution)
//! - Fill rate and partial fill tracking
//! - Latency distribution analysis
//! - Best execution scoring
//! - VWAP/TWAP benchmark comparison
//! - Execution cost attribution
//!
//! # Example
//! ```ignore
//! use polymarket_bot::execution_quality::{ExecutionAnalyzer, ExecutionRecord};
//!
//! let mut analyzer = ExecutionAnalyzer::new();
//! analyzer.record_execution(ExecutionRecord {
//!     order_id: "order123".to_string(),
//!     symbol: "BTC-USD".to_string(),
//!     side: OrderSide::Buy,
//!     decision_price: dec!(100000.0),
//!     arrival_price: dec!(100005.0),
//!     execution_price: dec!(100010.0),
//!     quantity_ordered: dec!(1.0),
//!     quantity_filled: dec!(1.0),
//!     decision_time: Utc::now() - Duration::seconds(5),
//!     arrival_time: Utc::now() - Duration::seconds(3),
//!     execution_time: Utc::now(),
//!     fees: dec!(10.0),
//! });
//!
//! let metrics = analyzer.compute_metrics();
//! println!("Average slippage: {} bps", metrics.avg_slippage_bps);
//! ```

use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Order side for execution tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Complete record of a single execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    /// Unique order identifier
    pub order_id: String,
    /// Trading symbol
    pub symbol: String,
    /// Buy or sell
    pub side: OrderSide,
    /// Price when decision was made (signal generated)
    pub decision_price: Decimal,
    /// Price when order arrived at exchange
    pub arrival_price: Decimal,
    /// Actual execution price (or average for multiple fills)
    pub execution_price: Decimal,
    /// Original order quantity
    pub quantity_ordered: Decimal,
    /// Actually filled quantity
    pub quantity_filled: Decimal,
    /// Time of trading decision
    pub decision_time: DateTime<Utc>,
    /// Time order arrived at exchange
    pub arrival_time: DateTime<Utc>,
    /// Time of execution (last fill)
    pub execution_time: DateTime<Utc>,
    /// Total fees paid
    pub fees: Decimal,
    /// Optional VWAP benchmark for comparison
    pub vwap_benchmark: Option<Decimal>,
    /// Optional TWAP benchmark for comparison
    pub twap_benchmark: Option<Decimal>,
}

/// Breakdown of execution costs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostAttribution {
    /// Cost from price delay (decision to arrival)
    pub delay_cost_bps: Decimal,
    /// Cost from market impact
    pub market_impact_bps: Decimal,
    /// Cost from timing (arrival to execution)
    pub timing_cost_bps: Decimal,
    /// Explicit fee cost
    pub fee_cost_bps: Decimal,
    /// Total implementation shortfall
    pub total_shortfall_bps: Decimal,
}

/// Latency statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Mean decision-to-execution latency (ms)
    pub mean_latency_ms: f64,
    /// Median latency (ms)
    pub median_latency_ms: f64,
    /// 95th percentile latency (ms)
    pub p95_latency_ms: f64,
    /// 99th percentile latency (ms)
    pub p99_latency_ms: f64,
    /// Minimum latency (ms)
    pub min_latency_ms: f64,
    /// Maximum latency (ms)
    pub max_latency_ms: f64,
    /// Standard deviation (ms)
    pub std_dev_ms: f64,
}

/// Fill quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FillMetrics {
    /// Overall fill rate (filled / ordered)
    pub fill_rate: Decimal,
    /// Count of complete fills
    pub complete_fills: usize,
    /// Count of partial fills
    pub partial_fills: usize,
    /// Count of zero fills (order rejected/expired)
    pub zero_fills: usize,
    /// Average fill ratio for partial fills
    pub avg_partial_fill_ratio: Decimal,
}

/// Slippage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SlippageStats {
    /// Average slippage in basis points
    pub avg_slippage_bps: Decimal,
    /// Median slippage in basis points
    pub median_slippage_bps: Decimal,
    /// Maximum adverse slippage
    pub max_adverse_slippage_bps: Decimal,
    /// Maximum favorable slippage (price improvement)
    pub max_favorable_slippage_bps: Decimal,
    /// Percentage of trades with positive slippage (worse than expected)
    pub adverse_slippage_pct: Decimal,
    /// Percentage of trades with price improvement
    pub price_improvement_pct: Decimal,
}

/// Benchmark comparison metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Average performance vs VWAP (bps, negative = beat)
    pub vs_vwap_bps: Decimal,
    /// Average performance vs TWAP (bps, negative = beat)
    pub vs_twap_bps: Decimal,
    /// Percentage of trades that beat VWAP
    pub vwap_beat_rate: Decimal,
    /// Percentage of trades that beat TWAP
    pub twap_beat_rate: Decimal,
}

/// Comprehensive execution quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionQualityMetrics {
    /// Total number of executions analyzed
    pub total_executions: usize,
    /// Total volume traded
    pub total_volume: Decimal,
    /// Slippage statistics
    pub slippage: SlippageStats,
    /// Fill quality metrics
    pub fills: FillMetrics,
    /// Latency statistics
    pub latency: LatencyStats,
    /// Cost attribution breakdown
    pub cost_attribution: CostAttribution,
    /// Benchmark comparison
    pub benchmarks: BenchmarkMetrics,
    /// Best execution score (0-100, higher is better)
    pub best_execution_score: Decimal,
    /// Per-symbol metrics
    pub by_symbol: HashMap<String, SymbolMetrics>,
}

/// Per-symbol execution metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolMetrics {
    pub executions: usize,
    pub volume: Decimal,
    pub avg_slippage_bps: Decimal,
    pub fill_rate: Decimal,
    pub avg_latency_ms: f64,
}

/// Execution quality analyzer
#[derive(Debug)]
pub struct ExecutionAnalyzer {
    /// All recorded executions
    records: VecDeque<ExecutionRecord>,
    /// Maximum records to keep
    max_records: usize,
    /// Per-symbol statistics
    symbol_stats: HashMap<String, SymbolExecutionStats>,
}

#[derive(Debug, Default)]
struct SymbolExecutionStats {
    total_volume: Decimal,
    total_slippage_bps_weighted: Decimal,
    total_latency_ms: f64,
    count: usize,
    filled_volume: Decimal,
    ordered_volume: Decimal,
}

impl ExecutionAnalyzer {
    /// Create a new execution analyzer
    pub fn new() -> Self {
        Self::with_capacity(10000)
    }

    /// Create analyzer with specific capacity
    pub fn with_capacity(max_records: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(max_records),
            max_records,
            symbol_stats: HashMap::new(),
        }
    }

    /// Record a new execution
    pub fn record_execution(&mut self, record: ExecutionRecord) {
        // Calculate slippage first before borrowing symbol_stats
        let volume = record.quantity_filled * record.execution_price;
        let slippage_bps = self.calculate_slippage_bps(&record);
        let latency_ms = (record.execution_time - record.decision_time)
            .num_milliseconds() as f64;
        let symbol = record.symbol.clone();
        let filled = record.quantity_filled;
        let ordered = record.quantity_ordered;

        // Update symbol stats
        let stats = self.symbol_stats.entry(symbol).or_default();
        stats.total_volume += volume;
        stats.total_slippage_bps_weighted += slippage_bps * volume;
        stats.total_latency_ms += latency_ms;
        stats.count += 1;
        stats.filled_volume += filled;
        stats.ordered_volume += ordered;

        // Add record
        if self.records.len() >= self.max_records {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    /// Calculate slippage in basis points for a single record
    fn calculate_slippage_bps(&self, record: &ExecutionRecord) -> Decimal {
        if record.decision_price.is_zero() {
            return Decimal::ZERO;
        }

        let price_diff = match record.side {
            OrderSide::Buy => record.execution_price - record.decision_price,
            OrderSide::Sell => record.decision_price - record.execution_price,
        };

        (price_diff / record.decision_price) * dec!(10000)
    }

    /// Calculate implementation shortfall breakdown
    fn calculate_cost_attribution(&self, record: &ExecutionRecord) -> CostAttribution {
        if record.decision_price.is_zero() {
            return CostAttribution::default();
        }

        let notional = record.quantity_filled * record.decision_price;
        if notional.is_zero() {
            return CostAttribution::default();
        }

        // Delay cost: decision to arrival
        let delay_cost = match record.side {
            OrderSide::Buy => record.arrival_price - record.decision_price,
            OrderSide::Sell => record.decision_price - record.arrival_price,
        };
        let delay_cost_bps = (delay_cost / record.decision_price) * dec!(10000);

        // Market impact: arrival to execution
        let market_impact = match record.side {
            OrderSide::Buy => record.execution_price - record.arrival_price,
            OrderSide::Sell => record.arrival_price - record.execution_price,
        };
        let market_impact_bps = (market_impact / record.decision_price) * dec!(10000);

        // Fee cost
        let fee_cost_bps = (record.fees / notional) * dec!(10000);

        // Timing cost (simplified - could be more sophisticated)
        let timing_cost_bps = Decimal::ZERO;

        let total_shortfall_bps = delay_cost_bps + market_impact_bps + timing_cost_bps + fee_cost_bps;

        CostAttribution {
            delay_cost_bps,
            market_impact_bps,
            timing_cost_bps,
            fee_cost_bps,
            total_shortfall_bps,
        }
    }

    /// Compute comprehensive execution quality metrics
    pub fn compute_metrics(&self) -> ExecutionQualityMetrics {
        if self.records.is_empty() {
            return ExecutionQualityMetrics::default();
        }

        let mut metrics = ExecutionQualityMetrics {
            total_executions: self.records.len(),
            ..Default::default()
        };

        // Collect data for analysis
        let mut slippages_bps: Vec<Decimal> = Vec::new();
        let mut latencies_ms: Vec<f64> = Vec::new();
        let mut total_volume = Decimal::ZERO;
        let mut total_ordered = Decimal::ZERO;
        let mut total_filled = Decimal::ZERO;
        let mut complete_fills = 0usize;
        let mut partial_fills = 0usize;
        let mut zero_fills = 0usize;
        let mut partial_fill_ratios: Vec<Decimal> = Vec::new();
        let mut adverse_slippage_count = 0usize;
        let mut price_improvement_count = 0usize;
        let mut total_delay_cost = Decimal::ZERO;
        let mut total_impact_cost = Decimal::ZERO;
        let mut total_fee_cost = Decimal::ZERO;
        let mut vwap_comparisons: Vec<Decimal> = Vec::new();
        let mut twap_comparisons: Vec<Decimal> = Vec::new();
        let mut vwap_beats = 0usize;
        let mut twap_beats = 0usize;

        for record in &self.records {
            let volume = record.quantity_filled * record.execution_price;
            total_volume += volume;
            total_ordered += record.quantity_ordered;
            total_filled += record.quantity_filled;

            // Slippage
            let slippage = self.calculate_slippage_bps(record);
            slippages_bps.push(slippage);
            if slippage > Decimal::ZERO {
                adverse_slippage_count += 1;
            } else if slippage < Decimal::ZERO {
                price_improvement_count += 1;
            }

            // Latency
            let latency = (record.execution_time - record.decision_time)
                .num_milliseconds() as f64;
            latencies_ms.push(latency);

            // Fill tracking
            if record.quantity_filled.is_zero() {
                zero_fills += 1;
            } else if record.quantity_filled >= record.quantity_ordered {
                complete_fills += 1;
            } else {
                partial_fills += 1;
                if !record.quantity_ordered.is_zero() {
                    partial_fill_ratios.push(record.quantity_filled / record.quantity_ordered);
                }
            }

            // Cost attribution
            let costs = self.calculate_cost_attribution(record);
            total_delay_cost += costs.delay_cost_bps * volume;
            total_impact_cost += costs.market_impact_bps * volume;
            total_fee_cost += costs.fee_cost_bps * volume;

            // Benchmark comparison
            if let Some(vwap) = record.vwap_benchmark {
                if !vwap.is_zero() {
                    let vs_vwap = match record.side {
                        OrderSide::Buy => (record.execution_price - vwap) / vwap * dec!(10000),
                        OrderSide::Sell => (vwap - record.execution_price) / vwap * dec!(10000),
                    };
                    vwap_comparisons.push(vs_vwap);
                    if vs_vwap < Decimal::ZERO {
                        vwap_beats += 1;
                    }
                }
            }

            if let Some(twap) = record.twap_benchmark {
                if !twap.is_zero() {
                    let vs_twap = match record.side {
                        OrderSide::Buy => (record.execution_price - twap) / twap * dec!(10000),
                        OrderSide::Sell => (twap - record.execution_price) / twap * dec!(10000),
                    };
                    twap_comparisons.push(vs_twap);
                    if vs_twap < Decimal::ZERO {
                        twap_beats += 1;
                    }
                }
            }
        }

        metrics.total_volume = total_volume;

        // Slippage stats
        if !slippages_bps.is_empty() {
            slippages_bps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = slippages_bps.len();
            let sum: Decimal = slippages_bps.iter().copied().sum();
            metrics.slippage.avg_slippage_bps = sum / Decimal::from(n);
            metrics.slippage.median_slippage_bps = slippages_bps[n / 2];
            metrics.slippage.max_adverse_slippage_bps = *slippages_bps.last().unwrap_or(&Decimal::ZERO);
            metrics.slippage.max_favorable_slippage_bps = *slippages_bps.first().unwrap_or(&Decimal::ZERO);
            metrics.slippage.adverse_slippage_pct = Decimal::from(adverse_slippage_count) / Decimal::from(n) * dec!(100);
            metrics.slippage.price_improvement_pct = Decimal::from(price_improvement_count) / Decimal::from(n) * dec!(100);
        }

        // Fill metrics
        metrics.fills.complete_fills = complete_fills;
        metrics.fills.partial_fills = partial_fills;
        metrics.fills.zero_fills = zero_fills;
        if !total_ordered.is_zero() {
            metrics.fills.fill_rate = total_filled / total_ordered;
        }
        if !partial_fill_ratios.is_empty() {
            let sum: Decimal = partial_fill_ratios.iter().copied().sum();
            metrics.fills.avg_partial_fill_ratio = sum / Decimal::from(partial_fill_ratios.len());
        }

        // Latency stats
        if !latencies_ms.is_empty() {
            latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = latencies_ms.len();
            let sum: f64 = latencies_ms.iter().sum();
            metrics.latency.mean_latency_ms = sum / n as f64;
            metrics.latency.median_latency_ms = latencies_ms[n / 2];
            metrics.latency.p95_latency_ms = latencies_ms[(n as f64 * 0.95) as usize];
            metrics.latency.p99_latency_ms = latencies_ms[(n as f64 * 0.99).min((n - 1) as f64) as usize];
            metrics.latency.min_latency_ms = latencies_ms[0];
            metrics.latency.max_latency_ms = latencies_ms[n - 1];

            // Standard deviation
            let mean = metrics.latency.mean_latency_ms;
            let variance: f64 = latencies_ms.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / n as f64;
            metrics.latency.std_dev_ms = variance.sqrt();
        }

        // Cost attribution (volume-weighted)
        if !total_volume.is_zero() {
            metrics.cost_attribution.delay_cost_bps = total_delay_cost / total_volume;
            metrics.cost_attribution.market_impact_bps = total_impact_cost / total_volume;
            metrics.cost_attribution.fee_cost_bps = total_fee_cost / total_volume;
            metrics.cost_attribution.total_shortfall_bps = 
                metrics.cost_attribution.delay_cost_bps +
                metrics.cost_attribution.market_impact_bps +
                metrics.cost_attribution.fee_cost_bps;
        }

        // Benchmark metrics
        if !vwap_comparisons.is_empty() {
            let sum: Decimal = vwap_comparisons.iter().copied().sum();
            metrics.benchmarks.vs_vwap_bps = sum / Decimal::from(vwap_comparisons.len());
            metrics.benchmarks.vwap_beat_rate = Decimal::from(vwap_beats) / Decimal::from(vwap_comparisons.len()) * dec!(100);
        }
        if !twap_comparisons.is_empty() {
            let sum: Decimal = twap_comparisons.iter().copied().sum();
            metrics.benchmarks.vs_twap_bps = sum / Decimal::from(twap_comparisons.len());
            metrics.benchmarks.twap_beat_rate = Decimal::from(twap_beats) / Decimal::from(twap_comparisons.len()) * dec!(100);
        }

        // Per-symbol metrics
        for (symbol, stats) in &self.symbol_stats {
            if stats.count > 0 {
                let avg_slippage = if stats.total_volume.is_zero() {
                    Decimal::ZERO
                } else {
                    stats.total_slippage_bps_weighted / stats.total_volume
                };
                let fill_rate = if stats.ordered_volume.is_zero() {
                    Decimal::ZERO
                } else {
                    stats.filled_volume / stats.ordered_volume
                };

                metrics.by_symbol.insert(symbol.clone(), SymbolMetrics {
                    executions: stats.count,
                    volume: stats.total_volume,
                    avg_slippage_bps: avg_slippage,
                    fill_rate,
                    avg_latency_ms: stats.total_latency_ms / stats.count as f64,
                });
            }
        }

        // Best Execution Score (0-100)
        // Composite score based on multiple factors
        metrics.best_execution_score = self.calculate_best_execution_score(&metrics);

        metrics
    }

    /// Calculate best execution score (0-100)
    fn calculate_best_execution_score(&self, metrics: &ExecutionQualityMetrics) -> Decimal {
        // Weight factors for scoring
        let mut score = dec!(100);

        // Slippage penalty (max 30 points)
        // -1 point per 2 bps of adverse slippage
        let slippage_penalty = (metrics.slippage.avg_slippage_bps / dec!(2))
            .min(dec!(30))
            .max(Decimal::ZERO);
        score -= slippage_penalty;

        // Fill rate bonus/penalty (max 20 points)
        let fill_rate_score = metrics.fills.fill_rate * dec!(20);
        score = score - dec!(20) + fill_rate_score;

        // Latency penalty (max 20 points)
        // -1 point per 100ms above 50ms baseline
        let latency_penalty = Decimal::from_f64_retain(
            ((metrics.latency.mean_latency_ms - 50.0).max(0.0) / 100.0).min(20.0)
        ).unwrap_or(Decimal::ZERO);
        score -= latency_penalty;

        // Implementation shortfall penalty (max 20 points)
        let shortfall_penalty = (metrics.cost_attribution.total_shortfall_bps / dec!(5))
            .min(dec!(20))
            .max(Decimal::ZERO);
        score -= shortfall_penalty;

        // Benchmark beat bonus (max 10 points)
        let benchmark_bonus = (metrics.benchmarks.vwap_beat_rate / dec!(10)).min(dec!(10));
        score += benchmark_bonus;

        score.max(Decimal::ZERO).min(dec!(100))
    }

    /// Get execution records for a specific symbol
    pub fn get_symbol_records(&self, symbol: &str) -> Vec<&ExecutionRecord> {
        self.records.iter()
            .filter(|r| r.symbol == symbol)
            .collect()
    }

    /// Get recent executions
    pub fn get_recent(&self, count: usize) -> Vec<&ExecutionRecord> {
        self.records.iter().rev().take(count).collect()
    }

    /// Get executions within a time range
    pub fn get_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&ExecutionRecord> {
        self.records.iter()
            .filter(|r| r.execution_time >= start && r.execution_time <= end)
            .collect()
    }

    /// Clear all records
    pub fn clear(&mut self) {
        self.records.clear();
        self.symbol_stats.clear();
    }

    /// Get total number of records
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if analyzer is empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl Default for ExecutionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Real-time execution monitor for live trading
#[derive(Debug)]
pub struct ExecutionMonitor {
    analyzer: ExecutionAnalyzer,
    /// Alert threshold for slippage (bps)
    slippage_alert_threshold_bps: Decimal,
    /// Alert threshold for latency (ms)
    latency_alert_threshold_ms: f64,
    /// Alert threshold for fill rate
    fill_rate_alert_threshold: Decimal,
    /// Recent alerts
    alerts: VecDeque<ExecutionAlert>,
    /// Max alerts to keep
    max_alerts: usize,
}

/// Execution quality alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionAlert {
    pub timestamp: DateTime<Utc>,
    pub alert_type: AlertType,
    pub message: String,
    pub order_id: String,
    pub value: String,
}

/// Types of execution alerts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertType {
    HighSlippage,
    HighLatency,
    LowFillRate,
    PartialFill,
    ZeroFill,
}

impl ExecutionMonitor {
    /// Create a new execution monitor
    pub fn new() -> Self {
        Self {
            analyzer: ExecutionAnalyzer::new(),
            slippage_alert_threshold_bps: dec!(50), // 50 bps = 0.5%
            latency_alert_threshold_ms: 1000.0,      // 1 second
            fill_rate_alert_threshold: dec!(0.8),    // 80%
            alerts: VecDeque::with_capacity(1000),
            max_alerts: 1000,
        }
    }

    /// Configure alert thresholds
    pub fn set_thresholds(
        &mut self,
        slippage_bps: Decimal,
        latency_ms: f64,
        fill_rate: Decimal,
    ) {
        self.slippage_alert_threshold_bps = slippage_bps;
        self.latency_alert_threshold_ms = latency_ms;
        self.fill_rate_alert_threshold = fill_rate;
    }

    /// Process an execution and check for alerts
    pub fn process_execution(&mut self, record: ExecutionRecord) -> Vec<ExecutionAlert> {
        let mut new_alerts = Vec::new();

        // Check slippage
        let slippage_bps = if !record.decision_price.is_zero() {
            let diff = match record.side {
                OrderSide::Buy => record.execution_price - record.decision_price,
                OrderSide::Sell => record.decision_price - record.execution_price,
            };
            (diff / record.decision_price) * dec!(10000)
        } else {
            Decimal::ZERO
        };

        if slippage_bps > self.slippage_alert_threshold_bps {
            new_alerts.push(ExecutionAlert {
                timestamp: Utc::now(),
                alert_type: AlertType::HighSlippage,
                message: format!("High slippage detected: {} bps", slippage_bps),
                order_id: record.order_id.clone(),
                value: slippage_bps.to_string(),
            });
        }

        // Check latency
        let latency_ms = (record.execution_time - record.decision_time)
            .num_milliseconds() as f64;
        if latency_ms > self.latency_alert_threshold_ms {
            new_alerts.push(ExecutionAlert {
                timestamp: Utc::now(),
                alert_type: AlertType::HighLatency,
                message: format!("High latency detected: {:.1} ms", latency_ms),
                order_id: record.order_id.clone(),
                value: format!("{:.1}", latency_ms),
            });
        }

        // Check fill rate
        if record.quantity_filled.is_zero() {
            new_alerts.push(ExecutionAlert {
                timestamp: Utc::now(),
                alert_type: AlertType::ZeroFill,
                message: "Order not filled".to_string(),
                order_id: record.order_id.clone(),
                value: "0%".to_string(),
            });
        } else if !record.quantity_ordered.is_zero() {
            let fill_rate = record.quantity_filled / record.quantity_ordered;
            if fill_rate < self.fill_rate_alert_threshold && fill_rate < dec!(1) {
                new_alerts.push(ExecutionAlert {
                    timestamp: Utc::now(),
                    alert_type: AlertType::PartialFill,
                    message: format!("Partial fill: {}%", fill_rate * dec!(100)),
                    order_id: record.order_id.clone(),
                    value: format!("{}%", fill_rate * dec!(100)),
                });
            }
        }

        // Store alerts
        for alert in &new_alerts {
            if self.alerts.len() >= self.max_alerts {
                self.alerts.pop_front();
            }
            self.alerts.push_back(alert.clone());
        }

        // Record execution
        self.analyzer.record_execution(record);

        new_alerts
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> ExecutionQualityMetrics {
        self.analyzer.compute_metrics()
    }

    /// Get recent alerts
    pub fn get_alerts(&self, count: usize) -> Vec<&ExecutionAlert> {
        self.alerts.iter().rev().take(count).collect()
    }

    /// Get alerts by type
    pub fn get_alerts_by_type(&self, alert_type: AlertType) -> Vec<&ExecutionAlert> {
        self.alerts.iter()
            .filter(|a| a.alert_type == alert_type)
            .collect()
    }

    /// Get underlying analyzer
    pub fn analyzer(&self) -> &ExecutionAnalyzer {
        &self.analyzer
    }
}

impl Default for ExecutionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Trade Cost Analysis (TCA) report generator
#[derive(Debug)]
pub struct TcaReportGenerator;

impl TcaReportGenerator {
    /// Generate a comprehensive TCA report
    pub fn generate_report(metrics: &ExecutionQualityMetrics) -> String {
        let mut report = String::new();

        report.push_str("=== Trade Cost Analysis Report ===\n\n");
        
        report.push_str(&format!("Total Executions: {}\n", metrics.total_executions));
        report.push_str(&format!("Total Volume: ${:.2}\n\n", metrics.total_volume));

        report.push_str("--- Slippage Analysis ---\n");
        report.push_str(&format!("Average Slippage: {:.2} bps\n", metrics.slippage.avg_slippage_bps));
        report.push_str(&format!("Median Slippage: {:.2} bps\n", metrics.slippage.median_slippage_bps));
        report.push_str(&format!("Max Adverse: {:.2} bps\n", metrics.slippage.max_adverse_slippage_bps));
        report.push_str(&format!("Price Improvement Rate: {:.1}%\n\n", metrics.slippage.price_improvement_pct));

        report.push_str("--- Fill Quality ---\n");
        report.push_str(&format!("Fill Rate: {:.1}%\n", metrics.fills.fill_rate * dec!(100)));
        report.push_str(&format!("Complete Fills: {}\n", metrics.fills.complete_fills));
        report.push_str(&format!("Partial Fills: {}\n", metrics.fills.partial_fills));
        report.push_str(&format!("Zero Fills: {}\n\n", metrics.fills.zero_fills));

        report.push_str("--- Latency Statistics ---\n");
        report.push_str(&format!("Mean: {:.1} ms\n", metrics.latency.mean_latency_ms));
        report.push_str(&format!("Median: {:.1} ms\n", metrics.latency.median_latency_ms));
        report.push_str(&format!("P95: {:.1} ms\n", metrics.latency.p95_latency_ms));
        report.push_str(&format!("P99: {:.1} ms\n\n", metrics.latency.p99_latency_ms));

        report.push_str("--- Implementation Shortfall ---\n");
        report.push_str(&format!("Delay Cost: {:.2} bps\n", metrics.cost_attribution.delay_cost_bps));
        report.push_str(&format!("Market Impact: {:.2} bps\n", metrics.cost_attribution.market_impact_bps));
        report.push_str(&format!("Fee Cost: {:.2} bps\n", metrics.cost_attribution.fee_cost_bps));
        report.push_str(&format!("Total Shortfall: {:.2} bps\n\n", metrics.cost_attribution.total_shortfall_bps));

        if !metrics.benchmarks.vs_vwap_bps.is_zero() || !metrics.benchmarks.vs_twap_bps.is_zero() {
            report.push_str("--- Benchmark Comparison ---\n");
            report.push_str(&format!("vs VWAP: {:.2} bps (beat rate: {:.1}%)\n", 
                metrics.benchmarks.vs_vwap_bps, metrics.benchmarks.vwap_beat_rate));
            report.push_str(&format!("vs TWAP: {:.2} bps (beat rate: {:.1}%)\n\n",
                metrics.benchmarks.vs_twap_bps, metrics.benchmarks.twap_beat_rate));
        }

        report.push_str(&format!("*** Best Execution Score: {:.1}/100 ***\n", 
            metrics.best_execution_score));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_record(
        price_decision: f64,
        price_arrival: f64,
        price_execution: f64,
        side: OrderSide,
    ) -> ExecutionRecord {
        ExecutionRecord {
            order_id: format!("test_{}", rand::random::<u32>()),
            symbol: "BTC-USD".to_string(),
            side,
            decision_price: Decimal::from_f64_retain(price_decision).unwrap(),
            arrival_price: Decimal::from_f64_retain(price_arrival).unwrap(),
            execution_price: Decimal::from_f64_retain(price_execution).unwrap(),
            quantity_ordered: dec!(1.0),
            quantity_filled: dec!(1.0),
            decision_time: Utc::now() - Duration::milliseconds(100),
            arrival_time: Utc::now() - Duration::milliseconds(50),
            execution_time: Utc::now(),
            fees: dec!(1.0),
            vwap_benchmark: None,
            twap_benchmark: None,
        }
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ExecutionAnalyzer::new();
        assert!(analyzer.is_empty());
        assert_eq!(analyzer.len(), 0);
    }

    #[test]
    fn test_record_execution() {
        let mut analyzer = ExecutionAnalyzer::new();
        let record = create_test_record(100.0, 100.05, 100.10, OrderSide::Buy);
        analyzer.record_execution(record);
        assert_eq!(analyzer.len(), 1);
    }

    #[test]
    fn test_slippage_calculation_buy() {
        let mut analyzer = ExecutionAnalyzer::new();
        // Buy at 100.10 when decided at 100.00 = 10 bps slippage
        let record = create_test_record(100.0, 100.0, 100.10, OrderSide::Buy);
        analyzer.record_execution(record);
        
        let metrics = analyzer.compute_metrics();
        assert!(metrics.slippage.avg_slippage_bps > dec!(9) && 
                metrics.slippage.avg_slippage_bps < dec!(11));
    }

    #[test]
    fn test_slippage_calculation_sell() {
        let mut analyzer = ExecutionAnalyzer::new();
        // Sell at 99.90 when decided at 100.00 = 10 bps slippage
        let record = create_test_record(100.0, 100.0, 99.90, OrderSide::Sell);
        analyzer.record_execution(record);
        
        let metrics = analyzer.compute_metrics();
        assert!(metrics.slippage.avg_slippage_bps > dec!(9) && 
                metrics.slippage.avg_slippage_bps < dec!(11));
    }

    #[test]
    fn test_price_improvement() {
        let mut analyzer = ExecutionAnalyzer::new();
        // Buy at 99.90 when decided at 100.00 = -10 bps (price improvement)
        let record = create_test_record(100.0, 100.0, 99.90, OrderSide::Buy);
        analyzer.record_execution(record);
        
        let metrics = analyzer.compute_metrics();
        assert!(metrics.slippage.avg_slippage_bps < Decimal::ZERO);
        assert!(metrics.slippage.price_improvement_pct > Decimal::ZERO);
    }

    #[test]
    fn test_fill_metrics_complete() {
        let mut analyzer = ExecutionAnalyzer::new();
        let record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
        analyzer.record_execution(record);
        
        let metrics = analyzer.compute_metrics();
        assert_eq!(metrics.fills.complete_fills, 1);
        assert_eq!(metrics.fills.partial_fills, 0);
        assert_eq!(metrics.fills.fill_rate, dec!(1));
    }

    #[test]
    fn test_fill_metrics_partial() {
        let mut analyzer = ExecutionAnalyzer::new();
        let mut record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
        record.quantity_filled = dec!(0.5);
        analyzer.record_execution(record);
        
        let metrics = analyzer.compute_metrics();
        assert_eq!(metrics.fills.complete_fills, 0);
        assert_eq!(metrics.fills.partial_fills, 1);
        assert_eq!(metrics.fills.fill_rate, dec!(0.5));
    }

    #[test]
    fn test_fill_metrics_zero() {
        let mut analyzer = ExecutionAnalyzer::new();
        let mut record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
        record.quantity_filled = Decimal::ZERO;
        analyzer.record_execution(record);
        
        let metrics = analyzer.compute_metrics();
        assert_eq!(metrics.fills.zero_fills, 1);
    }

    #[test]
    fn test_latency_stats() {
        let mut analyzer = ExecutionAnalyzer::new();
        
        for _ in 0..100 {
            let mut record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
            record.decision_time = Utc::now() - Duration::milliseconds(100);
            record.execution_time = Utc::now();
            analyzer.record_execution(record);
        }
        
        let metrics = analyzer.compute_metrics();
        assert!(metrics.latency.mean_latency_ms > 90.0);
        assert!(metrics.latency.p95_latency_ms > 90.0);
    }

    #[test]
    fn test_cost_attribution() {
        let mut analyzer = ExecutionAnalyzer::new();
        // Decision: 100, Arrival: 100.05 (5 bps delay), Execution: 100.10 (5 bps impact)
        let record = create_test_record(100.0, 100.05, 100.10, OrderSide::Buy);
        analyzer.record_execution(record);
        
        let metrics = analyzer.compute_metrics();
        assert!(metrics.cost_attribution.delay_cost_bps > dec!(4));
        assert!(metrics.cost_attribution.market_impact_bps > dec!(4));
    }

    #[test]
    fn test_benchmark_comparison() {
        let mut analyzer = ExecutionAnalyzer::new();
        let mut record = create_test_record(100.0, 100.0, 99.95, OrderSide::Buy);
        record.vwap_benchmark = Some(dec!(100.0));
        record.twap_benchmark = Some(dec!(100.0));
        analyzer.record_execution(record);
        
        let metrics = analyzer.compute_metrics();
        assert!(metrics.benchmarks.vs_vwap_bps < Decimal::ZERO); // Beat VWAP
        assert!(metrics.benchmarks.vwap_beat_rate > Decimal::ZERO);
    }

    #[test]
    fn test_best_execution_score() {
        let mut analyzer = ExecutionAnalyzer::new();
        
        // Add good executions
        for _ in 0..10 {
            let record = create_test_record(100.0, 100.0, 100.01, OrderSide::Buy);
            analyzer.record_execution(record);
        }
        
        let metrics = analyzer.compute_metrics();
        assert!(metrics.best_execution_score > dec!(70)); // Should be good score
    }

    #[test]
    fn test_symbol_filtering() {
        let mut analyzer = ExecutionAnalyzer::new();
        
        let mut btc_record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
        btc_record.symbol = "BTC-USD".to_string();
        analyzer.record_execution(btc_record);
        
        let mut eth_record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
        eth_record.symbol = "ETH-USD".to_string();
        analyzer.record_execution(eth_record);
        
        let btc_records = analyzer.get_symbol_records("BTC-USD");
        assert_eq!(btc_records.len(), 1);
    }

    #[test]
    fn test_execution_monitor() {
        let mut monitor = ExecutionMonitor::new();
        
        let record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
        let alerts = monitor.process_execution(record);
        
        // No alerts for normal execution
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_high_slippage_alert() {
        let mut monitor = ExecutionMonitor::new();
        monitor.set_thresholds(dec!(10), 1000.0, dec!(0.8));
        
        // 100 bps slippage (above 10 bps threshold)
        let record = create_test_record(100.0, 100.0, 101.0, OrderSide::Buy);
        let alerts = monitor.process_execution(record);
        
        assert!(alerts.iter().any(|a| a.alert_type == AlertType::HighSlippage));
    }

    #[test]
    fn test_zero_fill_alert() {
        let mut monitor = ExecutionMonitor::new();
        
        let mut record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
        record.quantity_filled = Decimal::ZERO;
        let alerts = monitor.process_execution(record);
        
        assert!(alerts.iter().any(|a| a.alert_type == AlertType::ZeroFill));
    }

    #[test]
    fn test_partial_fill_alert() {
        let mut monitor = ExecutionMonitor::new();
        
        let mut record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
        record.quantity_filled = dec!(0.5);
        let alerts = monitor.process_execution(record);
        
        assert!(alerts.iter().any(|a| a.alert_type == AlertType::PartialFill));
    }

    #[test]
    fn test_tca_report_generation() {
        let mut analyzer = ExecutionAnalyzer::new();
        
        for _ in 0..10 {
            let record = create_test_record(100.0, 100.02, 100.05, OrderSide::Buy);
            analyzer.record_execution(record);
        }
        
        let metrics = analyzer.compute_metrics();
        let report = TcaReportGenerator::generate_report(&metrics);
        
        assert!(report.contains("Trade Cost Analysis Report"));
        assert!(report.contains("Slippage Analysis"));
        assert!(report.contains("Best Execution Score"));
    }

    #[test]
    fn test_capacity_limit() {
        let mut analyzer = ExecutionAnalyzer::with_capacity(10);
        
        for i in 0..20 {
            let mut record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
            record.order_id = format!("order_{}", i);
            analyzer.record_execution(record);
        }
        
        assert_eq!(analyzer.len(), 10);
    }

    #[test]
    fn test_per_symbol_metrics() {
        let mut analyzer = ExecutionAnalyzer::new();
        
        for _ in 0..5 {
            let mut record = create_test_record(100.0, 100.0, 100.05, OrderSide::Buy);
            record.symbol = "BTC-USD".to_string();
            analyzer.record_execution(record);
        }
        
        for _ in 0..3 {
            let mut record = create_test_record(50.0, 50.0, 50.02, OrderSide::Buy);
            record.symbol = "ETH-USD".to_string();
            analyzer.record_execution(record);
        }
        
        let metrics = analyzer.compute_metrics();
        assert!(metrics.by_symbol.contains_key("BTC-USD"));
        assert!(metrics.by_symbol.contains_key("ETH-USD"));
        assert_eq!(metrics.by_symbol["BTC-USD"].executions, 5);
        assert_eq!(metrics.by_symbol["ETH-USD"].executions, 3);
    }

    #[test]
    fn test_clear() {
        let mut analyzer = ExecutionAnalyzer::new();
        let record = create_test_record(100.0, 100.0, 100.0, OrderSide::Buy);
        analyzer.record_execution(record);
        assert!(!analyzer.is_empty());
        
        analyzer.clear();
        assert!(analyzer.is_empty());
    }
}
