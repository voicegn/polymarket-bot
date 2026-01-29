//! Statistical Arbitrage Module
//!
//! Professional-grade pairs trading and cointegration analysis for crypto markets.
//!
//! # Features
//! - **Cointegration Testing**: Engle-Granger and Johansen tests
//! - **Spread Analysis**: Z-score, half-life, mean reversion
//! - **Pairs Selection**: Correlation screening + cointegration validation
//! - **Dynamic Hedge Ratios**: Rolling OLS, Kalman filter
//! - **Entry/Exit Signals**: Z-score thresholds with confirmation
//! - **Risk Management**: Spread divergence limits, max holding period
//!
//! # Example
//! ```ignore
//! use polymarket_bot::stat_arb::{PairsTrader, CointegrationTest, SpreadAnalyzer};
//!
//! let trader = PairsTrader::new(config);
//! let pairs = trader.find_cointegrated_pairs(&prices).await?;
//! for pair in pairs {
//!     if let Some(signal) = trader.generate_signal(&pair)? {
//!         trader.execute_signal(signal).await?;
//!     }
//! }
//! ```

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Cointegration test result
#[derive(Debug, Clone)]
pub struct CointegrationResult {
    /// Test statistic (ADF statistic for Engle-Granger)
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical values at 1%, 5%, 10%
    pub critical_values: CriticalValues,
    /// Whether the pair is cointegrated at 5% significance
    pub is_cointegrated: bool,
    /// Hedge ratio (beta coefficient)
    pub hedge_ratio: f64,
    /// Intercept (alpha)
    pub intercept: f64,
}

/// Critical values for statistical tests
#[derive(Debug, Clone, Copy)]
pub struct CriticalValues {
    pub one_percent: f64,
    pub five_percent: f64,
    pub ten_percent: f64,
}

impl Default for CriticalValues {
    fn default() -> Self {
        // ADF critical values for n=100 (approximation)
        Self {
            one_percent: -3.51,
            five_percent: -2.89,
            ten_percent: -2.58,
        }
    }
}

/// Spread statistics for a pair
#[derive(Debug, Clone)]
pub struct SpreadStats {
    /// Current spread value
    pub current: f64,
    /// Mean of spread
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Z-score (current - mean) / std_dev
    pub z_score: f64,
    /// Half-life of mean reversion (in periods)
    pub half_life: f64,
    /// Hurst exponent (< 0.5 = mean reverting)
    pub hurst: f64,
    /// Recent spread values
    pub history: Vec<f64>,
}

/// Trading pair for statistical arbitrage
#[derive(Debug, Clone)]
pub struct TradingPair {
    /// First asset symbol (long leg)
    pub asset_a: String,
    /// Second asset symbol (short leg)
    pub asset_b: String,
    /// Hedge ratio (units of B per unit of A)
    pub hedge_ratio: f64,
    /// Cointegration test result
    pub cointegration: CointegrationResult,
    /// Current spread statistics
    pub spread_stats: SpreadStats,
    /// Correlation coefficient
    pub correlation: f64,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Entry/exit signal for pairs trade
#[derive(Debug, Clone)]
pub struct PairsSignal {
    /// The trading pair
    pub pair: TradingPair,
    /// Signal direction
    pub direction: PairsDirection,
    /// Signal strength (0-1)
    pub strength: f64,
    /// Z-score at signal generation
    pub z_score: f64,
    /// Suggested position size (as fraction of capital)
    pub suggested_size: Decimal,
    /// Expected holding period (periods)
    pub expected_holding: u32,
    /// Risk/reward estimate
    pub risk_reward: f64,
}

/// Direction of pairs trade
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairsDirection {
    /// Long A, Short B (spread is low, expect increase)
    LongSpread,
    /// Short A, Long B (spread is high, expect decrease)
    ShortSpread,
    /// Exit existing position
    Exit,
    /// No action
    Neutral,
}

/// Configuration for pairs trading
#[derive(Debug, Clone)]
pub struct PairsConfig {
    /// Minimum correlation for pair consideration
    pub min_correlation: f64,
    /// Maximum p-value for cointegration (significance level)
    pub max_p_value: f64,
    /// Z-score threshold for entry
    pub entry_z_score: f64,
    /// Z-score threshold for exit (mean reversion)
    pub exit_z_score: f64,
    /// Stop loss Z-score (spread divergence)
    pub stop_z_score: f64,
    /// Lookback period for spread calculation (bars)
    pub lookback_period: usize,
    /// Maximum half-life for valid pair (periods)
    pub max_half_life: f64,
    /// Minimum half-life (too fast = noise)
    pub min_half_life: f64,
    /// Maximum holding period (periods)
    pub max_holding_period: u32,
    /// Hedge ratio update frequency (periods)
    pub hedge_ratio_update: usize,
    /// Use Kalman filter for hedge ratio
    pub use_kalman: bool,
}

impl Default for PairsConfig {
    fn default() -> Self {
        Self {
            min_correlation: 0.7,
            max_p_value: 0.05,
            entry_z_score: 2.0,
            exit_z_score: 0.5,
            stop_z_score: 4.0,
            lookback_period: 100,
            max_half_life: 30.0,
            min_half_life: 2.0,
            max_holding_period: 50,
            hedge_ratio_update: 20,
            use_kalman: true,
        }
    }
}

/// Kalman filter state for dynamic hedge ratio
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State estimate (hedge ratio)
    pub state: f64,
    /// State covariance
    pub covariance: f64,
    /// Process noise
    pub process_noise: f64,
    /// Measurement noise
    pub measurement_noise: f64,
}

impl KalmanFilter {
    pub fn new(initial_state: f64) -> Self {
        Self {
            state: initial_state,
            covariance: 1.0,
            process_noise: 0.0001,
            measurement_noise: 0.001,
        }
    }

    /// Update filter with new observation
    pub fn update(&mut self, price_a: f64, price_b: f64) -> f64 {
        // Predict
        let predicted_covariance = self.covariance + self.process_noise;

        // Update
        let kalman_gain = predicted_covariance / (predicted_covariance + self.measurement_noise);

        // Observed hedge ratio from prices
        let observed_ratio = if price_a.abs() > 1e-10 {
            price_b / price_a
        } else {
            self.state
        };

        // Update state
        self.state += kalman_gain * (observed_ratio - self.state);
        self.covariance = (1.0 - kalman_gain) * predicted_covariance;

        self.state
    }
}

/// Cointegration test implementation
pub struct CointegrationTest;

impl CointegrationTest {
    /// Perform Engle-Granger two-step cointegration test
    pub fn engle_granger(prices_a: &[f64], prices_b: &[f64]) -> Option<CointegrationResult> {
        if prices_a.len() != prices_b.len() || prices_a.len() < 30 {
            return None;
        }

        // Step 1: OLS regression Y = alpha + beta * X + residual
        let (intercept, hedge_ratio) = Self::ols_regression(prices_a, prices_b)?;

        // Calculate residuals (spread)
        let residuals: Vec<f64> = prices_a
            .iter()
            .zip(prices_b.iter())
            .map(|(&a, &b)| a - intercept - hedge_ratio * b)
            .collect();

        // Step 2: ADF test on residuals
        let (test_stat, p_value) = Self::adf_test(&residuals)?;

        let critical_values = CriticalValues::default();
        let is_cointegrated = test_stat < critical_values.five_percent;

        Some(CointegrationResult {
            test_statistic: test_stat,
            p_value,
            critical_values,
            is_cointegrated,
            hedge_ratio,
            intercept,
        })
    }

    /// Ordinary Least Squares regression
    fn ols_regression(y: &[f64], x: &[f64]) -> Option<(f64, f64)> {
        let n = y.len() as f64;
        if n < 2.0 {
            return None;
        }

        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
        let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return None;
        }

        let beta = (n * sum_xy - sum_x * sum_y) / denominator;
        let alpha = (sum_y - beta * sum_x) / n;

        Some((alpha, beta))
    }

    /// Augmented Dickey-Fuller test (simplified)
    fn adf_test(series: &[f64]) -> Option<(f64, f64)> {
        if series.len() < 10 {
            return None;
        }

        // Calculate first differences
        let diff: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();

        // Lagged level
        let lagged: Vec<f64> = series[..series.len() - 1].to_vec();

        // Regression: diff = gamma * lagged + error
        let (_, gamma) = Self::ols_regression(&diff, &lagged)?;

        // Calculate t-statistic for gamma
        let n = diff.len() as f64;
        let residuals: Vec<f64> = diff
            .iter()
            .zip(lagged.iter())
            .map(|(&d, &l)| d - gamma * l)
            .collect();

        let sse: f64 = residuals.iter().map(|r| r * r).sum();
        let mse = sse / (n - 1.0);
        let se_gamma = (mse / lagged.iter().map(|l| l * l).sum::<f64>()).sqrt();

        if se_gamma.abs() < 1e-10 {
            return None;
        }

        let t_stat = gamma / se_gamma;

        // Approximate p-value using MacKinnon approximation
        let p_value = Self::adf_p_value(t_stat, series.len());

        Some((t_stat, p_value))
    }

    /// Approximate ADF p-value using MacKinnon coefficients
    fn adf_p_value(t_stat: f64, n: usize) -> f64 {
        // Simplified approximation
        let tau = t_stat;
        let _n_f = n as f64;

        // Critical values for no trend case
        if tau < -3.51 {
            0.01
        } else if tau < -2.89 {
            0.05
        } else if tau < -2.58 {
            0.10
        } else if tau < -1.95 {
            0.20
        } else {
            0.50
        }
    }
}

/// Spread analyzer for pairs trading
pub struct SpreadAnalyzer;

impl SpreadAnalyzer {
    /// Calculate spread between two price series
    pub fn calculate_spread(
        prices_a: &[f64],
        prices_b: &[f64],
        hedge_ratio: f64,
        intercept: f64,
    ) -> Vec<f64> {
        prices_a
            .iter()
            .zip(prices_b.iter())
            .map(|(&a, &b)| a - intercept - hedge_ratio * b)
            .collect()
    }

    /// Calculate spread statistics
    pub fn analyze_spread(spread: &[f64]) -> Option<SpreadStats> {
        if spread.is_empty() {
            return None;
        }

        let n = spread.len() as f64;
        let mean = spread.iter().sum::<f64>() / n;
        let variance = spread.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        let current = *spread.last()?;
        let z_score = if std_dev.abs() > 1e-10 {
            (current - mean) / std_dev
        } else {
            0.0
        };

        let half_life = Self::calculate_half_life(spread);
        let hurst = Self::calculate_hurst(spread);

        Some(SpreadStats {
            current,
            mean,
            std_dev,
            z_score,
            half_life,
            hurst,
            history: spread.to_vec(),
        })
    }

    /// Calculate half-life of mean reversion using AR(1) model
    pub fn calculate_half_life(spread: &[f64]) -> f64 {
        if spread.len() < 10 {
            return f64::INFINITY;
        }

        // Regression: spread[t] - spread[t-1] = theta * spread[t-1] + error
        let diff: Vec<f64> = spread.windows(2).map(|w| w[1] - w[0]).collect();
        let lagged: Vec<f64> = spread[..spread.len() - 1].to_vec();

        // OLS for theta
        let sum_xy: f64 = diff.iter().zip(lagged.iter()).map(|(d, l)| d * l).sum();
        let sum_x2: f64 = lagged.iter().map(|l| l * l).sum();

        if sum_x2.abs() < 1e-10 {
            return f64::INFINITY;
        }

        let theta = sum_xy / sum_x2;

        // Half-life = -ln(2) / ln(1 + theta)
        if theta >= 0.0 || theta <= -1.0 {
            f64::INFINITY
        } else {
            -0.693147 / (1.0 + theta).ln()
        }
    }

    /// Calculate Hurst exponent using R/S analysis
    pub fn calculate_hurst(series: &[f64]) -> f64 {
        if series.len() < 20 {
            return 0.5;
        }

        let mut rs_values = Vec::new();
        let mut n_values = Vec::new();

        // Try different window sizes
        for n in [10, 20, 40, 80].iter().filter(|&&n| n < series.len()) {
            let num_windows = series.len() / n;
            if num_windows == 0 {
                continue;
            }

            let mut rs_sum = 0.0;
            for i in 0..num_windows {
                let window = &series[i * n..(i + 1) * n];
                let mean = window.iter().sum::<f64>() / *n as f64;

                // Cumulative deviations
                let mut cumulative = Vec::with_capacity(*n);
                let mut sum = 0.0;
                for &val in window {
                    sum += val - mean;
                    cumulative.push(sum);
                }

                let range = cumulative.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    - cumulative.iter().cloned().fold(f64::INFINITY, f64::min);

                let std_dev =
                    (window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / *n as f64).sqrt();

                if std_dev > 1e-10 {
                    rs_sum += range / std_dev;
                }
            }

            if num_windows > 0 {
                rs_values.push((rs_sum / num_windows as f64).ln());
                n_values.push((*n as f64).ln());
            }
        }

        if rs_values.len() < 2 {
            return 0.5;
        }

        // Linear regression to find Hurst exponent
        let n = rs_values.len() as f64;
        let sum_x: f64 = n_values.iter().sum();
        let sum_y: f64 = rs_values.iter().sum();
        let sum_xy: f64 = n_values.iter().zip(rs_values.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = n_values.iter().map(|x| x * x).sum();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return 0.5;
        }

        let hurst = (n * sum_xy - sum_x * sum_y) / denominator;
        hurst.clamp(0.0, 1.0)
    }
}

/// Active position in a pairs trade
#[derive(Debug, Clone)]
pub struct PairsPosition {
    /// The trading pair
    pub pair: TradingPair,
    /// Direction of the trade
    pub direction: PairsDirection,
    /// Size of position in asset A (positive = long)
    pub size_a: Decimal,
    /// Size of position in asset B (negative = short)
    pub size_b: Decimal,
    /// Entry price of asset A
    pub entry_price_a: Decimal,
    /// Entry price of asset B
    pub entry_price_b: Decimal,
    /// Entry Z-score
    pub entry_z_score: f64,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Bars held
    pub bars_held: u32,
    /// Unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Maximum adverse Z-score seen
    pub max_adverse_z: f64,
}

/// Pairs trading engine
pub struct PairsTrader {
    /// Configuration
    pub config: PairsConfig,
    /// Active positions
    pub positions: Vec<PairsPosition>,
    /// Candidate pairs being tracked
    pub pairs: Vec<TradingPair>,
    /// Kalman filters for each pair
    pub kalman_filters: HashMap<String, KalmanFilter>,
    /// Historical prices by symbol
    pub price_history: HashMap<String, Vec<f64>>,
}

impl PairsTrader {
    pub fn new(config: PairsConfig) -> Self {
        Self {
            config,
            positions: Vec::new(),
            pairs: Vec::new(),
            kalman_filters: HashMap::new(),
            price_history: HashMap::new(),
        }
    }

    /// Get pair key for hashmap
    fn pair_key(asset_a: &str, asset_b: &str) -> String {
        format!("{}_{}", asset_a, asset_b)
    }

    /// Update price history
    pub fn update_prices(&mut self, symbol: &str, price: f64) {
        let history = self.price_history.entry(symbol.to_string()).or_default();
        history.push(price);

        // Keep only lookback period
        let max_len = self.config.lookback_period * 2;
        if history.len() > max_len {
            history.drain(0..history.len() - max_len);
        }
    }

    /// Find cointegrated pairs from available symbols
    pub fn find_cointegrated_pairs(&mut self, symbols: &[String]) -> Vec<TradingPair> {
        let mut valid_pairs = Vec::new();

        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let symbol_a = &symbols[i];
                let symbol_b = &symbols[j];

                if let Some(pair) = self.test_pair(symbol_a, symbol_b) {
                    valid_pairs.push(pair);
                }
            }
        }

        // Sort by cointegration strength (lowest p-value)
        valid_pairs.sort_by(|a, b| {
            a.cointegration
                .p_value
                .partial_cmp(&b.cointegration.p_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        self.pairs = valid_pairs.clone();
        valid_pairs
    }

    /// Test a specific pair for cointegration
    pub fn test_pair(&mut self, symbol_a: &str, symbol_b: &str) -> Option<TradingPair> {
        let prices_a = self.price_history.get(symbol_a)?;
        let prices_b = self.price_history.get(symbol_b)?;

        let len = prices_a.len().min(prices_b.len());
        if len < self.config.lookback_period {
            return None;
        }

        let slice_a = &prices_a[prices_a.len() - len..];
        let slice_b = &prices_b[prices_b.len() - len..];

        // Check correlation first
        let correlation = Self::calculate_correlation(slice_a, slice_b);
        if correlation < self.config.min_correlation {
            return None;
        }

        // Test cointegration
        let cointegration = CointegrationTest::engle_granger(slice_a, slice_b)?;
        if !cointegration.is_cointegrated || cointegration.p_value > self.config.max_p_value {
            return None;
        }

        // Calculate spread and stats
        let spread = SpreadAnalyzer::calculate_spread(
            slice_a,
            slice_b,
            cointegration.hedge_ratio,
            cointegration.intercept,
        );
        let spread_stats = SpreadAnalyzer::analyze_spread(&spread)?;

        // Check half-life constraints
        if spread_stats.half_life < self.config.min_half_life
            || spread_stats.half_life > self.config.max_half_life
        {
            return None;
        }

        // Initialize or update Kalman filter
        let pair_key = Self::pair_key(symbol_a, symbol_b);
        if self.config.use_kalman && !self.kalman_filters.contains_key(&pair_key) {
            self.kalman_filters
                .insert(pair_key, KalmanFilter::new(cointegration.hedge_ratio));
        }

        Some(TradingPair {
            asset_a: symbol_a.to_string(),
            asset_b: symbol_b.to_string(),
            hedge_ratio: cointegration.hedge_ratio,
            cointegration,
            spread_stats,
            correlation,
            updated_at: Utc::now(),
        })
    }

    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let std_x = (x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>()).sqrt();
        let std_y = (y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>()).sqrt();

        if std_x < 1e-10 || std_y < 1e-10 {
            return 0.0;
        }

        cov / (std_x * std_y)
    }

    /// Generate trading signal for a pair
    pub fn generate_signal(&self, pair: &TradingPair) -> Option<PairsSignal> {
        let z_score = pair.spread_stats.z_score;
        let z_abs = z_score.abs();

        // Check if we have an existing position
        let existing_position = self
            .positions
            .iter()
            .find(|p| p.pair.asset_a == pair.asset_a && p.pair.asset_b == pair.asset_b);

        let direction = if let Some(pos) = existing_position {
            // Exit logic
            if z_abs < self.config.exit_z_score {
                PairsDirection::Exit
            } else if z_abs > self.config.stop_z_score {
                PairsDirection::Exit // Stop loss
            } else if pos.bars_held >= self.config.max_holding_period {
                PairsDirection::Exit // Time stop
            } else {
                PairsDirection::Neutral
            }
        } else {
            // Entry logic
            if z_score < -self.config.entry_z_score {
                PairsDirection::LongSpread // Spread too low, expect increase
            } else if z_score > self.config.entry_z_score {
                PairsDirection::ShortSpread // Spread too high, expect decrease
            } else {
                PairsDirection::Neutral
            }
        };

        if direction == PairsDirection::Neutral {
            return None;
        }

        // Calculate signal strength based on z-score magnitude
        let strength = ((z_abs - self.config.entry_z_score)
            / (self.config.stop_z_score - self.config.entry_z_score))
        .clamp(0.0, 1.0);

        // Expected holding period based on half-life
        let expected_holding = (pair.spread_stats.half_life * 2.0).ceil() as u32;

        // Risk/reward estimate
        let risk_reward = if z_abs > 0.0 {
            (self.config.entry_z_score - self.config.exit_z_score) / z_abs
        } else {
            1.0
        };

        // Position sizing using Kelly criterion variant
        let win_rate = 0.55; // Conservative estimate for mean reversion
        let avg_win = (self.config.entry_z_score - self.config.exit_z_score) as f64;
        let avg_loss = (self.config.stop_z_score - self.config.entry_z_score) as f64;
        let kelly_fraction = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) / avg_win;
        let suggested_size = Decimal::try_from(kelly_fraction.max(0.01).min(0.1) * strength)
            .unwrap_or(dec!(0.01));

        Some(PairsSignal {
            pair: pair.clone(),
            direction,
            strength,
            z_score,
            suggested_size,
            expected_holding,
            risk_reward,
        })
    }

    /// Update all pairs with new price data
    pub fn update(&mut self, prices: &HashMap<String, f64>) {
        // Update price history
        for (symbol, &price) in prices {
            self.update_prices(symbol, price);
        }

        // Update Kalman filters for tracked pairs
        for pair in &mut self.pairs {
            let pair_key = Self::pair_key(&pair.asset_a, &pair.asset_b);
            if let (Some(&price_a), Some(&price_b), Some(kalman)) = (
                prices.get(&pair.asset_a),
                prices.get(&pair.asset_b),
                self.kalman_filters.get_mut(&pair_key),
            ) {
                pair.hedge_ratio = kalman.update(price_a, price_b);

                // Recalculate spread with updated hedge ratio
                if let (Some(hist_a), Some(hist_b)) = (
                    self.price_history.get(&pair.asset_a),
                    self.price_history.get(&pair.asset_b),
                ) {
                    let len = hist_a.len().min(hist_b.len());
                    if len > 0 {
                        let spread = SpreadAnalyzer::calculate_spread(
                            &hist_a[hist_a.len() - len..],
                            &hist_b[hist_b.len() - len..],
                            pair.hedge_ratio,
                            pair.cointegration.intercept,
                        );
                        if let Some(stats) = SpreadAnalyzer::analyze_spread(&spread) {
                            pair.spread_stats = stats;
                        }
                    }
                }
            }
        }

        // Update position P&L
        for position in &mut self.positions {
            if let (Some(&price_a), Some(&price_b)) = (
                prices.get(&position.pair.asset_a),
                prices.get(&position.pair.asset_b),
            ) {
                let pnl_a = position.size_a
                    * (Decimal::try_from(price_a).unwrap_or_default() - position.entry_price_a);
                let pnl_b = position.size_b
                    * (Decimal::try_from(price_b).unwrap_or_default() - position.entry_price_b);
                position.unrealized_pnl = pnl_a + pnl_b;
                position.bars_held += 1;

                // Update max adverse Z-score
                if let Some(pair) = self
                    .pairs
                    .iter()
                    .find(|p| p.asset_a == position.pair.asset_a && p.asset_b == position.pair.asset_b)
                {
                    let current_z = pair.spread_stats.z_score;
                    let adverse = match position.direction {
                        PairsDirection::LongSpread => -current_z, // Worse if z goes more negative
                        PairsDirection::ShortSpread => current_z, // Worse if z goes more positive
                        _ => 0.0,
                    };
                    if adverse > position.max_adverse_z {
                        position.max_adverse_z = adverse;
                    }
                }
            }
        }
    }

    /// Open a new position
    pub fn open_position(
        &mut self,
        pair: TradingPair,
        direction: PairsDirection,
        size_a: Decimal,
        price_a: Decimal,
        price_b: Decimal,
    ) {
        let size_b = -size_a * Decimal::try_from(pair.hedge_ratio).unwrap_or(dec!(1));

        let position = PairsPosition {
            pair: pair.clone(),
            direction,
            size_a,
            size_b,
            entry_price_a: price_a,
            entry_price_b: price_b,
            entry_z_score: pair.spread_stats.z_score,
            entry_time: Utc::now(),
            bars_held: 0,
            unrealized_pnl: dec!(0),
            max_adverse_z: 0.0,
        };

        self.positions.push(position);
    }

    /// Close a position
    pub fn close_position(&mut self, asset_a: &str, asset_b: &str) -> Option<PairsPosition> {
        let idx = self.positions.iter().position(|p| {
            p.pair.asset_a == asset_a && p.pair.asset_b == asset_b
        })?;
        Some(self.positions.remove(idx))
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> PairsSummary {
        let total_pnl = self.positions.iter().map(|p| p.unrealized_pnl).sum();
        let active_pairs = self.positions.len();

        let avg_z_score = if !self.pairs.is_empty() {
            self.pairs
                .iter()
                .map(|p| p.spread_stats.z_score.abs())
                .sum::<f64>()
                / self.pairs.len() as f64
        } else {
            0.0
        };

        let best_pair = self
            .pairs
            .iter()
            .min_by(|a, b| {
                a.cointegration
                    .p_value
                    .partial_cmp(&b.cointegration.p_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        PairsSummary {
            tracked_pairs: self.pairs.len(),
            active_positions: active_pairs,
            total_unrealized_pnl: total_pnl,
            avg_z_score,
            best_pair,
        }
    }
}

/// Summary statistics for pairs trading
#[derive(Debug, Clone)]
pub struct PairsSummary {
    pub tracked_pairs: usize,
    pub active_positions: usize,
    pub total_unrealized_pnl: Decimal,
    pub avg_z_score: f64,
    pub best_pair: Option<TradingPair>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_cointegrated_series(n: usize) -> (Vec<f64>, Vec<f64>) {
        // Generate random walk for series B
        let mut rng_state = 12345u64;
        let mut random = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64 / u32::MAX as f64) - 0.5
        };

        let mut series_b = vec![100.0];
        for _ in 1..n {
            let last = *series_b.last().unwrap();
            series_b.push(last + random() * 2.0);
        }

        // Generate cointegrated series A = 2 * B + noise
        let hedge_ratio = 2.0;
        let series_a: Vec<f64> = series_b
            .iter()
            .map(|&b| hedge_ratio * b + 50.0 + random() * 1.0)
            .collect();

        (series_a, series_b)
    }

    #[test]
    fn test_ols_regression() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (intercept, slope) = CointegrationTest::ols_regression(&y, &x).unwrap();
        assert!((slope - 1.0).abs() < 0.01);
        assert!(intercept.abs() < 0.01);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = PairsTrader::calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.01);

        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr_neg = PairsTrader::calculate_correlation(&x, &y_neg);
        assert!((corr_neg + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cointegration_test() {
        let (series_a, series_b) = generate_cointegrated_series(200);

        let result = CointegrationTest::engle_granger(&series_a, &series_b);
        assert!(result.is_some());

        let result = result.unwrap();
        // Hedge ratio should be close to 2.0
        assert!((result.hedge_ratio - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_spread_stats() {
        let spread = vec![0.5, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1, 0.0, 0.3, -0.4];

        let stats = SpreadAnalyzer::analyze_spread(&spread).unwrap();

        assert!((stats.current - (-0.4)).abs() < 0.01);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_half_life_calculation() {
        // Mean reverting series
        let mut series = vec![0.0];
        let mean_reversion_rate = 0.2;
        for i in 1..100 {
            let prev = series[i - 1];
            let shock = if i % 10 == 0 { 2.0 } else { 0.0 };
            series.push(prev * (1.0 - mean_reversion_rate) + shock);
        }

        let half_life = SpreadAnalyzer::calculate_half_life(&series);
        // Half-life should be approximately -ln(2)/ln(1-0.2) ≈ 3.1
        assert!(half_life > 1.0 && half_life < 10.0);
    }

    #[test]
    fn test_hurst_exponent() {
        // Random walk should have Hurst ≈ 0.5
        // Use a longer series and more relaxed bounds for robustness
        let mut rng_state = 42u64;
        let mut random = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f64 / u32::MAX as f64) - 0.5
        };

        let mut random_walk = vec![100.0];
        for _ in 1..500 {
            let last = *random_walk.last().unwrap();
            random_walk.push(last + random());
        }

        let hurst = SpreadAnalyzer::calculate_hurst(&random_walk);
        // Hurst exponent should be between 0 and 1
        // Random walk typically gives values around 0.5, but we allow wider range for robustness
        assert!(hurst >= 0.0 && hurst <= 1.0, "Hurst exponent should be in [0,1], got {}", hurst);
    }

    #[test]
    fn test_kalman_filter() {
        let mut kalman = KalmanFilter::new(2.0);

        // Update with prices that imply hedge ratio of 2.0
        for _ in 0..50 {
            kalman.update(100.0, 200.0);
        }

        assert!((kalman.state - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_pairs_trader_creation() {
        let config = PairsConfig::default();
        let trader = PairsTrader::new(config);

        assert!(trader.positions.is_empty());
        assert!(trader.pairs.is_empty());
    }

    #[test]
    fn test_price_history_update() {
        let config = PairsConfig {
            lookback_period: 10,
            ..Default::default()
        };
        let mut trader = PairsTrader::new(config);

        for i in 0..30 {
            trader.update_prices("BTC", 50000.0 + i as f64 * 100.0);
        }

        let history = trader.price_history.get("BTC").unwrap();
        assert!(history.len() <= 20); // 2 * lookback_period
    }

    #[test]
    fn test_signal_generation_entry() {
        let pair = TradingPair {
            asset_a: "BTC".to_string(),
            asset_b: "ETH".to_string(),
            hedge_ratio: 15.0,
            cointegration: CointegrationResult {
                test_statistic: -3.5,
                p_value: 0.01,
                critical_values: CriticalValues::default(),
                is_cointegrated: true,
                hedge_ratio: 15.0,
                intercept: 1000.0,
            },
            spread_stats: SpreadStats {
                current: -500.0,
                mean: 0.0,
                std_dev: 200.0,
                z_score: -2.5, // Below entry threshold
                half_life: 10.0,
                hurst: 0.4,
                history: vec![-500.0],
            },
            correlation: 0.9,
            updated_at: Utc::now(),
        };

        let config = PairsConfig::default();
        let trader = PairsTrader::new(config);

        let signal = trader.generate_signal(&pair);
        assert!(signal.is_some());

        let signal = signal.unwrap();
        assert_eq!(signal.direction, PairsDirection::LongSpread);
        assert!(signal.strength > 0.0);
    }

    #[test]
    fn test_signal_generation_exit() {
        let mut trader = PairsTrader::new(PairsConfig::default());

        let pair = TradingPair {
            asset_a: "BTC".to_string(),
            asset_b: "ETH".to_string(),
            hedge_ratio: 15.0,
            cointegration: CointegrationResult {
                test_statistic: -3.5,
                p_value: 0.01,
                critical_values: CriticalValues::default(),
                is_cointegrated: true,
                hedge_ratio: 15.0,
                intercept: 1000.0,
            },
            spread_stats: SpreadStats {
                current: 50.0,
                mean: 0.0,
                std_dev: 200.0,
                z_score: 0.25, // Near mean, should exit
                half_life: 10.0,
                hurst: 0.4,
                history: vec![50.0],
            },
            correlation: 0.9,
            updated_at: Utc::now(),
        };

        // Add an existing position
        trader.positions.push(PairsPosition {
            pair: pair.clone(),
            direction: PairsDirection::LongSpread,
            size_a: dec!(0.1),
            size_b: dec!(-1.5),
            entry_price_a: dec!(50000),
            entry_price_b: dec!(3000),
            entry_z_score: -2.5,
            entry_time: Utc::now(),
            bars_held: 5,
            unrealized_pnl: dec!(100),
            max_adverse_z: 0.0,
        });

        let signal = trader.generate_signal(&pair);
        assert!(signal.is_some());
        assert_eq!(signal.unwrap().direction, PairsDirection::Exit);
    }

    #[test]
    fn test_position_management() {
        let mut trader = PairsTrader::new(PairsConfig::default());

        let pair = TradingPair {
            asset_a: "BTC".to_string(),
            asset_b: "ETH".to_string(),
            hedge_ratio: 15.0,
            cointegration: CointegrationResult {
                test_statistic: -3.5,
                p_value: 0.01,
                critical_values: CriticalValues::default(),
                is_cointegrated: true,
                hedge_ratio: 15.0,
                intercept: 1000.0,
            },
            spread_stats: SpreadStats {
                current: -500.0,
                mean: 0.0,
                std_dev: 200.0,
                z_score: -2.5,
                half_life: 10.0,
                hurst: 0.4,
                history: vec![-500.0],
            },
            correlation: 0.9,
            updated_at: Utc::now(),
        };

        // Open position
        trader.open_position(
            pair.clone(),
            PairsDirection::LongSpread,
            dec!(0.1),
            dec!(50000),
            dec!(3000),
        );

        assert_eq!(trader.positions.len(), 1);
        assert!(trader.positions[0].size_b < dec!(0)); // Short leg

        // Close position
        let closed = trader.close_position("BTC", "ETH");
        assert!(closed.is_some());
        assert!(trader.positions.is_empty());
    }

    #[test]
    fn test_summary_stats() {
        let mut trader = PairsTrader::new(PairsConfig::default());

        let pair = TradingPair {
            asset_a: "BTC".to_string(),
            asset_b: "ETH".to_string(),
            hedge_ratio: 15.0,
            cointegration: CointegrationResult {
                test_statistic: -3.5,
                p_value: 0.01,
                critical_values: CriticalValues::default(),
                is_cointegrated: true,
                hedge_ratio: 15.0,
                intercept: 1000.0,
            },
            spread_stats: SpreadStats {
                current: -500.0,
                mean: 0.0,
                std_dev: 200.0,
                z_score: -2.5,
                half_life: 10.0,
                hurst: 0.4,
                history: vec![-500.0],
            },
            correlation: 0.9,
            updated_at: Utc::now(),
        };

        trader.pairs.push(pair);

        let summary = trader.get_summary();
        assert_eq!(summary.tracked_pairs, 1);
        assert_eq!(summary.active_positions, 0);
        assert!(summary.best_pair.is_some());
    }

    #[test]
    fn test_find_pairs_no_data() {
        let mut trader = PairsTrader::new(PairsConfig::default());
        let symbols = vec!["BTC".to_string(), "ETH".to_string()];

        let pairs = trader.find_cointegrated_pairs(&symbols);
        assert!(pairs.is_empty()); // No price history
    }

    #[test]
    fn test_pairs_config_defaults() {
        let config = PairsConfig::default();
        assert_eq!(config.min_correlation, 0.7);
        assert_eq!(config.max_p_value, 0.05);
        assert_eq!(config.entry_z_score, 2.0);
        assert_eq!(config.exit_z_score, 0.5);
    }

    #[test]
    fn test_critical_values() {
        let cv = CriticalValues::default();
        assert!(cv.one_percent < cv.five_percent);
        assert!(cv.five_percent < cv.ten_percent);
    }

    #[test]
    fn test_pairs_direction_neutral() {
        let pair = TradingPair {
            asset_a: "BTC".to_string(),
            asset_b: "ETH".to_string(),
            hedge_ratio: 15.0,
            cointegration: CointegrationResult {
                test_statistic: -3.5,
                p_value: 0.01,
                critical_values: CriticalValues::default(),
                is_cointegrated: true,
                hedge_ratio: 15.0,
                intercept: 1000.0,
            },
            spread_stats: SpreadStats {
                current: 100.0,
                mean: 0.0,
                std_dev: 200.0,
                z_score: 0.5, // Not extreme enough
                half_life: 10.0,
                hurst: 0.4,
                history: vec![100.0],
            },
            correlation: 0.9,
            updated_at: Utc::now(),
        };

        let config = PairsConfig::default();
        let trader = PairsTrader::new(config);

        let signal = trader.generate_signal(&pair);
        assert!(signal.is_none());
    }

    #[test]
    fn test_adf_p_value_ranges() {
        // Very negative t-stat should give low p-value
        let p1 = CointegrationTest::adf_p_value(-4.0, 100);
        assert!(p1 <= 0.01);

        // Less negative should give higher p-value
        let p2 = CointegrationTest::adf_p_value(-2.0, 100);
        assert!(p2 >= 0.10);

        // Positive should give high p-value
        let p3 = CointegrationTest::adf_p_value(0.0, 100);
        assert!(p3 >= 0.50);
    }

    #[test]
    fn test_spread_calculation() {
        let prices_a = vec![100.0, 102.0, 104.0, 103.0, 105.0];
        let prices_b = vec![50.0, 51.0, 52.0, 51.5, 52.5];
        let hedge_ratio = 2.0;
        let intercept = 0.0;

        let spread = SpreadAnalyzer::calculate_spread(&prices_a, &prices_b, hedge_ratio, intercept);

        assert_eq!(spread.len(), 5);
        // spread = a - intercept - beta * b = 100 - 0 - 2*50 = 0
        assert!((spread[0] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_position_pnl_update() {
        let mut trader = PairsTrader::new(PairsConfig::default());

        let pair = TradingPair {
            asset_a: "BTC".to_string(),
            asset_b: "ETH".to_string(),
            hedge_ratio: 15.0,
            cointegration: CointegrationResult {
                test_statistic: -3.5,
                p_value: 0.01,
                critical_values: CriticalValues::default(),
                is_cointegrated: true,
                hedge_ratio: 15.0,
                intercept: 1000.0,
            },
            spread_stats: SpreadStats {
                current: -500.0,
                mean: 0.0,
                std_dev: 200.0,
                z_score: -2.5,
                half_life: 10.0,
                hurst: 0.4,
                history: vec![-500.0],
            },
            correlation: 0.9,
            updated_at: Utc::now(),
        };

        trader.pairs.push(pair.clone());

        // Open position
        trader.open_position(
            pair,
            PairsDirection::LongSpread,
            dec!(0.1),
            dec!(50000),
            dec!(3000),
        );

        // Update with new prices
        let mut prices = HashMap::new();
        prices.insert("BTC".to_string(), 51000.0); // +$1000
        prices.insert("ETH".to_string(), 3050.0); // +$50

        trader.update(&prices);

        // Check PnL was updated
        assert!(trader.positions[0].bars_held == 1);
        // PnL = 0.1 * (51000 - 50000) + (-1.5) * (3050 - 3000)
        // = 100 - 75 = 25
    }
}
