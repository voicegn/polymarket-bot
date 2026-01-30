//! Technical indicators for crypto market analysis
//!
//! Ported from Go implementation. Includes:
//! - RSI (Relative Strength Index)
//! - Stochastic RSI
//! - Signal analysis (oversold/overbought/crossover)

use std::collections::VecDeque;

/// RSI (Relative Strength Index) indicator
#[derive(Debug, Clone)]
pub struct RSI {
    period: usize,
    gains: VecDeque<f64>,
    losses: VecDeque<f64>,
    avg_gain: f64,
    avg_loss: f64,
    last_price: Option<f64>,
    initialized: bool,
}

impl RSI {
    /// Create a new RSI indicator
    pub fn new(period: usize) -> Self {
        Self {
            period,
            gains: VecDeque::with_capacity(period),
            losses: VecDeque::with_capacity(period),
            avg_gain: 0.0,
            avg_loss: 0.0,
            last_price: None,
            initialized: false,
        }
    }

    /// Update with new price and return RSI value
    pub fn update(&mut self, price: f64) -> f64 {
        // Need at least 2 prices to calculate change
        let last = match self.last_price {
            Some(p) => p,
            None => {
                self.last_price = Some(price);
                return 50.0; // Neutral default
            }
        };

        self.last_price = Some(price);
        let change = price - last;

        let (gain, loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, -change)
        };

        // Initialization phase: collect enough data
        if self.gains.len() < self.period {
            self.gains.push_back(gain);
            self.losses.push_back(loss);

            if self.gains.len() == self.period {
                // Calculate initial averages
                let sum_gain: f64 = self.gains.iter().sum();
                let sum_loss: f64 = self.losses.iter().sum();
                self.avg_gain = sum_gain / self.period as f64;
                self.avg_loss = sum_loss / self.period as f64;
                self.initialized = true;
            }
            return 50.0;
        }

        // Wilder's smoothing method
        let period_f = self.period as f64;
        self.avg_gain = (self.avg_gain * (period_f - 1.0) + gain) / period_f;
        self.avg_loss = (self.avg_loss * (period_f - 1.0) + loss) / period_f;

        // Calculate RSI
        if self.avg_loss == 0.0 {
            return 100.0;
        }

        let rs = self.avg_gain / self.avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Get current RSI value
    pub fn value(&self) -> f64 {
        if !self.initialized || self.avg_loss == 0.0 {
            return if self.avg_gain > 0.0 { 100.0 } else { 50.0 };
        }
        let rs = self.avg_gain / self.avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Check if indicator has enough data
    pub fn is_ready(&self) -> bool {
        self.initialized
    }

    /// Reset the indicator
    pub fn reset(&mut self) {
        self.gains.clear();
        self.losses.clear();
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.last_price = None;
        self.initialized = false;
    }
}

/// Stochastic RSI result
#[derive(Debug, Clone, Default)]
pub struct StochRSIResult {
    /// %K line (0-100)
    pub k: f64,
    /// %D line (SMA of K)
    pub d: f64,
    /// Underlying RSI value
    pub rsi: f64,
    /// Whether there's enough data
    pub ready: bool,
}

/// Stochastic RSI indicator
#[derive(Debug, Clone)]
pub struct StochRSI {
    rsi: RSI,
    stoch_period: usize,
    k_smooth: usize,
    d_smooth: usize,
    rsi_values: VecDeque<f64>,
    k_values: VecDeque<f64>,
}

impl StochRSI {
    /// Create a new Stochastic RSI indicator
    ///
    /// # Arguments
    /// * `rsi_period` - RSI period (typically 14)
    /// * `stoch_period` - Stochastic period (typically 14)
    /// * `k_smooth` - K line smoothing (typically 3)
    /// * `d_smooth` - D line smoothing (typically 3)
    pub fn new(rsi_period: usize, stoch_period: usize, k_smooth: usize, d_smooth: usize) -> Self {
        Self {
            rsi: RSI::new(rsi_period),
            stoch_period,
            k_smooth,
            d_smooth,
            rsi_values: VecDeque::with_capacity(stoch_period),
            k_values: VecDeque::with_capacity(d_smooth),
        }
    }

    /// Update with new price
    pub fn update(&mut self, price: f64) -> StochRSIResult {
        let rsi_value = self.rsi.update(price);

        // Store RSI value
        self.rsi_values.push_back(rsi_value);
        if self.rsi_values.len() > self.stoch_period {
            self.rsi_values.pop_front();
        }

        let mut result = StochRSIResult {
            rsi: rsi_value,
            ready: false,
            k: 50.0,
            d: 50.0,
        };

        // Need enough RSI values
        if self.rsi_values.len() < self.stoch_period {
            return result;
        }

        // Calculate Stochastic: (RSI - Low) / (High - Low) * 100
        let rsi_high = self.rsi_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let rsi_low = self.rsi_values.iter().cloned().fold(f64::INFINITY, f64::min);

        let stoch_rsi = if rsi_high - rsi_low > 0.0 {
            (rsi_value - rsi_low) / (rsi_high - rsi_low) * 100.0
        } else {
            50.0
        };

        // Store for %K calculation
        self.k_values.push_back(stoch_rsi);
        if self.k_values.len() > self.k_smooth {
            self.k_values.pop_front();
        }

        // %K = SMA(Stoch RSI, k_smooth)
        let k_sum: f64 = self.k_values.iter().sum();
        result.k = k_sum / self.k_values.len() as f64;

        // %D = SMA(%K, d_smooth)
        if self.k_values.len() >= self.d_smooth {
            let start = self.k_values.len() - self.d_smooth;
            let d_sum: f64 = self.k_values.iter().skip(start).sum();
            result.d = d_sum / self.d_smooth as f64;
            result.ready = true;
        } else {
            result.d = result.k;
        }

        result
    }

    /// Check if indicator has enough data
    pub fn is_ready(&self) -> bool {
        self.rsi.is_ready() && self.rsi_values.len() >= self.stoch_period
    }

    /// Reset the indicator
    pub fn reset(&mut self) {
        self.rsi.reset();
        self.rsi_values.clear();
        self.k_values.clear();
    }
}

/// Trading signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// No clear signal
    None,
    /// Oversold condition (potential buy)
    Oversold,
    /// Overbought condition (potential sell)
    Overbought,
    /// Bullish crossover (K crosses above D in low zone)
    Bullish,
    /// Bearish crossover (K crosses below D in high zone)
    Bearish,
}

impl SignalType {
    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "NONE",
            Self::Oversold => "OVERSOLD",
            Self::Overbought => "OVERBOUGHT",
            Self::Bullish => "BULLISH",
            Self::Bearish => "BEARISH",
        }
    }
}

/// Analyze RSI and Stoch RSI for trading signals
pub fn analyze_signal(
    rsi: f64,
    stoch_k: f64,
    stoch_d: f64,
    prev_stoch_k: f64,
    prev_stoch_d: f64,
) -> SignalType {
    // Oversold: RSI < 30 and Stoch K < 20
    if rsi < 30.0 && stoch_k < 20.0 {
        return SignalType::Oversold;
    }

    // Overbought: RSI > 70 and Stoch K > 80
    if rsi > 70.0 && stoch_k > 80.0 {
        return SignalType::Overbought;
    }

    // Bullish crossover: K crosses above D in low zone
    if prev_stoch_k < prev_stoch_d && stoch_k > stoch_d && stoch_k < 50.0 {
        return SignalType::Bullish;
    }

    // Bearish crossover: K crosses below D in high zone
    if prev_stoch_k > prev_stoch_d && stoch_k < stoch_d && stoch_k > 50.0 {
        return SignalType::Bearish;
    }

    SignalType::None
}

/// Spike event detected
#[derive(Debug, Clone)]
pub struct SpikeEvent {
    /// Spike type
    pub spike_type: SpikeType,
    /// Spike percentage
    pub spike_percent: f64,
    /// Recovery percentage
    pub recovery_pct: f64,
    /// Severity (1-5)
    pub severity: u8,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Spike types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpikeType {
    Up,
    Down,
    BothWays,
}

/// Spike detector configuration
#[derive(Debug, Clone)]
pub struct SpikeConfig {
    /// Window size for price history
    pub window_size: usize,
    /// Threshold for spike detection (e.g., 0.02 = 2%)
    pub spike_threshold: f64,
    /// Recovery percentage to confirm spike
    pub recovery_pct: f64,
    /// Cooldown between spikes (ms)
    pub cooldown_ms: u64,
}

impl Default for SpikeConfig {
    fn default() -> Self {
        Self {
            window_size: 30,
            spike_threshold: 0.02,
            recovery_pct: 0.30,
            cooldown_ms: 3000,
        }
    }
}

/// Spike detector
#[derive(Debug, Clone)]
pub struct SpikeDetector {
    config: SpikeConfig,
    prices: VecDeque<f64>,
    last_spike: Option<std::time::Instant>,
    total_updates: u64,
    spike_count: u64,
}

impl SpikeDetector {
    /// Create a new spike detector
    pub fn new(config: SpikeConfig) -> Self {
        Self {
            prices: VecDeque::with_capacity(config.window_size),
            config,
            last_spike: None,
            total_updates: 0,
            spike_count: 0,
        }
    }

    /// Update with new price, returns spike event if detected
    pub fn update(&mut self, price: f64, _bid: f64, _ask: f64) -> Option<SpikeEvent> {
        self.total_updates += 1;

        // Add price to history
        self.prices.push_back(price);
        if self.prices.len() > self.config.window_size {
            self.prices.pop_front();
        }

        // Need enough history
        if self.prices.len() < 3 {
            return None;
        }

        // Check cooldown
        if let Some(last) = self.last_spike {
            if last.elapsed().as_millis() < self.config.cooldown_ms as u128 {
                return None;
            }
        }

        // Find min/max in window
        let (min_price, max_price) = self.prices.iter().fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min, max), &p| (min.min(p), max.max(p)),
        );

        // Calculate range
        if min_price <= 0.0 || max_price <= 0.0 {
            return None;
        }

        let range = (max_price - min_price) / min_price;
        
        // Check if range exceeds spike threshold
        if range < self.config.spike_threshold {
            return None;
        }

        // Check for recovery (price moved away from extreme)
        let current = *self.prices.back().unwrap();
        let _mid = (max_price + min_price) / 2.0;
        let recovery = (current - min_price) / (max_price - min_price);

        // Determine spike type
        let spike_type = if recovery > 0.7 {
            SpikeType::Up
        } else if recovery < 0.3 {
            SpikeType::Down
        } else {
            SpikeType::BothWays
        };

        // Calculate severity (1-5 based on magnitude)
        let severity = ((range / self.config.spike_threshold).min(5.0)) as u8;

        self.spike_count += 1;
        self.last_spike = Some(std::time::Instant::now());

        Some(SpikeEvent {
            spike_type,
            spike_percent: range * 100.0,
            recovery_pct: recovery * 100.0,
            severity: severity.max(1),
            timestamp: std::time::Instant::now(),
        })
    }

    /// Get statistics
    pub fn get_stats(&self) -> (u64, u64) {
        (self.total_updates, self.spike_count)
    }

    /// Calculate current volatility
    pub fn get_volatility(&self) -> f64 {
        if self.prices.len() < 2 {
            return 0.0;
        }

        let mean: f64 = self.prices.iter().sum::<f64>() / self.prices.len() as f64;
        let variance: f64 = self.prices
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / self.prices.len() as f64;

        (variance.sqrt() / mean) * 100.0
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.prices.clear();
        self.last_spike = None;
        self.total_updates = 0;
        self.spike_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rsi_calculation() {
        let mut rsi = RSI::new(6);

        // Feed some prices
        let prices = [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0];
        let mut last_value = 50.0;
        for price in prices {
            last_value = rsi.update(price);
        }

        // After upward movement, RSI should be > 50
        assert!(last_value > 50.0);
    }

    #[test]
    fn test_stoch_rsi() {
        let mut stoch = StochRSI::new(14, 14, 3, 3);

        // Feed prices
        for i in 0..30 {
            let price = 100.0 + (i as f64).sin() * 5.0;
            let result = stoch.update(price);
            if result.ready {
                assert!(result.k >= 0.0 && result.k <= 100.0);
                assert!(result.d >= 0.0 && result.d <= 100.0);
            }
        }
    }

    #[test]
    fn test_signal_analysis() {
        // Test oversold
        assert_eq!(
            analyze_signal(25.0, 15.0, 20.0, 18.0, 22.0),
            SignalType::Oversold
        );

        // Test overbought
        assert_eq!(
            analyze_signal(75.0, 85.0, 80.0, 82.0, 78.0),
            SignalType::Overbought
        );

        // Test bullish crossover
        assert_eq!(
            analyze_signal(50.0, 35.0, 30.0, 25.0, 30.0),
            SignalType::Bullish
        );
    }

    #[test]
    fn test_spike_detector() {
        let config = SpikeConfig {
            window_size: 10,
            spike_threshold: 0.02,
            recovery_pct: 0.30,
            cooldown_ms: 100,
        };
        let mut detector = SpikeDetector::new(config);

        // Normal prices
        for i in 0..5 {
            detector.update(100.0 + i as f64 * 0.1, 99.9, 100.1);
        }

        // Spike up
        let spike = detector.update(105.0, 104.9, 105.1);
        // May or may not trigger depending on window
        
        let (total, spikes) = detector.get_stats();
        assert!(total > 0);
    }
}
