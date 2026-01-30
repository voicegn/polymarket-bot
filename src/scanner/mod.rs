//! Continuous arbitrage scanner
//!
//! Background service that continuously scans markets for arbitrage opportunities.
//! Ported from Go implementation.

mod arbitrage_loop;
mod continuous;
mod crypto_market;
mod crypto15m_monitor;
mod indicators;
mod negative_risk;
mod realtime;

pub use arbitrage_loop::{ArbitrageLoop, ArbitrageLoopConfig, LoopStats};
pub use continuous::ContinuousScanner;
pub use crypto_market::{
    CryptoMarket, CryptoMarketDiscovery, MarketInterval, 
    CRYPTO_SYMBOLS, get_aligned_timestamp, get_remaining_time,
};
pub use crypto15m_monitor::{
    Crypto15mMonitor, MarketIndicators, MarketStatus,
    IndicatorStatus, SpikeStats,
};
pub use indicators::{
    RSI, StochRSI, StochRSIResult, SignalType, analyze_signal,
    SpikeDetector, SpikeConfig, SpikeEvent, SpikeType,
};
pub use negative_risk::NegativeRiskScanner;
pub use realtime::RealtimeArbitrageScanner;

use rust_decimal::Decimal;
use std::time::Duration;
use tokio::sync::mpsc;

/// Scanner configuration
#[derive(Debug, Clone)]
pub struct ScannerConfig {
    /// Scan interval (default: 500ms)
    pub scan_interval: Duration,
    /// Market refresh interval (default: 5 min)
    pub refresh_interval: Duration,
    /// Minimum spread to consider (default: 0.5%)
    pub min_spread: Decimal,
    /// Minimum liquidity in shares
    pub min_liquidity: u32,
    /// Minimum volume (USD)
    pub min_volume: Decimal,
    /// Maximum concurrent scans
    pub max_concurrent: usize,
    /// Taker fee rate
    pub taker_fee_rate: Decimal,
    /// Estimated gas cost (USDC)
    pub gas_cost: Decimal,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        use rust_decimal_macros::dec;
        Self {
            scan_interval: Duration::from_millis(500),
            refresh_interval: Duration::from_secs(300),
            min_spread: dec!(0.005),  // 0.5%
            min_liquidity: 10,
            min_volume: dec!(1000),
            max_concurrent: 10,
            taker_fee_rate: dec!(0),  // Polymarket currently 0
            gas_cost: dec!(0.01),
        }
    }
}

/// Detected arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbitrageOpp {
    /// Market condition ID
    pub condition_id: String,
    /// Market question
    pub question: String,
    /// Market slug
    pub slug: String,
    /// Yes token ID
    pub yes_token_id: String,
    /// No token ID  
    pub no_token_id: String,
    /// Yes best ask price
    pub yes_ask: Decimal,
    /// No best ask price
    pub no_ask: Decimal,
    /// Total cost (yes_ask + no_ask)
    pub total_cost: Decimal,
    /// Spread (1.0 - total_cost)
    pub spread: Decimal,
    /// Available liquidity (min of yes/no ask sizes)
    pub max_size: u32,
    /// Profit margin percentage
    pub profit_margin: Decimal,
    /// Net profit after fees
    pub net_profit: Decimal,
    /// Confidence score (0-1)
    pub confidence: Decimal,
    /// Detection timestamp
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

/// Channel for receiving arbitrage opportunities
pub type OpportunityReceiver = mpsc::Receiver<ArbitrageOpp>;
pub type OpportunitySender = mpsc::Sender<ArbitrageOpp>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ScannerConfig::default();
        assert_eq!(config.scan_interval, Duration::from_millis(500));
        assert_eq!(config.max_concurrent, 10);
    }
}
