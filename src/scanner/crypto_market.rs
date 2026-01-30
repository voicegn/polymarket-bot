//! Crypto 15m/30m/60m Market Discovery
//!
//! Discovers dynamically created crypto Up/Down markets using timestamp-based slugs.
//! Ported from Go implementation.
//!
//! ## Key Insight
//!
//! Polymarket creates crypto markets with predictable slug patterns:
//! - `{symbol}-updown-15m-{unix_timestamp}` where timestamp is 15-min aligned
//! - `{symbol}-updown-30m-{unix_timestamp}` for 30-min markets
//! - `{symbol}-updown-hourly-{unix_timestamp}` for hourly markets

use crate::error::Result;
use chrono::{DateTime, Duration, Utc};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Supported crypto symbols
pub const CRYPTO_SYMBOLS: &[&str] = &["btc", "eth", "xrp", "sol", "doge"];

/// Market interval types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketInterval {
    /// 15-minute markets (900 seconds)
    Min15,
    /// 30-minute markets (1800 seconds)
    Min30,
    /// Hourly markets (3600 seconds)
    Hourly,
    /// 4-hour markets
    Hour4,
    /// Daily markets
    Daily,
}

impl MarketInterval {
    /// Get interval duration in seconds
    pub fn duration_secs(&self) -> i64 {
        match self {
            Self::Min15 => 900,
            Self::Min30 => 1800,
            Self::Hourly => 3600,
            Self::Hour4 => 14400,
            Self::Daily => 86400,
        }
    }

    /// Get slug suffix for this interval
    pub fn slug_suffix(&self) -> &'static str {
        match self {
            Self::Min15 => "updown-15m",
            Self::Min30 => "updown-30m",
            Self::Hourly => "updown-hourly",
            Self::Hour4 => "updown-4h",
            Self::Daily => "updown-daily",
        }
    }

    /// Align timestamp to this interval
    pub fn align_timestamp(&self, unix_ts: i64) -> i64 {
        let duration = self.duration_secs();
        (unix_ts / duration) * duration
    }
}

/// Discovered crypto market
#[derive(Debug, Clone)]
pub struct CryptoMarket {
    /// Symbol (BTC, ETH, etc.)
    pub symbol: String,
    /// Interval type
    pub interval: MarketInterval,
    /// Event slug
    pub slug: String,
    /// Event title
    pub title: String,
    /// Condition ID
    pub condition_id: String,
    /// Up token ID (outcome: price goes up)
    pub up_token_id: String,
    /// Down token ID (outcome: price goes down)
    pub down_token_id: String,
    /// Current Up price
    pub up_price: Decimal,
    /// Current Down price
    pub down_price: Decimal,
    /// Sum of prices (should be ~1.0)
    pub sum: Decimal,
    /// Spread (1.0 - sum)
    pub spread: Decimal,
    /// Market start time
    pub start_time: DateTime<Utc>,
    /// Market end time
    pub end_time: DateTime<Utc>,
    /// Is market active
    pub active: bool,
}

impl CryptoMarket {
    /// Get remaining time until market ends
    pub fn remaining(&self) -> Duration {
        let now = Utc::now();
        if now >= self.end_time {
            Duration::zero()
        } else {
            self.end_time - now
        }
    }

    /// Check if there's arbitrage opportunity (sum < 1.0)
    pub fn has_arbitrage(&self, min_spread: Decimal) -> bool {
        self.spread > min_spread
    }
}

/// API response structures
#[derive(Debug, Deserialize)]
struct GammaEventResponse {
    title: String,
    slug: String,
    markets: Vec<GammaMarketResponse>,
}

#[derive(Debug, Deserialize)]
struct GammaMarketResponse {
    question: String,
    #[serde(rename = "conditionId")]
    condition_id: String,
    #[serde(rename = "clobTokenIds")]
    clob_token_ids: Option<String>,
    #[serde(rename = "outcomePrices")]
    outcome_prices: Option<String>,
    active: Option<bool>,
    closed: Option<bool>,
}

/// Crypto market discovery service
pub struct CryptoMarketDiscovery {
    http: Client,
    gamma_url: String,
}

impl CryptoMarketDiscovery {
    /// Create a new discovery service
    pub fn new(gamma_url: &str) -> Self {
        Self {
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap(),
            gamma_url: gamma_url.trim_end_matches('/').to_string(),
        }
    }

    /// Get current market for a symbol and interval
    ///
    /// This is the KEY discovery mechanism:
    /// 1. Calculate the aligned timestamp for current window
    /// 2. Build the slug: `{symbol}-{interval_suffix}-{timestamp}`
    /// 3. Query the Gamma API
    pub async fn get_current_market(
        &self,
        symbol: &str,
        interval: MarketInterval,
    ) -> Result<Option<CryptoMarket>> {
        let now = Utc::now().timestamp();
        let aligned = interval.align_timestamp(now);
        
        self.get_market_by_timestamp(symbol, interval, aligned).await
    }

    /// Get next upcoming market
    pub async fn get_next_market(
        &self,
        symbol: &str,
        interval: MarketInterval,
    ) -> Result<Option<CryptoMarket>> {
        let now = Utc::now().timestamp();
        let aligned = interval.align_timestamp(now);
        let next_ts = aligned + interval.duration_secs();
        
        self.get_market_by_timestamp(symbol, interval, next_ts).await
    }

    /// Get market by specific timestamp
    pub async fn get_market_by_timestamp(
        &self,
        symbol: &str,
        interval: MarketInterval,
        timestamp: i64,
    ) -> Result<Option<CryptoMarket>> {
        let slug = format!("{}-{}-{}", symbol.to_lowercase(), interval.slug_suffix(), timestamp);
        let url = format!("{}/events?slug={}", self.gamma_url, slug);

        debug!("[CryptoDiscovery] Fetching market: {}", slug);

        let resp = self.http.get(&url).send().await?;
        
        if !resp.status().is_success() {
            debug!("[CryptoDiscovery] Market not found: {}", slug);
            return Ok(None);
        }

        let events: Vec<GammaEventResponse> = resp.json().await?;
        
        if events.is_empty() || events[0].markets.is_empty() {
            return Ok(None);
        }

        let event = &events[0];
        let market = &event.markets[0];

        // Parse token IDs
        let token_ids: Vec<String> = market
            .clob_token_ids
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        if token_ids.len() < 2 {
            return Ok(None);
        }

        // Parse prices
        let prices: Vec<String> = market
            .outcome_prices
            .as_ref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        let up_price: Decimal = prices
            .get(0)
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();
        let down_price: Decimal = prices
            .get(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or_default();

        let sum = up_price + down_price;
        let spread = Decimal::ONE - sum;

        let start_time = DateTime::from_timestamp(timestamp, 0)
            .unwrap_or_else(Utc::now);
        let end_time = DateTime::from_timestamp(timestamp + interval.duration_secs(), 0)
            .unwrap_or_else(Utc::now);

        let is_active = market.active.unwrap_or(true) && !market.closed.unwrap_or(false);

        Ok(Some(CryptoMarket {
            symbol: symbol.to_uppercase(),
            interval,
            slug: event.slug.clone(),
            title: event.title.clone(),
            condition_id: market.condition_id.clone(),
            up_token_id: token_ids[0].clone(),
            down_token_id: token_ids[1].clone(),
            up_price,
            down_price,
            sum,
            spread,
            start_time,
            end_time,
            active: is_active,
        }))
    }

    /// Get all current markets for all symbols and a given interval
    pub async fn get_all_current_markets(
        &self,
        interval: MarketInterval,
    ) -> Vec<CryptoMarket> {
        let mut markets = Vec::new();

        for symbol in CRYPTO_SYMBOLS {
            match self.get_current_market(symbol, interval).await {
                Ok(Some(m)) => {
                    info!(
                        "[CryptoDiscovery] Found {}-{}: UP={:.3} DOWN={:.3} SUM={:.4}",
                        m.symbol, 
                        interval.slug_suffix(),
                        m.up_price, 
                        m.down_price, 
                        m.sum
                    );
                    markets.push(m);
                }
                Ok(None) => {
                    debug!("[CryptoDiscovery] No market for {}-{}", symbol, interval.slug_suffix());
                }
                Err(e) => {
                    warn!("[CryptoDiscovery] Error fetching {}-{}: {}", symbol, interval.slug_suffix(), e);
                }
            }
        }

        markets
    }

    /// Discover markets across multiple intervals
    pub async fn discover_all_crypto_markets(&self) -> HashMap<MarketInterval, Vec<CryptoMarket>> {
        let intervals = [
            MarketInterval::Min15,
            MarketInterval::Min30,
            MarketInterval::Hourly,
        ];

        let mut result = HashMap::new();

        for interval in intervals {
            let markets = self.get_all_current_markets(interval).await;
            if !markets.is_empty() {
                result.insert(interval, markets);
            }
        }

        result
    }
}

/// Helper to get aligned timestamp for current window
pub fn get_aligned_timestamp(interval: MarketInterval) -> i64 {
    let now = Utc::now().timestamp();
    interval.align_timestamp(now)
}

/// Calculate remaining time in current window
pub fn get_remaining_time(interval: MarketInterval) -> Duration {
    let now = Utc::now().timestamp();
    let aligned = interval.align_timestamp(now);
    let end = aligned + interval.duration_secs();
    let remaining = end - now;
    Duration::seconds(remaining)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_interval_alignment() {
        // Test 15-min alignment
        let ts = 1706590850; // Some arbitrary timestamp
        let aligned = MarketInterval::Min15.align_timestamp(ts);
        assert_eq!(aligned % 900, 0);
        assert!(aligned <= ts);
        assert!(aligned + 900 > ts);
    }

    #[test]
    fn test_slug_format() {
        let ts = 1706590800;
        let interval = MarketInterval::Min15;
        let slug = format!("btc-{}-{}", interval.slug_suffix(), ts);
        assert_eq!(slug, "btc-updown-15m-1706590800");
    }

    #[test]
    fn test_hourly_alignment() {
        let ts = 1706594000; // Some timestamp
        let aligned = MarketInterval::Hourly.align_timestamp(ts);
        assert_eq!(aligned % 3600, 0);
    }

    #[test]
    fn test_arbitrage_check() {
        let market = CryptoMarket {
            symbol: "BTC".into(),
            interval: MarketInterval::Min15,
            slug: "test".into(),
            title: "Test".into(),
            condition_id: "cond".into(),
            up_token_id: "up".into(),
            down_token_id: "down".into(),
            up_price: dec!(0.48),
            down_price: dec!(0.48),
            sum: dec!(0.96),
            spread: dec!(0.04),  // 4% spread
            start_time: Utc::now(),
            end_time: Utc::now(),
            active: true,
        };

        assert!(market.has_arbitrage(dec!(0.01)));  // 1% threshold
        assert!(market.has_arbitrage(dec!(0.03)));  // 3% threshold
        assert!(!market.has_arbitrage(dec!(0.05))); // 5% threshold
    }
}
