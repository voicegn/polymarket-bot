//! High-Frequency Crypto Up/Down Strategy
//!
//! Trades 15-minute BTC/ETH/SOL/XRP Up/Down markets based on
//! real-time price momentum.

use crate::error::Result;
use crate::types::{Market, Side, Signal};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;
use std::collections::VecDeque;

/// Crypto price tracker
pub struct CryptoPriceTracker {
    http: reqwest::Client,
    // Recent prices for momentum calculation
    btc_prices: VecDeque<PricePoint>,
    eth_prices: VecDeque<PricePoint>,
    sol_prices: VecDeque<PricePoint>,
    xrp_prices: VecDeque<PricePoint>,
    max_history: usize,
}

#[derive(Debug, Clone)]
pub struct PricePoint {
    pub price: Decimal,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
struct BinancePrice {
    #[allow(dead_code)]
    symbol: String,
    price: String,
}

/// High-frequency crypto strategy
pub struct CryptoHfStrategy {
    /// Minimum momentum to trigger trade (e.g., 0.002 = 0.2%)
    pub min_momentum: Decimal,
    /// Minutes before market close to enter
    pub entry_minutes_before_close: u32,
    /// Minimum market price to consider "certain" direction
    pub certainty_threshold: Decimal,
    /// Maximum position size in USD
    pub max_position_usd: Decimal,
}

impl Default for CryptoHfStrategy {
    fn default() -> Self {
        Self {
            min_momentum: dec!(0.001),           // 0.1% minimum move (more aggressive)
            entry_minutes_before_close: 3,       // Enter 3 mins before close
            certainty_threshold: dec!(0.85),     // 85% certainty
            max_position_usd: dec!(20),          // $20 max per trade
        }
    }
}

impl CryptoPriceTracker {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            btc_prices: VecDeque::with_capacity(100),
            eth_prices: VecDeque::with_capacity(100),
            sol_prices: VecDeque::with_capacity(100),
            xrp_prices: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }

    /// Initialize history using Binance klines API
    pub async fn init_history(&mut self) -> Result<()> {
        let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"];
        
        for symbol in symbols {
            // Get 1-minute klines for last 15 minutes
            let url = format!(
                "https://api.binance.com/api/v3/klines?symbol={}&interval=1m&limit=15",
                symbol
            );
            
            let resp: Vec<Vec<serde_json::Value>> = match self.http.get(&url).send().await {
                Ok(r) => match r.json().await {
                    Ok(j) => j,
                    Err(_) => continue,
                },
                Err(_) => continue,
            };
            
            let queue = match symbol {
                "BTCUSDT" => &mut self.btc_prices,
                "ETHUSDT" => &mut self.eth_prices,
                "SOLUSDT" => &mut self.sol_prices,
                "XRPUSDT" => &mut self.xrp_prices,
                _ => continue,
            };
            
            // Parse klines: [open_time, open, high, low, close, ...]
            for kline in resp {
                if kline.len() < 5 {
                    continue;
                }
                let timestamp_ms = kline[0].as_i64().unwrap_or(0);
                let close_price = kline[4].as_str().unwrap_or("0");
                
                if let Ok(price) = close_price.parse::<Decimal>() {
                    let timestamp = DateTime::from_timestamp_millis(timestamp_ms)
                        .unwrap_or_else(Utc::now);
                    queue.push_back(PricePoint { price, timestamp });
                }
            }
        }
        
        tracing::info!("Initialized crypto price history: BTC={}, ETH={}, SOL={}, XRP={} points",
            self.btc_prices.len(), self.eth_prices.len(), 
            self.sol_prices.len(), self.xrp_prices.len());
        
        Ok(())
    }

    /// Fetch current prices from Binance
    pub async fn update_prices(&mut self) -> Result<()> {
        let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"];
        
        for symbol in symbols {
            let url = format!(
                "https://api.binance.com/api/v3/ticker/price?symbol={}",
                symbol
            );
            
            let resp: BinancePrice = self.http
                .get(&url)
                .send()
                .await?
                .json()
                .await?;
            
            let price: Decimal = resp.price.parse().unwrap_or(Decimal::ZERO);
            let point = PricePoint {
                price,
                timestamp: Utc::now(),
            };
            
            let queue = match symbol {
                "BTCUSDT" => &mut self.btc_prices,
                "ETHUSDT" => &mut self.eth_prices,
                "SOLUSDT" => &mut self.sol_prices,
                "XRPUSDT" => &mut self.xrp_prices,
                _ => continue,
            };
            
            if queue.len() >= self.max_history {
                queue.pop_front();
            }
            queue.push_back(point);
        }
        
        Ok(())
    }

    /// Calculate momentum over last N minutes
    pub fn calculate_momentum(&self, asset: &str, minutes: u32) -> Option<Decimal> {
        let queue = match asset.to_uppercase().as_str() {
            "BTC" | "BTCUSDT" => &self.btc_prices,
            "ETH" | "ETHUSDT" => &self.eth_prices,
            "SOL" | "SOLUSDT" => &self.sol_prices,
            "XRP" | "XRPUSDT" => &self.xrp_prices,
            _ => return None,
        };
        
        if queue.len() < 2 {
            return None;
        }
        
        let now = Utc::now();
        let cutoff = now - chrono::Duration::minutes(minutes as i64);
        
        // Find oldest price within window
        let oldest = queue.iter()
            .find(|p| p.timestamp >= cutoff)?;
        
        let latest = queue.back()?;
        
        // Calculate percentage change
        if oldest.price == Decimal::ZERO {
            return None;
        }
        
        Some((latest.price - oldest.price) / oldest.price)
    }

    /// Get current price
    pub fn current_price(&self, asset: &str) -> Option<Decimal> {
        let queue = match asset.to_uppercase().as_str() {
            "BTC" | "BTCUSDT" => &self.btc_prices,
            "ETH" | "ETHUSDT" => &self.eth_prices,
            "SOL" | "SOLUSDT" => &self.sol_prices,
            "XRP" | "XRPUSDT" => &self.xrp_prices,
            _ => return None,
        };
        
        queue.back().map(|p| p.price)
    }
}

impl CryptoHfStrategy {
    /// Check if this is a crypto Up/Down market
    pub fn is_crypto_hf_market(market: &Market) -> Option<CryptoMarketInfo> {
        let question = market.question.to_lowercase();
        
        // Match patterns like "Bitcoin Up or Down - January 28, 10:45PM-11:00PM ET"
        let asset = if question.contains("bitcoin") || question.contains("btc") {
            "BTC"
        } else if question.contains("ethereum") || question.contains("eth") {
            "ETH"
        } else if question.contains("solana") || question.contains("sol") {
            "SOL"
        } else if question.contains("xrp") {
            "XRP"
        } else {
            return None;
        };
        
        if !question.contains("up or down") {
            return None;
        }
        
        // Try to parse the time window
        // Format: "10:45PM-11:00PM ET"
        // For now, just check if market is active and near closing
        
        Some(CryptoMarketInfo {
            asset: asset.to_string(),
            is_15_min: question.contains(":45") || question.contains(":00") || question.contains(":15") || question.contains(":30"),
        })
    }

    /// Generate signal for crypto HF market
    pub fn generate_signal(
        &self,
        market: &Market,
        tracker: &CryptoPriceTracker,
    ) -> Option<Signal> {
        let info = Self::is_crypto_hf_market(market)?;
        
        // Get current momentum
        let momentum = match tracker.calculate_momentum(&info.asset, 10) {
            Some(m) => m,
            None => {
                tracing::debug!("Crypto {}: No momentum data yet (need more price history)", info.asset);
                return None;
            }
        };
        
        tracing::info!("Crypto {}: momentum = {:.4}%", info.asset, momentum * dec!(100));
        
        // Check if momentum is strong enough
        if momentum.abs() < self.min_momentum {
            tracing::debug!("Crypto {}: momentum {:.4}% below threshold {:.4}%", 
                info.asset, momentum.abs() * dec!(100), self.min_momentum * dec!(100));
            return None;
        }
        
        // Get current market prices
        let up_price = market.outcomes.iter()
            .find(|o| o.outcome.to_lowercase() == "up")
            .map(|o| o.price)?;
        
        let down_price = market.outcomes.iter()
            .find(|o| o.outcome.to_lowercase() == "down")
            .map(|o| o.price)?;
        
        // Determine direction based on momentum
        let (side, token_id, model_prob, market_prob) = if momentum > Decimal::ZERO {
            // Price going up -> buy "Up"
            let token = market.outcomes.iter()
                .find(|o| o.outcome.to_lowercase() == "up")?
                .token_id.clone();
            
            // Our model probability based on momentum strength
            let prob = dec!(0.5) + (momentum * dec!(100)).min(dec!(0.4));
            
            (Side::Buy, token, prob, up_price)
        } else {
            // Price going down -> buy "Down"
            let token = market.outcomes.iter()
                .find(|o| o.outcome.to_lowercase() == "down")?
                .token_id.clone();
            
            let prob = dec!(0.5) + (momentum.abs() * dec!(100)).min(dec!(0.4));
            
            (Side::Buy, token, prob, down_price)
        };
        
        // Check if market price already reflects the momentum (no edge)
        let edge = model_prob - market_prob;
        if edge < dec!(0.05) {
            return None; // Not enough edge
        }
        
        // Calculate position size (fixed small amount for HF)
        let size = self.max_position_usd.min(dec!(10));
        
        Some(Signal {
            market_id: market.id.clone(),
            token_id,
            side,
            model_probability: model_prob,
            market_probability: market_prob,
            edge,
            confidence: dec!(0.7), // Medium confidence for HF trades
            suggested_size: size / dec!(100), // As fraction of portfolio
            timestamp: Utc::now(),
        })
    }
}

#[derive(Debug)]
pub struct CryptoMarketInfo {
    pub asset: String,
    pub is_15_min: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Outcome;

    #[test]
    fn test_parse_crypto_market() {
        let market = Market {
            id: "test".to_string(),
            question: "Bitcoin Up or Down - January 28, 10:45PM-11:00PM ET".to_string(),
            description: None,
            end_date: None,
            volume: Decimal::ZERO,
            liquidity: Decimal::ZERO,
            outcomes: vec![],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market);
        assert!(info.is_some());
        assert_eq!(info.unwrap().asset, "BTC");
    }

    #[test]
    fn test_parse_eth_market() {
        let market = Market {
            id: "eth1".to_string(),
            question: "Ethereum Up or Down - January 28, 9PM ET".to_string(),
            description: None,
            end_date: None,
            volume: Decimal::ZERO,
            liquidity: Decimal::ZERO,
            outcomes: vec![],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market);
        assert!(info.is_some());
        assert_eq!(info.unwrap().asset, "ETH");
    }

    #[test]
    fn test_parse_sol_market() {
        let market = Market {
            id: "sol1".to_string(),
            question: "Solana Up or Down - January 28".to_string(),
            description: None,
            end_date: None,
            volume: Decimal::ZERO,
            liquidity: Decimal::ZERO,
            outcomes: vec![],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market);
        assert!(info.is_some());
        assert_eq!(info.unwrap().asset, "SOL");
    }

    #[test]
    fn test_parse_xrp_market() {
        let market = Market {
            id: "xrp1".to_string(),
            question: "XRP Up or Down - January 28, 11PM ET".to_string(),
            description: None,
            end_date: None,
            volume: Decimal::ZERO,
            liquidity: Decimal::ZERO,
            outcomes: vec![],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market);
        assert!(info.is_some());
        assert_eq!(info.unwrap().asset, "XRP");
    }

    #[test]
    fn test_non_crypto_market() {
        let market = Market {
            id: "politics1".to_string(),
            question: "Will Trump win in 2024?".to_string(),
            description: None,
            end_date: None,
            volume: Decimal::ZERO,
            liquidity: Decimal::ZERO,
            outcomes: vec![],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market);
        assert!(info.is_none());
    }

    #[test]
    fn test_crypto_hf_strategy_default() {
        let strategy = CryptoHfStrategy::default();
        assert_eq!(strategy.min_momentum, dec!(0.003));
        assert_eq!(strategy.entry_minutes_before_close, 3);
        assert_eq!(strategy.certainty_threshold, dec!(0.85));
        assert_eq!(strategy.max_position_usd, dec!(20));
    }

    #[test]
    fn test_crypto_price_tracker_new() {
        let tracker = CryptoPriceTracker::new();
        assert_eq!(tracker.max_history, 100);
    }

    #[test]
    fn test_price_point_creation() {
        let point = PricePoint {
            price: dec!(50000),
            timestamp: Utc::now(),
        };
        assert_eq!(point.price, dec!(50000));
    }

    #[test]
    fn test_price_point_clone() {
        let point = PricePoint {
            price: dec!(3000),
            timestamp: Utc::now(),
        };
        let cloned = point.clone();
        assert_eq!(point.price, cloned.price);
    }

    #[test]
    fn test_crypto_market_info() {
        let info = CryptoMarketInfo {
            asset: "BTC".to_string(),
            is_15_min: true,
        };
        assert_eq!(info.asset, "BTC");
        assert!(info.is_15_min);
    }

    #[test]
    fn test_15_min_market_detection() {
        let market = Market {
            id: "test".to_string(),
            question: "Bitcoin Up or Down - January 28, 10:45PM-11:00PM ET".to_string(),
            description: None,
            end_date: None,
            volume: Decimal::ZERO,
            liquidity: Decimal::ZERO,
            outcomes: vec![],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market).unwrap();
        assert!(info.is_15_min);
    }

    #[test]
    fn test_hourly_market_detection() {
        let market = Market {
            id: "test".to_string(),
            question: "Bitcoin Up or Down - January 28, 10PM ET".to_string(),
            description: None,
            end_date: None,
            volume: Decimal::ZERO,
            liquidity: Decimal::ZERO,
            outcomes: vec![],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market).unwrap();
        assert!(!info.is_15_min);
    }

    #[test]
    fn test_market_with_outcomes() {
        let market = Market {
            id: "btc1".to_string(),
            question: "Bitcoin Up or Down - January 28, 11PM ET".to_string(),
            description: None,
            end_date: None,
            volume: dec!(10000),
            liquidity: dec!(5000),
            outcomes: vec![
                Outcome {
                    token_id: "up".to_string(),
                    outcome: "Up".to_string(),
                    price: dec!(0.65),
                },
                Outcome {
                    token_id: "down".to_string(),
                    outcome: "Down".to_string(),
                    price: dec!(0.35),
                },
            ],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market);
        assert!(info.is_some());
        assert_eq!(market.outcomes.len(), 2);
    }
}
