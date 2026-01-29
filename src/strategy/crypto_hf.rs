//! High-Frequency Crypto Up/Down Strategy
//!
//! Trades 15-minute BTC/ETH/SOL/XRP Up/Down markets based on
//! real-time price momentum.

use crate::error::Result;
use crate::types::{Market, Side, Signal};
use chrono::{DateTime, Utc, Timelike};
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
            min_momentum: dec!(0.003),           // 0.3% minimum move
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
        let momentum = tracker.calculate_momentum(&info.asset, 10)?;
        
        // Check if momentum is strong enough
        if momentum.abs() < self.min_momentum {
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
}
