//! High-Frequency Crypto Up/Down Strategy
//!
//! Trades 15-minute BTC/ETH/SOL/XRP Up/Down markets based on
//! real-time price momentum.

use crate::error::Result;
use crate::strategy::trend_detector::{PriceBar, TrendDetector, TrendSignal};
use crate::types::{Market, Side, Signal};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;
use std::collections::VecDeque;

/// Crypto price tracker with full OHLCV data
pub struct CryptoPriceTracker {
    http: reqwest::Client,
    // Recent prices for backward compatibility
    btc_prices: VecDeque<PricePoint>,
    eth_prices: VecDeque<PricePoint>,
    sol_prices: VecDeque<PricePoint>,
    xrp_prices: VecDeque<PricePoint>,
    // Full OHLCV klines for trend analysis
    btc_bars: VecDeque<PriceBar>,
    eth_bars: VecDeque<PriceBar>,
    sol_bars: VecDeque<PriceBar>,
    xrp_bars: VecDeque<PriceBar>,
    max_history: usize,
    // Trend detector
    trend_detector: TrendDetector,
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
            min_momentum: dec!(0.0003),           // 0.1% minimum move (more aggressive)
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
            btc_prices: VecDeque::with_capacity(500),
            eth_prices: VecDeque::with_capacity(500),
            sol_prices: VecDeque::with_capacity(500),
            xrp_prices: VecDeque::with_capacity(500),
            btc_bars: VecDeque::with_capacity(500),
            eth_bars: VecDeque::with_capacity(500),
            sol_bars: VecDeque::with_capacity(500),
            xrp_bars: VecDeque::with_capacity(500),
            max_history: 500,  // ~8 hours of 1-minute data
            trend_detector: TrendDetector::new(),
        }
    }

    /// Initialize history using Binance klines API
    /// Fetches enough 1m klines to support all market durations (15min to 6h momentum windows)
    pub async fn init_history(&mut self) -> Result<()> {
        let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"];
        
        for symbol in symbols {
            // Get 1-minute klines for last ~6 hours (360 minutes) to support all momentum windows
            // This covers: 10min (15m markets), 30min (1h markets), 120min (4h markets), 360min (daily)
            let url = format!(
                "https://api.binance.com/api/v3/klines?symbol={}&interval=1m&limit=360",
                symbol
            );
            
            let resp: Vec<Vec<serde_json::Value>> = match self.http.get(&url).send().await {
                Ok(r) => match r.json().await {
                    Ok(j) => j,
                    Err(_) => continue,
                },
                Err(_) => continue,
            };
            
            let (price_queue, bar_queue) = match symbol {
                "BTCUSDT" => (&mut self.btc_prices, &mut self.btc_bars),
                "ETHUSDT" => (&mut self.eth_prices, &mut self.eth_bars),
                "SOLUSDT" => (&mut self.sol_prices, &mut self.sol_bars),
                "XRPUSDT" => (&mut self.xrp_prices, &mut self.xrp_bars),
                _ => continue,
            };
            
            // Parse klines: [open_time, open, high, low, close, volume, ...]
            for kline in resp {
                if kline.len() < 6 {
                    continue;
                }
                let timestamp_ms = kline[0].as_i64().unwrap_or(0);
                let open = kline[1].as_str().unwrap_or("0").parse::<Decimal>().unwrap_or_default();
                let high = kline[2].as_str().unwrap_or("0").parse::<Decimal>().unwrap_or_default();
                let low = kline[3].as_str().unwrap_or("0").parse::<Decimal>().unwrap_or_default();
                let close = kline[4].as_str().unwrap_or("0").parse::<Decimal>().unwrap_or_default();
                let volume = kline[5].as_str().unwrap_or("0").parse::<Decimal>().unwrap_or_default();
                
                // Add to legacy price queue
                let timestamp = DateTime::from_timestamp_millis(timestamp_ms)
                    .unwrap_or_else(Utc::now);
                price_queue.push_back(PricePoint { price: close, timestamp });
                
                // Add to OHLCV bar queue
                bar_queue.push_back(PriceBar {
                    open, high, low, close, volume,
                    timestamp_ms,
                });
            }
        }
        
        tracing::info!("Initialized crypto price history: BTC={}, ETH={}, SOL={}, XRP={} bars",
            self.btc_bars.len(), self.eth_bars.len(), 
            self.sol_bars.len(), self.xrp_bars.len());
        
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
    
    /// 使用多指标趋势检测器分析资产
    pub fn analyze_trend(&self, asset: &str) -> Option<TrendSignal> {
        let bars = match asset.to_uppercase().as_str() {
            "BTC" | "BTCUSDT" => &self.btc_bars,
            "ETH" | "ETHUSDT" => &self.eth_bars,
            "SOL" | "SOLUSDT" => &self.sol_bars,
            "XRP" | "XRPUSDT" => &self.xrp_bars,
            _ => return None,
        };
        
        let bars_vec: Vec<PriceBar> = bars.iter().cloned().collect();
        self.trend_detector.analyze(&bars_vec)
    }
    
    /// 检查是否应该交易 (基于趋势信号)
    pub fn should_trade(&self, asset: &str) -> bool {
        self.analyze_trend(asset)
            .map(|s| self.trend_detector.should_trade(&s))
            .unwrap_or(false)
    }
}

impl CryptoHfStrategy {
    /// Check if this is a crypto Up/Down market and parse time window
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
        
        // Parse time window duration
        // Patterns: "10:45PM-11:00PM" (15 min), "10PM-11PM" (1 hour), "8AM-12PM" (4 hours)
        let duration_minutes = Self::parse_duration(&question);
        
        Some(CryptoMarketInfo {
            asset: asset.to_string(),
            duration_minutes,
        })
    }
    
    /// Parse market duration from question text
    fn parse_duration(question: &str) -> u32 {
        // Check for explicit duration markers
        if question.contains("15 min") || question.contains("15-min") || question.contains("15min") {
            return 15;
        }
        if question.contains("4 hour") || question.contains("4-hour") || question.contains("4h") {
            return 240;
        }
        if question.contains("daily") || question.contains("24 hour") || question.contains("day") {
            return 1440;
        }
        
        // Try to parse time range like "10:45PM-11:00PM" or "10PM-11PM"
        // If we see :15, :30, :45 patterns, it's likely a 15-min market
        if question.contains(":15") || question.contains(":30") || question.contains(":45") {
            // Contains minute markers → likely 15-minute market
            return 15;
        }
        
        // Check for hour spans like "8AM-12PM" (4 hours) vs "10PM-11PM" (1 hour)
        // Simple heuristic: if no minute markers, assume 1 hour
        if question.contains("am-") || question.contains("pm-") 
           || question.contains("am -") || question.contains("pm -") {
            // Has time range, assume 1 hour if no minute markers
            return 60;
        }
        
        // Default to 1 hour for unrecognized patterns
        60
    }

    /// Generate signal for crypto HF market using multi-indicator trend detection
    pub fn generate_signal(
        &self,
        market: &Market,
        tracker: &CryptoPriceTracker,
    ) -> Option<Signal> {
        let info = Self::is_crypto_hf_market(market)?;
        
        // 使用多指标趋势检测器分析
        let trend_signal = match tracker.analyze_trend(&info.asset) {
            Some(s) => s,
            None => {
                tracing::debug!("Crypto {}: 数据不足，无法分析趋势", info.asset);
                return None;
            }
        };
        
        tracing::info!("Crypto {} ({}min市场): 趋势={:?} 置信度={:.1}% | {}",
            info.asset, info.duration_minutes, 
            trend_signal.trend, trend_signal.confidence * dec!(100),
            trend_signal.reason);
        
        // 检查是否应该交易
        if !tracker.trend_detector.should_trade(&trend_signal) {
            tracing::debug!("Crypto {}: 置信度不足 ({:.1}% < 65%)，不交易",
                info.asset, trend_signal.confidence * dec!(100));
            return None;
        }
        
        // 获取建议方向
        let direction = match trend_signal.suggested_direction() {
            Some(d) => d,
            None => {
                tracing::debug!("Crypto {}: 趋势中性，不交易", info.asset);
                return None;
            }
        };
        
        // Get current market prices
        let up_price = market.outcomes.iter()
            .find(|o| o.outcome.to_lowercase() == "up")
            .map(|o| o.price)?;
        
        let down_price = market.outcomes.iter()
            .find(|o| o.outcome.to_lowercase() == "down")
            .map(|o| o.price)?;
        
        // 根据趋势方向确定交易
        let (side, token_id, model_prob, market_prob) = if direction == "Up" {
            let token = market.outcomes.iter()
                .find(|o| o.outcome.to_lowercase() == "up")?
                .token_id.clone();
            
            // 基于置信度计算模型概率
            let prob = dec!(0.5) + (trend_signal.confidence * dec!(0.4));
            
            (Side::Buy, token, prob, up_price)
        } else {
            let token = market.outcomes.iter()
                .find(|o| o.outcome.to_lowercase() == "down")?
                .token_id.clone();
            
            let prob = dec!(0.5) + (trend_signal.confidence * dec!(0.4));
            
            (Side::Buy, token, prob, down_price)
        };
        
        // 计算 edge，扣除手续费后的净收益
        // Polymarket 手续费: ~2% taker fee
        let fee_rate = dec!(0.02);
        let gross_edge = model_prob - market_prob;
        let net_edge = gross_edge - fee_rate;
        
        // 最小净 edge 要求 (扣费后还要有 3% 利润空间)
        let min_net_edge = dec!(0.03);
        
        if net_edge < min_net_edge {
            tracing::debug!("Crypto {}: 净edge不足 ({:.1}% < {:.1}%)，毛edge={:.1}%，手续费={:.1}%",
                info.asset, 
                net_edge * dec!(100), min_net_edge * dec!(100),
                gross_edge * dec!(100), fee_rate * dec!(100));
            return None;
        }
        
        tracing::info!("Crypto {}: ✅ 净edge={:.1}% (毛{:.1}% - 手续费{:.1}%)",
            info.asset, net_edge * dec!(100), gross_edge * dec!(100), fee_rate * dec!(100));
        
        // 根据置信度和 edge 调整仓位
        let base_size = self.max_position_usd;
        let size_factor = trend_signal.position_size_factor(); // 强趋势=1.0, 弱趋势=0.5
        let size = (base_size * size_factor).min(dec!(20));
        
        Some(Signal {
            market_id: market.id.clone(),
            token_id,
            side,
            model_probability: model_prob,
            market_probability: market_prob,
            edge: net_edge, // 使用净 edge
            confidence: trend_signal.confidence,
            suggested_size: size / dec!(100), // As fraction of portfolio
            timestamp: Utc::now(),
        })
    }
}

/// Market time window information
#[derive(Debug, Clone)]
pub struct CryptoMarketInfo {
    pub asset: String,
    pub duration_minutes: u32,  // 15, 60, 240, 1440
}

impl CryptoMarketInfo {
    /// Get the recommended kline interval for this market
    pub fn kline_interval(&self) -> &'static str {
        match self.duration_minutes {
            0..=15 => "1m",      // 15 分钟市场 → 1 分钟 K 线
            16..=60 => "5m",     // 1 小时市场 → 5 分钟 K 线
            61..=240 => "15m",   // 4 小时市场 → 15 分钟 K 线
            _ => "1h",           // 日线市场 → 1 小时 K 线
        }
    }
    
    /// Get number of klines to fetch for history
    pub fn kline_limit(&self) -> u32 {
        match self.duration_minutes {
            0..=15 => 15,    // 15 条 1m K 线 = 15 分钟历史
            16..=60 => 12,   // 12 条 5m K 线 = 1 小时历史
            61..=240 => 16,  // 16 条 15m K 线 = 4 小时历史
            _ => 24,         // 24 条 1h K 线 = 24 小时历史
        }
    }
    
    /// Get momentum calculation window in minutes
    pub fn momentum_minutes(&self) -> u32 {
        match self.duration_minutes {
            0..=15 => 10,    // 最近 10 分钟动量
            16..=60 => 30,   // 最近 30 分钟动量
            61..=240 => 120, // 最近 2 小时动量
            _ => 360,        // 最近 6 小时动量
        }
    }
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
        assert_eq!(strategy.min_momentum, dec!(0.0003)); // 0.03%
        assert_eq!(strategy.entry_minutes_before_close, 3);
        assert_eq!(strategy.certainty_threshold, dec!(0.85));
        assert_eq!(strategy.max_position_usd, dec!(20));
    }

    #[test]
    fn test_crypto_price_tracker_new() {
        let tracker = CryptoPriceTracker::new();
        assert_eq!(tracker.max_history, 500); // ~8 hours of 1-minute data
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
            duration_minutes: 15,
        };
        assert_eq!(info.asset, "BTC");
        assert_eq!(info.duration_minutes, 15);
        assert_eq!(info.kline_interval(), "1m");
        assert_eq!(info.momentum_minutes(), 10);
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
        assert_eq!(info.duration_minutes, 15);
        assert_eq!(info.kline_interval(), "1m");
    }

    #[test]
    fn test_hourly_market_detection() {
        let market = Market {
            id: "test".to_string(),
            question: "Bitcoin Up or Down - January 28, 10PM-11PM ET".to_string(),
            description: None,
            end_date: None,
            volume: Decimal::ZERO,
            liquidity: Decimal::ZERO,
            outcomes: vec![],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market).unwrap();
        assert_eq!(info.duration_minutes, 60);
        assert_eq!(info.kline_interval(), "5m");
    }
    
    #[test]
    fn test_4h_market_detection() {
        let market = Market {
            id: "test".to_string(),
            question: "Bitcoin Up or Down 4-hour - January 28".to_string(),
            description: None,
            end_date: None,
            volume: Decimal::ZERO,
            liquidity: Decimal::ZERO,
            outcomes: vec![],
            active: true,
            closed: false,
        };
        
        let info = CryptoHfStrategy::is_crypto_hf_market(&market).unwrap();
        assert_eq!(info.duration_minutes, 240);
        assert_eq!(info.kline_interval(), "15m");
        assert_eq!(info.momentum_minutes(), 120);
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
