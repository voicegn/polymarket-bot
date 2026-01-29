//! 分钟级趋势判断系统
//!
//! 多指标融合判断短期趋势，只在高置信度时触发交易信号

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;

/// 趋势方向
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Trend {
    StrongUp,    // 强势上涨
    WeakUp,      // 弱势上涨
    Neutral,     // 震荡/无趋势
    WeakDown,    // 弱势下跌
    StrongDown,  // 强势下跌
}

/// 趋势信号结果
#[derive(Debug, Clone)]
pub struct TrendSignal {
    pub trend: Trend,
    pub confidence: Decimal,     // 0-1 置信度
    pub momentum: Decimal,       // 动量值
    pub rsi: Decimal,           // RSI 指标
    pub macd_signal: Decimal,   // MACD 信号
    pub volume_ratio: Decimal,  // 成交量比率
    pub reason: String,         // 判断理由
}

/// 价格数据点（带成交量）
#[derive(Debug, Clone)]
pub struct PriceBar {
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub timestamp_ms: i64,
}

/// 趋势检测器
pub struct TrendDetector {
    // 配置参数
    rsi_period: usize,
    macd_fast: usize,
    macd_slow: usize,
    macd_signal: usize,
    volume_ma_period: usize,
    
    // 阈值
    strong_trend_threshold: Decimal,  // 强趋势需要的置信度
    min_trade_confidence: Decimal,    // 最小交易置信度
}

impl Default for TrendDetector {
    fn default() -> Self {
        Self {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            volume_ma_period: 20,
            strong_trend_threshold: dec!(0.75),
            min_trade_confidence: dec!(0.65),
        }
    }
}

impl TrendDetector {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 分析趋势（核心方法）
    pub fn analyze(&self, bars: &[PriceBar]) -> Option<TrendSignal> {
        if bars.len() < 30 {
            return None; // 数据不足
        }
        
        let closes: Vec<Decimal> = bars.iter().map(|b| b.close).collect();
        let volumes: Vec<Decimal> = bars.iter().map(|b| b.volume).collect();
        
        // 1. 计算各项指标
        let rsi = self.calculate_rsi(&closes)?;
        let (macd, signal, histogram) = self.calculate_macd(&closes)?;
        let momentum = self.calculate_momentum(&closes, 10)?;
        let volume_ratio = self.calculate_volume_ratio(&volumes)?;
        let price_action = self.analyze_price_action(bars)?;
        
        // 2. 多指标融合判断
        let mut up_score = dec!(0);
        let mut down_score = dec!(0);
        let mut reasons = Vec::new();
        
        // RSI 信号 (权重 25%)
        if rsi > dec!(70) {
            down_score += dec!(0.25); // 超买 → 可能回调
            reasons.push(format!("RSI超买({:.1})", rsi));
        } else if rsi < dec!(30) {
            up_score += dec!(0.25); // 超卖 → 可能反弹
            reasons.push(format!("RSI超卖({:.1})", rsi));
        } else if rsi > dec!(55) {
            up_score += dec!(0.15);
            reasons.push(format!("RSI偏多({:.1})", rsi));
        } else if rsi < dec!(45) {
            down_score += dec!(0.15);
            reasons.push(format!("RSI偏空({:.1})", rsi));
        }
        
        // MACD 信号 (权重 30%)
        if histogram > dec!(0) && macd > signal {
            up_score += dec!(0.30);
            reasons.push("MACD金叉".to_string());
        } else if histogram < dec!(0) && macd < signal {
            down_score += dec!(0.30);
            reasons.push("MACD死叉".to_string());
        }
        
        // 动量信号 (权重 25%)
        let momentum_pct = momentum * dec!(100);
        if momentum_pct > dec!(0.3) {
            up_score += dec!(0.25);
            reasons.push(format!("强势动量(+{:.2}%)", momentum_pct));
        } else if momentum_pct < dec!(-0.3) {
            down_score += dec!(0.25);
            reasons.push(format!("弱势动量({:.2}%)", momentum_pct));
        } else if momentum_pct > dec!(0.1) {
            up_score += dec!(0.15);
            reasons.push(format!("动量偏多(+{:.2}%)", momentum_pct));
        } else if momentum_pct < dec!(-0.1) {
            down_score += dec!(0.15);
            reasons.push(format!("动量偏空({:.2}%)", momentum_pct));
        }
        
        // 成交量确认 (权重 10%)
        if volume_ratio > dec!(1.5) {
            // 放量 → 加强当前方向
            if momentum > dec!(0) {
                up_score += dec!(0.10);
                reasons.push(format!("放量上涨({:.1}x)", volume_ratio));
            } else {
                down_score += dec!(0.10);
                reasons.push(format!("放量下跌({:.1}x)", volume_ratio));
            }
        }
        
        // 价格行为确认 (权重 10%)
        match price_action {
            PriceAction::HigherHighs => {
                up_score += dec!(0.10);
                reasons.push("创新高".to_string());
            }
            PriceAction::LowerLows => {
                down_score += dec!(0.10);
                reasons.push("创新低".to_string());
            }
            PriceAction::Consolidating => {
                reasons.push("盘整中".to_string());
            }
        }
        
        // 3. 计算最终趋势和置信度
        let (trend, confidence) = if up_score > down_score {
            let conf = up_score;
            let t = if conf >= self.strong_trend_threshold {
                Trend::StrongUp
            } else if conf >= self.min_trade_confidence {
                Trend::WeakUp
            } else {
                Trend::Neutral
            };
            (t, conf)
        } else if down_score > up_score {
            let conf = down_score;
            let t = if conf >= self.strong_trend_threshold {
                Trend::StrongDown
            } else if conf >= self.min_trade_confidence {
                Trend::WeakDown
            } else {
                Trend::Neutral
            };
            (t, conf)
        } else {
            (Trend::Neutral, dec!(0.5))
        };
        
        Some(TrendSignal {
            trend,
            confidence,
            momentum,
            rsi,
            macd_signal: histogram,
            volume_ratio,
            reason: reasons.join(" | "),
        })
    }
    
    /// 是否应该交易
    pub fn should_trade(&self, signal: &TrendSignal) -> bool {
        signal.confidence >= self.min_trade_confidence 
            && signal.trend != Trend::Neutral
    }
    
    /// 计算 RSI
    fn calculate_rsi(&self, closes: &[Decimal]) -> Option<Decimal> {
        if closes.len() < self.rsi_period + 1 {
            return None;
        }
        
        let mut gains = dec!(0);
        let mut losses = dec!(0);
        
        for i in (closes.len() - self.rsi_period)..closes.len() {
            let change = closes[i] - closes[i - 1];
            if change > dec!(0) {
                gains += change;
            } else {
                losses += change.abs();
            }
        }
        
        let avg_gain = gains / Decimal::from(self.rsi_period);
        let avg_loss = losses / Decimal::from(self.rsi_period);
        
        if avg_loss == dec!(0) {
            return Some(dec!(100));
        }
        
        let rs = avg_gain / avg_loss;
        let rsi = dec!(100) - (dec!(100) / (dec!(1) + rs));
        
        Some(rsi)
    }
    
    /// 计算 MACD
    fn calculate_macd(&self, closes: &[Decimal]) -> Option<(Decimal, Decimal, Decimal)> {
        if closes.len() < self.macd_slow + self.macd_signal {
            return None;
        }
        
        let ema_fast = self.ema(closes, self.macd_fast)?;
        let ema_slow = self.ema(closes, self.macd_slow)?;
        
        let macd_line = ema_fast - ema_slow;
        
        // 简化: 用最近的 macd 值近似信号线
        // 实际应该计算 macd 历史的 EMA
        let signal_line = macd_line * dec!(0.9); // 简化处理
        let histogram = macd_line - signal_line;
        
        Some((macd_line, signal_line, histogram))
    }
    
    /// 计算 EMA
    fn ema(&self, data: &[Decimal], period: usize) -> Option<Decimal> {
        if data.len() < period {
            return None;
        }
        
        let multiplier = dec!(2) / Decimal::from(period + 1);
        
        // 初始 SMA
        let mut ema: Decimal = data[..period].iter().sum::<Decimal>() / Decimal::from(period);
        
        // 计算 EMA
        for price in &data[period..] {
            ema = (*price - ema) * multiplier + ema;
        }
        
        Some(ema)
    }
    
    /// 计算动量 (最近 N 根 K 线的涨跌幅)
    fn calculate_momentum(&self, closes: &[Decimal], period: usize) -> Option<Decimal> {
        if closes.len() < period + 1 {
            return None;
        }
        
        let current = closes.last()?;
        let past = closes.get(closes.len() - period - 1)?;
        
        if *past == dec!(0) {
            return None;
        }
        
        Some((*current - *past) / *past)
    }
    
    /// 计算成交量比率 (当前 vs 平均)
    fn calculate_volume_ratio(&self, volumes: &[Decimal]) -> Option<Decimal> {
        if volumes.len() < self.volume_ma_period + 1 {
            return Some(dec!(1)); // 数据不足，返回中性值
        }
        
        let recent: Decimal = volumes[volumes.len() - 5..].iter().sum::<Decimal>() / dec!(5);
        let avg: Decimal = volumes[volumes.len() - self.volume_ma_period - 1..volumes.len() - 1]
            .iter()
            .sum::<Decimal>() / Decimal::from(self.volume_ma_period);
        
        if avg == dec!(0) {
            return Some(dec!(1));
        }
        
        Some(recent / avg)
    }
    
    /// 分析价格行为
    fn analyze_price_action(&self, bars: &[PriceBar]) -> Option<PriceAction> {
        if bars.len() < 10 {
            return None;
        }
        
        let recent = &bars[bars.len() - 5..];
        let prior = &bars[bars.len() - 10..bars.len() - 5];
        
        let recent_high = recent.iter().map(|b| b.high).max()?;
        let recent_low = recent.iter().map(|b| b.low).min()?;
        let prior_high = prior.iter().map(|b| b.high).max()?;
        let prior_low = prior.iter().map(|b| b.low).min()?;
        
        if recent_high > prior_high && recent_low > prior_low {
            Some(PriceAction::HigherHighs)
        } else if recent_low < prior_low && recent_high < prior_high {
            Some(PriceAction::LowerLows)
        } else {
            Some(PriceAction::Consolidating)
        }
    }
}

#[derive(Debug)]
enum PriceAction {
    HigherHighs,    // 更高的高点和更高的低点
    LowerLows,      // 更低的低点和更低的高点
    Consolidating,  // 盘整
}

impl TrendSignal {
    /// 获取建议方向 (用于 Polymarket Up/Down 市场)
    pub fn suggested_direction(&self) -> Option<&'static str> {
        match self.trend {
            Trend::StrongUp | Trend::WeakUp => Some("Up"),
            Trend::StrongDown | Trend::WeakDown => Some("Down"),
            Trend::Neutral => None,
        }
    }
    
    /// 获取建议仓位比例 (基于置信度)
    pub fn position_size_factor(&self) -> Decimal {
        match self.trend {
            Trend::StrongUp | Trend::StrongDown => dec!(1.0),   // 满仓
            Trend::WeakUp | Trend::WeakDown => dec!(0.5),       // 半仓
            Trend::Neutral => dec!(0),                          // 不交易
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_bars(prices: &[(f64, f64)]) -> Vec<PriceBar> {
        prices.iter().enumerate().map(|(i, (price, vol))| {
            let p = Decimal::from_f64_retain(*price).unwrap();
            let v = Decimal::from_f64_retain(*vol).unwrap();
            PriceBar {
                open: p,
                high: p * dec!(1.001),
                low: p * dec!(0.999),
                close: p,
                volume: v,
                timestamp_ms: i as i64 * 60000,
            }
        }).collect()
    }
    
    #[test]
    fn test_trend_detector_new() {
        let detector = TrendDetector::new();
        assert_eq!(detector.rsi_period, 14);
    }
    
    #[test]
    fn test_uptrend_detection() {
        // 模拟上涨趋势
        let prices: Vec<(f64, f64)> = (0..50)
            .map(|i| (100.0 + i as f64 * 0.5, 1000.0))
            .collect();
        
        let bars = make_bars(&prices);
        let detector = TrendDetector::new();
        let signal = detector.analyze(&bars);
        
        assert!(signal.is_some());
        let s = signal.unwrap();
        assert!(s.momentum > dec!(0));
    }
    
    #[test]
    fn test_downtrend_detection() {
        // 模拟下跌趋势
        let prices: Vec<(f64, f64)> = (0..50)
            .map(|i| (100.0 - i as f64 * 0.5, 1000.0))
            .collect();
        
        let bars = make_bars(&prices);
        let detector = TrendDetector::new();
        let signal = detector.analyze(&bars);
        
        assert!(signal.is_some());
        let s = signal.unwrap();
        assert!(s.momentum < dec!(0));
    }
    
    #[test]
    fn test_insufficient_data() {
        let bars = make_bars(&[(100.0, 1000.0); 10]);
        let detector = TrendDetector::new();
        let signal = detector.analyze(&bars);
        
        assert!(signal.is_none());
    }
}
