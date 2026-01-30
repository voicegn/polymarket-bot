//! ML Predictor - Unified Interface for Live Trading
//!
//! Provides a simple, high-level interface for ML-based predictions
//! that integrates:
//! - Technical indicator features
//! - Multi-factor fusion
//! - Ensemble prediction
//! - Probability calibration
//!
//! Usage:
//! ```ignore
//! let predictor = MLPredictor::new(MLPredictorConfig::default());
//! let result = predictor.predict(&market_data).await;
//! ```

use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use rust_decimal_macros::dec;
use std::collections::HashMap;
use chrono::Utc;

use super::features::{FeatureExtractor, FeatureConfig};
use super::calibration::{ProbabilityCalibrator, CalibrationMethod};
use super::ensemble::{EnsemblePredictor, ModelPrediction, EnsembleConfig, EnsembleMethod};
use super::factors::{MultiFactorFusion, Factor, FactorCategory, FusionConfig};

/// Market data input for prediction
#[derive(Debug, Clone)]
pub struct MarketDataInput {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Current price
    pub price: f64,
    /// OHLCV data for feature extraction
    pub klines: Vec<KlineData>,
    /// Current order book depth (optional)
    pub orderbook_imbalance: Option<f64>,
    /// Recent volume
    pub volume_24h: f64,
    /// External sentiment score (optional)
    pub sentiment_score: Option<f64>,
    /// Market question context
    pub question: String,
}

/// Single kline data point
#[derive(Debug, Clone)]
pub struct KlineData {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// ML prediction result
#[derive(Debug, Clone)]
pub struct MLPredictionResult {
    /// Predicted probability of "Up" outcome
    pub up_probability: f64,
    /// Confidence in the prediction (0-1)
    pub confidence: f64,
    /// Individual factor signals
    pub factor_signals: HashMap<String, f64>,
    /// Recommended side ("Yes" or "No")
    pub recommended_side: String,
    /// Calculated edge vs market price
    pub edge: f64,
    /// Feature summary
    pub features: FeatureSummary,
    /// Model agreement score
    pub model_agreement: f64,
}

/// Summary of extracted features
#[derive(Debug, Clone)]
pub struct FeatureSummary {
    pub rsi: f64,
    pub macd_signal: f64,
    pub bollinger_position: f64,
    pub adx: f64,
    pub momentum_1h: f64,
    pub volume_trend: f64,
    pub volatility: f64,
}

/// Configuration for ML predictor
#[derive(Debug, Clone)]
pub struct MLPredictorConfig {
    /// Enable technical analysis
    pub use_technical: bool,
    /// Enable sentiment analysis
    pub use_sentiment: bool,
    /// Enable orderbook analysis
    pub use_orderbook: bool,
    /// Calibration method
    pub calibration_method: CalibrationMethod,
    /// Ensemble method
    pub ensemble_method: EnsembleMethod,
    /// Factor weights
    pub factor_weights: FactorWeights,
}

/// Pre-configured factor weights
#[derive(Debug, Clone)]
pub struct FactorWeights {
    pub technical: f64,
    pub momentum: f64,
    pub mean_reversion: f64,
    pub volume: f64,
    pub sentiment: f64,
    pub orderbook: f64,
}

impl Default for FactorWeights {
    fn default() -> Self {
        Self {
            technical: 0.25,
            momentum: 0.20,
            mean_reversion: 0.20,
            volume: 0.15,
            sentiment: 0.10,
            orderbook: 0.10,
        }
    }
}

impl Default for MLPredictorConfig {
    fn default() -> Self {
        Self {
            use_technical: true,
            use_sentiment: true,
            use_orderbook: true,
            calibration_method: CalibrationMethod::PlattScaling,
            ensemble_method: EnsembleMethod::WeightedAverage,
            factor_weights: FactorWeights::default(),
        }
    }
}

/// ML-based predictor for crypto hourly markets
pub struct MLPredictor {
    config: MLPredictorConfig,
    feature_extractor: FeatureExtractor,
    calibrator: ProbabilityCalibrator,
    ensemble: EnsemblePredictor,
    factor_fusion: MultiFactorFusion,
}

impl MLPredictor {
    /// Create new ML predictor with given configuration
    pub fn new(config: MLPredictorConfig) -> Self {
        let feature_extractor = FeatureExtractor::new(FeatureConfig::default());
        let calibrator = ProbabilityCalibrator::new(config.calibration_method);
        let ensemble = EnsemblePredictor::new(EnsembleConfig {
            method: config.ensemble_method,
            ..Default::default()
        });
        let factor_fusion = MultiFactorFusion::new(FusionConfig::default());

        Self {
            config,
            feature_extractor,
            calibrator,
            ensemble,
            factor_fusion,
        }
    }

    /// Generate prediction for given market data
    pub fn predict(&self, data: &MarketDataInput, market_price: f64) -> MLPredictionResult {
        // Extract technical features
        let features = self.extract_features(data);
        
        // Generate individual model predictions
        let model_predictions = self.generate_model_predictions(data, &features);
        
        // Create factors for fusion
        let factors = self.create_factors(data, &features);
        
        // Combine predictions using ensemble
        let ensemble_result = self.ensemble.predict(&model_predictions);
        
        // Fuse factors for final probability
        let fusion_result = self.factor_fusion.fuse(&factors);
        
        // Get ensemble values (with defaults if None)
        let (ensemble_prob, ensemble_agreement) = match ensemble_result {
            Some(ref result) => (
                result.probability.to_f64().unwrap_or(0.5),
                result.agreement.to_f64().unwrap_or(0.5),
            ),
            None => (0.5, 0.5),
        };
        
        // Combine ensemble and factor fusion
        let raw_prob = self.combine_predictions(
            ensemble_prob,
            fusion_result.probability.to_f64().unwrap_or(0.5),
        );
        
        // Calibrate probability
        let calibration_result = self.calibrator.calibrate(
            Decimal::from_f64(raw_prob).unwrap_or(dec!(0.5))
        );
        let calibrated_prob = calibration_result.calibrated_probability.to_f64().unwrap_or(raw_prob);
        
        // Calculate confidence
        let confidence = self.calculate_confidence(
            &features,
            ensemble_agreement,
            fusion_result.confidence.to_f64().unwrap_or(0.5),
        );
        
        // Determine recommended side
        let (recommended_side, edge) = self.determine_side(
            calibrated_prob,
            market_price,
            &data.question,
        );
        
        // Collect factor signals
        let factor_signals = self.collect_factor_signals(&factors);
        
        MLPredictionResult {
            up_probability: calibrated_prob,
            confidence,
            factor_signals,
            recommended_side,
            edge,
            features,
            model_agreement: ensemble_agreement,
        }
    }

    /// Extract features from market data
    fn extract_features(&self, data: &MarketDataInput) -> FeatureSummary {
        if data.klines.is_empty() {
            return FeatureSummary {
                rsi: 50.0,
                macd_signal: 0.0,
                bollinger_position: 0.5,
                adx: 25.0,
                momentum_1h: 0.0,
                volume_trend: 0.0,
                volatility: 0.02,
            };
        }

        let closes: Vec<f64> = data.klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = data.klines.iter().map(|k| k.volume).collect();
        let highs: Vec<f64> = data.klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = data.klines.iter().map(|k| k.low).collect();

        FeatureSummary {
            rsi: self.calculate_rsi(&closes, 14),
            macd_signal: self.calculate_macd_signal(&closes),
            bollinger_position: self.calculate_bollinger_position(&closes, data.price),
            adx: self.calculate_adx(&highs, &lows, &closes, 14),
            momentum_1h: self.calculate_momentum(&closes),
            volume_trend: self.calculate_volume_trend(&volumes),
            volatility: self.calculate_volatility(&closes),
        }
    }

    /// Calculate RSI
    fn calculate_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in (prices.len() - period)..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Calculate MACD signal (positive = bullish crossover)
    fn calculate_macd_signal(&self, prices: &[f64]) -> f64 {
        if prices.len() < 26 {
            return 0.0;
        }

        let ema_12 = self.calculate_ema(prices, 12);
        let ema_26 = self.calculate_ema(prices, 26);
        let macd_line = ema_12 - ema_26;

        // Calculate signal line (9-period EMA of MACD)
        // Simplified: just return normalized MACD
        (macd_line / prices.last().unwrap_or(&1.0)) * 100.0
    }

    /// Calculate EMA
    fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }
        if prices.len() < period {
            return prices.iter().sum::<f64>() / prices.len() as f64;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[..period].iter().sum::<f64>() / period as f64;

        for price in prices.iter().skip(period) {
            ema = (*price - ema) * multiplier + ema;
        }

        ema
    }

    /// Calculate Bollinger Band position (0 = lower band, 1 = upper band)
    fn calculate_bollinger_position(&self, prices: &[f64], current_price: f64) -> f64 {
        if prices.len() < 20 {
            return 0.5;
        }

        let recent = &prices[prices.len().saturating_sub(20)..];
        let sma: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 = recent.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / recent.len() as f64;
        let std_dev = variance.sqrt();

        let upper_band = sma + 2.0 * std_dev;
        let lower_band = sma - 2.0 * std_dev;

        if upper_band == lower_band {
            return 0.5;
        }

        ((current_price - lower_band) / (upper_band - lower_band)).clamp(0.0, 1.0)
    }

    /// Calculate ADX (trend strength)
    fn calculate_adx(&self, highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
        if highs.len() < period + 1 || lows.len() < period + 1 || closes.len() < period + 1 {
            return 25.0; // Neutral
        }

        let mut plus_dm_sum = 0.0;
        let mut minus_dm_sum = 0.0;
        let mut tr_sum = 0.0;

        for i in 1..=period.min(highs.len() - 1) {
            let idx = highs.len() - period + i - 1;
            if idx == 0 { continue; }
            
            let high_diff = highs[idx] - highs[idx - 1];
            let low_diff = lows[idx - 1] - lows[idx];
            
            if high_diff > low_diff && high_diff > 0.0 {
                plus_dm_sum += high_diff;
            }
            if low_diff > high_diff && low_diff > 0.0 {
                minus_dm_sum += low_diff;
            }
            
            // True Range
            let tr = (highs[idx] - lows[idx])
                .max((highs[idx] - closes[idx - 1]).abs())
                .max((lows[idx] - closes[idx - 1]).abs());
            tr_sum += tr;
        }

        if tr_sum == 0.0 {
            return 25.0;
        }

        let plus_di = (plus_dm_sum / tr_sum) * 100.0;
        let minus_di = (minus_dm_sum / tr_sum) * 100.0;
        
        let di_diff = (plus_di - minus_di).abs();
        let di_sum = plus_di + minus_di;
        
        if di_sum == 0.0 {
            return 25.0;
        }

        (di_diff / di_sum) * 100.0
    }

    /// Calculate momentum
    fn calculate_momentum(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }
        
        let current = prices.last().unwrap();
        let previous = prices.get(prices.len().saturating_sub(2)).unwrap_or(current);
        
        if *previous == 0.0 {
            return 0.0;
        }
        
        ((current - previous) / previous) * 100.0
    }

    /// Calculate volume trend
    fn calculate_volume_trend(&self, volumes: &[f64]) -> f64 {
        if volumes.len() < 5 {
            return 0.0;
        }

        let recent = &volumes[volumes.len().saturating_sub(5)..];
        let older = &volumes[volumes.len().saturating_sub(10)..volumes.len().saturating_sub(5)];
        
        if older.is_empty() {
            return 0.0;
        }

        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;
        
        if older_avg == 0.0 {
            return 0.0;
        }

        ((recent_avg - older_avg) / older_avg) * 100.0
    }

    /// Calculate volatility
    fn calculate_volatility(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.02;
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        
        if returns.is_empty() {
            return 0.02;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }

    /// Generate model predictions
    fn generate_model_predictions(&self, _data: &MarketDataInput, features: &FeatureSummary) -> Vec<ModelPrediction> {
        let mut predictions = Vec::new();
        let now = Utc::now();

        // RSI-based prediction
        let rsi_prob = self.rsi_to_probability(features.rsi);
        predictions.push(ModelPrediction {
            model_id: "rsi_model".to_string(),
            probability: Decimal::from_f64(rsi_prob).unwrap_or(dec!(0.5)),
            confidence: Decimal::from_f64(self.rsi_confidence(features.rsi)).unwrap_or(dec!(0.5)),
            uncertainty: Some(dec!(0.1)),
            timestamp: now,
            metadata: None,
        });

        // MACD-based prediction
        let macd_prob = if features.macd_signal > 0.0 {
            0.5 + (features.macd_signal.min(1.0) * 0.2)
        } else {
            0.5 + (features.macd_signal.max(-1.0) * 0.2)
        };
        predictions.push(ModelPrediction {
            model_id: "macd_model".to_string(),
            probability: Decimal::from_f64(macd_prob).unwrap_or(dec!(0.5)),
            confidence: Decimal::from_f64(features.macd_signal.abs().min(1.0) * 0.8).unwrap_or(dec!(0.5)),
            uncertainty: Some(dec!(0.15)),
            timestamp: now,
            metadata: None,
        });

        // Bollinger-based prediction (mean reversion)
        let bb_prob = if features.bollinger_position > 0.8 {
            0.4 // Overbought, expect down
        } else if features.bollinger_position < 0.2 {
            0.6 // Oversold, expect up
        } else {
            0.5
        };
        predictions.push(ModelPrediction {
            model_id: "bollinger_model".to_string(),
            probability: Decimal::from_f64(bb_prob).unwrap_or(dec!(0.5)),
            confidence: Decimal::from_f64((features.bollinger_position - 0.5).abs() * 2.0).unwrap_or(dec!(0.5)),
            uncertainty: Some(dec!(0.12)),
            timestamp: now,
            metadata: None,
        });

        // Momentum-based prediction
        let momentum_prob = (0.5 + features.momentum_1h * 0.1).clamp(0.2, 0.8);
        predictions.push(ModelPrediction {
            model_id: "momentum_model".to_string(),
            probability: Decimal::from_f64(momentum_prob).unwrap_or(dec!(0.5)),
            confidence: Decimal::from_f64(features.momentum_1h.abs().min(1.0) * 0.7).unwrap_or(dec!(0.5)),
            uncertainty: Some(dec!(0.15)),
            timestamp: now,
            metadata: None,
        });

        // ADX trend strength adjustment
        if features.adx > 25.0 {
            // Strong trend - boost momentum prediction confidence
            if let Some(p) = predictions.iter_mut().find(|p| p.model_id == "momentum_model") {
                p.confidence = (p.confidence * dec!(1.3)).min(dec!(1.0));
            }
        }

        predictions
    }

    /// Convert RSI to probability
    fn rsi_to_probability(&self, rsi: f64) -> f64 {
        // Mean reversion logic
        if rsi > 70.0 {
            0.35 + (100.0 - rsi) / 100.0 // Overbought → lower prob of up
        } else if rsi < 30.0 {
            0.65 - rsi / 100.0 // Oversold → higher prob of up
        } else {
            0.5 + (50.0 - rsi) / 200.0 // Neutral zone, slight mean reversion
        }
    }

    /// Calculate RSI prediction confidence
    fn rsi_confidence(&self, rsi: f64) -> f64 {
        // Higher confidence at extremes
        let distance_from_50 = (rsi - 50.0).abs();
        (distance_from_50 / 50.0).min(1.0) * 0.8
    }

    /// Create factors for fusion
    fn create_factors(&self, data: &MarketDataInput, features: &FeatureSummary) -> Vec<Factor> {
        let mut factors = Vec::new();
        let now = Utc::now();

        // Technical factor
        let tech_signal = (features.rsi - 50.0) / 50.0;
        factors.push(Factor {
            id: "technical".to_string(),
            name: "Technical Analysis".to_string(),
            category: FactorCategory::Technical,
            value: Decimal::from_f64(tech_signal).unwrap_or(dec!(0)),
            signal: Decimal::from_f64(self.rsi_to_probability(features.rsi) - 0.5).unwrap_or(dec!(0)),
            confidence: Decimal::from_f64(self.rsi_confidence(features.rsi)).unwrap_or(dec!(0.5)),
            timestamp: now,
        });

        // Momentum factor
        let momentum_signal = features.momentum_1h.clamp(-1.0, 1.0);
        factors.push(Factor {
            id: "momentum".to_string(),
            name: "Momentum".to_string(),
            category: FactorCategory::Technical,
            value: Decimal::from_f64(momentum_signal).unwrap_or(dec!(0)),
            signal: Decimal::from_f64(momentum_signal * 0.2).unwrap_or(dec!(0)),
            confidence: Decimal::from_f64(features.momentum_1h.abs().min(1.0) * 0.7).unwrap_or(dec!(0.5)),
            timestamp: now,
        });

        // Volume factor
        let volume_signal = features.volume_trend.clamp(-100.0, 100.0) / 100.0;
        factors.push(Factor {
            id: "volume".to_string(),
            name: "Volume Trend".to_string(),
            category: FactorCategory::Microstructure,
            value: Decimal::from_f64(volume_signal).unwrap_or(dec!(0)),
            signal: Decimal::from_f64(volume_signal * 0.1).unwrap_or(dec!(0)),
            confidence: Decimal::from_f64(0.5).unwrap_or(dec!(0.5)),
            timestamp: now,
        });

        // Volatility factor (inverse - high vol = lower confidence)
        let vol_signal = 1.0 - (features.volatility * 20.0).min(1.0);
        factors.push(Factor {
            id: "volatility".to_string(),
            name: "Volatility".to_string(),
            category: FactorCategory::Technical,
            value: Decimal::from_f64(vol_signal).unwrap_or(dec!(0.5)),
            signal: dec!(0), // Volatility doesn't have directional signal
            confidence: Decimal::from_f64(vol_signal).unwrap_or(dec!(0.5)),
            timestamp: now,
        });

        // Sentiment factor (if available)
        if let Some(sentiment) = data.sentiment_score {
            factors.push(Factor {
                id: "sentiment".to_string(),
                name: "Sentiment".to_string(),
                category: FactorCategory::Sentiment,
                value: Decimal::from_f64(sentiment).unwrap_or(dec!(0)),
                signal: Decimal::from_f64(sentiment * 0.15).unwrap_or(dec!(0)),
                confidence: Decimal::from_f64(sentiment.abs()).unwrap_or(dec!(0.5)),
                timestamp: now,
            });
        }

        // Orderbook factor (if available)
        if let Some(imbalance) = data.orderbook_imbalance {
            factors.push(Factor {
                id: "orderbook".to_string(),
                name: "Order Book Imbalance".to_string(),
                category: FactorCategory::Microstructure,
                value: Decimal::from_f64(imbalance).unwrap_or(dec!(0)),
                signal: Decimal::from_f64(imbalance * 0.1).unwrap_or(dec!(0)),
                confidence: Decimal::from_f64(imbalance.abs()).unwrap_or(dec!(0.5)),
                timestamp: now,
            });
        }

        factors
    }

    /// Combine ensemble and factor predictions
    fn combine_predictions(&self, ensemble_prob: f64, factor_prob: f64) -> f64 {
        // Weight ensemble more heavily as it combines multiple models
        ensemble_prob * 0.6 + (factor_prob + 0.5) * 0.4
    }

    /// Calculate overall confidence
    fn calculate_confidence(&self, features: &FeatureSummary, agreement: f64, factor_conf: f64) -> f64 {
        // Base confidence from model agreement
        let mut confidence = agreement * 0.4 + factor_conf * 0.3;

        // Boost confidence when signals align
        let signals_aligned = (features.rsi > 50.0 && features.macd_signal > 0.0 && features.momentum_1h > 0.0)
            || (features.rsi < 50.0 && features.macd_signal < 0.0 && features.momentum_1h < 0.0);
        
        if signals_aligned {
            confidence += 0.15;
        }

        // Reduce confidence in high volatility
        if features.volatility > 0.03 {
            confidence *= 0.8;
        }

        // Reduce confidence when ADX is low (weak trend)
        if features.adx < 20.0 {
            confidence *= 0.9;
        }

        confidence.clamp(0.1, 0.95)
    }

    /// Determine recommended side and edge
    fn determine_side(&self, up_prob: f64, market_price: f64, question: &str) -> (String, f64) {
        let q = question.to_lowercase();
        
        // Parse market direction from question
        let is_up_market = q.contains("go up") || q.contains("上涨") || 
                          (q.contains("up or down") && !q.contains("go down"));
        
        if is_up_market {
            // Market asks about "up"
            if up_prob > 0.5 {
                let edge = up_prob - market_price;
                ("Yes".to_string(), edge)
            } else {
                let edge = (1.0 - up_prob) - (1.0 - market_price);
                ("No".to_string(), edge)
            }
        } else {
            // Market asks about "down"
            let down_prob = 1.0 - up_prob;
            if down_prob > 0.5 {
                let edge = down_prob - market_price;
                ("Yes".to_string(), edge)
            } else {
                let edge = (1.0 - down_prob) - (1.0 - market_price);
                ("No".to_string(), edge)
            }
        }
    }

    /// Collect factor signals for debugging
    fn collect_factor_signals(&self, factors: &[Factor]) -> HashMap<String, f64> {
        factors.iter()
            .map(|f| (f.id.clone(), f.signal.to_f64().unwrap_or(0.0)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines() -> Vec<KlineData> {
        let base_price = 85000.0;
        (0..50).map(|i| {
            let price = base_price + (i as f64 * 100.0) + ((i as f64 * 0.5).sin() * 500.0);
            KlineData {
                timestamp: 1706500000 + i * 3600,
                open: price - 50.0,
                high: price + 100.0,
                low: price - 100.0,
                close: price,
                volume: 1000000.0 + (i as f64 * 10000.0),
            }
        }).collect()
    }

    #[test]
    fn test_ml_predictor_creation() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        assert!(predictor.config.use_technical);
    }

    #[test]
    fn test_feature_extraction() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        let data = MarketDataInput {
            symbol: "BTCUSDT".to_string(),
            price: 85000.0,
            klines: create_test_klines(),
            orderbook_imbalance: Some(0.1),
            volume_24h: 50000000.0,
            sentiment_score: Some(0.3),
            question: "Will Bitcoin go up?".to_string(),
        };

        let features = predictor.extract_features(&data);
        assert!(features.rsi >= 0.0 && features.rsi <= 100.0);
        assert!(features.bollinger_position >= 0.0 && features.bollinger_position <= 1.0);
        assert!(features.adx >= 0.0);
    }

    #[test]
    fn test_rsi_calculation() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        
        // Create uptrending prices
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + i as f64).collect();
        let rsi = predictor.calculate_rsi(&prices, 14);
        assert!(rsi > 50.0, "Uptrend should have RSI > 50");

        // Create downtrending prices
        let prices: Vec<f64> = (0..20).map(|i| 100.0 - i as f64).collect();
        let rsi = predictor.calculate_rsi(&prices, 14);
        assert!(rsi < 50.0, "Downtrend should have RSI < 50");
    }

    #[test]
    fn test_full_prediction() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        let data = MarketDataInput {
            symbol: "BTCUSDT".to_string(),
            price: 85000.0,
            klines: create_test_klines(),
            orderbook_imbalance: Some(0.1),
            volume_24h: 50000000.0,
            sentiment_score: Some(0.3),
            question: "Will Bitcoin go up in the next hour?".to_string(),
        };

        let result = predictor.predict(&data, 0.55);
        
        assert!(result.up_probability >= 0.0 && result.up_probability <= 1.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(!result.factor_signals.is_empty());
        assert!(result.recommended_side == "Yes" || result.recommended_side == "No");
    }

    #[test]
    fn test_momentum_calculation() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        
        let prices = vec![100.0, 102.0]; // 2% up
        let momentum = predictor.calculate_momentum(&prices);
        assert!((momentum - 2.0).abs() < 0.01);
        
        let prices = vec![100.0, 98.0]; // 2% down
        let momentum = predictor.calculate_momentum(&prices);
        assert!((momentum - (-2.0)).abs() < 0.01);
    }

    #[test]
    fn test_volatility_calculation() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        
        // Low volatility
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 * 0.1)).collect();
        let vol_low = predictor.calculate_volatility(&prices);
        
        // High volatility
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + ((i % 2) as f64 * 10.0)).collect();
        let vol_high = predictor.calculate_volatility(&prices);
        
        assert!(vol_high > vol_low);
    }

    #[test]
    fn test_bollinger_position() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        
        // Create prices with some variance around 100
        let prices: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64 - 10.0) * 0.5).collect();
        // Mean is around 100, std_dev is small
        
        // Price at mean
        let pos = predictor.calculate_bollinger_position(&prices, 100.0);
        assert!((pos - 0.5).abs() < 0.2, "Position at mean should be ~0.5, got {}", pos);
        
        // Price well above upper band (use same data but extreme price)
        let high_price = 110.0; // Well above the band
        let pos_high = predictor.calculate_bollinger_position(&prices, high_price);
        assert!(pos_high > 0.8 || pos_high == 1.0, "High price should have position > 0.8, got {}", pos_high);
        
        // Price below lower band
        let low_price = 90.0;
        let pos_low = predictor.calculate_bollinger_position(&prices, low_price);
        assert!(pos_low < 0.2 || pos_low == 0.0, "Low price should have position < 0.2, got {}", pos_low);
    }

    #[test]
    fn test_empty_klines() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        let data = MarketDataInput {
            symbol: "BTCUSDT".to_string(),
            price: 85000.0,
            klines: vec![], // Empty
            orderbook_imbalance: None,
            volume_24h: 0.0,
            sentiment_score: None,
            question: "Will Bitcoin go up?".to_string(),
        };

        let result = predictor.predict(&data, 0.5);
        
        // Should return neutral prediction
        assert!((result.up_probability - 0.5).abs() < 0.2);
    }

    #[test]
    fn test_determine_side_up_market() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        
        // High probability of up, market underpriced
        let (side, edge) = predictor.determine_side(0.7, 0.5, "Will Bitcoin go up?");
        assert_eq!(side, "Yes");
        assert!(edge > 0.0);
        
        // Low probability of up
        let (side, edge) = predictor.determine_side(0.3, 0.5, "Will Bitcoin go up?");
        assert_eq!(side, "No");
        assert!(edge > 0.0);
    }

    #[test]
    fn test_determine_side_down_market() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        
        // High probability of down (low prob of up)
        let (side, _edge) = predictor.determine_side(0.3, 0.5, "Will Bitcoin go down?");
        assert_eq!(side, "Yes");
        
        // Low probability of down (high prob of up)
        let (side, _edge) = predictor.determine_side(0.7, 0.5, "Will Bitcoin go down?");
        assert_eq!(side, "No");
    }

    #[test]
    fn test_ema_calculation() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        
        let prices: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let ema = predictor.calculate_ema(&prices, 5);
        
        // EMA should be weighted towards recent values
        assert!(ema > 10.0); // Higher than simple average of 1-20
    }

    #[test]
    fn test_factor_creation() {
        let predictor = MLPredictor::new(MLPredictorConfig::default());
        let features = FeatureSummary {
            rsi: 65.0,
            macd_signal: 0.5,
            bollinger_position: 0.7,
            adx: 30.0,
            momentum_1h: 0.5,
            volume_trend: 10.0,
            volatility: 0.02,
        };
        let data = MarketDataInput {
            symbol: "BTCUSDT".to_string(),
            price: 85000.0,
            klines: vec![],
            orderbook_imbalance: Some(0.2),
            volume_24h: 50000000.0,
            sentiment_score: Some(0.4),
            question: "Will Bitcoin go up?".to_string(),
        };

        let factors = predictor.create_factors(&data, &features);
        
        // Should have at least technical, momentum, volume, volatility
        assert!(factors.len() >= 4);
        
        // With sentiment and orderbook
        assert!(factors.len() >= 6);
    }
}
