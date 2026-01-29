//! Trading strategy implementation

pub mod crypto_hf;

use crate::config::{RiskConfig, StrategyConfig};
use crate::model::Prediction;
use crate::types::{Market, Side, Signal};
use chrono::Utc;
use rust_decimal::Decimal;

pub use crypto_hf::{CryptoHfStrategy, CryptoPriceTracker};

/// Signal generator based on model predictions
pub struct SignalGenerator {
    config: StrategyConfig,
    risk_config: RiskConfig,
}

impl SignalGenerator {
    pub fn new(config: StrategyConfig, risk_config: RiskConfig) -> Self {
        Self { config, risk_config }
    }

    /// Generate trading signal from market and prediction
    pub fn generate(&self, market: &Market, prediction: &Prediction) -> Option<Signal> {
        let market_prob = market.yes_price()?;
        let model_prob = prediction.probability;
        let edge = model_prob - market_prob;

        // Check if edge is significant
        if edge.abs() < self.config.min_edge {
            return None;
        }

        // Check confidence threshold
        if prediction.confidence < self.config.min_confidence {
            return None;
        }

        // Determine side and token
        // edge > 0: Model thinks Yes underpriced -> Buy Yes
        // edge < 0: Model thinks Yes overpriced -> Sell Yes (not Buy No!)
        //           Selling Yes is usually more liquid than buying No
        let yes_token = market
            .outcomes
            .iter()
            .find(|o| o.outcome.to_lowercase() == "yes")
            .map(|o| o.token_id.clone())?;

        let (side, token_id, effective_prob) = if edge > Decimal::ZERO {
            (Side::Buy, yes_token, model_prob)
        } else {
            // Sell Yes when overpriced
            // For Kelly calculation, we're betting on "not Yes" at price (1 - market_prob)
            (Side::Sell, yes_token, Decimal::ONE - model_prob)
        };

        // Calculate position size using Kelly criterion
        let market_price = if edge > Decimal::ZERO {
            market_prob
        } else {
            Decimal::ONE - market_prob // Selling Yes = buying at (1 - price)
        };
        let suggested_size = self.calculate_kelly_size(effective_prob, market_price, prediction.confidence);

        Some(Signal {
            market_id: market.id.clone(),
            token_id,
            side,
            model_probability: model_prob,
            market_probability: market_prob,
            edge,
            confidence: prediction.confidence,
            suggested_size,
            timestamp: Utc::now(),
        })
    }

    /// Calculate position size using fractional Kelly criterion
    ///
    /// Kelly formula for binary bets: f* = (p * b - q) / b
    /// Where:
    ///   p = probability of winning (model's estimate)
    ///   q = probability of losing (1 - p)
    ///   b = odds ratio = (1 - market_price) / market_price
    ///
    /// Simplified: f* = p - q/b = p - (1-p) * market_price / (1 - market_price)
    ///           = (p * (1 - market_price) - (1 - p) * market_price) / (1 - market_price)
    ///           = (p - market_price) / (1 - market_price)
    ///           = edge / (1 - market_price)
    fn calculate_kelly_size(
        &self,
        model_prob: Decimal,
        market_price: Decimal,
        confidence: Decimal,
    ) -> Decimal {
        // Edge = model_prob - market_price
        let edge = model_prob - market_price;
        
        // Potential profit if we win = 1 - market_price (we pay market_price, get 1)
        let potential_profit = Decimal::ONE - market_price;
        
        // Avoid division by zero
        if potential_profit <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        // Full Kelly: f* = edge / potential_profit
        let full_kelly = edge / potential_profit;
        
        // Never bet negative (shouldn't happen if edge > 0, but safety first)
        if full_kelly <= Decimal::ZERO {
            return Decimal::ZERO;
        }

        // Fractional Kelly for safety (typically 0.25 - 0.5)
        let fractional_kelly = full_kelly * self.config.kelly_fraction;

        // Adjust by model confidence
        let adjusted = fractional_kelly * confidence;

        // Cap at max position size
        adjusted.min(self.risk_config.max_position_pct)
    }
}
