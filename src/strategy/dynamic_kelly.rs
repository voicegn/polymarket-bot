//! Dynamic Kelly Position Sizing
//!
//! Advanced Kelly criterion implementation that adjusts position size based on:
//! 1. Recent performance (win/loss streaks)
//! 2. Daily risk budget remaining
//! 3. Market volatility
//! 4. Account drawdown level
//! 5. Confidence in the edge estimate

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;
use std::sync::RwLock;

/// Configuration for dynamic Kelly sizing
#[derive(Debug, Clone)]
pub struct DynamicKellyConfig {
    /// Base Kelly fraction (e.g., 0.25 = quarter Kelly)
    pub base_fraction: Decimal,
    /// Minimum Kelly fraction (floor)
    pub min_fraction: Decimal,
    /// Maximum Kelly fraction (ceiling)
    pub max_fraction: Decimal,
    /// Number of recent trades to consider for streak analysis
    pub lookback_trades: usize,
    /// Drawdown level to start reducing size (e.g., 0.05 = 5%)
    pub drawdown_reduction_start: Decimal,
    /// Drawdown level for minimum size (e.g., 0.10 = 10%)
    pub drawdown_reduction_full: Decimal,
    /// Win streak multiplier per consecutive win (e.g., 1.05 = +5% per win)
    pub win_streak_multiplier: Decimal,
    /// Loss streak reduction per consecutive loss (e.g., 0.85 = -15% per loss)  
    pub loss_streak_multiplier: Decimal,
    /// Maximum multiplier from streaks
    pub max_streak_multiplier: Decimal,
    /// Minimum multiplier from streaks
    pub min_streak_multiplier: Decimal,
}

impl Default for DynamicKellyConfig {
    fn default() -> Self {
        Self {
            base_fraction: dec!(0.25),           // Quarter Kelly default
            min_fraction: dec!(0.05),            // Never go below 5%
            max_fraction: dec!(0.50),            // Never exceed half Kelly
            lookback_trades: 10,                  // Look at last 10 trades
            drawdown_reduction_start: dec!(0.03), // Start reducing at 3% drawdown
            drawdown_reduction_full: dec!(0.08),  // Full reduction at 8% drawdown
            win_streak_multiplier: dec!(1.08),    // +8% per consecutive win
            loss_streak_multiplier: dec!(0.80),   // -20% per consecutive loss
            max_streak_multiplier: dec!(1.30),    // Max +30% from win streak
            min_streak_multiplier: dec!(0.50),    // Max -50% from loss streak
        }
    }
}

/// A single trade result for streak tracking
#[derive(Debug, Clone)]
pub struct TradeResult {
    pub pnl: Decimal,
    pub is_win: bool,
}

/// Market context for volatility adjustment
#[derive(Debug, Clone)]
pub struct MarketContext {
    /// Recent price volatility (std dev of returns)
    pub volatility: Decimal,
    /// Market liquidity score 0-1
    pub liquidity_score: Decimal,
    /// Time pressure (0 = plenty of time, 1 = urgent)
    pub time_pressure: Decimal,
}

impl Default for MarketContext {
    fn default() -> Self {
        Self {
            volatility: dec!(0.05),      // 5% default volatility
            liquidity_score: dec!(1.0),  // Assume full liquidity
            time_pressure: dec!(0.0),    // No time pressure
        }
    }
}

/// Dynamic Kelly position sizer
pub struct DynamicKelly {
    config: DynamicKellyConfig,
    /// Recent trade results for streak analysis
    recent_trades: RwLock<VecDeque<TradeResult>>,
    /// Current account drawdown from peak
    current_drawdown: RwLock<Decimal>,
    /// Peak account value
    peak_value: RwLock<Decimal>,
    /// Current account value
    current_value: RwLock<Decimal>,
}

impl DynamicKelly {
    pub fn new(config: DynamicKellyConfig, initial_value: Decimal) -> Self {
        Self {
            config,
            recent_trades: RwLock::new(VecDeque::new()),
            current_drawdown: RwLock::new(dec!(0)),
            peak_value: RwLock::new(initial_value),
            current_value: RwLock::new(initial_value),
        }
    }

    /// Calculate optimal position size
    pub fn calculate_position_size(
        &self,
        model_prob: Decimal,
        market_price: Decimal,
        confidence: Decimal,
        remaining_budget_pct: Decimal,
        market_context: Option<&MarketContext>,
    ) -> KellyResult {
        // 1. Calculate base Kelly
        let edge = model_prob - market_price;
        let potential_profit = Decimal::ONE - market_price;
        
        if potential_profit <= Decimal::ZERO || edge <= Decimal::ZERO {
            return KellyResult {
                position_size: dec!(0),
                effective_fraction: dec!(0),
                adjustments: KellyAdjustments::default(),
                reasoning: "No positive edge or invalid price".to_string(),
            };
        }

        let full_kelly = edge / potential_profit;
        
        // 2. Apply base fraction
        let mut fraction = self.config.base_fraction;
        let mut adjustments = KellyAdjustments::default();

        // 3. Streak adjustment
        let streak_mult = self.calculate_streak_multiplier();
        adjustments.streak_multiplier = streak_mult;
        fraction = fraction * streak_mult;

        // 4. Drawdown adjustment
        let drawdown = *self.current_drawdown.read().unwrap();
        let drawdown_mult = self.calculate_drawdown_multiplier(drawdown);
        adjustments.drawdown_multiplier = drawdown_mult;
        fraction = fraction * drawdown_mult;

        // 5. Volatility adjustment
        if let Some(ctx) = market_context {
            let vol_mult = self.calculate_volatility_multiplier(ctx.volatility);
            adjustments.volatility_multiplier = vol_mult;
            fraction = fraction * vol_mult;

            // Liquidity adjustment
            let liq_mult = ctx.liquidity_score.max(dec!(0.3)); // At least 30%
            adjustments.liquidity_multiplier = liq_mult;
            fraction = fraction * liq_mult;

            // Time pressure adjustment (reduce size under pressure)
            let time_mult = Decimal::ONE - (ctx.time_pressure * dec!(0.3)); // Up to -30%
            adjustments.time_multiplier = time_mult;
            fraction = fraction * time_mult;
        }

        // 6. Confidence adjustment
        adjustments.confidence_multiplier = confidence;
        fraction = fraction * confidence;

        // 7. Risk budget adjustment
        // If we've used most of our budget, reduce further
        let budget_mult = if remaining_budget_pct < dec!(0.20) {
            dec!(0.5) // Only 50% if < 20% budget left
        } else if remaining_budget_pct < dec!(0.50) {
            dec!(0.75) // 75% if < 50% budget left
        } else {
            dec!(1.0)
        };
        adjustments.budget_multiplier = budget_mult;
        fraction = fraction * budget_mult;

        // 8. Clamp to bounds
        fraction = fraction
            .max(self.config.min_fraction)
            .min(self.config.max_fraction);
        adjustments.final_fraction = fraction;

        // 9. Calculate final position size
        let position_size = full_kelly * fraction;

        KellyResult {
            position_size,
            effective_fraction: fraction,
            adjustments,
            reasoning: self.build_reasoning(&adjustments),
        }
    }

    /// Calculate streak-based multiplier
    fn calculate_streak_multiplier(&self) -> Decimal {
        let trades = self.recent_trades.read().unwrap();
        
        if trades.is_empty() {
            return dec!(1.0);
        }

        // Count consecutive wins/losses from most recent
        let mut consecutive = 0i32;
        let mut last_was_win: Option<bool> = None;

        for trade in trades.iter().rev() {
            match last_was_win {
                None => {
                    last_was_win = Some(trade.is_win);
                    consecutive = 1;
                }
                Some(was_win) if was_win == trade.is_win => {
                    consecutive += 1;
                }
                _ => break,
            }
        }

        let multiplier = if last_was_win.unwrap_or(true) {
            // Win streak
            self.config.win_streak_multiplier.powu(consecutive as u64)
        } else {
            // Loss streak
            self.config.loss_streak_multiplier.powu(consecutive as u64)
        };

        multiplier
            .max(self.config.min_streak_multiplier)
            .min(self.config.max_streak_multiplier)
    }

    /// Calculate drawdown-based multiplier
    fn calculate_drawdown_multiplier(&self, drawdown: Decimal) -> Decimal {
        if drawdown <= self.config.drawdown_reduction_start {
            return dec!(1.0);
        }

        if drawdown >= self.config.drawdown_reduction_full {
            return dec!(0.5); // Minimum 50% at full drawdown
        }

        // Linear interpolation between start and full
        let range = self.config.drawdown_reduction_full - self.config.drawdown_reduction_start;
        let progress = (drawdown - self.config.drawdown_reduction_start) / range;
        
        dec!(1.0) - (progress * dec!(0.5)) // 100% -> 50%
    }

    /// Calculate volatility-based multiplier
    fn calculate_volatility_multiplier(&self, volatility: Decimal) -> Decimal {
        // Higher volatility = smaller position
        // Baseline: 5% volatility = 1.0 multiplier
        // 10% volatility = 0.7 multiplier
        // 2% volatility = 1.2 multiplier
        
        let baseline_vol = dec!(0.05);
        let vol_ratio = baseline_vol / volatility.max(dec!(0.01));
        
        vol_ratio
            .sqrt()
            .max(dec!(0.5))  // Don't reduce more than 50%
            .min(dec!(1.5))  // Don't increase more than 50%
    }

    /// Record a trade result
    pub fn record_trade(&self, pnl: Decimal) {
        let result = TradeResult {
            pnl,
            is_win: pnl > Decimal::ZERO,
        };

        let mut trades = self.recent_trades.write().unwrap();
        trades.push_back(result);
        
        // Keep only lookback window
        while trades.len() > self.config.lookback_trades {
            trades.pop_front();
        }
    }

    /// Update account value and drawdown
    pub fn update_account_value(&self, value: Decimal) {
        let mut current = self.current_value.write().unwrap();
        let mut peak = self.peak_value.write().unwrap();
        let mut drawdown = self.current_drawdown.write().unwrap();

        *current = value;
        
        if value > *peak {
            *peak = value;
            *drawdown = dec!(0);
        } else {
            *drawdown = (*peak - value) / *peak;
        }
    }

    /// Get current stats
    pub fn get_stats(&self) -> KellyStats {
        let trades = self.recent_trades.read().unwrap();
        let wins = trades.iter().filter(|t| t.is_win).count();
        let losses = trades.len() - wins;
        
        KellyStats {
            recent_wins: wins,
            recent_losses: losses,
            win_rate: if trades.is_empty() {
                dec!(0)
            } else {
                Decimal::from(wins as u32) / Decimal::from(trades.len() as u32)
            },
            current_drawdown: *self.current_drawdown.read().unwrap(),
            streak_multiplier: self.calculate_streak_multiplier(),
        }
    }

    fn build_reasoning(&self, adj: &KellyAdjustments) -> String {
        let mut parts = Vec::new();
        
        if adj.streak_multiplier != dec!(1) {
            parts.push(format!("streak:{:.0}%", adj.streak_multiplier * dec!(100)));
        }
        if adj.drawdown_multiplier != dec!(1) {
            parts.push(format!("dd:{:.0}%", adj.drawdown_multiplier * dec!(100)));
        }
        if adj.volatility_multiplier != dec!(1) {
            parts.push(format!("vol:{:.0}%", adj.volatility_multiplier * dec!(100)));
        }
        if adj.budget_multiplier != dec!(1) {
            parts.push(format!("budget:{:.0}%", adj.budget_multiplier * dec!(100)));
        }
        
        if parts.is_empty() {
            format!("Kelly @ {:.0}%", adj.final_fraction * dec!(100))
        } else {
            format!("Kelly @ {:.0}% [{}]", adj.final_fraction * dec!(100), parts.join(", "))
        }
    }
}

#[derive(Debug, Clone)]
pub struct KellyResult {
    /// Final position size as fraction of bankroll
    pub position_size: Decimal,
    /// Effective Kelly fraction after all adjustments
    pub effective_fraction: Decimal,
    /// Breakdown of adjustments
    pub adjustments: KellyAdjustments,
    /// Human-readable reasoning
    pub reasoning: String,
}

#[derive(Debug, Clone, Default)]
pub struct KellyAdjustments {
    pub streak_multiplier: Decimal,
    pub drawdown_multiplier: Decimal,
    pub volatility_multiplier: Decimal,
    pub liquidity_multiplier: Decimal,
    pub time_multiplier: Decimal,
    pub confidence_multiplier: Decimal,
    pub budget_multiplier: Decimal,
    pub final_fraction: Decimal,
}

#[derive(Debug, Clone)]
pub struct KellyStats {
    pub recent_wins: usize,
    pub recent_losses: usize,
    pub win_rate: Decimal,
    pub current_drawdown: Decimal,
    pub streak_multiplier: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kelly() -> DynamicKelly {
        DynamicKelly::new(DynamicKellyConfig::default(), dec!(10000))
    }

    #[test]
    fn test_basic_kelly_calculation() {
        let kelly = make_kelly();
        
        // 60% model prob, 50% market price, 80% confidence
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0), // Full budget
            None,
        );
        
        // Edge = 0.10, potential profit = 0.50
        // Full Kelly = 0.20 (20%)
        // Fractional = 0.20 * 0.25 = 0.05
        // With confidence = 0.05 * 0.80 = 0.04
        assert!(result.position_size > dec!(0));
        assert!(result.position_size < dec!(0.10));
        println!("Position size: {}, Reasoning: {}", result.position_size, result.reasoning);
    }

    #[test]
    fn test_no_edge_returns_zero() {
        let kelly = make_kelly();
        
        // No edge (model = market)
        let result = kelly.calculate_position_size(
            dec!(0.50),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        assert_eq!(result.position_size, dec!(0));
    }

    #[test]
    fn test_negative_edge_returns_zero() {
        let kelly = make_kelly();
        
        // Negative edge
        let result = kelly.calculate_position_size(
            dec!(0.40),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        assert_eq!(result.position_size, dec!(0));
    }

    #[test]
    fn test_win_streak_increases_size() {
        let kelly = make_kelly();
        
        // Record 3 wins
        kelly.record_trade(dec!(100));
        kelly.record_trade(dec!(50));
        kelly.record_trade(dec!(75));
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        // Streak multiplier should be > 1
        assert!(result.adjustments.streak_multiplier > dec!(1));
        println!("Streak mult after 3 wins: {}", result.adjustments.streak_multiplier);
    }

    #[test]
    fn test_loss_streak_decreases_size() {
        let kelly = make_kelly();
        
        // Record 3 losses
        kelly.record_trade(dec!(-100));
        kelly.record_trade(dec!(-50));
        kelly.record_trade(dec!(-75));
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        // Streak multiplier should be < 1
        assert!(result.adjustments.streak_multiplier < dec!(1));
        println!("Streak mult after 3 losses: {}", result.adjustments.streak_multiplier);
    }

    #[test]
    fn test_drawdown_reduces_size() {
        let kelly = make_kelly();
        
        // Simulate 5% drawdown
        kelly.update_account_value(dec!(10000));
        kelly.update_account_value(dec!(9500));
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        // Drawdown multiplier should be < 1 (we're past 3% threshold)
        assert!(result.adjustments.drawdown_multiplier < dec!(1));
        println!("Drawdown mult at 5%: {}", result.adjustments.drawdown_multiplier);
    }

    #[test]
    fn test_high_volatility_reduces_size() {
        let kelly = make_kelly();
        
        let ctx = MarketContext {
            volatility: dec!(0.15), // 15% volatility (high)
            liquidity_score: dec!(1.0),
            time_pressure: dec!(0),
        };
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            Some(&ctx),
        );
        
        // Volatility multiplier should be < 1
        assert!(result.adjustments.volatility_multiplier < dec!(1));
        println!("Vol mult at 15%: {}", result.adjustments.volatility_multiplier);
    }

    #[test]
    fn test_low_budget_reduces_size() {
        let kelly = make_kelly();
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(0.15), // Only 15% budget remaining
            None,
        );
        
        // Budget multiplier should be 0.5
        assert_eq!(result.adjustments.budget_multiplier, dec!(0.5));
    }

    #[test]
    fn test_streak_resets_on_alternation() {
        let kelly = make_kelly();
        
        // Win, loss, win (no streak)
        kelly.record_trade(dec!(100));
        kelly.record_trade(dec!(-50));
        kelly.record_trade(dec!(75));
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            None,
        );
        
        // Single win at end
        assert!(result.adjustments.streak_multiplier > dec!(1));
        assert!(result.adjustments.streak_multiplier < dec!(1.15)); // Less than 2-streak
    }

    #[test]
    fn test_combined_adjustments() {
        let kelly = make_kelly();
        
        // 3 losses + 5% drawdown + high volatility + low budget
        kelly.record_trade(dec!(-100));
        kelly.record_trade(dec!(-100));
        kelly.record_trade(dec!(-100));
        kelly.update_account_value(dec!(10000));
        kelly.update_account_value(dec!(9500));
        
        let ctx = MarketContext {
            volatility: dec!(0.10),
            liquidity_score: dec!(0.7),
            time_pressure: dec!(0.5),
        };
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(0.30),
            Some(&ctx),
        );
        
        // All adjustments should compound to reduce size significantly
        println!("Combined result: {:?}", result);
        assert!(result.effective_fraction < dec!(0.15)); // Significantly reduced
    }

    #[test]
    fn test_kelly_stats() {
        let kelly = make_kelly();
        
        kelly.record_trade(dec!(100));
        kelly.record_trade(dec!(-50));
        kelly.record_trade(dec!(75));
        kelly.record_trade(dec!(25));
        kelly.update_account_value(dec!(10150));
        
        let stats = kelly.get_stats();
        
        assert_eq!(stats.recent_wins, 3);
        assert_eq!(stats.recent_losses, 1);
        assert_eq!(stats.win_rate, dec!(0.75));
    }

    #[test]
    fn test_fraction_bounds() {
        let kelly = make_kelly();
        
        // Even with massive streak, should cap at max_fraction
        for _ in 0..10 {
            kelly.record_trade(dec!(1000));
        }
        
        let result = kelly.calculate_position_size(
            dec!(0.90), // Huge edge
            dec!(0.10),
            dec!(1.0),  // Full confidence
            dec!(1.0),
            None,
        );
        
        // Should not exceed max_fraction
        assert!(result.effective_fraction <= dec!(0.50));
    }

    #[test]
    fn test_low_liquidity_reduces_size() {
        let kelly = make_kelly();
        
        let ctx = MarketContext {
            volatility: dec!(0.05),
            liquidity_score: dec!(0.4), // Low liquidity
            time_pressure: dec!(0),
        };
        
        let result = kelly.calculate_position_size(
            dec!(0.60),
            dec!(0.50),
            dec!(0.80),
            dec!(1.0),
            Some(&ctx),
        );
        
        // Should reduce due to low liquidity
        assert!(result.adjustments.liquidity_multiplier < dec!(1));
        assert!(result.adjustments.liquidity_multiplier >= dec!(0.3));
    }
}
