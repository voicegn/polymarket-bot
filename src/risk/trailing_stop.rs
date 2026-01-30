//! Trailing Stop Loss Module
//!
//! Provides dynamic stop-loss management that:
//! - Follows price when position is profitable
//! - Locks in profits by raising stop level
//! - Supports multiple trailing modes (fixed %, ATR-based, stepped)
//! - Handles breakeven stops after initial profit
//!
//! Industry-standard feature for professional trading bots.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Trailing stop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingStopConfig {
    /// Enable trailing stops
    pub enabled: bool,
    /// Default trailing mode
    pub default_mode: TrailingMode,
    /// Default trail percentage (for Percentage mode)
    pub default_trail_pct: Decimal,
    /// ATR multiplier (for ATR mode)
    pub atr_multiplier: Decimal,
    /// Move to breakeven after this profit %
    pub breakeven_trigger_pct: Decimal,
    /// Lock in this % of profit when stepping
    pub profit_lock_pct: Decimal,
    /// Step increment for stepped trailing
    pub step_increment_pct: Decimal,
    /// Minimum profit before trailing activates
    pub min_profit_to_activate_pct: Decimal,
    /// Maximum position age before forced exit (hours)
    pub max_position_age_hours: Option<u64>,
}

impl Default for TrailingStopConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_mode: TrailingMode::Stepped,
            default_trail_pct: dec!(0.05),        // 5% trail
            atr_multiplier: dec!(2.0),            // 2x ATR
            breakeven_trigger_pct: dec!(0.03),    // Move to breakeven at 3% profit
            profit_lock_pct: dec!(0.50),          // Lock 50% of max profit
            step_increment_pct: dec!(0.02),       // Step every 2%
            min_profit_to_activate_pct: dec!(0.01), // Activate at 1% profit
            max_position_age_hours: Some(72),     // 72 hour max hold
        }
    }
}

/// Trailing stop mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrailingMode {
    /// Fixed percentage below high water mark
    Percentage,
    /// ATR-based dynamic trailing
    ATR,
    /// Step up stop at discrete profit levels
    Stepped,
    /// Hybrid: starts fixed, tightens as profit grows
    Adaptive,
}

/// State of a trailing stop for one position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingStopState {
    /// Position identifier (token_id)
    pub position_id: String,
    /// Entry price
    pub entry_price: Decimal,
    /// Position side (true = long, false = short)
    pub is_long: bool,
    /// Current trailing mode
    pub mode: TrailingMode,
    /// High water mark (best price seen)
    pub high_water_mark: Decimal,
    /// Low water mark (for shorts)
    pub low_water_mark: Decimal,
    /// Current stop price
    pub stop_price: Decimal,
    /// Maximum unrealized profit seen
    pub max_profit_pct: Decimal,
    /// Whether breakeven stop is active
    pub breakeven_active: bool,
    /// Position opened at
    pub opened_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// ATR value at entry (if using ATR mode)
    pub entry_atr: Option<Decimal>,
    /// Number of step-ups performed
    pub step_count: u32,
}

impl TrailingStopState {
    /// Create new trailing stop state for a position
    pub fn new(
        position_id: String,
        entry_price: Decimal,
        is_long: bool,
        mode: TrailingMode,
        config: &TrailingStopConfig,
    ) -> Self {
        let initial_stop = if is_long {
            entry_price * (Decimal::ONE - config.default_trail_pct)
        } else {
            entry_price * (Decimal::ONE + config.default_trail_pct)
        };

        let now = Utc::now();
        Self {
            position_id,
            entry_price,
            is_long,
            mode,
            high_water_mark: entry_price,
            low_water_mark: entry_price,
            stop_price: initial_stop,
            max_profit_pct: Decimal::ZERO,
            breakeven_active: false,
            opened_at: now,
            updated_at: now,
            entry_atr: None,
            step_count: 0,
        }
    }

    /// Calculate current unrealized profit percentage
    pub fn current_profit_pct(&self, current_price: Decimal) -> Decimal {
        if self.is_long {
            (current_price - self.entry_price) / self.entry_price
        } else {
            (self.entry_price - current_price) / self.entry_price
        }
    }

    /// Check if stop has been hit
    pub fn is_stopped(&self, current_price: Decimal) -> bool {
        if self.is_long {
            current_price <= self.stop_price
        } else {
            current_price >= self.stop_price
        }
    }
}

/// Trailing stop action to take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrailingStopAction {
    /// No action needed
    None,
    /// Stop triggered - exit position
    Exit {
        position_id: String,
        reason: ExitReason,
        stop_price: Decimal,
        current_price: Decimal,
    },
    /// Stop level updated
    Updated {
        position_id: String,
        old_stop: Decimal,
        new_stop: Decimal,
        reason: String,
    },
}

/// Reason for exit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExitReason {
    /// Regular trailing stop hit
    TrailingStopHit,
    /// Breakeven stop hit
    BreakevenStopHit,
    /// Position age exceeded
    MaxAgeExceeded,
    /// Manual stop
    Manual,
}

/// Trailing stop manager
pub struct TrailingStopManager {
    config: TrailingStopConfig,
    stops: HashMap<String, TrailingStopState>,
}

impl TrailingStopManager {
    /// Create new trailing stop manager
    pub fn new(config: TrailingStopConfig) -> Self {
        Self {
            config,
            stops: HashMap::new(),
        }
    }

    /// Register a new position for trailing stop management
    pub fn register_position(
        &mut self,
        position_id: String,
        entry_price: Decimal,
        is_long: bool,
        atr: Option<Decimal>,
    ) -> &TrailingStopState {
        let mut state = TrailingStopState::new(
            position_id.clone(),
            entry_price,
            is_long,
            self.config.default_mode,
            &self.config,
        );
        
        if let Some(atr_value) = atr {
            state.entry_atr = Some(atr_value);
        }
        
        self.stops.insert(position_id.clone(), state);
        self.stops.get(&position_id).unwrap()
    }

    /// Remove position from tracking
    pub fn unregister_position(&mut self, position_id: &str) -> Option<TrailingStopState> {
        self.stops.remove(position_id)
    }

    /// Update trailing stop with new price data
    pub fn update(
        &mut self,
        position_id: &str,
        current_price: Decimal,
        current_atr: Option<Decimal>,
    ) -> TrailingStopAction {
        if !self.config.enabled {
            return TrailingStopAction::None;
        }

        // First, check basic conditions without mutable borrow
        let state_info = match self.stops.get(position_id) {
            Some(s) => {
                // Extract all needed info first
                (
                    s.stop_price,
                    s.is_long,
                    s.breakeven_active,
                    s.mode,
                    s.opened_at,
                    s.is_stopped(current_price),
                    s.entry_price,
                    s.high_water_mark,
                    s.low_water_mark,
                    s.max_profit_pct,
                    s.current_profit_pct(current_price),
                    s.entry_atr,
                    s.step_count,
                )
            }
            None => return TrailingStopAction::None,
        };

        let (stop_price, is_long, breakeven_active, mode, opened_at, is_stopped,
             entry_price, high_water_mark, low_water_mark, max_profit_pct,
             current_profit, entry_atr, _step_count) = state_info;

        // Check for max age
        if let Some(max_hours) = self.config.max_position_age_hours {
            let age = Utc::now() - opened_at;
            if age.num_hours() as u64 >= max_hours {
                return TrailingStopAction::Exit {
                    position_id: position_id.to_string(),
                    reason: ExitReason::MaxAgeExceeded,
                    stop_price,
                    current_price,
                };
            }
        }

        // Check if stop hit
        if is_stopped {
            let reason = if breakeven_active {
                ExitReason::BreakevenStopHit
            } else {
                ExitReason::TrailingStopHit
            };
            
            return TrailingStopAction::Exit {
                position_id: position_id.to_string(),
                reason,
                stop_price,
                current_price,
            };
        }

        // Update high/low water marks
        let new_high_water_mark = if is_long && current_price > high_water_mark {
            current_price
        } else {
            high_water_mark
        };
        
        let new_low_water_mark = if !is_long && current_price < low_water_mark {
            current_price
        } else {
            low_water_mark
        };

        // Calculate new max profit
        let new_max_profit = if current_profit > max_profit_pct {
            current_profit
        } else {
            max_profit_pct
        };

        // Check if we should activate trailing (need minimum profit)
        if new_max_profit < self.config.min_profit_to_activate_pct {
            // Still update water marks
            if let Some(state) = self.stops.get_mut(position_id) {
                state.high_water_mark = new_high_water_mark;
                state.low_water_mark = new_low_water_mark;
                state.max_profit_pct = new_max_profit;
            }
            return TrailingStopAction::None;
        }

        // Calculate new stop based on mode (using extracted values)
        let old_stop = stop_price;
        let new_stop = match mode {
            TrailingMode::Percentage => {
                if is_long {
                    new_high_water_mark * (Decimal::ONE - self.config.default_trail_pct)
                } else {
                    new_low_water_mark * (Decimal::ONE + self.config.default_trail_pct)
                }
            }
            TrailingMode::ATR => {
                let atr = current_atr
                    .or(entry_atr)
                    .unwrap_or(entry_price * dec!(0.02));
                let trail_amount = atr * self.config.atr_multiplier;
                if is_long {
                    new_high_water_mark - trail_amount
                } else {
                    new_low_water_mark + trail_amount
                }
            }
            TrailingMode::Stepped => {
                let profit = current_profit;
                let steps = (profit / self.config.step_increment_pct).floor();
                let locked_profit = steps * self.config.step_increment_pct * self.config.profit_lock_pct;
                if is_long {
                    entry_price * (Decimal::ONE + locked_profit)
                } else {
                    entry_price * (Decimal::ONE - locked_profit)
                }
            }
            TrailingMode::Adaptive => {
                let base_trail = self.config.default_trail_pct;
                let profit_factor = (Decimal::ONE - new_max_profit * dec!(2.0)).max(dec!(0.4));
                let adaptive_trail = (base_trail * profit_factor).max(dec!(0.02));

                if let Some(atr) = current_atr.or(entry_atr) {
                    let atr_pct = atr / entry_price;
                    let combined_trail = (adaptive_trail + atr_pct * self.config.atr_multiplier) / dec!(2);
                    if is_long {
                        new_high_water_mark * (Decimal::ONE - combined_trail)
                    } else {
                        new_low_water_mark * (Decimal::ONE + combined_trail)
                    }
                } else {
                    if is_long {
                        new_high_water_mark * (Decimal::ONE - adaptive_trail)
                    } else {
                        new_low_water_mark * (Decimal::ONE + adaptive_trail)
                    }
                }
            }
        };

        // Apply breakeven logic
        let should_activate_breakeven = current_profit >= self.config.breakeven_trigger_pct;
        let new_breakeven_active = breakeven_active || should_activate_breakeven;
        
        let new_stop = if new_breakeven_active {
            if is_long {
                new_stop.max(entry_price)
            } else {
                new_stop.min(entry_price)
            }
        } else {
            new_stop
        };

        // Only update if new stop is more favorable (higher for long, lower for short)
        let should_update = if is_long {
            new_stop > stop_price
        } else {
            new_stop < stop_price
        };

        // Now do the actual mutable update
        if let Some(state) = self.stops.get_mut(position_id) {
            state.high_water_mark = new_high_water_mark;
            state.low_water_mark = new_low_water_mark;
            state.max_profit_pct = new_max_profit;
            state.breakeven_active = new_breakeven_active;
            
            // Update step count for stepped mode
            if mode == TrailingMode::Stepped {
                let profit = current_profit;
                let steps = (profit / self.config.step_increment_pct).floor();
                if let Ok(s) = steps.try_into() {
                    if s > state.step_count {
                        state.step_count = s;
                    }
                }
            }
            
            if should_update {
                state.stop_price = new_stop;
                state.updated_at = Utc::now();
            }
        }

        if should_update {
            let reason = if new_breakeven_active && old_stop < entry_price {
                "Moved to breakeven".to_string()
            } else {
                format!("Trail update ({:?})", mode)
            };

            return TrailingStopAction::Updated {
                position_id: position_id.to_string(),
                old_stop,
                new_stop,
                reason,
            };
        }

        TrailingStopAction::None
    }

    /// Get current state for a position
    pub fn get_state(&self, position_id: &str) -> Option<&TrailingStopState> {
        self.stops.get(position_id)
    }

    /// Get all active trailing stops
    pub fn get_all_stops(&self) -> &HashMap<String, TrailingStopState> {
        &self.stops
    }

    /// Update config
    pub fn update_config(&mut self, config: TrailingStopConfig) {
        self.config = config;
    }

    /// Get config reference
    pub fn config(&self) -> &TrailingStopConfig {
        &self.config
    }

    /// Force exit a position (manual stop)
    pub fn force_exit(&mut self, position_id: &str, current_price: Decimal) -> TrailingStopAction {
        if let Some(state) = self.stops.get(position_id) {
            TrailingStopAction::Exit {
                position_id: position_id.to_string(),
                reason: ExitReason::Manual,
                stop_price: state.stop_price,
                current_price,
            }
        } else {
            TrailingStopAction::None
        }
    }
}

/// Summary of trailing stop status for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrailingStopSummary {
    pub position_id: String,
    pub entry_price: Decimal,
    pub current_stop: Decimal,
    pub stop_distance_pct: Decimal,
    pub max_profit_pct: Decimal,
    pub breakeven_active: bool,
    pub mode: TrailingMode,
    pub age_hours: u64,
}

impl From<&TrailingStopState> for TrailingStopSummary {
    fn from(state: &TrailingStopState) -> Self {
        let stop_distance = if state.is_long {
            (state.high_water_mark - state.stop_price) / state.high_water_mark
        } else {
            (state.stop_price - state.low_water_mark) / state.low_water_mark
        };

        Self {
            position_id: state.position_id.clone(),
            entry_price: state.entry_price,
            current_stop: state.stop_price,
            stop_distance_pct: stop_distance * Decimal::ONE_HUNDRED,
            max_profit_pct: state.max_profit_pct * Decimal::ONE_HUNDRED,
            breakeven_active: state.breakeven_active,
            mode: state.mode,
            age_hours: (Utc::now() - state.opened_at).num_hours() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TrailingStopConfig {
        TrailingStopConfig::default()
    }

    #[test]
    fn test_new_trailing_stop_state() {
        let config = default_config();
        let state = TrailingStopState::new(
            "test-1".to_string(),
            dec!(100),
            true,
            TrailingMode::Percentage,
            &config,
        );

        assert_eq!(state.entry_price, dec!(100));
        assert!(state.is_long);
        assert_eq!(state.stop_price, dec!(95)); // 5% below
        assert!(!state.breakeven_active);
    }

    #[test]
    fn test_trailing_stop_short_position() {
        let config = default_config();
        let state = TrailingStopState::new(
            "test-short".to_string(),
            dec!(100),
            false, // short
            TrailingMode::Percentage,
            &config,
        );

        assert_eq!(state.stop_price, dec!(105)); // 5% above for shorts
    }

    #[test]
    fn test_profit_calculation_long() {
        let config = default_config();
        let state = TrailingStopState::new(
            "test-1".to_string(),
            dec!(100),
            true,
            TrailingMode::Percentage,
            &config,
        );

        // 10% profit
        let profit = state.current_profit_pct(dec!(110));
        assert_eq!(profit, dec!(0.10));

        // 5% loss
        let loss = state.current_profit_pct(dec!(95));
        assert_eq!(loss, dec!(-0.05));
    }

    #[test]
    fn test_profit_calculation_short() {
        let config = default_config();
        let state = TrailingStopState::new(
            "test-1".to_string(),
            dec!(100),
            false, // short
            TrailingMode::Percentage,
            &config,
        );

        // 10% profit (price went down)
        let profit = state.current_profit_pct(dec!(90));
        assert_eq!(profit, dec!(0.10));

        // 5% loss (price went up)
        let loss = state.current_profit_pct(dec!(105));
        assert_eq!(loss, dec!(-0.05));
    }

    #[test]
    fn test_stop_triggered_long() {
        let config = default_config();
        let state = TrailingStopState::new(
            "test-1".to_string(),
            dec!(100),
            true,
            TrailingMode::Percentage,
            &config,
        );

        assert!(!state.is_stopped(dec!(96))); // Above stop
        assert!(state.is_stopped(dec!(95)));  // At stop
        assert!(state.is_stopped(dec!(90)));  // Below stop
    }

    #[test]
    fn test_stop_triggered_short() {
        let config = default_config();
        let state = TrailingStopState::new(
            "test-1".to_string(),
            dec!(100),
            false,
            TrailingMode::Percentage,
            &config,
        );

        assert!(!state.is_stopped(dec!(104))); // Below stop
        assert!(state.is_stopped(dec!(105)));  // At stop
        assert!(state.is_stopped(dec!(110)));  // Above stop
    }

    #[test]
    fn test_trailing_update_raises_stop() {
        let config = default_config();
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            None,
        );

        // Initial stop at 95
        assert_eq!(manager.get_state("test-1").unwrap().stop_price, dec!(95));

        // Price rises to 110 (10% gain, triggers trailing)
        let action = manager.update("test-1", dec!(110), None);
        
        // Stop should move up
        if let TrailingStopAction::Updated { new_stop, .. } = action {
            assert!(new_stop > dec!(95));
        }
    }

    #[test]
    fn test_breakeven_activation() {
        let mut config = default_config();
        config.breakeven_trigger_pct = dec!(0.05); // 5%
        config.min_profit_to_activate_pct = dec!(0.01);
        
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            None,
        );

        // Price rises to 106 (6% profit, should trigger breakeven)
        manager.update("test-1", dec!(106), None);
        
        let state = manager.get_state("test-1").unwrap();
        assert!(state.breakeven_active);
        assert!(state.stop_price >= dec!(100)); // Stop at or above entry
    }

    #[test]
    fn test_stepped_mode() {
        let mut config = default_config();
        config.default_mode = TrailingMode::Stepped;
        config.step_increment_pct = dec!(0.02);  // 2% steps
        config.profit_lock_pct = dec!(0.50);     // Lock 50%
        config.min_profit_to_activate_pct = dec!(0.01);
        
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            None,
        );

        // Price rises to 105 (5% profit = 2 steps of 2%)
        manager.update("test-1", dec!(105), None);
        
        let state = manager.get_state("test-1").unwrap();
        // 2 steps * 2% * 50% = 2% locked, stop at 102
        assert!(state.stop_price >= dec!(101));
    }

    #[test]
    fn test_exit_on_stop_hit() {
        let config = default_config();
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            None,
        );

        // Price drops to stop level
        let action = manager.update("test-1", dec!(94), None);
        
        assert!(matches!(
            action,
            TrailingStopAction::Exit { reason: ExitReason::TrailingStopHit, .. }
        ));
    }

    #[test]
    fn test_atr_based_stop() {
        let mut config = default_config();
        config.default_mode = TrailingMode::ATR;
        config.atr_multiplier = dec!(2.0);
        config.min_profit_to_activate_pct = dec!(0.01);
        
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            Some(dec!(2.5)), // ATR of 2.5
        );

        // Price rises to 110
        manager.update("test-1", dec!(110), Some(dec!(2.5)));
        
        let state = manager.get_state("test-1").unwrap();
        // Stop should be 110 - (2.5 * 2) = 105
        assert_eq!(state.stop_price, dec!(105));
    }

    #[test]
    fn test_max_age_exit() {
        let mut config = default_config();
        config.max_position_age_hours = Some(0); // Immediate expiry for test
        
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            None,
        );

        // Any update should trigger age exit
        let action = manager.update("test-1", dec!(100), None);
        
        assert!(matches!(
            action,
            TrailingStopAction::Exit { reason: ExitReason::MaxAgeExceeded, .. }
        ));
    }

    #[test]
    fn test_stop_never_goes_lower() {
        let config = default_config();
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            None,
        );

        // Price rises to 110
        manager.update("test-1", dec!(110), None);
        let stop_after_rise = manager.get_state("test-1").unwrap().stop_price;

        // Price drops back to 105
        manager.update("test-1", dec!(105), None);
        let stop_after_drop = manager.get_state("test-1").unwrap().stop_price;

        // Stop should not have moved lower
        assert!(stop_after_drop >= stop_after_rise);
    }

    #[test]
    fn test_unregister_position() {
        let config = default_config();
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            None,
        );

        assert!(manager.get_state("test-1").is_some());
        
        let removed = manager.unregister_position("test-1");
        assert!(removed.is_some());
        assert!(manager.get_state("test-1").is_none());
    }

    #[test]
    fn test_trailing_stop_summary() {
        let config = default_config();
        let mut state = TrailingStopState::new(
            "test-1".to_string(),
            dec!(100),
            true,
            TrailingMode::Percentage,
            &config,
        );
        state.high_water_mark = dec!(110);
        state.max_profit_pct = dec!(0.10);

        let summary = TrailingStopSummary::from(&state);
        
        assert_eq!(summary.entry_price, dec!(100));
        assert_eq!(summary.max_profit_pct, dec!(10)); // 10%
    }

    #[test]
    fn test_force_exit() {
        let config = default_config();
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            None,
        );

        let action = manager.force_exit("test-1", dec!(105));
        
        assert!(matches!(
            action,
            TrailingStopAction::Exit { reason: ExitReason::Manual, .. }
        ));
    }

    #[test]
    fn test_adaptive_mode_tightens() {
        let mut config = default_config();
        config.default_mode = TrailingMode::Adaptive;
        config.min_profit_to_activate_pct = dec!(0.01);
        
        let mut manager = TrailingStopManager::new(config);

        manager.register_position(
            "test-1".to_string(),
            dec!(100),
            true,
            None,
        );

        // Price rises to 110 (10% profit)
        manager.update("test-1", dec!(110), None);
        let state = manager.get_state("test-1").unwrap();
        
        // Adaptive mode should tighten the trail as profit grows
        // So stop should be closer to current price than basic percentage
        let distance_pct = (dec!(110) - state.stop_price) / dec!(110);
        assert!(distance_pct < dec!(0.05)); // Less than 5% distance
    }

    #[test]
    fn test_config_update() {
        let config = default_config();
        let mut manager = TrailingStopManager::new(config);

        let mut new_config = default_config();
        new_config.default_trail_pct = dec!(0.10);
        
        manager.update_config(new_config);
        
        assert_eq!(manager.config().default_trail_pct, dec!(0.10));
    }
}
