//! Tests for telegram module

#[cfg(test)]
mod tests {
    use super::super::{BotCommand, BotState};
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    #[test]
    fn test_bot_state_default() {
        let state = BotState::default();
        assert!(!state.paused);
        assert_eq!(state.daily_pnl, Decimal::ZERO);
        assert!(!state.daily_loss_limit_hit);
    }

    #[test]
    fn test_bot_state_paused() {
        let state = BotState {
            paused: true,
            daily_pnl: dec!(100),
            daily_loss_limit_hit: false,
        };
        assert!(state.paused);
    }

    #[test]
    fn test_bot_state_loss_limit() {
        let state = BotState {
            paused: false,
            daily_pnl: dec!(-500),
            daily_loss_limit_hit: true,
        };
        assert!(state.daily_loss_limit_hit);
        assert!(state.daily_pnl < Decimal::ZERO);
    }

    #[test]
    fn test_bot_state_clone() {
        let state = BotState {
            paused: true,
            daily_pnl: dec!(250),
            daily_loss_limit_hit: false,
        };
        let cloned = state.clone();
        assert_eq!(state.paused, cloned.paused);
        assert_eq!(state.daily_pnl, cloned.daily_pnl);
    }

    #[test]
    fn test_bot_command_pause() {
        let cmd = BotCommand::Pause;
        match cmd {
            BotCommand::Pause => assert!(true),
            _ => panic!("Expected Pause"),
        }
    }

    #[test]
    fn test_bot_command_resume() {
        let cmd = BotCommand::Resume;
        match cmd {
            BotCommand::Resume => assert!(true),
            _ => panic!("Expected Resume"),
        }
    }

    #[test]
    fn test_bot_command_status() {
        let cmd = BotCommand::Status;
        match cmd {
            BotCommand::Status => assert!(true),
            _ => panic!("Expected Status"),
        }
    }

    #[test]
    fn test_bot_command_markets() {
        let cmd = BotCommand::Markets { limit: 10 };
        match cmd {
            BotCommand::Markets { limit } => assert_eq!(limit, 10),
            _ => panic!("Expected Markets"),
        }
    }

    #[test]
    fn test_bot_command_buy() {
        let cmd = BotCommand::Buy {
            market_id: "market1".to_string(),
            amount: dec!(100),
        };
        match cmd {
            BotCommand::Buy { market_id, amount } => {
                assert_eq!(market_id, "market1");
                assert_eq!(amount, dec!(100));
            }
            _ => panic!("Expected Buy"),
        }
    }

    #[test]
    fn test_bot_command_sell() {
        let cmd = BotCommand::Sell {
            market_id: "market2".to_string(),
            amount: dec!(50),
        };
        match cmd {
            BotCommand::Sell { market_id, amount } => {
                assert_eq!(market_id, "market2");
                assert_eq!(amount, dec!(50));
            }
            _ => panic!("Expected Sell"),
        }
    }

    #[test]
    fn test_bot_command_pnl() {
        let cmd = BotCommand::Pnl;
        match cmd {
            BotCommand::Pnl => assert!(true),
            _ => panic!("Expected Pnl"),
        }
    }

    #[test]
    fn test_bot_command_positions() {
        let cmd = BotCommand::Positions;
        match cmd {
            BotCommand::Positions => assert!(true),
            _ => panic!("Expected Positions"),
        }
    }

    #[test]
    fn test_bot_command_set_risk() {
        let cmd = BotCommand::SetRisk {
            param: "max_position_pct".to_string(),
            value: dec!(0.10),
        };
        match cmd {
            BotCommand::SetRisk { param, value } => {
                assert_eq!(param, "max_position_pct");
                assert_eq!(value, dec!(0.10));
            }
            _ => panic!("Expected SetRisk"),
        }
    }

    #[test]
    fn test_bot_command_help() {
        let cmd = BotCommand::Help;
        match cmd {
            BotCommand::Help => assert!(true),
            _ => panic!("Expected Help"),
        }
    }

    #[test]
    fn test_bot_command_clone() {
        let cmd = BotCommand::Markets { limit: 5 };
        let cloned = cmd.clone();
        match cloned {
            BotCommand::Markets { limit } => assert_eq!(limit, 5),
            _ => panic!("Clone failed"),
        }
    }

    #[test]
    fn test_bot_command_debug() {
        let cmd = BotCommand::Status;
        let debug = format!("{:?}", cmd);
        assert!(debug.contains("Status"));
    }

    #[test]
    fn test_positive_pnl_state() {
        let state = BotState {
            paused: false,
            daily_pnl: dec!(1500),
            daily_loss_limit_hit: false,
        };
        assert!(state.daily_pnl > Decimal::ZERO);
    }

    #[test]
    fn test_negative_pnl_state() {
        let state = BotState {
            paused: false,
            daily_pnl: dec!(-300),
            daily_loss_limit_hit: false,
        };
        assert!(state.daily_pnl < Decimal::ZERO);
    }
}
