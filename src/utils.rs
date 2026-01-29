//! Shared utility functions

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Approximate square root using Newton's method
/// 
/// # Arguments
/// * `x` - The decimal value to calculate square root of
/// 
/// # Returns
/// The approximate square root of x, or zero if x <= 0
pub fn sqrt_decimal(x: Decimal) -> Decimal {
    if x <= Decimal::ZERO {
        return Decimal::ZERO;
    }

    let mut guess = x / dec!(2);
    let tolerance = dec!(0.0001);
    
    for _ in 0..20 {
        let new_guess = (guess + x / guess) / dec!(2);
        if (new_guess - guess).abs() < tolerance {
            return new_guess;
        }
        guess = new_guess;
    }
    
    guess
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_perfect_squares() {
        let result = sqrt_decimal(dec!(4));
        assert!((result - dec!(2)).abs() < dec!(0.001));
        
        let result = sqrt_decimal(dec!(9));
        assert!((result - dec!(3)).abs() < dec!(0.001));
        
        let result = sqrt_decimal(dec!(16));
        assert!((result - dec!(4)).abs() < dec!(0.001));
    }

    #[test]
    fn test_sqrt_zero() {
        let result = sqrt_decimal(dec!(0));
        assert_eq!(result, Decimal::ZERO);
    }

    #[test]
    fn test_sqrt_negative() {
        let result = sqrt_decimal(dec!(-1));
        assert_eq!(result, Decimal::ZERO);
    }

    #[test]
    fn test_sqrt_small_number() {
        let result = sqrt_decimal(dec!(0.25));
        assert!((result - dec!(0.5)).abs() < dec!(0.001));
    }

    #[test]
    fn test_sqrt_large_number() {
        let result = sqrt_decimal(dec!(10000));
        assert!((result - dec!(100)).abs() < dec!(0.1));
    }
}
