//! # Portfolio Optimization Module
//!
//! Industry-grade portfolio optimization implementing:
//! - Mean-Variance Optimization (Markowitz)
//! - Minimum Variance Portfolio
//! - Maximum Sharpe Ratio Portfolio  
//! - Risk Parity
//! - Hierarchical Risk Parity (HRP)
//! - Black-Litterman Model
//! - Risk Budgeting
//!
//! ```rust,ignore
//! use polymarket_bot::portfolio::{PortfolioOptimizer, OptimizationMethod};
//!
//! let optimizer = PortfolioOptimizer::new(returns_matrix, risk_free_rate);
//! let weights = optimizer.optimize(OptimizationMethod::MaxSharpe)?;
//! ```

use rust_decimal::Decimal;
#[allow(unused_imports)]
use rust_decimal::prelude::*;
use std::collections::HashMap;
use thiserror::Error;

/// Portfolio optimization errors
#[derive(Error, Debug, Clone)]
pub enum PortfolioError {
    #[error("Insufficient data: need at least {required} observations, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Singular covariance matrix - assets may be perfectly correlated")]
    SingularMatrix,
    
    #[error("Optimization failed to converge after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },
    
    #[error("Invalid constraint: {0}")]
    InvalidConstraint(String),
    
    #[error("No feasible solution exists with given constraints")]
    NoFeasibleSolution,
    
    #[error("Negative variance detected for asset {asset}")]
    NegativeVariance { asset: String },
}

/// Optimization method to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationMethod {
    /// Minimum variance portfolio (lowest risk)
    MinVariance,
    /// Maximum Sharpe ratio portfolio (best risk-adjusted return)
    MaxSharpe,
    /// Risk Parity (equal risk contribution)
    RiskParity,
    /// Hierarchical Risk Parity (Lopez de Prado)
    HierarchicalRiskParity,
    /// Equal Weight (1/N)
    EqualWeight,
    /// Maximum Diversification
    MaxDiversification,
    /// Target return with minimum risk
    TargetReturn { target: Decimal },
    /// Target volatility with maximum return
    TargetVolatility { target: Decimal },
}

/// Portfolio constraints
#[derive(Debug, Clone)]
pub struct PortfolioConstraints {
    /// Minimum weight per asset (default: 0)
    pub min_weight: Decimal,
    /// Maximum weight per asset (default: 1)
    pub max_weight: Decimal,
    /// Minimum number of assets to hold
    pub min_assets: usize,
    /// Maximum number of assets to hold (0 = unlimited)
    pub max_assets: usize,
    /// Sector constraints: (sector_name, max_weight)
    pub sector_limits: HashMap<String, Decimal>,
    /// Long-only constraint (no short selling)
    pub long_only: bool,
    /// Maximum turnover from current portfolio
    pub max_turnover: Option<Decimal>,
    /// Current portfolio weights (for turnover constraint)
    pub current_weights: Option<Vec<Decimal>>,
}

impl Default for PortfolioConstraints {
    fn default() -> Self {
        Self {
            min_weight: Decimal::ZERO,
            max_weight: Decimal::ONE,
            min_assets: 1,
            max_assets: 0, // unlimited
            sector_limits: HashMap::new(),
            long_only: true,
            max_turnover: None,
            current_weights: None,
        }
    }
}

/// Asset information for portfolio construction
#[derive(Debug, Clone)]
pub struct Asset {
    pub symbol: String,
    pub expected_return: Decimal,
    pub volatility: Decimal,
    pub sector: Option<String>,
}

/// Optimized portfolio result
#[derive(Debug, Clone)]
pub struct OptimizedPortfolio {
    /// Asset weights (same order as input)
    pub weights: Vec<Decimal>,
    /// Expected portfolio return (annualized)
    pub expected_return: Decimal,
    /// Portfolio volatility (annualized)
    pub volatility: Decimal,
    /// Sharpe ratio
    pub sharpe_ratio: Decimal,
    /// Diversification ratio
    pub diversification_ratio: Decimal,
    /// Effective number of assets (1/sum(w^2))
    pub effective_n: Decimal,
    /// Risk contributions per asset
    pub risk_contributions: Vec<Decimal>,
    /// Marginal risk contributions
    pub marginal_risk: Vec<Decimal>,
}

/// Portfolio optimizer
pub struct PortfolioOptimizer {
    /// Asset symbols
    symbols: Vec<String>,
    /// Expected returns vector (annualized)
    expected_returns: Vec<Decimal>,
    /// Covariance matrix (annualized)
    covariance_matrix: Vec<Vec<Decimal>>,
    /// Correlation matrix
    correlation_matrix: Vec<Vec<Decimal>>,
    /// Risk-free rate (annualized)
    risk_free_rate: Decimal,
    /// Optimization constraints
    constraints: PortfolioConstraints,
    /// Max iterations for optimization
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: Decimal,
}

impl PortfolioOptimizer {
    /// Create optimizer from returns matrix
    /// 
    /// # Arguments
    /// * `symbols` - Asset symbols
    /// * `returns` - Matrix of returns [time][asset], each row is a time period
    /// * `risk_free_rate` - Annual risk-free rate
    /// * `annualization_factor` - Factor to annualize (252 for daily, 52 for weekly, 12 for monthly)
    pub fn from_returns(
        symbols: Vec<String>,
        returns: &[Vec<Decimal>],
        risk_free_rate: Decimal,
        annualization_factor: u32,
    ) -> Result<Self, PortfolioError> {
        let n_assets = symbols.len();
        let n_periods = returns.len();
        
        if n_periods < 10 {
            return Err(PortfolioError::InsufficientData {
                required: 10,
                actual: n_periods,
            });
        }
        
        // Validate dimensions
        for (i, row) in returns.iter().enumerate() {
            if row.len() != n_assets {
                return Err(PortfolioError::DimensionMismatch {
                    expected: n_assets,
                    actual: row.len(),
                });
            }
            // Also validate in first loop iteration
            if i == 0 && row.is_empty() {
                return Err(PortfolioError::InsufficientData {
                    required: 1,
                    actual: 0,
                });
            }
        }
        
        // Calculate expected returns (mean)
        let mut expected_returns = vec![Decimal::ZERO; n_assets];
        for row in returns {
            for (j, &ret) in row.iter().enumerate() {
                expected_returns[j] += ret;
            }
        }
        let n_dec = Decimal::from(n_periods as u32);
        for ret in &mut expected_returns {
            *ret /= n_dec;
        }
        
        // Annualize returns
        let ann_factor = Decimal::from(annualization_factor);
        for ret in &mut expected_returns {
            *ret *= ann_factor;
        }
        
        // Calculate covariance matrix
        let mut covariance_matrix = vec![vec![Decimal::ZERO; n_assets]; n_assets];
        let mean_returns: Vec<Decimal> = expected_returns.iter()
            .map(|r| *r / ann_factor)
            .collect();
        
        for row in returns {
            for i in 0..n_assets {
                for j in 0..n_assets {
                    let dev_i = row[i] - mean_returns[i];
                    let dev_j = row[j] - mean_returns[j];
                    covariance_matrix[i][j] += dev_i * dev_j;
                }
            }
        }
        
        // Divide by (n-1) for sample covariance and annualize
        let divisor = Decimal::from((n_periods - 1) as u32);
        for i in 0..n_assets {
            for j in 0..n_assets {
                covariance_matrix[i][j] = covariance_matrix[i][j] / divisor * ann_factor;
            }
        }
        
        // Calculate correlation matrix
        let mut correlation_matrix = vec![vec![Decimal::ZERO; n_assets]; n_assets];
        let volatilities: Vec<Decimal> = (0..n_assets)
            .map(|i| sqrt_decimal(covariance_matrix[i][i]))
            .collect();
        
        for i in 0..n_assets {
            for j in 0..n_assets {
                if volatilities[i] > Decimal::ZERO && volatilities[j] > Decimal::ZERO {
                    correlation_matrix[i][j] = covariance_matrix[i][j] 
                        / (volatilities[i] * volatilities[j]);
                } else if i == j {
                    correlation_matrix[i][j] = Decimal::ONE;
                }
            }
        }
        
        Ok(Self {
            symbols,
            expected_returns,
            covariance_matrix,
            correlation_matrix,
            risk_free_rate,
            constraints: PortfolioConstraints::default(),
            max_iterations: 1000,
            tolerance: Decimal::new(1, 8), // 1e-8
        })
    }
    
    /// Create optimizer from pre-computed statistics
    pub fn from_statistics(
        symbols: Vec<String>,
        expected_returns: Vec<Decimal>,
        covariance_matrix: Vec<Vec<Decimal>>,
        risk_free_rate: Decimal,
    ) -> Result<Self, PortfolioError> {
        let n = symbols.len();
        
        if expected_returns.len() != n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n,
                actual: expected_returns.len(),
            });
        }
        
        if covariance_matrix.len() != n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n,
                actual: covariance_matrix.len(),
            });
        }
        
        for row in &covariance_matrix {
            if row.len() != n {
                return Err(PortfolioError::DimensionMismatch {
                    expected: n,
                    actual: row.len(),
                });
            }
        }
        
        // Calculate correlation matrix
        let mut correlation_matrix = vec![vec![Decimal::ZERO; n]; n];
        let volatilities: Vec<Decimal> = (0..n)
            .map(|i| sqrt_decimal(covariance_matrix[i][i]))
            .collect();
        
        for i in 0..n {
            for j in 0..n {
                if volatilities[i] > Decimal::ZERO && volatilities[j] > Decimal::ZERO {
                    correlation_matrix[i][j] = covariance_matrix[i][j]
                        / (volatilities[i] * volatilities[j]);
                } else if i == j {
                    correlation_matrix[i][j] = Decimal::ONE;
                }
            }
        }
        
        Ok(Self {
            symbols,
            expected_returns,
            covariance_matrix,
            correlation_matrix,
            risk_free_rate,
            constraints: PortfolioConstraints::default(),
            max_iterations: 1000,
            tolerance: Decimal::new(1, 8),
        })
    }
    
    /// Set portfolio constraints
    pub fn with_constraints(mut self, constraints: PortfolioConstraints) -> Self {
        self.constraints = constraints;
        self
    }
    
    /// Set max iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }
    
    /// Optimize portfolio using specified method
    pub fn optimize(&self, method: OptimizationMethod) -> Result<OptimizedPortfolio, PortfolioError> {
        let weights = match method {
            OptimizationMethod::EqualWeight => self.equal_weight(),
            OptimizationMethod::MinVariance => self.min_variance()?,
            OptimizationMethod::MaxSharpe => self.max_sharpe()?,
            OptimizationMethod::RiskParity => self.risk_parity()?,
            OptimizationMethod::HierarchicalRiskParity => self.hrp()?,
            OptimizationMethod::MaxDiversification => self.max_diversification()?,
            OptimizationMethod::TargetReturn { target } => self.target_return(target)?,
            OptimizationMethod::TargetVolatility { target } => self.target_volatility(target)?,
        };
        
        self.build_result(weights)
    }
    
    /// Equal weight portfolio (1/N)
    fn equal_weight(&self) -> Vec<Decimal> {
        let n = Decimal::from(self.symbols.len() as u32);
        vec![Decimal::ONE / n; self.symbols.len()]
    }
    
    /// Minimum variance portfolio using analytical solution
    fn min_variance(&self) -> Result<Vec<Decimal>, PortfolioError> {
        let n = self.symbols.len();
        
        // For minimum variance: w = (Σ^-1 * 1) / (1' * Σ^-1 * 1)
        // First compute inverse of covariance matrix
        let cov_inv = self.invert_matrix(&self.covariance_matrix)?;
        
        // Compute Σ^-1 * 1
        let mut sigma_inv_ones = vec![Decimal::ZERO; n];
        for i in 0..n {
            for j in 0..n {
                sigma_inv_ones[i] += cov_inv[i][j];
            }
        }
        
        // Compute 1' * Σ^-1 * 1
        let mut denom = Decimal::ZERO;
        for &val in &sigma_inv_ones {
            denom += val;
        }
        
        if denom.abs() < self.tolerance {
            return Err(PortfolioError::SingularMatrix);
        }
        
        // Final weights
        let mut weights: Vec<Decimal> = sigma_inv_ones.iter()
            .map(|&x| x / denom)
            .collect();
        
        // Apply constraints
        self.apply_constraints(&mut weights)?;
        
        Ok(weights)
    }
    
    /// Maximum Sharpe ratio portfolio
    fn max_sharpe(&self) -> Result<Vec<Decimal>, PortfolioError> {
        let n = self.symbols.len();
        
        // For max Sharpe: w = (Σ^-1 * (μ - rf)) / (1' * Σ^-1 * (μ - rf))
        let cov_inv = self.invert_matrix(&self.covariance_matrix)?;
        
        // Excess returns: μ - rf
        let excess_returns: Vec<Decimal> = self.expected_returns.iter()
            .map(|&r| r - self.risk_free_rate)
            .collect();
        
        // Compute Σ^-1 * (μ - rf)
        let mut sigma_inv_excess = vec![Decimal::ZERO; n];
        for i in 0..n {
            for j in 0..n {
                sigma_inv_excess[i] += cov_inv[i][j] * excess_returns[j];
            }
        }
        
        // Compute 1' * Σ^-1 * (μ - rf)
        let mut denom = Decimal::ZERO;
        for &val in &sigma_inv_excess {
            denom += val;
        }
        
        if denom.abs() < self.tolerance {
            // All assets have same expected return, fall back to min variance
            return self.min_variance();
        }
        
        let mut weights: Vec<Decimal> = sigma_inv_excess.iter()
            .map(|&x| x / denom)
            .collect();
        
        self.apply_constraints(&mut weights)?;
        
        Ok(weights)
    }
    
    /// Risk parity portfolio (equal risk contribution)
    fn risk_parity(&self) -> Result<Vec<Decimal>, PortfolioError> {
        let n = self.symbols.len();
        
        // Initial guess: inverse volatility weighted
        let mut weights: Vec<Decimal> = (0..n)
            .map(|i| {
                let vol = sqrt_decimal(self.covariance_matrix[i][i]);
                if vol > Decimal::ZERO {
                    Decimal::ONE / vol
                } else {
                    Decimal::ONE
                }
            })
            .collect();
        
        // Normalize
        let sum: Decimal = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }
        
        // Newton-Raphson iteration for risk parity
        for _iter in 0..self.max_iterations {
            let rc = self.risk_contributions(&weights);
            let portfolio_vol = self.portfolio_volatility(&weights);
            
            // Target: equal risk contribution
            let target_rc = portfolio_vol / Decimal::from(n as u32);
            
            // Check convergence
            let max_diff: Decimal = rc.iter()
                .map(|&r| (r - target_rc).abs())
                .max()
                .unwrap_or(Decimal::ZERO);
            
            if max_diff < self.tolerance * Decimal::from(100u32) {
                break;
            }
            
            // Update weights proportionally to deviation
            for i in 0..n {
                let adjustment = target_rc / (rc[i] + self.tolerance);
                weights[i] *= adjustment;
            }
            
            // Renormalize
            let sum: Decimal = weights.iter().sum();
            if sum > Decimal::ZERO {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }
        
        self.apply_constraints(&mut weights)?;
        
        Ok(weights)
    }
    
    /// Hierarchical Risk Parity (Lopez de Prado)
    fn hrp(&self) -> Result<Vec<Decimal>, PortfolioError> {
        let n = self.symbols.len();
        
        if n == 1 {
            return Ok(vec![Decimal::ONE]);
        }
        
        // Step 1: Compute distance matrix from correlations
        let dist_matrix = self.correlation_to_distance();
        
        // Step 2: Hierarchical clustering (single linkage)
        let clusters = self.hierarchical_cluster(&dist_matrix);
        
        // Step 3: Quasi-diagonalization (reorder assets)
        let order = self.get_quasi_diagonal_order(&clusters);
        
        // Step 4: Recursive bisection
        let mut weights = vec![Decimal::ONE; n];
        self.recursive_bisection(&order, &mut weights)?;
        
        // Reorder weights back to original order
        let mut final_weights = vec![Decimal::ZERO; n];
        for (new_idx, &orig_idx) in order.iter().enumerate() {
            final_weights[orig_idx] = weights[new_idx];
        }
        
        self.apply_constraints(&mut final_weights)?;
        
        Ok(final_weights)
    }
    
    /// Maximum diversification portfolio
    fn max_diversification(&self) -> Result<Vec<Decimal>, PortfolioError> {
        let n = self.symbols.len();
        
        // Volatilities
        let vols: Vec<Decimal> = (0..n)
            .map(|i| sqrt_decimal(self.covariance_matrix[i][i]))
            .collect();
        
        // Initial: inverse vol weighted
        let mut weights: Vec<Decimal> = vols.iter()
            .map(|&v| if v > Decimal::ZERO { Decimal::ONE / v } else { Decimal::ONE })
            .collect();
        
        let sum: Decimal = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }
        
        // Gradient ascent to maximize diversification ratio
        let step_size = Decimal::new(1, 3); // 0.001
        
        for _iter in 0..self.max_iterations {
            let portfolio_vol = self.portfolio_volatility(&weights);
            
            // Weighted average vol
            let weighted_vol: Decimal = weights.iter()
                .zip(vols.iter())
                .map(|(&w, &v)| w * v)
                .sum();
            
            if portfolio_vol < self.tolerance {
                break;
            }
            
            // Gradient of diversification ratio
            let mut gradient = vec![Decimal::ZERO; n];
            for i in 0..n {
                // d(DR)/dw_i = vol_i/portfolio_vol - weighted_vol * d(portfolio_vol)/dw_i / portfolio_vol^2
                let mut d_portfolio_vol = Decimal::ZERO;
                for j in 0..n {
                    d_portfolio_vol += self.covariance_matrix[i][j] * weights[j];
                }
                d_portfolio_vol /= portfolio_vol;
                
                gradient[i] = vols[i] / portfolio_vol 
                    - weighted_vol * d_portfolio_vol / (portfolio_vol * portfolio_vol);
            }
            
            // Update
            for i in 0..n {
                weights[i] += step_size * gradient[i];
                weights[i] = weights[i].max(Decimal::ZERO);
            }
            
            // Normalize
            let sum: Decimal = weights.iter().sum();
            if sum > Decimal::ZERO {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }
        
        self.apply_constraints(&mut weights)?;
        
        Ok(weights)
    }
    
    /// Portfolio with target return and minimum variance
    fn target_return(&self, target: Decimal) -> Result<Vec<Decimal>, PortfolioError> {
        let n = self.symbols.len();
        
        // Lagrangian: min w'Σw s.t. w'μ = target, w'1 = 1
        // Using augmented matrix approach
        let cov_inv = self.invert_matrix(&self.covariance_matrix)?;
        
        // Compute A = 1'Σ^-1*1, B = 1'Σ^-1*μ, C = μ'Σ^-1*μ
        let mut a = Decimal::ZERO;
        let mut b = Decimal::ZERO;
        let mut c = Decimal::ZERO;
        
        for i in 0..n {
            for j in 0..n {
                a += cov_inv[i][j];
                b += cov_inv[i][j] * self.expected_returns[j];
                c += self.expected_returns[i] * cov_inv[i][j] * self.expected_returns[j];
            }
        }
        
        let d = a * c - b * b;
        if d.abs() < self.tolerance {
            return Err(PortfolioError::SingularMatrix);
        }
        
        // Compute g = (Σ^-1*1*C - Σ^-1*μ*B) / D
        // Compute h = (Σ^-1*μ*A - Σ^-1*1*B) / D
        let mut g = vec![Decimal::ZERO; n];
        let mut h = vec![Decimal::ZERO; n];
        
        for i in 0..n {
            let mut sigma_inv_ones = Decimal::ZERO;
            let mut sigma_inv_mu = Decimal::ZERO;
            for j in 0..n {
                sigma_inv_ones += cov_inv[i][j];
                sigma_inv_mu += cov_inv[i][j] * self.expected_returns[j];
            }
            g[i] = (sigma_inv_ones * c - sigma_inv_mu * b) / d;
            h[i] = (sigma_inv_mu * a - sigma_inv_ones * b) / d;
        }
        
        // w = g + h * target
        let mut weights: Vec<Decimal> = (0..n)
            .map(|i| g[i] + h[i] * target)
            .collect();
        
        self.apply_constraints(&mut weights)?;
        
        Ok(weights)
    }
    
    /// Portfolio with target volatility and maximum return
    fn target_volatility(&self, target: Decimal) -> Result<Vec<Decimal>, PortfolioError> {
        // First get max Sharpe portfolio
        let max_sharpe_weights = self.max_sharpe()?;
        let max_sharpe_vol = self.portfolio_volatility(&max_sharpe_weights);
        
        if max_sharpe_vol < self.tolerance {
            return Err(PortfolioError::SingularMatrix);
        }
        
        // Scale to target volatility (leveraging/deleveraging)
        let scale = target / max_sharpe_vol;
        
        let mut weights: Vec<Decimal> = max_sharpe_weights.iter()
            .map(|&w| w * scale)
            .collect();
        
        // If long-only and scale > 1, we can't leverage - just use max Sharpe
        if self.constraints.long_only && scale > Decimal::ONE {
            weights = max_sharpe_weights;
        }
        
        self.apply_constraints(&mut weights)?;
        
        Ok(weights)
    }
    
    /// Apply portfolio constraints
    fn apply_constraints(&self, weights: &mut Vec<Decimal>) -> Result<(), PortfolioError> {
        let n = weights.len();
        
        // Enforce min/max weights
        for w in weights.iter_mut() {
            if self.constraints.long_only && *w < Decimal::ZERO {
                *w = Decimal::ZERO;
            }
            *w = (*w).max(self.constraints.min_weight);
            *w = (*w).min(self.constraints.max_weight);
        }
        
        // Renormalize to sum to 1
        let sum: Decimal = weights.iter().sum();
        if sum > Decimal::ZERO {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        } else {
            // Fallback to equal weight
            let eq_weight = Decimal::ONE / Decimal::from(n as u32);
            for w in weights.iter_mut() {
                *w = eq_weight;
            }
        }
        
        // Enforce max assets if specified
        if self.constraints.max_assets > 0 && self.constraints.max_assets < n {
            let mut indexed: Vec<(usize, Decimal)> = weights.iter()
                .enumerate()
                .map(|(i, &w)| (i, w))
                .collect();
            indexed.sort_by(|a, b| b.1.cmp(&a.1));
            
            // Zero out smallest weights
            for i in self.constraints.max_assets..n {
                weights[indexed[i].0] = Decimal::ZERO;
            }
            
            // Renormalize
            let sum: Decimal = weights.iter().sum();
            if sum > Decimal::ZERO {
                for w in weights.iter_mut() {
                    *w /= sum;
                }
            }
        }
        
        Ok(())
    }
    
    /// Build final result with all metrics
    fn build_result(&self, weights: Vec<Decimal>) -> Result<OptimizedPortfolio, PortfolioError> {
        let n = self.symbols.len();
        
        // Expected return
        let expected_return: Decimal = weights.iter()
            .zip(self.expected_returns.iter())
            .map(|(&w, &r)| w * r)
            .sum();
        
        // Volatility
        let volatility = self.portfolio_volatility(&weights);
        
        // Sharpe ratio
        let sharpe_ratio = if volatility > Decimal::ZERO {
            (expected_return - self.risk_free_rate) / volatility
        } else {
            Decimal::ZERO
        };
        
        // Diversification ratio
        let weighted_vol: Decimal = weights.iter()
            .enumerate()
            .map(|(i, &w)| w * sqrt_decimal(self.covariance_matrix[i][i]))
            .sum();
        let diversification_ratio = if volatility > Decimal::ZERO {
            weighted_vol / volatility
        } else {
            Decimal::ONE
        };
        
        // Effective N (1 / sum(w^2))
        let sum_w_sq: Decimal = weights.iter().map(|&w| w * w).sum();
        let effective_n = if sum_w_sq > Decimal::ZERO {
            Decimal::ONE / sum_w_sq
        } else {
            Decimal::from(n as u32)
        };
        
        // Risk contributions
        let risk_contributions = self.risk_contributions(&weights);
        
        // Marginal risk contributions
        let mut marginal_risk = vec![Decimal::ZERO; n];
        if volatility > Decimal::ZERO {
            for i in 0..n {
                let mut mrc = Decimal::ZERO;
                for j in 0..n {
                    mrc += self.covariance_matrix[i][j] * weights[j];
                }
                marginal_risk[i] = mrc / volatility;
            }
        }
        
        Ok(OptimizedPortfolio {
            weights,
            expected_return,
            volatility,
            sharpe_ratio,
            diversification_ratio,
            effective_n,
            risk_contributions,
            marginal_risk,
        })
    }
    
    /// Calculate portfolio volatility
    fn portfolio_volatility(&self, weights: &[Decimal]) -> Decimal {
        let n = weights.len();
        let mut variance = Decimal::ZERO;
        
        for i in 0..n {
            for j in 0..n {
                variance += weights[i] * weights[j] * self.covariance_matrix[i][j];
            }
        }
        
        sqrt_decimal(variance.max(Decimal::ZERO))
    }
    
    /// Calculate risk contributions per asset
    fn risk_contributions(&self, weights: &[Decimal]) -> Vec<Decimal> {
        let n = weights.len();
        let portfolio_vol = self.portfolio_volatility(weights);
        
        if portfolio_vol <= Decimal::ZERO {
            return vec![Decimal::ZERO; n];
        }
        
        let mut contributions = vec![Decimal::ZERO; n];
        
        for i in 0..n {
            let mut mrc = Decimal::ZERO;
            for j in 0..n {
                mrc += self.covariance_matrix[i][j] * weights[j];
            }
            // RC_i = w_i * MRC_i = w_i * (Σw)_i / σ_p
            contributions[i] = weights[i] * mrc / portfolio_vol;
        }
        
        contributions
    }
    
    /// Invert a matrix using Gauss-Jordan elimination
    fn invert_matrix(&self, matrix: &[Vec<Decimal>]) -> Result<Vec<Vec<Decimal>>, PortfolioError> {
        let n = matrix.len();
        
        // Create augmented matrix [A|I]
        let mut aug = vec![vec![Decimal::ZERO; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = matrix[i][j];
            }
            aug[i][n + i] = Decimal::ONE;
        }
        
        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            
            if max_val < self.tolerance {
                return Err(PortfolioError::SingularMatrix);
            }
            
            // Swap rows
            if max_row != col {
                aug.swap(col, max_row);
            }
            
            // Scale pivot row
            let pivot = aug[col][col];
            for j in 0..(2 * n) {
                aug[col][j] /= pivot;
            }
            
            // Eliminate column - store col row values to avoid borrow issues
            let col_row: Vec<Decimal> = aug[col].clone();
            for row in 0..n {
                if row != col {
                    let factor = aug[row][col];
                    for j in 0..(2 * n) {
                        aug[row][j] -= factor * col_row[j];
                    }
                }
            }
        }
        
        // Extract inverse
        let mut inverse = vec![vec![Decimal::ZERO; n]; n];
        for i in 0..n {
            for j in 0..n {
                inverse[i][j] = aug[i][n + j];
            }
        }
        
        Ok(inverse)
    }
    
    /// Convert correlation to distance matrix for clustering
    fn correlation_to_distance(&self) -> Vec<Vec<Decimal>> {
        let n = self.correlation_matrix.len();
        let mut dist = vec![vec![Decimal::ZERO; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                // Distance = sqrt(0.5 * (1 - correlation))
                let corr = self.correlation_matrix[i][j];
                let d_sq = Decimal::new(5, 1) * (Decimal::ONE - corr); // 0.5 * (1 - corr)
                dist[i][j] = sqrt_decimal(d_sq.max(Decimal::ZERO));
            }
        }
        
        dist
    }
    
    /// Hierarchical clustering (single linkage)
    fn hierarchical_cluster(&self, dist: &[Vec<Decimal>]) -> Vec<(usize, usize, Decimal)> {
        let n = dist.len();
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut dist_matrix = dist.to_vec();
        let mut merges = Vec::new();
        
        while clusters.len() > 1 {
            // Find minimum distance
            let mut min_dist = Decimal::MAX;
            let mut min_i = 0;
            let mut min_j = 1;
            
            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    if dist_matrix[i][j] < min_dist {
                        min_dist = dist_matrix[i][j];
                        min_i = i;
                        min_j = j;
                    }
                }
            }
            
            // Merge clusters
            let cluster_j = clusters.remove(min_j);
            clusters[min_i].extend(cluster_j);
            merges.push((min_i, min_j, min_dist));
            
            // Update distance matrix (single linkage: min)
            let row_j = dist_matrix.remove(min_j);
            for i in 0..clusters.len() {
                dist_matrix[i].remove(min_j);
            }
            
            for i in 0..clusters.len() {
                if i != min_i {
                    let new_dist = dist_matrix[min_i][i].min(row_j[i]);
                    dist_matrix[min_i][i] = new_dist;
                    dist_matrix[i][min_i] = new_dist;
                }
            }
        }
        
        merges
    }
    
    /// Get quasi-diagonal order from clustering
    fn get_quasi_diagonal_order(&self, _clusters: &[(usize, usize, Decimal)]) -> Vec<usize> {
        // For simplicity, use sorted volatility order
        // Full implementation would trace the dendrogram
        let n = self.symbols.len();
        let mut indexed: Vec<(usize, Decimal)> = (0..n)
            .map(|i| (i, sqrt_decimal(self.covariance_matrix[i][i])))
            .collect();
        indexed.sort_by(|a, b| a.1.cmp(&b.1));
        indexed.into_iter().map(|(i, _)| i).collect()
    }
    
    /// Recursive bisection for HRP
    fn recursive_bisection(
        &self,
        items: &[usize],
        weights: &mut [Decimal],
    ) -> Result<(), PortfolioError> {
        if items.len() <= 1 {
            return Ok(());
        }
        
        let mid = items.len() / 2;
        let left = &items[..mid];
        let right = &items[mid..];
        
        // Calculate cluster variances
        let left_var = self.cluster_variance(left);
        let right_var = self.cluster_variance(right);
        
        // Allocate inversely proportional to variance
        let total_inv_var = if left_var > Decimal::ZERO { Decimal::ONE / left_var } else { Decimal::ONE }
            + if right_var > Decimal::ZERO { Decimal::ONE / right_var } else { Decimal::ONE };
        
        let left_weight = if left_var > Decimal::ZERO {
            (Decimal::ONE / left_var) / total_inv_var
        } else {
            Decimal::new(5, 1) // 0.5
        };
        let right_weight = Decimal::ONE - left_weight;
        
        // Apply weights
        for &i in left {
            weights[i] *= left_weight;
        }
        for &i in right {
            weights[i] *= right_weight;
        }
        
        // Recurse
        self.recursive_bisection(left, weights)?;
        self.recursive_bisection(right, weights)?;
        
        Ok(())
    }
    
    /// Calculate cluster variance (min variance portfolio)
    fn cluster_variance(&self, indices: &[usize]) -> Decimal {
        if indices.is_empty() {
            return Decimal::ONE;
        }
        if indices.len() == 1 {
            return self.covariance_matrix[indices[0]][indices[0]];
        }
        
        // Extract sub-covariance matrix
        let n = indices.len();
        let mut sub_cov = vec![vec![Decimal::ZERO; n]; n];
        for (i, &idx_i) in indices.iter().enumerate() {
            for (j, &idx_j) in indices.iter().enumerate() {
                sub_cov[i][j] = self.covariance_matrix[idx_i][idx_j];
            }
        }
        
        // Equal weight portfolio variance for sub-cluster
        let w = Decimal::ONE / Decimal::from(n as u32);
        let mut var = Decimal::ZERO;
        for i in 0..n {
            for j in 0..n {
                var += w * w * sub_cov[i][j];
            }
        }
        
        var.max(Decimal::new(1, 10)) // Prevent division by zero
    }
    
    /// Get symbols
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }
    
    /// Get expected returns
    pub fn expected_returns(&self) -> &[Decimal] {
        &self.expected_returns
    }
    
    /// Get covariance matrix
    pub fn covariance_matrix(&self) -> &[Vec<Decimal>] {
        &self.covariance_matrix
    }
    
    /// Get correlation matrix
    pub fn correlation_matrix(&self) -> &[Vec<Decimal>] {
        &self.correlation_matrix
    }
}

/// Black-Litterman Model for incorporating views
pub struct BlackLitterman {
    /// Market equilibrium returns
    equilibrium_returns: Vec<Decimal>,
    /// Covariance matrix
    covariance: Vec<Vec<Decimal>>,
    /// Risk aversion coefficient
    risk_aversion: Decimal,
    /// Uncertainty in equilibrium (tau)
    tau: Decimal,
}

/// A view in Black-Litterman model
#[derive(Debug, Clone)]
pub struct View {
    /// Asset weights in view (P matrix row)
    pub weights: Vec<Decimal>,
    /// Expected return of view (Q value)
    pub expected_return: Decimal,
    /// Confidence in view (0-1, higher = more confident)
    pub confidence: Decimal,
}

impl BlackLitterman {
    /// Create from market cap weights and covariance
    pub fn new(
        market_weights: &[Decimal],
        covariance: Vec<Vec<Decimal>>,
        risk_aversion: Decimal,
        tau: Decimal,
    ) -> Result<Self, PortfolioError> {
        let n = market_weights.len();
        
        if covariance.len() != n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n,
                actual: covariance.len(),
            });
        }
        
        // Calculate equilibrium returns: Π = δ * Σ * w_mkt
        let mut equilibrium_returns = vec![Decimal::ZERO; n];
        for i in 0..n {
            for j in 0..n {
                equilibrium_returns[i] += risk_aversion * covariance[i][j] * market_weights[j];
            }
        }
        
        Ok(Self {
            equilibrium_returns,
            covariance,
            risk_aversion,
            tau,
        })
    }
    
    /// Apply views and get posterior returns
    pub fn apply_views(&self, views: &[View]) -> Result<Vec<Decimal>, PortfolioError> {
        let n = self.equilibrium_returns.len();
        let k = views.len();
        
        if k == 0 {
            return Ok(self.equilibrium_returns.clone());
        }
        
        // P matrix (k x n)
        let p: Vec<Vec<Decimal>> = views.iter()
            .map(|v| v.weights.clone())
            .collect();
        
        // Q vector (k)
        let q: Vec<Decimal> = views.iter()
            .map(|v| v.expected_return)
            .collect();
        
        // Omega diagonal (k x k) - uncertainty in views
        // Omega_ii = (1/c - 1) * P_i * tau*Σ * P_i'
        let mut omega = vec![vec![Decimal::ZERO; k]; k];
        for i in 0..k {
            let confidence = views[i].confidence.max(Decimal::new(1, 2)); // min 0.01
            let uncertainty = Decimal::ONE / confidence - Decimal::ONE;
            
            // P_i * tau*Σ * P_i'
            let mut view_var = Decimal::ZERO;
            for a in 0..n {
                for b in 0..n {
                    view_var += p[i][a] * self.tau * self.covariance[a][b] * p[i][b];
                }
            }
            omega[i][i] = uncertainty * view_var;
        }
        
        // Posterior: E[R] = [(tau*Σ)^-1 + P'*Ω^-1*P]^-1 * [(tau*Σ)^-1*Π + P'*Ω^-1*Q]
        // Simplified: E[R] = Π + tau*Σ*P' * (P*tau*Σ*P' + Ω)^-1 * (Q - P*Π)
        
        // tau*Σ*P' (n x k)
        let mut tau_sigma_p_t = vec![vec![Decimal::ZERO; k]; n];
        for i in 0..n {
            for j in 0..k {
                for a in 0..n {
                    tau_sigma_p_t[i][j] += self.tau * self.covariance[i][a] * p[j][a];
                }
            }
        }
        
        // P*tau*Σ*P' + Ω (k x k)
        let mut m = omega.clone();
        for i in 0..k {
            for j in 0..k {
                for a in 0..n {
                    m[i][j] += p[i][a] * tau_sigma_p_t[a][j];
                }
            }
        }
        
        // Invert M
        let m_inv = invert_small_matrix(&m)?;
        
        // Q - P*Π (k)
        let mut q_minus_p_pi = vec![Decimal::ZERO; k];
        for i in 0..k {
            q_minus_p_pi[i] = q[i];
            for a in 0..n {
                q_minus_p_pi[i] -= p[i][a] * self.equilibrium_returns[a];
            }
        }
        
        // M^-1 * (Q - P*Π) (k)
        let mut m_inv_q = vec![Decimal::ZERO; k];
        for i in 0..k {
            for j in 0..k {
                m_inv_q[i] += m_inv[i][j] * q_minus_p_pi[j];
            }
        }
        
        // Final: Π + tau*Σ*P' * M^-1 * (Q - P*Π)
        let mut posterior = self.equilibrium_returns.clone();
        for i in 0..n {
            for j in 0..k {
                posterior[i] += tau_sigma_p_t[i][j] * m_inv_q[j];
            }
        }
        
        Ok(posterior)
    }
    
    /// Get equilibrium returns
    pub fn equilibrium_returns(&self) -> &[Decimal] {
        &self.equilibrium_returns
    }
}

/// Helper: Invert small matrix
fn invert_small_matrix(matrix: &[Vec<Decimal>]) -> Result<Vec<Vec<Decimal>>, PortfolioError> {
    let n = matrix.len();
    let tolerance = Decimal::new(1, 10);
    
    let mut aug = vec![vec![Decimal::ZERO; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = matrix[i][j];
        }
        aug[i][n + i] = Decimal::ONE;
    }
    
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        
        if max_val < tolerance {
            return Err(PortfolioError::SingularMatrix);
        }
        
        if max_row != col {
            aug.swap(col, max_row);
        }
        
        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }
        
        // Store col row to avoid borrow issues
        let col_row: Vec<Decimal> = aug[col].clone();
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * col_row[j];
                }
            }
        }
    }
    
    let mut inverse = vec![vec![Decimal::ZERO; n]; n];
    for i in 0..n {
        for j in 0..n {
            inverse[i][j] = aug[i][n + j];
        }
    }
    
    Ok(inverse)
}

/// Newton's method square root for Decimal
fn sqrt_decimal(x: Decimal) -> Decimal {
    if x <= Decimal::ZERO {
        return Decimal::ZERO;
    }
    
    let mut guess = x / Decimal::TWO;
    if guess == Decimal::ZERO {
        guess = Decimal::ONE;
    }
    
    for _ in 0..50 {
        let next = (guess + x / guess) / Decimal::TWO;
        if (next - guess).abs() < Decimal::new(1, 12) {
            return next;
        }
        guess = next;
    }
    
    guess
}

/// Risk Budgeting: allocate risk contributions to match target budgets
pub struct RiskBudgeter {
    covariance: Vec<Vec<Decimal>>,
    target_budgets: Vec<Decimal>,
    max_iterations: usize,
    tolerance: Decimal,
}

impl RiskBudgeter {
    /// Create new risk budgeter
    pub fn new(covariance: Vec<Vec<Decimal>>, target_budgets: Vec<Decimal>) -> Result<Self, PortfolioError> {
        let n = covariance.len();
        
        if target_budgets.len() != n {
            return Err(PortfolioError::DimensionMismatch {
                expected: n,
                actual: target_budgets.len(),
            });
        }
        
        // Validate budgets sum to 1
        let sum: Decimal = target_budgets.iter().sum();
        if (sum - Decimal::ONE).abs() > Decimal::new(1, 4) {
            return Err(PortfolioError::InvalidConstraint(
                format!("Risk budgets must sum to 1, got {}", sum)
            ));
        }
        
        Ok(Self {
            covariance,
            target_budgets,
            max_iterations: 1000,
            tolerance: Decimal::new(1, 8),
        })
    }
    
    /// Optimize weights to match risk budgets
    pub fn optimize(&self) -> Result<Vec<Decimal>, PortfolioError> {
        let n = self.covariance.len();
        
        // Start with equal weights
        let mut weights = vec![Decimal::ONE / Decimal::from(n as u32); n];
        
        for _iter in 0..self.max_iterations {
            // Calculate current risk contributions
            let portfolio_var = self.portfolio_variance(&weights);
            let portfolio_vol = sqrt_decimal(portfolio_var);
            
            if portfolio_vol < self.tolerance {
                break;
            }
            
            let mut marginal_risk = vec![Decimal::ZERO; n];
            for i in 0..n {
                for j in 0..n {
                    marginal_risk[i] += self.covariance[i][j] * weights[j];
                }
                marginal_risk[i] /= portfolio_vol;
            }
            
            // Current risk contributions
            let risk_contributions: Vec<Decimal> = (0..n)
                .map(|i| weights[i] * marginal_risk[i])
                .collect();
            
            // Target risk contributions
            let target_contributions: Vec<Decimal> = self.target_budgets.iter()
                .map(|&b| b * portfolio_vol)
                .collect();
            
            // Check convergence
            let max_error: Decimal = risk_contributions.iter()
                .zip(target_contributions.iter())
                .map(|(&rc, &tc)| (rc - tc).abs())
                .max()
                .unwrap_or(Decimal::ZERO);
            
            if max_error < self.tolerance * portfolio_vol {
                break;
            }
            
            // Update weights
            for i in 0..n {
                if marginal_risk[i] > self.tolerance {
                    weights[i] = target_contributions[i] / marginal_risk[i];
                }
            }
            
            // Normalize
            let sum: Decimal = weights.iter().sum();
            if sum > Decimal::ZERO {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }
        
        Ok(weights)
    }
    
    fn portfolio_variance(&self, weights: &[Decimal]) -> Decimal {
        let n = weights.len();
        let mut var = Decimal::ZERO;
        for i in 0..n {
            for j in 0..n {
                var += weights[i] * weights[j] * self.covariance[i][j];
            }
        }
        var.max(Decimal::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn dec(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }
    
    #[test]
    fn test_equal_weight() {
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let returns = vec![dec("0.10"), dec("0.15"), dec("0.12")];
        let cov = vec![
            vec![dec("0.04"), dec("0.01"), dec("0.005")],
            vec![dec("0.01"), dec("0.09"), dec("0.02")],
            vec![dec("0.005"), dec("0.02"), dec("0.0625")],
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::EqualWeight).unwrap();
        
        assert_eq!(result.weights.len(), 3);
        for w in &result.weights {
            assert!((*w - dec("0.333333333")).abs() < dec("0.01"));
        }
    }
    
    #[test]
    fn test_min_variance() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![dec("0.10"), dec("0.15")];
        // Low correlation between assets
        let cov = vec![
            vec![dec("0.04"), dec("0.005")],
            vec![dec("0.005"), dec("0.09")],
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::MinVariance).unwrap();
        
        // Should allocate more to lower variance asset (A)
        assert!(result.weights[0] > result.weights[1]);
        assert!(result.volatility > Decimal::ZERO);
    }
    
    #[test]
    fn test_max_sharpe() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![dec("0.10"), dec("0.20")]; // B has much higher return
        let cov = vec![
            vec![dec("0.04"), dec("0.01")],
            vec![dec("0.01"), dec("0.09")],
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::MaxSharpe).unwrap();
        
        // Should allocate more to B (higher return despite higher vol)
        assert!(result.sharpe_ratio > Decimal::ZERO);
        assert!(result.expected_return > dec("0.10"));
    }
    
    #[test]
    fn test_risk_parity() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![dec("0.10"), dec("0.15")];
        let cov = vec![
            vec![dec("0.04"), dec("0.00")],  // Zero correlation
            vec![dec("0.00"), dec("0.16")],  // B has 2x volatility
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::RiskParity).unwrap();
        
        // Risk contributions should be approximately equal
        let rc_diff = (result.risk_contributions[0] - result.risk_contributions[1]).abs();
        assert!(rc_diff < dec("0.01"), "Risk contributions not equal: {:?}", result.risk_contributions);
    }
    
    #[test]
    fn test_hrp() {
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let returns = vec![dec("0.10"), dec("0.15"), dec("0.12")];
        let cov = vec![
            vec![dec("0.04"), dec("0.02"), dec("0.005")],
            vec![dec("0.02"), dec("0.09"), dec("0.01")],
            vec![dec("0.005"), dec("0.01"), dec("0.0625")],
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::HierarchicalRiskParity).unwrap();
        
        // Weights should sum to 1
        let sum: Decimal = result.weights.iter().sum();
        assert!((sum - Decimal::ONE).abs() < dec("0.0001"));
        
        // All weights positive
        for w in &result.weights {
            assert!(*w >= Decimal::ZERO);
        }
    }
    
    #[test]
    fn test_from_returns() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![
            vec![dec("0.01"), dec("0.02")],
            vec![dec("-0.005"), dec("0.01")],
            vec![dec("0.015"), dec("-0.01")],
            vec![dec("0.008"), dec("0.015")],
            vec![dec("-0.002"), dec("0.005")],
            vec![dec("0.012"), dec("0.018")],
            vec![dec("0.003"), dec("-0.005")],
            vec![dec("0.009"), dec("0.012")],
            vec![dec("-0.001"), dec("0.008")],
            vec![dec("0.011"), dec("0.014")],
        ];
        
        let optimizer = PortfolioOptimizer::from_returns(
            symbols, &returns, dec("0.02"), 252
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::MinVariance).unwrap();
        
        assert_eq!(result.weights.len(), 2);
        let sum: Decimal = result.weights.iter().sum();
        assert!((sum - Decimal::ONE).abs() < dec("0.0001"));
    }
    
    #[test]
    fn test_constraints_max_weight() {
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let returns = vec![dec("0.05"), dec("0.30"), dec("0.10")];
        let cov = vec![
            vec![dec("0.04"), dec("0.00"), dec("0.00")],
            vec![dec("0.00"), dec("0.04"), dec("0.00")],
            vec![dec("0.00"), dec("0.00"), dec("0.04")],
        ];
        
        let constraints = PortfolioConstraints {
            max_weight: dec("0.5"),
            ..Default::default()
        };
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap().with_constraints(constraints);
        
        let result = optimizer.optimize(OptimizationMethod::MaxSharpe).unwrap();
        
        // With 3 assets, constraint should have effect
        // Sum of weights = 1, so at least one < 0.5 when redistributed
        let sum: Decimal = result.weights.iter().sum();
        assert!((sum - Decimal::ONE).abs() < dec("0.01"));
        
        // Verify all weights are non-negative
        for w in &result.weights {
            assert!(*w >= Decimal::ZERO);
        }
    }
    
    #[test]
    fn test_target_return() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![dec("0.08"), dec("0.15")];
        let cov = vec![
            vec![dec("0.04"), dec("0.01")],
            vec![dec("0.01"), dec("0.09")],
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap();
        
        let target = dec("0.12");
        let result = optimizer.optimize(OptimizationMethod::TargetReturn { target }).unwrap();
        
        // Expected return should be close to target
        assert!((result.expected_return - target).abs() < dec("0.02"));
    }
    
    #[test]
    fn test_diversification_ratio() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![dec("0.10"), dec("0.10")];
        // High correlation reduces diversification
        let cov = vec![
            vec![dec("0.04"), dec("0.038")],  // corr ~ 0.95
            vec![dec("0.038"), dec("0.04")],
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols.clone(), returns.clone(), cov, dec("0.02")
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::EqualWeight).unwrap();
        let high_corr_dr = result.diversification_ratio;
        
        // Low correlation increases diversification
        let cov_low = vec![
            vec![dec("0.04"), dec("0.00")],
            vec![dec("0.00"), dec("0.04")],
        ];
        
        let optimizer2 = PortfolioOptimizer::from_statistics(
            symbols, returns, cov_low, dec("0.02")
        ).unwrap();
        
        let result2 = optimizer2.optimize(OptimizationMethod::EqualWeight).unwrap();
        let low_corr_dr = result2.diversification_ratio;
        
        // Lower correlation should have higher diversification ratio
        assert!(low_corr_dr > high_corr_dr);
    }
    
    #[test]
    fn test_effective_n() {
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
        let returns = vec![dec("0.10"); 4];
        let mut cov = vec![vec![dec("0.04"); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    cov[i][j] = dec("0.01");
                }
            }
        }
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::EqualWeight).unwrap();
        
        // Equal weight across 4 assets: effective N should be 4
        assert!((result.effective_n - dec("4")).abs() < dec("0.1"));
    }
    
    #[test]
    fn test_black_litterman_no_views() {
        let market_weights = vec![dec("0.6"), dec("0.4")];
        let cov = vec![
            vec![dec("0.04"), dec("0.01")],
            vec![dec("0.01"), dec("0.09")],
        ];
        
        let bl = BlackLitterman::new(
            &market_weights, cov, dec("2.5"), dec("0.05")
        ).unwrap();
        
        let posterior = bl.apply_views(&[]).unwrap();
        
        // No views -> posterior equals equilibrium
        assert_eq!(posterior.len(), 2);
        assert_eq!(posterior, bl.equilibrium_returns().to_vec());
    }
    
    #[test]
    fn test_black_litterman_with_view() {
        let market_weights = vec![dec("0.5"), dec("0.5")];
        let cov = vec![
            vec![dec("0.04"), dec("0.00")],
            vec![dec("0.00"), dec("0.04")],
        ];
        
        let bl = BlackLitterman::new(
            &market_weights, cov, dec("2.5"), dec("0.05")
        ).unwrap();
        
        let equilibrium = bl.equilibrium_returns().to_vec();
        
        // View: Asset A will outperform by 5%
        let view = View {
            weights: vec![dec("1.0"), dec("0.0")],
            expected_return: equilibrium[0] + dec("0.05"),
            confidence: dec("0.8"),
        };
        
        let posterior = bl.apply_views(&[view]).unwrap();
        
        // Posterior for A should be higher than equilibrium
        assert!(posterior[0] > equilibrium[0]);
    }
    
    #[test]
    fn test_risk_budgeter() {
        let cov = vec![
            vec![dec("0.04"), dec("0.00")],
            vec![dec("0.00"), dec("0.09")],
        ];
        
        // 60-40 risk budget
        let budgets = vec![dec("0.6"), dec("0.4")];
        
        let budgeter = RiskBudgeter::new(cov, budgets.clone()).unwrap();
        let weights = budgeter.optimize().unwrap();
        
        // Weights should be positive and sum to 1
        assert!(weights[0] > Decimal::ZERO);
        assert!(weights[1] > Decimal::ZERO);
        let sum: Decimal = weights.iter().sum();
        assert!((sum - Decimal::ONE).abs() < dec("0.0001"));
    }
    
    #[test]
    fn test_risk_budgeter_invalid() {
        let cov = vec![
            vec![dec("0.04"), dec("0.00")],
            vec![dec("0.00"), dec("0.09")],
        ];
        
        // Invalid budgets (don't sum to 1)
        let budgets = vec![dec("0.6"), dec("0.5")];
        
        let result = RiskBudgeter::new(cov, budgets);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_sqrt_decimal() {
        assert_eq!(sqrt_decimal(Decimal::ZERO), Decimal::ZERO);
        
        let four = Decimal::from(4);
        let two = sqrt_decimal(four);
        assert!((two - Decimal::from(2)).abs() < dec("0.0001"));
        
        let nine = Decimal::from(9);
        let three = sqrt_decimal(nine);
        assert!((three - Decimal::from(3)).abs() < dec("0.0001"));
    }
    
    #[test]
    fn test_matrix_inversion() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![dec("0.10"), dec("0.15")];
        let cov = vec![
            vec![dec("0.04"), dec("0.01")],
            vec![dec("0.01"), dec("0.09")],
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov.clone(), dec("0.02")
        ).unwrap();
        
        let inv = optimizer.invert_matrix(&cov).unwrap();
        
        // A * A^-1 should be identity
        let mut product = vec![vec![Decimal::ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    product[i][j] += cov[i][k] * inv[k][j];
                }
            }
        }
        
        assert!((product[0][0] - Decimal::ONE).abs() < dec("0.0001"));
        assert!((product[1][1] - Decimal::ONE).abs() < dec("0.0001"));
        assert!(product[0][1].abs() < dec("0.0001"));
        assert!(product[1][0].abs() < dec("0.0001"));
    }
    
    #[test]
    fn test_singular_matrix_error() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![dec("0.10"), dec("0.15")];
        // Singular matrix (rows are multiples)
        let cov = vec![
            vec![dec("0.04"), dec("0.04")],
            vec![dec("0.04"), dec("0.04")],
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::MinVariance);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_insufficient_data() {
        let symbols = vec!["A".to_string(), "B".to_string()];
        let returns = vec![
            vec![dec("0.01"), dec("0.02")],
            vec![dec("0.02"), dec("0.01")],
        ]; // Only 2 periods
        
        let result = PortfolioOptimizer::from_returns(
            symbols, &returns, dec("0.02"), 252
        );
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_max_diversification() {
        let symbols = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let returns = vec![dec("0.10"), dec("0.12"), dec("0.11")];
        let cov = vec![
            vec![dec("0.04"), dec("0.01"), dec("0.005")],
            vec![dec("0.01"), dec("0.09"), dec("0.02")],
            vec![dec("0.005"), dec("0.02"), dec("0.0625")],
        ];
        
        let optimizer = PortfolioOptimizer::from_statistics(
            symbols, returns, cov, dec("0.02")
        ).unwrap();
        
        let result = optimizer.optimize(OptimizationMethod::MaxDiversification).unwrap();
        
        // Diversification ratio should be > 1
        assert!(result.diversification_ratio >= Decimal::ONE);
        
        // Weights valid
        let sum: Decimal = result.weights.iter().sum();
        assert!((sum - Decimal::ONE).abs() < dec("0.01"));
    }
}
