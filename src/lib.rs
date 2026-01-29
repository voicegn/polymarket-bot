//! Polymarket Probability Trading Bot
//! 
//! A Rust-based automated trading system for Polymarket prediction markets.
//! 
//! ## Architecture
//! 
//! ```text
//! Ingester (TG/X/Chain) → Processor (LLM) → Strategy → Executor → Notifier
//!                                              ↑                    ↑
//!                            Analysis (Pattern Recognition, Copy Trade)
//!                                              ↑
//!                                    Risk Management (Daily P&L, Volatility, Correlation)
//! ```

pub mod analysis;
pub mod client;
pub mod config;
pub mod error;
pub mod executor;
pub mod ingester;
pub mod model;
pub mod monitor;
pub mod notify;
pub mod risk;
pub mod storage;
pub mod strategy;
pub mod telegram;
pub mod testing;
pub mod types;
pub mod utils;

#[cfg(test)]
mod types_tests;
#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod error_tests;
#[cfg(test)]
mod integration_tests;
