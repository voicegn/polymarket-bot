//! Negative Risk Market Scanner
//!
//! Scans multi-outcome markets (e.g., "Who will win the election?")
//! for arbitrage opportunities where sum of all Yes prices < 1.0
//!
//! Strategy: Buy Yes on ALL candidates. One must win, guaranteeing $1 payout.
//! If total cost < $1, profit = $1 - total_cost

use super::{ScannerConfig};
use crate::client::clob::ClobClient;
use crate::error::Result;
use chrono::{DateTime, Utc};
use reqwest::Client;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::Deserialize;
use tracing::{info, warn};

/// A negative risk arbitrage opportunity
#[derive(Debug, Clone)]
pub struct NegativeRiskOpp {
    /// Event ID
    pub event_id: String,
    /// Event title
    pub event_title: String,
    /// All outcomes with prices
    pub outcomes: Vec<OutcomeInfo>,
    /// Sum of all Yes prices
    pub total_yes_price: Decimal,
    /// Arbitrage value (1.0 - total_yes_price)
    pub arbitrage_value: Decimal,
    /// Maximum size (min of all ask sizes)
    pub max_size: u32,
    /// Net profit after fees
    pub net_profit: Decimal,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
}

/// Individual outcome info
#[derive(Debug, Clone)]
pub struct OutcomeInfo {
    pub name: String,
    pub token_id: String,
    pub yes_price: Decimal,
    pub yes_ask_size: u32,
}

/// API response for negative risk events
#[derive(Debug, Deserialize)]
struct NegRiskEvent {
    id: String,
    title: String,
    slug: String,
    #[serde(default)]
    markets: Vec<String>,
    #[serde(default)]
    neg_risk: bool,
    #[serde(default)]
    active: bool,
}

/// Market detail in event
#[derive(Debug, Deserialize)]
struct MarketDetail {
    question: String,
    #[serde(rename = "conditionId")]
    condition_id: String,
    #[serde(rename = "clobTokenIds")]
    clob_token_ids: Option<String>,
}

/// Event detail response
#[derive(Debug, Deserialize)]
struct EventDetail {
    id: String,
    title: String,
    markets: Vec<MarketDetail>,
}

/// Scanner for negative risk (multi-outcome) markets
pub struct NegativeRiskScanner {
    config: ScannerConfig,
    http: Client,
    clob: ClobClient,
    gamma_url: String,
}

impl NegativeRiskScanner {
    /// Create a new negative risk scanner
    pub fn new(config: ScannerConfig, clob: ClobClient, gamma_url: &str) -> Self {
        Self {
            config,
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap(),
            clob,
            gamma_url: gamma_url.to_string(),
        }
    }

    /// Scan for negative risk opportunities
    pub async fn scan(&self) -> Result<Vec<NegativeRiskOpp>> {
        let events = self.fetch_neg_risk_events().await?;
        let mut opportunities = Vec::new();

        for event in events {
            if !event.active || !event.neg_risk || event.markets.len() < 2 {
                continue;
            }

            match self.check_event(&event).await {
                Ok(Some(opp)) => {
                    info!(
                        "[NegRisk] Found opportunity: {} value={:.2}%, profit=${:.4}",
                        opp.event_title,
                        opp.arbitrage_value * dec!(100),
                        opp.net_profit
                    );
                    opportunities.push(opp);
                }
                Ok(None) => {}
                Err(e) => {
                    warn!("[NegRisk] Error checking event {}: {}", event.id, e);
                }
            }
        }

        // Sort by arbitrage value
        opportunities.sort_by(|a, b| b.arbitrage_value.cmp(&a.arbitrage_value));

        Ok(opportunities)
    }

    /// Fetch all negative risk events from API
    async fn fetch_neg_risk_events(&self) -> Result<Vec<NegRiskEvent>> {
        let url = format!(
            "{}/events?active=true&closed=false&neg_risk=true&limit=100",
            self.gamma_url
        );

        let resp: Vec<NegRiskEvent> = self.http.get(&url).send().await?.json().await?;

        Ok(resp)
    }

    /// Check a single event for arbitrage
    async fn check_event(&self, event: &NegRiskEvent) -> Result<Option<NegativeRiskOpp>> {
        // Fetch event details
        let url = format!("{}/events/{}", self.gamma_url, event.slug);
        let detail: EventDetail = self.http.get(&url).send().await?.json().await?;

        if detail.markets.len() < 2 {
            return Ok(None);
        }

        // Get prices for each outcome
        let mut outcomes = Vec::new();
        let mut total_yes_price = dec!(0);
        let mut min_ask_size: u32 = u32::MAX;

        for market in &detail.markets {
            // Parse token IDs
            let token_ids: Vec<String> = market
                .clob_token_ids
                .as_ref()
                .and_then(|s| serde_json::from_str(s).ok())
                .unwrap_or_default();

            if token_ids.is_empty() {
                continue;
            }

            let yes_token_id = &token_ids[0];

            // Get orderbook for Yes token
            let book = match self.clob.get_order_book(yes_token_id).await {
                Ok(b) => b,
                Err(_) => continue,
            };

            if book.asks.is_empty() {
                continue;
            }

            let yes_price = book.asks[0].price;
            let yes_size: u32 = book.asks[0].size.try_into().unwrap_or(0);

            // Extract outcome name from question
            let name = extract_outcome_name(&market.question);

            outcomes.push(OutcomeInfo {
                name,
                token_id: yes_token_id.clone(),
                yes_price,
                yes_ask_size: yes_size,
            });

            total_yes_price += yes_price;
            min_ask_size = min_ask_size.min(yes_size);
        }

        // Need at least 2 outcomes
        if outcomes.len() < 2 {
            return Ok(None);
        }

        // Check for arbitrage
        let arbitrage_value = dec!(1) - total_yes_price;
        if arbitrage_value <= self.config.min_spread {
            return Ok(None);
        }

        // Check liquidity
        if min_ask_size < self.config.min_liquidity {
            return Ok(None);
        }

        // Calculate profit
        let gross_profit = arbitrage_value * Decimal::from(min_ask_size);
        let fees = gross_profit * self.config.taker_fee_rate;
        // Gas cost per outcome
        let gas = self.config.gas_cost * Decimal::from(outcomes.len() as u32);
        let net_profit = gross_profit - fees - gas;

        if net_profit <= dec!(0) {
            return Ok(None);
        }

        Ok(Some(NegativeRiskOpp {
            event_id: detail.id,
            event_title: detail.title,
            outcomes,
            total_yes_price,
            arbitrage_value,
            max_size: min_ask_size,
            net_profit,
            detected_at: Utc::now(),
        }))
    }
}

/// Extract outcome name from question
fn extract_outcome_name(question: &str) -> String {
    if question.len() > 30 {
        format!("{}...", &question[..30])
    } else {
        question.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negative_risk_calculation() {
        // 4 candidates, each at 0.23 → total = 0.92 → arb = 0.08 (8%)
        let prices = vec![dec!(0.23), dec!(0.23), dec!(0.23), dec!(0.23)];
        let total: Decimal = prices.iter().sum();
        let arb = dec!(1) - total;

        assert_eq!(total, dec!(0.92));
        assert_eq!(arb, dec!(0.08));
    }

    #[test]
    fn test_no_opportunity() {
        // 4 candidates, each at 0.26 → total = 1.04 → no opportunity
        let prices = vec![dec!(0.26), dec!(0.26), dec!(0.26), dec!(0.26)];
        let total: Decimal = prices.iter().sum();

        assert!(total > dec!(1));
    }
}
