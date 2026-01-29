//! CLOB (Central Limit Order Book) API client
//!
//! Handles order placement, cancellation, and account queries.

use crate::client::auth::{ApiCredentials, PolySigner};
use crate::error::{BotError, Result};
use crate::types::{Order, OrderStatus, OrderType, Side};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// CLOB API client for trading operations
pub struct ClobClient {
    pub http: Client,
    base_url: String,
    signer: PolySigner,
    funder: Option<String>,
    credentials: Arc<RwLock<Option<ApiCredentials>>>,
}

#[derive(Debug, Serialize)]
struct CreateOrderRequest {
    token_id: String,
    price: String,
    size: String,
    side: String,
    order_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expiration: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct OrderResponse {
    #[serde(rename = "orderID")]
    order_id: String,
    status: String,
}

#[derive(Debug, Deserialize)]
struct BalanceResponse {
    balance: String,
}

impl ClobClient {
    /// Create a new CLOB client
    pub fn new(base_url: &str, signer: PolySigner, funder: Option<String>) -> Result<Self> {
        let http = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            signer,
            funder,
            credentials: Arc::new(RwLock::new(None)),
        })
    }

    /// Initialize API credentials (call this before trading)
    pub async fn initialize(&self) -> Result<()> {
        // Get nonce from server
        let nonce = self.get_nonce().await?;
        
        // Create and store credentials
        let creds = self.signer.create_api_credentials(nonce).await?;
        *self.credentials.write().await = Some(creds);
        
        Ok(())
    }

    /// Get server nonce for authentication
    async fn get_nonce(&self) -> Result<u64> {
        let url = format!("{}/auth/nonce", self.base_url);
        let addr = self.signer.address_hex();

        let resp: serde_json::Value = self
            .http
            .get(&url)
            .query(&[("address", &addr)])
            .send()
            .await?
            .json()
            .await?;

        resp["nonce"]
            .as_u64()
            .ok_or_else(|| BotError::Api("Invalid nonce response".into()))
    }

    /// Check if the CLOB API is healthy
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/", self.base_url);
        let resp = self.http.get(&url).send().await?;
        Ok(resp.status().is_success())
    }

    /// Get account USDC balance
    pub async fn get_balance(&self) -> Result<Decimal> {
        let creds = self.credentials.read().await;
        let creds = creds
            .as_ref()
            .ok_or_else(|| BotError::Auth("Not authenticated".into()))?;

        let url = format!("{}/balance", self.base_url);
        let resp: BalanceResponse = self
            .http
            .get(&url)
            .header("POLY_ADDRESS", &creds.api_passphrase)
            .header("POLY_SIGNATURE", &creds.api_secret)
            .header("POLY_TIMESTAMP", creds.timestamp.to_string())
            .send()
            .await?
            .json()
            .await?;

        resp.balance
            .parse()
            .map_err(|e| BotError::Api(format!("Invalid balance: {}", e)))
    }

    /// Place a limit order
    pub async fn place_order(&self, order: &Order) -> Result<OrderStatus> {
        let creds = self.credentials.read().await;
        let creds = creds
            .as_ref()
            .ok_or_else(|| BotError::Auth("Not authenticated".into()))?;

        let req = CreateOrderRequest {
            token_id: order.token_id.clone(),
            price: order.price.to_string(),
            size: order.size.to_string(),
            side: match order.side {
                Side::Buy => "BUY".to_string(),
                Side::Sell => "SELL".to_string(),
            },
            order_type: match order.order_type {
                OrderType::GTC => "GTC".to_string(),
                OrderType::FOK => "FOK".to_string(),
                OrderType::GTD => "GTD".to_string(),
            },
            expiration: None,
        };

        let url = format!("{}/order", self.base_url);
        let resp: OrderResponse = self
            .http
            .post(&url)
            .header("POLY_ADDRESS", &creds.api_passphrase)
            .header("POLY_SIGNATURE", &creds.api_secret)
            .header("POLY_TIMESTAMP", creds.timestamp.to_string())
            .json(&req)
            .send()
            .await?
            .json()
            .await?;

        Ok(OrderStatus {
            order_id: resp.order_id,
            status: resp.status,
            filled_size: Decimal::ZERO,
            remaining_size: order.size,
            avg_price: None,
        })
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let creds = self.credentials.read().await;
        let creds = creds
            .as_ref()
            .ok_or_else(|| BotError::Auth("Not authenticated".into()))?;

        let url = format!("{}/order/{}", self.base_url, order_id);
        self.http
            .delete(&url)
            .header("POLY_ADDRESS", &creds.api_passphrase)
            .header("POLY_SIGNATURE", &creds.api_secret)
            .header("POLY_TIMESTAMP", creds.timestamp.to_string())
            .send()
            .await?;

        Ok(())
    }

    /// Get order status
    pub async fn get_order(&self, order_id: &str) -> Result<OrderStatus> {
        let creds = self.credentials.read().await;
        let creds = creds
            .as_ref()
            .ok_or_else(|| BotError::Auth("Not authenticated".into()))?;

        let url = format!("{}/order/{}", self.base_url, order_id);
        let resp: serde_json::Value = self
            .http
            .get(&url)
            .header("POLY_ADDRESS", &creds.api_passphrase)
            .header("POLY_SIGNATURE", &creds.api_secret)
            .header("POLY_TIMESTAMP", creds.timestamp.to_string())
            .send()
            .await?
            .json()
            .await?;

        Ok(OrderStatus {
            order_id: resp["orderID"].as_str().unwrap_or_default().to_string(),
            status: resp["status"].as_str().unwrap_or_default().to_string(),
            filled_size: resp["sizeFilled"]
                .as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or(Decimal::ZERO),
            remaining_size: resp["sizeRemaining"]
                .as_str()
                .and_then(|s| s.parse().ok())
                .unwrap_or(Decimal::ZERO),
            avg_price: resp["avgPrice"].as_str().and_then(|s| s.parse().ok()),
        })
    }

    /// Get all open orders
    pub async fn get_open_orders(&self) -> Result<Vec<OrderStatus>> {
        let creds = self.credentials.read().await;
        let creds = creds
            .as_ref()
            .ok_or_else(|| BotError::Auth("Not authenticated".into()))?;

        let url = format!("{}/orders", self.base_url);
        let resp: Vec<serde_json::Value> = self
            .http
            .get(&url)
            .header("POLY_ADDRESS", &creds.api_passphrase)
            .header("POLY_SIGNATURE", &creds.api_secret)
            .header("POLY_TIMESTAMP", creds.timestamp.to_string())
            .query(&[("status", "open")])
            .send()
            .await?
            .json()
            .await?;

        Ok(resp
            .into_iter()
            .map(|o| OrderStatus {
                order_id: o["orderID"].as_str().unwrap_or_default().to_string(),
                status: o["status"].as_str().unwrap_or_default().to_string(),
                filled_size: o["sizeFilled"]
                    .as_str()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(Decimal::ZERO),
                remaining_size: o["sizeRemaining"]
                    .as_str()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(Decimal::ZERO),
                avg_price: o["avgPrice"].as_str().and_then(|s| s.parse().ok()),
            })
            .collect())
    }

    /// Get midpoint price for a token
    pub async fn get_midpoint(&self, token_id: &str) -> Result<Decimal> {
        let url = format!("{}/midpoint", self.base_url);
        let resp: serde_json::Value = self
            .http
            .get(&url)
            .query(&[("token_id", token_id)])
            .send()
            .await?
            .json()
            .await?;

        resp["mid"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| BotError::Api("Invalid midpoint response".into()))
    }

    /// Get order book for a token
    pub async fn get_order_book(&self, token_id: &str) -> Result<OrderBook> {
        let url = format!("{}/book", self.base_url);
        let resp: serde_json::Value = self
            .http
            .get(&url)
            .query(&[("token_id", token_id)])
            .send()
            .await?
            .json()
            .await?;

        let parse_levels = |arr: &serde_json::Value| -> Vec<OrderBookLevel> {
            arr.as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|l| {
                            Some(OrderBookLevel {
                                price: l["price"].as_str()?.parse().ok()?,
                                size: l["size"].as_str()?.parse().ok()?,
                            })
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        Ok(OrderBook {
            bids: parse_levels(&resp["bids"]),
            asks: parse_levels(&resp["asks"]),
        })
    }

    /// Get current positions
    pub async fn get_positions(&self) -> Result<Vec<crate::types::Position>> {
        let creds = self.credentials.read().await;
        let creds = creds
            .as_ref()
            .ok_or_else(|| BotError::Auth("Not authenticated".into()))?;

        let url = format!("{}/positions", self.base_url);
        let resp: Vec<serde_json::Value> = self
            .http
            .get(&url)
            .header("POLY_ADDRESS", &creds.api_passphrase)
            .header("POLY_SIGNATURE", &creds.api_secret)
            .header("POLY_TIMESTAMP", creds.timestamp.to_string())
            .send()
            .await?
            .json()
            .await?;

        Ok(resp
            .into_iter()
            .filter_map(|p| {
                let size: Decimal = p["size"].as_str()?.parse().ok()?;
                // Skip zero positions
                if size == Decimal::ZERO {
                    return None;
                }

                Some(crate::types::Position {
                    token_id: p["asset"].as_str()?.to_string(),
                    market_id: p["market"].as_str().unwrap_or("unknown").to_string(),
                    side: if size > Decimal::ZERO {
                        crate::types::Side::Buy
                    } else {
                        crate::types::Side::Sell
                    },
                    size: size.abs(),
                    avg_entry_price: p["avgCost"].as_str()?.parse().ok()?,
                    current_price: p["curPrice"]
                        .as_str()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(Decimal::ZERO),
                    unrealized_pnl: p["unrealizedPnl"]
                        .as_str()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(Decimal::ZERO),
                })
            })
            .collect())
    }
}

/// Order book data
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
}

/// Single level in order book
#[derive(Debug, Clone)]
pub struct OrderBookLevel {
    pub price: Decimal,
    pub size: Decimal,
}

impl OrderBook {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.first().map(|l| l.price)
    }

    /// Get spread
    pub fn spread(&self) -> Option<Decimal> {
        Some(self.best_ask()? - self.best_bid()?)
    }

    /// Get midpoint
    pub fn midpoint(&self) -> Option<Decimal> {
        Some((self.best_bid()? + self.best_ask()?) / Decimal::TWO)
    }
}
