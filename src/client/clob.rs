//! CLOB (Central Limit Order Book) API client
//!
//! Handles order placement, cancellation, and account queries.
//! Implements Polymarket's Level 1 (EIP-712) and Level 2 (HMAC) authentication.

use crate::client::auth::{ApiCredentials, PolySigner};
use crate::error::{BotError, Result};
use crate::types::{Order, OrderStatus, OrderType, Side};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use base64::{Engine as _, engine::general_purpose::{STANDARD as BASE64, URL_SAFE_NO_PAD as BASE64_URL_SAFE}};

/// Build HMAC signature for Level 2 authentication
fn build_hmac_signature(
    secret: &str,
    timestamp: i64,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> Result<String> {
    // Decode the base64 secret (try URL-safe first, then standard)
    let secret_bytes = BASE64_URL_SAFE.decode(secret)
        .or_else(|_| BASE64.decode(secret))
        .map_err(|e| BotError::Auth(format!("Invalid API secret: {}", e)))?;
    
    // Build the message: timestamp + method + path + body
    let body_str = body.unwrap_or("");
    let message = format!("{}{}{}{}", timestamp, method, path, body_str);
    
    // Create HMAC-SHA256
    let mut mac = Hmac::<Sha256>::new_from_slice(&secret_bytes)
        .map_err(|e| BotError::Auth(format!("HMAC creation failed: {}", e)))?;
    mac.update(message.as_bytes());
    
    // Return base64 encoded signature
    Ok(BASE64.encode(mac.finalize().into_bytes()))
}

/// CLOB API client for trading operations
#[derive(Clone)]
pub struct ClobClient {
    pub http: Client,
    base_url: String,
    signer: PolySigner,
    #[allow(dead_code)]
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

    /// Initialize API credentials by creating or deriving them from the server
    /// This uses Level 1 (EIP-712) authentication
    pub async fn initialize(&self) -> Result<()> {
        // Try to create new API key first, fall back to deriving existing one
        match self.create_api_key().await {
            Ok(creds) => {
                *self.credentials.write().await = Some(creds);
                Ok(())
            }
            Err(_) => {
                // If creation fails, try to derive existing key
                let creds = self.derive_api_key().await?;
                *self.credentials.write().await = Some(creds);
                Ok(())
            }
        }
    }

    /// Create Level 1 authentication headers (EIP-712 signed)
    async fn create_l1_headers(&self, nonce: u64) -> Result<Vec<(String, String)>> {
        let timestamp = chrono::Utc::now().timestamp();
        let signature = self.signer.sign_clob_auth(timestamp, nonce).await?;
        
        Ok(vec![
            ("POLY_ADDRESS".to_string(), self.signer.address_hex()),
            ("POLY_SIGNATURE".to_string(), signature),
            ("POLY_TIMESTAMP".to_string(), timestamp.to_string()),
            ("POLY_NONCE".to_string(), nonce.to_string()),
        ])
    }

    /// Create a new API key using Level 1 auth
    async fn create_api_key(&self) -> Result<ApiCredentials> {
        let url = format!("{}/auth/api-key", self.base_url);
        let headers = self.create_l1_headers(0).await?;
        
        let mut req = self.http.post(&url);
        for (key, value) in headers {
            req = req.header(&key, &value);
        }
        
        let resp: serde_json::Value = req.send().await?.json().await?;
        
        Ok(ApiCredentials {
            api_key: resp["apiKey"]
                .as_str()
                .ok_or_else(|| BotError::Api("Missing apiKey".into()))?
                .to_string(),
            api_secret: resp["secret"]
                .as_str()
                .ok_or_else(|| BotError::Api("Missing secret".into()))?
                .to_string(),
            api_passphrase: resp["passphrase"]
                .as_str()
                .ok_or_else(|| BotError::Api("Missing passphrase".into()))?
                .to_string(),
            timestamp: chrono::Utc::now().timestamp() as u64,
        })
    }

    /// Derive existing API key using Level 1 auth
    async fn derive_api_key(&self) -> Result<ApiCredentials> {
        let url = format!("{}/auth/derive-api-key", self.base_url);
        let headers = self.create_l1_headers(0).await?;
        
        let mut req = self.http.get(&url);
        for (key, value) in headers {
            req = req.header(&key, &value);
        }
        
        let resp: serde_json::Value = req.send().await?.json().await?;
        
        Ok(ApiCredentials {
            api_key: resp["apiKey"]
                .as_str()
                .ok_or_else(|| BotError::Api("Missing apiKey".into()))?
                .to_string(),
            api_secret: resp["secret"]
                .as_str()
                .ok_or_else(|| BotError::Api("Missing secret".into()))?
                .to_string(),
            api_passphrase: resp["passphrase"]
                .as_str()
                .ok_or_else(|| BotError::Api("Missing passphrase".into()))?
                .to_string(),
            timestamp: chrono::Utc::now().timestamp() as u64,
        })
    }

    /// Create Level 2 authentication headers (HMAC signed)
    fn create_l2_headers(
        &self, 
        creds: &ApiCredentials, 
        method: &str, 
        path: &str, 
        body: Option<&str>
    ) -> Result<Vec<(String, String)>> {
        let timestamp = chrono::Utc::now().timestamp();
        let hmac_sig = build_hmac_signature(
            &creds.api_secret,
            timestamp,
            method,
            path,
            body,
        )?;
        
        Ok(vec![
            ("POLY_ADDRESS".to_string(), self.signer.address_hex()),
            ("POLY_SIGNATURE".to_string(), hmac_sig),
            ("POLY_TIMESTAMP".to_string(), timestamp.to_string()),
            ("POLY_API_KEY".to_string(), creds.api_key.clone()),
            ("POLY_PASSPHRASE".to_string(), creds.api_passphrase.clone()),
        ])
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

        let path = "/balance";
        let url = format!("{}{}", self.base_url, path);
        let headers = self.create_l2_headers(creds, "GET", path, None)?;
        
        let mut req = self.http.get(&url);
        for (key, value) in headers {
            req = req.header(&key, &value);
        }
        
        let resp: BalanceResponse = req.send().await?.json().await?;

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

        let path = "/order";
        let url = format!("{}{}", self.base_url, path);
        let body = serde_json::to_string(&req)
            .map_err(|e| BotError::Api(format!("JSON serialization failed: {}", e)))?;
        let headers = self.create_l2_headers(creds, "POST", path, Some(&body))?;
        
        let mut http_req = self.http.post(&url);
        for (key, value) in headers {
            http_req = http_req.header(&key, &value);
        }
        
        let resp: OrderResponse = http_req
            .header("Content-Type", "application/json")
            .body(body)
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

        let path = format!("/order/{}", order_id);
        let url = format!("{}{}", self.base_url, path);
        let headers = self.create_l2_headers(creds, "DELETE", &path, None)?;
        
        let mut req = self.http.delete(&url);
        for (key, value) in headers {
            req = req.header(&key, &value);
        }
        
        req.send().await?;
        Ok(())
    }

    /// Get order status
    pub async fn get_order(&self, order_id: &str) -> Result<OrderStatus> {
        let creds = self.credentials.read().await;
        let creds = creds
            .as_ref()
            .ok_or_else(|| BotError::Auth("Not authenticated".into()))?;

        let path = format!("/order/{}", order_id);
        let url = format!("{}{}", self.base_url, path);
        let headers = self.create_l2_headers(creds, "GET", &path, None)?;
        
        let mut req = self.http.get(&url);
        for (key, value) in headers {
            req = req.header(&key, &value);
        }
        
        let resp: serde_json::Value = req.send().await?.json().await?;

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

        let path = "/orders";
        let url = format!("{}{}", self.base_url, path);
        let headers = self.create_l2_headers(creds, "GET", path, None)?;
        
        let mut req = self.http.get(&url);
        for (key, value) in headers {
            req = req.header(&key, &value);
        }
        
        let resp: Vec<serde_json::Value> = req
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

        let path = "/positions";
        let url = format!("{}{}", self.base_url, path);
        let headers = self.create_l2_headers(creds, "GET", path, None)?;
        
        let mut req = self.http.get(&url);
        for (key, value) in headers {
            req = req.header(&key, &value);
        }
        
        let resp: Vec<serde_json::Value> = req.send().await?.json().await?;

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
