//! Data storage and persistence

use crate::error::Result;
use crate::monitor::PerformanceStats;
use crate::types::Trade;
use rust_decimal::Decimal;
use sqlx::sqlite::{SqlitePool, SqlitePoolOptions};
use std::path::Path;

/// Database for storing trades and state
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    /// Connect to SQLite database (creates if not exists)
    pub async fn connect<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db_url = format!("sqlite:{}?mode=rwc", path.as_ref().display());
        
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await?;

        let db = Self { pool };
        db.run_migrations().await?;

        Ok(db)
    }

    /// Run database migrations
    async fn run_migrations(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                side TEXT NOT NULL,
                price TEXT NOT NULL,
                size TEXT NOT NULL,
                fee TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS positions (
                token_id TEXT PRIMARY KEY,
                market_id TEXT NOT NULL,
                size TEXT NOT NULL,
                avg_entry_price TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS market_cache (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Save a trade
    pub async fn save_trade(&self, trade: &Trade) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO trades (id, order_id, token_id, market_id, side, price, size, fee, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&trade.id)
        .bind(&trade.order_id)
        .bind(&trade.token_id)
        .bind(&trade.market_id)
        .bind(format!("{:?}", trade.side))
        .bind(trade.price.to_string())
        .bind(trade.size.to_string())
        .bind(trade.fee.to_string())
        .bind(trade.timestamp.to_rfc3339())
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Get recent trades
    pub async fn get_recent_trades(&self, limit: i64) -> Result<Vec<Trade>> {
        let rows = sqlx::query_as::<_, TradeRow>(
            r#"
            SELECT id, order_id, token_id, market_id, side, price, size, fee, timestamp
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().filter_map(|r| r.try_into().ok()).collect())
    }

    /// Get daily performance stats
    pub async fn get_daily_stats(&self) -> Result<PerformanceStats> {
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        
        let rows = sqlx::query_as::<_, TradeRow>(
            r#"
            SELECT id, order_id, token_id, market_id, side, price, size, fee, timestamp
            FROM trades
            WHERE timestamp LIKE ?
            ORDER BY timestamp DESC
            "#,
        )
        .bind(format!("{}%", today))
        .fetch_all(&self.pool)
        .await?;

        let trades: Vec<Trade> = rows.into_iter().filter_map(|r| r.try_into().ok()).collect();
        
        let total_trades = trades.len();
        // Note: PnL calculation requires position tracking - simplified here
        let total_pnl = Decimal::ZERO; // TODO: Calculate from closed positions
        
        Ok(PerformanceStats {
            total_trades,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: Decimal::ZERO,
            total_pnl,
            avg_pnl_per_trade: Decimal::ZERO,
            sharpe_ratio: None,
        })
    }
}

#[derive(Debug, sqlx::FromRow)]
struct TradeRow {
    id: String,
    order_id: String,
    token_id: String,
    market_id: String,
    side: String,
    price: String,
    size: String,
    fee: String,
    timestamp: String,
}

impl TryFrom<TradeRow> for Trade {
    type Error = anyhow::Error;

    fn try_from(row: TradeRow) -> std::result::Result<Self, Self::Error> {
        use crate::types::Side;

        Ok(Trade {
            id: row.id,
            order_id: row.order_id,
            token_id: row.token_id,
            market_id: row.market_id,
            side: if row.side.contains("Buy") {
                Side::Buy
            } else {
                Side::Sell
            },
            price: row.price.parse()?,
            size: row.size.parse()?,
            fee: row.fee.parse()?,
            timestamp: row.timestamp.parse()?,
        })
    }
}
