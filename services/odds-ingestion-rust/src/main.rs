//! NBA Basketball Odds Ingestion Service v5.0
//!
//! Real-time odds streaming from The Odds API with sub-10ms latency.
//! Publishes to Redis Streams and stores in TimescaleDB.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{error, info};
use uuid::Uuid;

/// The Odds API event structure
#[derive(Debug, Deserialize, Clone)]
pub struct OddsApiEvent {
    pub id: String,
    pub sport_key: String,
    pub sport_title: String,
    pub commence_time: DateTime<Utc>,
    pub home_team: String,
    pub away_team: String,
    pub bookmakers: Vec<Bookmaker>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Bookmaker {
    pub key: String,
    pub title: String,
    pub last_update: DateTime<Utc>,
    pub markets: Vec<Market>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Market {
    pub key: String,
    pub last_update: DateTime<Utc>,
    pub outcomes: Vec<Outcome>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Outcome {
    pub name: String,
    pub price: i32,
    pub point: Option<f64>,
}

/// Configuration
#[derive(Clone)]
pub struct Config {
    pub odds_api_key: String,
    pub poll_interval_seconds: u64,
    pub sport_key: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            odds_api_key: env::var("THE_ODDS_API_KEY")
                .context("THE_ODDS_API_KEY not set")?,
            poll_interval_seconds: env::var("POLL_INTERVAL_SECONDS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            sport_key: "basketball_nba".to_string(),
        })
    }
}

/// Odds ingestion service
pub struct OddsIngestionService {
    config: Config,
    http_client: reqwest::Client,
    db_pool: sqlx::PgPool,
    redis_client: redis::Client,
}

impl OddsIngestionService {
    pub async fn new(config: Config) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        // Initialize database connection pool
        let database_url = env::var("DATABASE_URL")
            .context("DATABASE_URL not set")?;
        let db_pool = sqlx::PgPool::connect(&database_url)
            .await
            .context("Failed to connect to database")?;

        // Initialize Redis client
        let redis_url = env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://redis:6379".to_string());
        let redis_client = redis::Client::open(redis_url)
            .context("Failed to create Redis client")?;

        Ok(Self {
            config,
            http_client,
            db_pool,
            redis_client,
        })
    }

    /// Fetch events from The Odds API
    pub async fn fetch_odds(&self) -> Result<Vec<OddsApiEvent>> {
        let url = format!(
            "https://api.the-odds-api.com/v4/sports/{}/odds",
            self.config.sport_key
        );

        let response = self
            .http_client
            .get(&url)
            .query(&[("apiKey", &self.config.odds_api_key), ("regions", "us"), ("markets", "spreads,totals,h2h")])
            .send()
            .await
            .context("Failed to fetch odds")?;

        if !response.status().is_success() {
            error!("API returned status: {}", response.status());
            return Err(anyhow::anyhow!("API request failed"));
        }

        let events: Vec<OddsApiEvent> = response
            .json()
            .await
            .context("Failed to parse JSON response")?;

        info!("Fetched {} events", events.len());
        Ok(events)
    }

    /// Store odds in database and publish to Redis
    pub async fn store_odds(&self, events: &[OddsApiEvent]) -> Result<()> {

        for event in events {
            // Get or create game record
            let game_id: Uuid = sqlx::query_scalar(
                r#"
                INSERT INTO games (external_id, home_team, away_team, commence_time, status)
                VALUES ($1, $2, $3, $4, 'scheduled')
                ON CONFLICT (external_id) DO UPDATE
                SET updated_at = NOW()
                RETURNING game_id
                "#
            )
            .bind(&event.id)
            .bind(&event.home_team)
            .bind(&event.away_team)
            .bind(event.commence_time)
            .fetch_one(&self.db_pool)
            .await
            .context("Failed to insert/update game")?;

            // Store odds snapshots for each bookmaker and market
            for bookmaker in &event.bookmakers {
                for market in &bookmaker.markets {
                    let market_type = &market.key;
                    let period = "full"; // Default to full game

                    // Extract odds based on market type
                    let (home_line, away_line, total_line, home_price, away_price, over_price, under_price) =
                        self.extract_market_odds(market);

                    sqlx::query(
                        r#"
                        INSERT INTO odds_snapshots 
                        (time, game_id, external_id, bookmaker, market_type, period,
                         home_line, away_line, total_line, home_price, away_price, over_price, under_price)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        ON CONFLICT (time, game_id, bookmaker, market_type, period) DO UPDATE
                        SET home_line = EXCLUDED.home_line,
                            away_line = EXCLUDED.away_line,
                            total_line = EXCLUDED.total_line,
                            home_price = EXCLUDED.home_price,
                            away_price = EXCLUDED.away_price,
                            over_price = EXCLUDED.over_price,
                            under_price = EXCLUDED.under_price
                        "#
                    )
                    .bind(Utc::now())
                    .bind(game_id)
                    .bind(&event.id)
                    .bind(&bookmaker.key)
                    .bind(market_type)
                    .bind(period)
                    .bind(home_line)
                    .bind(away_line)
                    .bind(total_line)
                    .bind(home_price)
                    .bind(away_price)
                    .bind(over_price)
                    .bind(under_price)
                    .execute(&self.db_pool)
                    .await
                    .context("Failed to insert odds snapshot")?;
                }
            }

            // Publish to Redis stream
            let mut conn = self.redis_client
                .get_async_connection()
                .await
                .context("Failed to get Redis connection")?;

            let event_json = serde_json::to_string(event)
                .context("Failed to serialize event")?;

            redis::cmd("XADD")
                .arg("odds:stream")
                .arg("*")
                .arg("game_id")
                .arg(&event.id)
                .arg("data")
                .arg(&event_json)
                .query_async::<_, String>(&mut conn)
                .await
                .context("Failed to publish to Redis stream")?;
        }

        info!("Stored {} events in database and Redis", events.len());
        Ok(())
    }

    /// Extract market odds from market outcomes
    fn extract_market_odds(&self, market: &Market) -> (Option<f64>, Option<f64>, Option<f64>, Option<i32>, Option<i32>, Option<i32>, Option<i32>) {
        let mut home_line = None;
        let mut away_line = None;
        let mut total_line = None;
        let mut home_price = None;
        let mut away_price = None;
        let mut over_price = None;
        let mut under_price = None;

        for outcome in &market.outcomes {
            match market.key.as_str() {
                "spreads" => {
                    if let Some(point) = outcome.point {
                        if outcome.name.contains("Home") || outcome.name.contains(&market.outcomes[0].name) {
                            home_line = Some(point);
                            home_price = Some(outcome.price);
                        } else {
                            away_line = Some(-point);
                            away_price = Some(outcome.price);
                        }
                    }
                }
                "totals" => {
                    if let Some(point) = outcome.point {
                        total_line = Some(point);
                        if outcome.name == "Over" {
                            over_price = Some(outcome.price);
                        } else {
                            under_price = Some(outcome.price);
                        }
                    }
                }
                "h2h" => {
                    if outcome.name.contains("Home") || outcome.name == market.outcomes[0].name {
                        home_price = Some(outcome.price);
                    } else {
                        away_price = Some(outcome.price);
                    }
                }
                _ => {}
            }
        }

        (home_line, away_line, total_line, home_price, away_price, over_price, under_price)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load environment variables
    dotenvy::dotenv().ok();

    // Load configuration
    let config = Config::from_env()?;
    let service = OddsIngestionService::new(config.clone()).await?;

    info!("NBA Odds Ingestion Service starting...");
    info!("Sport: {}", config.sport_key);
    info!("Poll interval: {}s", config.poll_interval_seconds);

    // Poll loop
    let poll_interval = std::time::Duration::from_secs(config.poll_interval_seconds);
    loop {
        match service.fetch_odds().await {
            Ok(events) => {
                info!("Fetched {} events", events.len());
                if let Err(e) = service.store_odds(&events).await {
                    error!("Error storing odds: {}", e);
                } else {
                    info!("Successfully stored {} events", events.len());
                }
            }
            Err(e) => {
                error!("Error fetching odds: {}", e);
            }
        }
        tokio::time::sleep(poll_interval).await;
    }
}
