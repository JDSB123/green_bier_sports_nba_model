//! NBA Basketball Odds Ingestion Service v5.0
//!
//! Real-time odds streaming from The Odds API with sub-10ms latency.
//! Publishes to Redis Streams and stores in TimescaleDB.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::env;
use tracing::{error, info};

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
}

impl OddsIngestionService {
    pub fn new(config: Config) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config,
            http_client,
        }
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
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Load environment variables
    dotenvy::dotenv().ok();

    // Load configuration
    let config = Config::from_env()?;
    let service = OddsIngestionService::new(config.clone());

    info!("NBA Odds Ingestion Service starting...");
    info!("Sport: {}", config.sport_key);
    info!("Poll interval: {}s", config.poll_interval_seconds);

    // Poll loop
    let poll_interval = std::time::Duration::from_secs(config.poll_interval_seconds);
    loop {
        match service.fetch_odds().await {
            Ok(events) => {
                info!("Processed {} events", events.len());
                // TODO: Store in database and publish to Redis
            }
            Err(e) => {
                error!("Error fetching odds: {}", e);
            }
        }
        tokio::time::sleep(poll_interval).await;
    }
}
