-- NBA v5.0 BETA Database Schema
-- Initial migration for PostgreSQL + TimescaleDB

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Games table
CREATE TABLE IF NOT EXISTS games (
    game_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(255) UNIQUE NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    commence_time TIMESTAMPTZ NOT NULL,
    status VARCHAR(50) DEFAULT 'scheduled',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Odds snapshots (time-series)
CREATE TABLE IF NOT EXISTS odds_snapshots (
    time TIMESTAMPTZ NOT NULL,
    game_id UUID REFERENCES games(game_id),
    external_id VARCHAR(255) NOT NULL,
    bookmaker VARCHAR(100) NOT NULL,
    market_type VARCHAR(50) NOT NULL,
    period VARCHAR(20) DEFAULT 'full',
    home_line FLOAT,
    away_line FLOAT,
    total_line FLOAT,
    home_price INTEGER,
    away_price INTEGER,
    over_price INTEGER,
    under_price INTEGER,
    PRIMARY KEY (time, game_id, bookmaker, market_type, period)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('odds_snapshots', 'time', if_not_exists => TRUE);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id UUID REFERENCES games(game_id),
    predicted_at TIMESTAMPTZ DEFAULT NOW(),
    market_type VARCHAR(50) NOT NULL,
    period VARCHAR(20) DEFAULT 'full',
    predicted_value FLOAT NOT NULL,
    confidence FLOAT,
    edge FLOAT,
    recommendation JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_games_commence_time ON games(commence_time);
CREATE INDEX IF NOT EXISTS idx_games_external_id ON games(external_id);
CREATE INDEX IF NOT EXISTS idx_odds_game_id ON odds_snapshots(game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_game_id ON predictions(game_id);
