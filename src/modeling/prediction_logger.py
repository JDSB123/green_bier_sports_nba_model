"""
Prediction logging system for retrospective analysis.

Logs all model predictions with full context for later analysis.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.config import settings


class PredictionLogger:
    """Log all model predictions for retrospective analysis."""
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize prediction logger.
        
        Args:
            log_dir: Directory to store prediction logs (defaults to data/processed/predictions)
        """
        if log_dir is None:
            log_dir = os.path.join(settings.data_processed_dir, "predictions")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_prediction(
        self,
        game_date: date,
        home_team: str,
        away_team: str,
        predictions: Dict[str, Any],
        features: Optional[Dict[str, Any]] = None,
        odds: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log a complete prediction set for a game.
        
        Args:
            game_date: Date of the game
            home_team: Home team name
            away_team: Away team name
            predictions: Dict with all predictions (spread, total, ML, etc.)
            features: Feature values used for prediction (optional)
            odds: Market odds at prediction time (optional)
            metadata: Additional metadata (model version, etc.)
            
        Returns:
            Log file path
        """
        timestamp = datetime.now()
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "game_date": game_date.isoformat(),
            "home_team": home_team,
            "away_team": away_team,
            "predictions": predictions,
            "features": features or {},
            "odds": odds or {},
            "metadata": metadata or {},
        }
        
        # Save to date-based file
        log_file = self.log_dir / f"predictions_{game_date.isoformat()}.jsonl"
        
        # Append to file (JSONL format - one JSON object per line)
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")
        
        return str(log_file)
    
    def load_predictions_for_date(
        self,
        game_date: date,
    ) -> List[Dict[str, Any]]:
        """
        Load all predictions for a specific date.
        
        Args:
            game_date: Date to load predictions for
            
        Returns:
            List of prediction log entries
        """
        log_file = self.log_dir / f"predictions_{game_date.isoformat()}.jsonl"
        
        if not log_file.exists():
            return []
        
        predictions = []
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        
        return predictions
    
    def load_all_predictions(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load all predictions within a date range.
        
        Args:
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            List of all prediction log entries
        """
        all_predictions = []
        
        # Get all log files
        log_files = sorted(self.log_dir.glob("predictions_*.jsonl"))
        
        for log_file in log_files:
            # Extract date from filename
            date_str = log_file.stem.replace("predictions_", "")
            try:
                file_date = date.fromisoformat(date_str)
                
                # Filter by date range
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                
                # Load predictions from this file
                with open(log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            all_predictions.append(json.loads(line))
            
            except ValueError:
                continue
        
        return all_predictions
