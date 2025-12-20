"""
Closing Line Value (CLV) tracking system.

CLV is the gold standard for model validation in sports betting.
If your predictions consistently beat the closing line, you have a profitable model.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

from src.config import settings


class CLVTracker:
    """Track Closing Line Value for model predictions."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize CLV tracker.
        
        Args:
            data_dir: Directory to store CLV data (defaults to data/processed/clv)
        """
        if data_dir is None:
            data_dir = os.path.join(settings.data_processed_dir, "clv")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.clv_file = self.data_dir / "clv_tracking.json"
        self.clv_data: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load existing CLV tracking data."""
        if self.clv_file.exists():
            try:
                with open(self.clv_file, "r") as f:
                    self.clv_data = json.load(f)
            except Exception:
                self.clv_data = []
        else:
            self.clv_data = []
    
    def _save_data(self) -> None:
        """Save CLV tracking data to disk."""
        with open(self.clv_file, "w") as f:
            json.dump(self.clv_data, f, indent=2, default=str)
    
    def record_prediction(
        self,
        game_date: date,
        home_team: str,
        away_team: str,
        bet_type: str,  # "spread", "total", "moneyline", "1h_spread", "1h_total"
        model_line: float,
        opening_line: Optional[float] = None,
        prediction_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a model prediction for CLV tracking.
        
        Args:
            game_date: Date of the game
            home_team: Home team name
            away_team: Away team name
            bet_type: Type of bet (spread, total, etc.)
            model_line: Model's predicted line
            opening_line: Opening line from sportsbook (if available)
            prediction_time: When prediction was made
            metadata: Additional metadata (edge, confidence, etc.)
            
        Returns:
            Prediction ID for later updating with closing line
        """
        if prediction_time is None:
            prediction_time = datetime.now()
        
        prediction_id = f"{game_date.isoformat()}_{home_team}_{away_team}_{bet_type}_{prediction_time.timestamp()}"
        
        record = {
            "prediction_id": prediction_id,
            "game_date": game_date.isoformat(),
            "home_team": home_team,
            "away_team": away_team,
            "bet_type": bet_type,
            "model_line": model_line,
            "opening_line": opening_line,
            "closing_line": None,
            "prediction_time": prediction_time.isoformat(),
            "closing_time": None,
            "clv": None,  # Closing Line Value (model_line - closing_line)
            "clv_advantage": None,  # Positive = model beat closing line
            "metadata": metadata or {},
        }
        
        self.clv_data.append(record)
        self._save_data()
        
        return prediction_id
    
    def update_closing_line(
        self,
        prediction_id: str,
        closing_line: float,
        closing_time: Optional[datetime] = None,
    ) -> bool:
        """
        Update a prediction record with the closing line.
        
        Args:
            prediction_id: ID from record_prediction()
            closing_line: The closing line from sportsbooks
            closing_time: When the line closed (defaults to now)
            
        Returns:
            True if updated successfully, False if prediction_id not found
        """
        if closing_time is None:
            closing_time = datetime.now()
        
        for record in self.clv_data:
            if record["prediction_id"] == prediction_id:
                record["closing_line"] = closing_line
                record["closing_time"] = closing_time.isoformat()
                
                # Calculate CLV
                model_line = record["model_line"]
                clv = model_line - closing_line
                record["clv"] = clv
                
                # CLV advantage: positive means model beat closing line
                # For spreads: if model predicted -3.5 and closing was -4.5, CLV = +1.0 (advantage)
                # For totals: if model predicted 220 and closing was 218, CLV = +2.0 (advantage)
                record["clv_advantage"] = clv
                
                self._save_data()
                return True
        
        return False
    
    def get_clv_stats(
        self,
        bet_type: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Calculate CLV statistics.
        
        Args:
            bet_type: Filter by bet type (optional)
            start_date: Filter by start date (optional)
            end_date: Filter by end date (optional)
            
        Returns:
            Dict with CLV statistics
        """
        # Filter records
        filtered = self.clv_data.copy()
        
        if bet_type:
            filtered = [r for r in filtered if r["bet_type"] == bet_type]
        
        if start_date:
            filtered = [r for r in filtered if date.fromisoformat(r["game_date"]) >= start_date]
        
        if end_date:
            filtered = [r for r in filtered if date.fromisoformat(r["game_date"]) <= end_date]
        
        # Only include records with closing lines
        completed = [r for r in filtered if r["closing_line"] is not None]
        
        if len(completed) == 0:
            return {
                "n_predictions": len(filtered),
                "n_completed": 0,
                "error": "No completed predictions with closing lines"
            }
        
        clv_values = [r["clv"] for r in completed]
        clv_advantages = [r["clv_advantage"] for r in completed]
        
        # Calculate statistics
        avg_clv = sum(clv_values) / len(clv_values)
        avg_advantage = sum(clv_advantages) / len(clv_advantages)
        
        # Count how many times model beat closing line
        beats_closing = sum(1 for adv in clv_advantages if adv > 0)
        beat_rate = beats_closing / len(completed) if completed else 0
        
        # For spreads: positive CLV means model liked the favorite more (or underdog less)
        # For totals: positive CLV means model predicted higher total
        # A model that consistently beats closing lines is profitable
        
        return {
            "n_predictions": len(filtered),
            "n_completed": len(completed),
            "completion_rate": len(completed) / len(filtered) if filtered else 0,
            "avg_clv": avg_clv,
            "avg_clv_advantage": avg_advantage,
            "beat_closing_rate": beat_rate,
            "n_beats_closing": beats_closing,
            "min_clv": min(clv_values),
            "max_clv": max(clv_values),
            "std_clv": pd.Series(clv_values).std() if len(clv_values) > 1 else 0,
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all CLV data to a pandas DataFrame."""
        if not self.clv_data:
            return pd.DataFrame()
        
        return pd.DataFrame(self.clv_data)
    
    def get_recent_predictions(
        self,
        n: int = 50,
        bet_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get most recent predictions.
        
        Args:
            n: Number of predictions to return
            bet_type: Filter by bet type (optional)
            
        Returns:
            List of prediction records, sorted by prediction_time (newest first)
        """
        filtered = self.clv_data.copy()
        
        if bet_type:
            filtered = [r for r in filtered if r["bet_type"] == bet_type]
        
        # Sort by prediction time (newest first)
        filtered.sort(
            key=lambda x: x.get("prediction_time", ""),
            reverse=True
        )
        
        return filtered[:n]
