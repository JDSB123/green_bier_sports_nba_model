"""
Moneyline prediction logic for full game and first half.
"""
from typing import Dict, Any, Optional
import pandas as pd

from src.prediction.moneyline.filters import FGMoneylineFilter, FirstHalfMoneylineFilter
from src.prediction.confidence import calculate_confidence_from_probabilities


def american_odds_to_implied_prob(odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Implied probability (0-1)
    """
    if odds < 0:
        # Favorite (e.g., -200 = 200/(200+100) = 66.7%)
        return abs(odds) / (abs(odds) + 100)
    else:
        # Underdog (e.g., +150 = 100/(150+100) = 40%)
        return 100 / (odds + 100)


class MoneylinePredictor:
    """
    Moneyline predictor for full game and first half markets.

    Uses spread model win probabilities for moneyline predictions.
    """

    def __init__(
        self,
        model,
        feature_columns: list,
        fg_filter: Optional[FGMoneylineFilter] = None,
        first_half_filter: Optional[FirstHalfMoneylineFilter] = None,
    ):
        """
        Initialize moneyline predictor.

        Args:
            model: Trained spread model (provides win probabilities)
            feature_columns: List of feature column names
            fg_filter: FGMoneylineFilter instance (None = use defaults)
            first_half_filter: FirstHalfMoneylineFilter instance (None = use defaults)
        """
        self.model = model
        self.feature_columns = feature_columns

        # Initialize filters
        self.fg_filter = fg_filter or FGMoneylineFilter()
        self.first_half_filter = first_half_filter or FirstHalfMoneylineFilter()

    def predict_full_game(
        self,
        features: Dict[str, float],
        home_odds: Optional[int] = None,
        away_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate full game moneyline prediction.

        Args:
            features: Feature dictionary for the game
            home_odds: Home team American odds (e.g., -150)
            away_odds: Away team American odds (e.g., +130)

        Returns:
            Dictionary with:
                - home_win_prob: Probability home wins
                - away_win_prob: Probability away wins
                - predicted_winner: "home" or "away"
                - confidence: Probability of predicted winner
                - home_implied_prob: Implied probability from home odds
                - away_implied_prob: Implied probability from away odds
                - home_edge: home_win_prob - home_implied_prob
                - away_edge: away_win_prob - away_implied_prob
                - recommended_bet: "home", "away", or None
                - passes_filter: bool
                - filter_reason: str or None
        """
        # Create feature dataframe
        feature_df = pd.DataFrame([features])

        # Add missing features with 0
        missing = set(self.feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0

        # Align columns
        X = feature_df[self.feature_columns]

        # Get win probabilities from spread model
        # spread_proba[1] = home covers, spread_proba[0] = away covers
        # For moneyline, these are direct win probabilities
        spread_proba = self.model.predict_proba(X)[0]
        home_win_prob = float(spread_proba[1])
        away_win_prob = float(spread_proba[0])

        # Calculate confidence from probabilities using entropy-based approach
        # This accounts for uncertainty rather than using raw probability
        confidence = calculate_confidence_from_probabilities(home_win_prob, away_win_prob)

        # Determine predicted winner
        if home_win_prob > 0.5:
            predicted_winner = "home"
        else:
            predicted_winner = "away"

        # Calculate implied probabilities from odds
        home_implied_prob = None
        away_implied_prob = None
        home_edge = None
        away_edge = None

        if home_odds is not None:
            home_implied_prob = american_odds_to_implied_prob(home_odds)
            home_edge = home_win_prob - home_implied_prob

        if away_odds is not None:
            away_implied_prob = american_odds_to_implied_prob(away_odds)
            away_edge = away_win_prob - away_implied_prob

        # Determine recommended bet (highest value)
        recommended_bet = None
        bet_edge = 0
        bet_implied_prob = None

        if home_edge is not None and away_edge is not None:
            if home_edge > away_edge and home_edge > 0:
                recommended_bet = "home"
                bet_edge = home_edge
                bet_implied_prob = home_implied_prob
            elif away_edge > 0:
                recommended_bet = "away"
                bet_edge = away_edge
                bet_implied_prob = away_implied_prob
        elif home_edge is not None and home_edge > 0:
            recommended_bet = "home"
            bet_edge = home_edge
            bet_implied_prob = home_implied_prob
        elif away_edge is not None and away_edge > 0:
            recommended_bet = "away"
            bet_edge = away_edge
            bet_implied_prob = away_implied_prob

        # Apply filter to recommended bet
        passes_filter = True
        filter_reason = None

        if recommended_bet == "home":
            passes_filter, filter_reason = self.fg_filter.should_bet(
                predicted_prob=home_win_prob,
                implied_prob=bet_implied_prob,
            )
        elif recommended_bet == "away":
            passes_filter, filter_reason = self.fg_filter.should_bet(
                predicted_prob=away_win_prob,
                implied_prob=bet_implied_prob,
            )

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "predicted_winner": predicted_winner,
            "confidence": confidence,
            "home_implied_prob": home_implied_prob,
            "away_implied_prob": away_implied_prob,
            "home_edge": home_edge,
            "away_edge": away_edge,
            "recommended_bet": recommended_bet if passes_filter else None,
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }

    def predict_first_half(
        self,
        features: Dict[str, float],
        home_odds: Optional[int] = None,
        away_odds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate first half moneyline prediction.

        Note: Uses FG spread model for 1H win probabilities.

        Args:
            features: Feature dictionary for the game
            home_odds: Home team 1H American odds
            away_odds: Away team 1H American odds

        Returns:
            Dictionary with 1H moneyline prediction
        """
        # Use FG model for win probabilities
        feature_df = pd.DataFrame([features])
        missing = set(self.feature_columns) - set(feature_df.columns)
        for col in missing:
            feature_df[col] = 0
        X = feature_df[self.feature_columns]

        # Get win probabilities
        spread_proba = self.model.predict_proba(X)[0]
        home_win_prob = float(spread_proba[1])
        away_win_prob = float(spread_proba[0])

        # Calculate confidence from probabilities using entropy-based approach
        # This accounts for uncertainty rather than using raw probability
        confidence = calculate_confidence_from_probabilities(home_win_prob, away_win_prob)

        # Determine predicted winner
        if home_win_prob > 0.5:
            predicted_winner = "home"
        else:
            predicted_winner = "away"

        # Calculate implied probabilities from 1H odds
        home_implied_prob = None
        away_implied_prob = None
        home_edge = None
        away_edge = None

        if home_odds is not None:
            home_implied_prob = american_odds_to_implied_prob(home_odds)
            home_edge = home_win_prob - home_implied_prob

        if away_odds is not None:
            away_implied_prob = american_odds_to_implied_prob(away_odds)
            away_edge = away_win_prob - away_implied_prob

        # Determine recommended bet (highest value)
        recommended_bet = None
        bet_edge = 0
        bet_implied_prob = None

        if home_edge is not None and away_edge is not None:
            if home_edge > away_edge and home_edge > 0:
                recommended_bet = "home"
                bet_edge = home_edge
                bet_implied_prob = home_implied_prob
            elif away_edge > 0:
                recommended_bet = "away"
                bet_edge = away_edge
                bet_implied_prob = away_implied_prob
        elif home_edge is not None and home_edge > 0:
            recommended_bet = "home"
            bet_edge = home_edge
            bet_implied_prob = home_implied_prob
        elif away_edge is not None and away_edge > 0:
            recommended_bet = "away"
            bet_edge = away_edge
            bet_implied_prob = away_implied_prob

        # Apply 1H filter to recommended bet
        passes_filter = True
        filter_reason = None

        if recommended_bet == "home":
            passes_filter, filter_reason = self.first_half_filter.should_bet(
                predicted_prob=home_win_prob,
                implied_prob=bet_implied_prob,
            )
        elif recommended_bet == "away":
            passes_filter, filter_reason = self.first_half_filter.should_bet(
                predicted_prob=away_win_prob,
                implied_prob=bet_implied_prob,
            )

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "predicted_winner": predicted_winner,
            "confidence": confidence,
            "home_implied_prob": home_implied_prob,
            "away_implied_prob": away_implied_prob,
            "home_edge": home_edge,
            "away_edge": away_edge,
            "recommended_bet": recommended_bet if passes_filter else None,
            "passes_filter": passes_filter,
            "filter_reason": filter_reason,
        }
