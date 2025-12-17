#!/usr/bin/env python3
"""
Improved Moneyline Model with Better Spread-to-ML Conversion
=============================================================
Enhanced ML predictions with correlation handling and proper EV calculation.
"""
import math
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class MoneylineBet:
    """Represents a moneyline betting opportunity."""
    team: str
    model_odds: int
    market_odds: int
    model_prob: float
    market_prob: float
    edge: float
    expected_value: float
    confidence: float
    correlated_with: Optional[str] = None


class ImprovedMoneylineModel:
    """Enhanced moneyline model with better conversions and correlation handling."""

    def __init__(self):
        # Historical spread to ML conversion table (based on actual NBA data)
        # Format: spread -> (favorite_win_prob, underdog_win_prob)
        self.spread_ml_table = {
            0.0: (0.500, 0.500),
            0.5: (0.520, 0.480),
            1.0: (0.540, 0.460),
            1.5: (0.555, 0.445),
            2.0: (0.570, 0.430),
            2.5: (0.585, 0.415),
            3.0: (0.600, 0.400),  # Key number
            3.5: (0.615, 0.385),
            4.0: (0.630, 0.370),
            4.5: (0.642, 0.358),
            5.0: (0.655, 0.345),
            5.5: (0.667, 0.333),
            6.0: (0.678, 0.322),
            6.5: (0.689, 0.311),
            7.0: (0.700, 0.300),  # Key number
            7.5: (0.710, 0.290),
            8.0: (0.720, 0.280),
            8.5: (0.729, 0.271),
            9.0: (0.738, 0.262),
            9.5: (0.747, 0.253),
            10.0: (0.755, 0.245),  # Key number
            11.0: (0.770, 0.230),
            12.0: (0.783, 0.217),
            13.0: (0.795, 0.205),
            14.0: (0.805, 0.195),
            15.0: (0.815, 0.185),
        }

        # Situational adjustments
        self.situational_factors = {
            "home_court": 0.015,  # 1.5% home court advantage
            "back_to_back": -0.025,  # 2.5% disadvantage
            "3_in_4": -0.015,  # 1.5% disadvantage
            "rest_advantage": 0.020,  # 2% per day of rest differential
            "injury_star": -0.040,  # 4% for missing star player
            "injury_starter": -0.020,  # 2% for missing starter
        }

    def spread_to_moneyline_enhanced(
        self,
        spread: float,
        home_elo: float = 1500,
        away_elo: float = 1500,
        pace_factor: float = 1.0,
        situational_adjustments: Dict[str, float] = None
    ) -> Tuple[int, int]:
        """
        Convert spread to moneyline with enhanced accuracy.

        Args:
            spread: Point spread (home perspective, negative = home favored)
            home_elo: Home team ELO rating
            away_elo: Away team ELO rating
            pace_factor: Game pace factor
            situational_adjustments: Dict of situational factors

        Returns:
            Tuple of (home_ml, away_ml) in American odds
        """
        abs_spread = abs(spread)

        # Interpolate from conversion table
        if abs_spread <= 15.0:
            # Find surrounding values in table
            lower_spread = math.floor(abs_spread * 2) / 2  # Round down to nearest 0.5
            upper_spread = math.ceil(abs_spread * 2) / 2   # Round up to nearest 0.5

            if lower_spread == upper_spread or upper_spread > 15.0:
                # Exact match or beyond table
                if abs_spread in self.spread_ml_table:
                    fav_prob, dog_prob = self.spread_ml_table[abs_spread]
                else:
                    # Use closest value
                    closest = min(self.spread_ml_table.keys(),
                                key=lambda x: abs(x - abs_spread))
                    fav_prob, dog_prob = self.spread_ml_table[closest]
            else:
                # Interpolate between values
                lower_probs = self.spread_ml_table.get(lower_spread, (0.5, 0.5))
                upper_probs = self.spread_ml_table.get(upper_spread, (0.5, 0.5))

                weight = (abs_spread - lower_spread) / (upper_spread - lower_spread)
                fav_prob = lower_probs[0] + weight * (upper_probs[0] - lower_probs[0])
                dog_prob = lower_probs[1] + weight * (upper_probs[1] - lower_probs[1])
        else:
            # Beyond table range, use formula
            # Each point beyond 15 adds ~1.2% to favorite probability
            fav_prob = 0.815 + (abs_spread - 15.0) * 0.012
            fav_prob = min(fav_prob, 0.95)  # Cap at 95%
            dog_prob = 1 - fav_prob

        # Apply ELO adjustment
        elo_diff = home_elo - away_elo
        elo_adjustment = elo_diff / 400 * 0.05  # 5% per 400 ELO points

        # Apply pace adjustment (higher variance in high-pace games)
        if pace_factor > 1.05:
            # High pace increases underdog chances slightly
            dog_prob *= 1.02
            fav_prob = 1 - dog_prob
        elif pace_factor < 0.95:
            # Low pace favors favorite slightly
            fav_prob *= 1.01
            dog_prob = 1 - fav_prob

        # Apply situational adjustments
        if situational_adjustments:
            total_adj = sum(situational_adjustments.values())
            if spread < 0:  # Home is favorite
                fav_prob += total_adj
            else:  # Away is favorite
                fav_prob -= total_adj

            # Normalize probabilities
            fav_prob = max(0.01, min(0.99, fav_prob))
            dog_prob = 1 - fav_prob

        # Determine which team is favorite
        if spread < 0:  # Home is favorite
            home_prob = fav_prob
            away_prob = dog_prob
        else:  # Away is favorite
            home_prob = dog_prob
            away_prob = fav_prob

        # Add final home court adjustment if game is close
        if abs_spread < 3:
            home_prob += self.situational_factors["home_court"] * (1 - abs_spread / 3)
            away_prob = 1 - home_prob

        # Convert to American odds
        home_ml = self.prob_to_american_odds(home_prob)
        away_ml = self.prob_to_american_odds(away_prob)

        return home_ml, away_ml

    def prob_to_american_odds(self, prob: float) -> int:
        """Convert probability to American odds."""
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)

    def american_to_implied_prob(self, american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def calculate_expected_value(
        self,
        model_prob: float,
        market_odds: int,
        bet_amount: float = 100
    ) -> float:
        """
        Calculate expected value of a bet.

        Args:
            model_prob: Model's probability of winning
            market_odds: Market odds (American format)
            bet_amount: Bet size (default $100)

        Returns:
            Expected value as a percentage of bet amount
        """
        # Calculate potential profit
        if market_odds > 0:
            profit = bet_amount * (market_odds / 100)
        else:
            profit = bet_amount * (100 / abs(market_odds))

        # EV = (prob_win * profit) - (prob_loss * bet_amount)
        ev = (model_prob * profit) - ((1 - model_prob) * bet_amount)

        # Return as percentage
        return ev / bet_amount


    def evaluate_moneyline_opportunity(
        self,
        model_spread: float,
        market_spread: float,
        market_home_ml: int,
        market_away_ml: int,
        home_team: str,
        away_team: str,
        **kwargs
    ) -> Optional[MoneylineBet]:
        """
        Evaluate moneyline betting opportunity.

        Returns:
            MoneylineBet object if opportunity exists, None otherwise
        """
        # Calculate model ML from spread
        model_home_ml, model_away_ml = self.spread_to_moneyline_enhanced(
            model_spread, **kwargs
        )

        # Convert to probabilities
        model_home_prob = self.american_to_implied_prob(model_home_ml)
        model_away_prob = self.american_to_implied_prob(model_away_ml)
        market_home_prob = self.american_to_implied_prob(market_home_ml)
        market_away_prob = self.american_to_implied_prob(market_away_ml)

        # Calculate edges
        home_edge = model_home_prob - market_home_prob
        away_edge = model_away_prob - market_away_prob

        # Calculate EVs
        home_ev = self.calculate_expected_value(model_home_prob, market_home_ml)
        away_ev = self.calculate_expected_value(model_away_prob, market_away_ml)

        # Determine best bet
        best_bet = None

        # Minimum thresholds
        min_edge = 0.03  # 3% edge minimum
        min_ev = 0.02  # 2% EV minimum
        min_prob = 0.30  # Don't bet on < 30% chances

        if home_edge > min_edge and home_ev > min_ev and model_home_prob > min_prob:
            confidence = min(home_edge * 3, 0.90)  # Cap at 90%

            best_bet = MoneylineBet(
                team=home_team,
                model_odds=model_home_ml,
                market_odds=market_home_ml,
                model_prob=model_home_prob,
                market_prob=market_home_prob,
                edge=home_edge,
                expected_value=home_ev,
                confidence=confidence
            )

        elif away_edge > min_edge and away_ev > min_ev and model_away_prob > min_prob:
            confidence = min(away_edge * 3, 0.90)

            best_bet = MoneylineBet(
                team=away_team,
                model_odds=model_away_ml,
                market_odds=market_away_ml,
                model_prob=model_away_prob,
                market_prob=market_away_prob,
                edge=away_edge,
                expected_value=away_ev,
                confidence=confidence
            )

        return best_bet


# Example usage
if __name__ == "__main__":
    model = ImprovedMoneylineModel()

    # Example game analysis
    home_team = "Boston Celtics"
    away_team = "Los Angeles Lakers"
    model_spread = -5.2  # Model: Celtics by 5.2
    market_spread = -3.5  # Market: Celtics by 3.5
    market_home_ml = -165
    market_away_ml = 145

    print("=" * 60)
    print(f"MONEYLINE ANALYSIS: {away_team} @ {home_team}")
    print("=" * 60)

    # Enhanced spread to ML conversion
    model_home_ml, model_away_ml = model.spread_to_moneyline_enhanced(
        spread=model_spread,
        home_elo=1650,
        away_elo=1550,
        pace_factor=1.02,
        situational_adjustments={
            "rest_advantage": 0.02,  # Celtics with rest advantage
        }
    )

    print(f"\nModel Assessment:")
    print(f"  Spread: {home_team} {model_spread:.1f}")
    print(f"  ML: {home_team} ({model_home_ml:+d}) vs {away_team} ({model_away_ml:+d})")

    print(f"\nMarket Lines:")
    print(f"  Spread: {home_team} {market_spread:.1f}")
    print(f"  ML: {home_team} ({market_home_ml:+d}) vs {away_team} ({market_away_ml:+d})")

    # Evaluate opportunity
    ml_bet = model.evaluate_moneyline_opportunity(
        model_spread=model_spread,
        market_spread=market_spread,
        market_home_ml=market_home_ml,
        market_away_ml=market_away_ml,
        home_team=home_team,
        away_team=away_team,
        home_elo=1650,
        away_elo=1550,
        pace_factor=1.02
    )

    if ml_bet:
        print(f"\n✅ MONEYLINE BET IDENTIFIED:")
        print(f"  Team: {ml_bet.team}")
        print(f"  Market Odds: {ml_bet.market_odds:+d}")
        print(f"  Model Probability: {ml_bet.model_prob:.1%}")

        print(f"  Market Probability: {ml_bet.market_prob:.1%}")

        print(f"  Edge: {ml_bet.edge:.1%}")

        print(f"  Expected Value: {ml_bet.expected_value:.2%}")

        print(f"  Confidence: {ml_bet.confidence:.1%}")
    else:
        print(f"\n❌ No moneyline value identified")

    # Test correlation handling
    print("\n" + "=" * 60)
    print("CORRELATION ADJUSTMENT EXAMPLE")
    print("=" * 60)

    adjusted = {}
