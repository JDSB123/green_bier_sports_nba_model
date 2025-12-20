#!/usr/bin/env python3
"""
Model Validation and Calibration Script
========================================
Tests model predictions against historical data and provides calibration metrics.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelValidator:
    """Validates and calibrates betting model predictions."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.results = []
        
    def calculate_calibration_metrics(
        self, 
        predictions: List[Dict],
        actuals: List[Dict]
    ) -> Dict:
        """
        Calculate model calibration metrics.
        
        Returns dict with:
        - brier_score: Mean squared difference between predicted prob and outcome
        - log_loss: Logarithmic loss
        - calibration_slope: Slope of calibration curve (ideal = 1.0)
        - expected_calibration_error: ECE metric
        """
        if not predictions or not actuals:
            return {}
            
        # Convert to arrays
        pred_probs = np.array([p.get("win_probability", 0.5) for p in predictions])
        outcomes = np.array([1 if a.get("won") else 0 for a in actuals])
        
        # Brier Score
        brier_score = np.mean((pred_probs - outcomes) ** 2)
        
        # Log Loss
        epsilon = 1e-7  # Small value to avoid log(0)
        log_loss = -np.mean(
            outcomes * np.log(pred_probs + epsilon) + 
            (1 - outcomes) * np.log(1 - pred_probs + epsilon)
        )
        
        # Calibration plot data (binned)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_accuracies.append(np.mean(outcomes[mask]))
                bin_counts.append(np.sum(mask))
        
        # Calculate calibration slope
        if len(bin_centers) > 1:
            z = np.polyfit(bin_centers, bin_accuracies, 1)
            calibration_slope = z[0]
            calibration_intercept = z[1]
        else:
            calibration_slope = 1.0
            calibration_intercept = 0.0
        
        # Expected Calibration Error (ECE)
        ece = 0
        total_samples = len(predictions)
        for i in range(len(bin_centers)):
            bin_confidence = bin_centers[i]
            bin_accuracy = bin_accuracies[i]
            bin_size = bin_counts[i]
            ece += (bin_size / total_samples) * abs(bin_accuracy - bin_confidence)
        
        return {
            "brier_score": float(brier_score),
            "log_loss": float(log_loss),
            "calibration_slope": float(calibration_slope),
            "calibration_intercept": float(calibration_intercept),
            "expected_calibration_error": float(ece),
            "n_predictions": len(predictions),
            "win_rate": float(np.mean(outcomes)),
            "avg_confidence": float(np.mean(pred_probs)),
        }
    
    def validate_spreads(
        self,
        predictions: List[Dict],
        actuals: List[Dict]
    ) -> Dict:
        """Validate spread predictions."""
        metrics = {
            "full_game": {},
            "first_half": {},
        }
        
        # Full game spreads
        fg_preds = []
        fg_actuals = []
        
        for pred, actual in zip(predictions, actuals):
            if "spread_prediction" in pred and "actual_margin" in actual:
                edge = pred.get("spread_edge", 0)
                if abs(edge) >= 2.0:  # Only count bets we would make
                    fg_preds.append({
                        "win_probability": pred.get("spread_win_prob", 0.5),
                        "edge": edge,
                        "pick": pred.get("spread_pick"),
                    })
                    
                    # Determine if bet won
                    actual_margin = actual["actual_margin"]
                    market_spread = pred.get("market_spread", 0)
                    
                    if pred["spread_pick"] == "home":
                        won = actual_margin > -market_spread
                    else:
                        won = actual_margin < -market_spread
                    
                    fg_actuals.append({"won": won})
        
        metrics["full_game"] = self.calculate_calibration_metrics(fg_preds, fg_actuals)
        
        # First half spreads (if available)
        fh_preds = []
        fh_actuals = []
        
        for pred, actual in zip(predictions, actuals):
            if "fh_spread_prediction" in pred and "fh_actual_margin" in actual:
                edge = pred.get("fh_spread_edge", 0)
                if abs(edge) >= 1.5:  # Lower threshold for FH
                    fh_preds.append({
                        "win_probability": pred.get("fh_spread_win_prob", 0.5),
                        "edge": edge,
                        "pick": pred.get("fh_spread_pick"),
                    })
                    
                    # Determine if bet won
                    actual_margin = actual["fh_actual_margin"]
                    market_spread = pred.get("fh_market_spread", 0)
                    
                    if pred["fh_spread_pick"] == "home":
                        won = actual_margin > -market_spread
                    else:
                        won = actual_margin < -market_spread
                    
                    fh_actuals.append({"won": won})
        
        if fh_preds:
            metrics["first_half"] = self.calculate_calibration_metrics(fh_preds, fh_actuals)
        
        return metrics
    
    def validate_totals(
        self,
        predictions: List[Dict],
        actuals: List[Dict]
    ) -> Dict:
        """Validate totals predictions."""
        metrics = {
            "full_game": {},
            "first_half": {},
        }
        
        # Full game totals
        fg_preds = []
        fg_actuals = []
        
        for pred, actual in zip(predictions, actuals):
            if "total_prediction" in pred and "actual_total" in actual:
                edge = pred.get("total_edge", 0)
                if abs(edge) >= 3.0:  # Only count bets we would make
                    fg_preds.append({
                        "win_probability": pred.get("total_win_prob", 0.5),
                        "edge": edge,
                        "pick": pred.get("total_pick"),
                    })
                    
                    # Determine if bet won
                    actual_total = actual["actual_total"]
                    market_total = pred.get("market_total", 0)
                    
                    if pred["total_pick"] == "OVER":
                        won = actual_total > market_total
                    else:
                        won = actual_total < market_total
                    
                    fg_actuals.append({"won": won})
        
        metrics["full_game"] = self.calculate_calibration_metrics(fg_preds, fg_actuals)
        
        # First half totals
        fh_preds = []
        fh_actuals = []
        
        for pred, actual in zip(predictions, actuals):
            if "fh_total_prediction" in pred and "fh_actual_total" in actual:
                edge = pred.get("fh_total_edge", 0)
                if abs(edge) >= 2.0:  # Lower threshold for FH
                    fh_preds.append({
                        "win_probability": pred.get("fh_total_win_prob", 0.5),
                        "edge": edge,
                        "pick": pred.get("fh_total_pick"),
                    })
                    
                    # Determine if bet won
                    actual_total = actual["fh_actual_total"]
                    market_total = pred.get("fh_market_total", 0)
                    
                    if pred["fh_total_pick"] == "OVER":
                        won = actual_total > market_total
                    else:
                        won = actual_total < market_total
                    
                    fh_actuals.append({"won": won})
        
        if fh_preds:
            metrics["first_half"] = self.calculate_calibration_metrics(fh_preds, fh_actuals)
        
        return metrics
    
    def calculate_roi(
        self,
        predictions: List[Dict],
        actuals: List[Dict],
        bet_type: str = "spread"
    ) -> Dict:
        """
        Calculate ROI for different betting strategies.
        
        Returns ROI metrics for various confidence thresholds.
        """
        results = []
        
        for pred, actual in zip(predictions, actuals):
            if bet_type == "spread":
                if "spread_pick" not in pred or "actual_margin" not in actual:
                    continue
                    
                edge = pred.get("spread_edge", 0)
                win_prob = pred.get("spread_win_prob", 0.5)
                confidence = pred.get("spread_confidence", 0)
                
                # Determine if bet won
                actual_margin = actual["actual_margin"]
                market_spread = pred.get("market_spread", 0)
                
                if pred["spread_pick"] == "home":
                    won = actual_margin > -market_spread
                else:
                    won = actual_margin < -market_spread
                
            elif bet_type == "total":
                if "total_pick" not in pred or "actual_total" not in actual:
                    continue
                    
                edge = pred.get("total_edge", 0)
                win_prob = pred.get("total_win_prob", 0.5)
                confidence = pred.get("total_confidence", 0)
                
                # Determine if bet won
                actual_total = actual["actual_total"]
                market_total = pred.get("market_total", 0)
                
                if pred["total_pick"] == "OVER":
                    won = actual_total > market_total
                else:
                    won = actual_total < market_total
            
            else:
                continue
            
            results.append({
                "edge": edge,
                "win_prob": win_prob,
                "confidence": confidence,
                "won": won,
                "odds": pred.get("odds", -110),
            })
        
        # Calculate ROI for different strategies
        roi_metrics = {}
        
        # All bets
        all_bets = [r for r in results if abs(r["edge"]) >= 2.0]
        if all_bets:
            wins = sum(1 for r in all_bets if r["won"])
            roi = self._calculate_roi_from_record(wins, len(all_bets), -110)
            roi_metrics["all_bets"] = {
                "count": len(all_bets),
                "win_rate": wins / len(all_bets),
                "roi": roi,
            }
        
        # High confidence (>60% or <40% win prob)
        high_conf = [r for r in results if r["win_prob"] > 0.60 or r["win_prob"] < 0.40]
        if high_conf:
            wins = sum(1 for r in high_conf if r["won"])
            roi = self._calculate_roi_from_record(wins, len(high_conf), -110)
            roi_metrics["high_confidence"] = {
                "count": len(high_conf),
                "win_rate": wins / len(high_conf),
                "roi": roi,
            }
        
        # Very high confidence (>65% or <35% win prob)  
        very_high = [r for r in results if r["win_prob"] > 0.65 or r["win_prob"] < 0.35]
        if very_high:
            wins = sum(1 for r in very_high if r["won"])
            roi = self._calculate_roi_from_record(wins, len(very_high), -110)
            roi_metrics["very_high_confidence"] = {
                "count": len(very_high),
                "win_rate": wins / len(very_high),
                "roi": roi,
            }
        
        # Large edges (>4 points)
        large_edge = [r for r in results if abs(r["edge"]) >= 4.0]
        if large_edge:
            wins = sum(1 for r in large_edge if r["won"])
            roi = self._calculate_roi_from_record(wins, len(large_edge), -110)
            roi_metrics["large_edge"] = {
                "count": len(large_edge),
                "win_rate": wins / len(large_edge),
                "roi": roi,
            }
        
        return roi_metrics
    
    def _calculate_roi_from_record(
        self,
        wins: int,
        total_bets: int,
        avg_odds: int = -110
    ) -> float:
        """Calculate ROI from win/loss record."""
        if total_bets == 0:
            return 0.0
        
        losses = total_bets - wins
        
        # Calculate profit/loss
        if avg_odds < 0:
            # Favorite odds
            win_amount = 100 / abs(avg_odds) * 100  # Profit on $100 bet
            loss_amount = 100  # Loss on $100 bet
        else:
            # Underdog odds
            win_amount = avg_odds  # Profit on $100 bet
            loss_amount = 100  # Loss on $100 bet
        
        total_profit = (wins * win_amount) - (losses * loss_amount)
        total_wagered = total_bets * 100
        
        return (total_profit / total_wagered) * 100
    
    def generate_calibration_report(
        self,
        spread_metrics: Dict,
        total_metrics: Dict,
        roi_metrics: Dict
    ) -> str:
        """Generate comprehensive calibration report."""
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL CALIBRATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        
        # Spread Calibration
        lines.append("SPREAD PREDICTIONS:")
        lines.append("-" * 40)
        
        if spread_metrics.get("full_game"):
            fg = spread_metrics["full_game"]
            lines.append("Full Game Spreads:")
            lines.append(f"  Brier Score: {fg['brier_score']:.4f} (lower is better, 0 = perfect)")
            lines.append(f"  Log Loss: {fg['log_loss']:.4f}")
            lines.append(f"  Calibration Slope: {fg['calibration_slope']:.3f} (ideal = 1.000)")
            lines.append(f"  Calibration Intercept: {fg['calibration_intercept']:.3f} (ideal = 0.000)")
            lines.append(f"  Expected Calibration Error: {fg['expected_calibration_error']:.3f}")
            lines.append(f"  Win Rate: {fg['win_rate']:.1%}")
            lines.append(f"  Avg Confidence: {fg['avg_confidence']:.1%}")
            lines.append("")
        
        if spread_metrics.get("first_half"):
            fh = spread_metrics["first_half"]
            lines.append("First Half Spreads:")
            lines.append(f"  Brier Score: {fh['brier_score']:.4f}")
            lines.append(f"  Log Loss: {fh['log_loss']:.4f}")
            lines.append(f"  Calibration Slope: {fh['calibration_slope']:.3f}")
            lines.append(f"  Expected Calibration Error: {fh['expected_calibration_error']:.3f}")
            lines.append(f"  Win Rate: {fh['win_rate']:.1%}")
            lines.append("")
        
        # Total Calibration
        lines.append("TOTAL PREDICTIONS:")
        lines.append("-" * 40)
        
        if total_metrics.get("full_game"):
            fg = total_metrics["full_game"]
            lines.append("Full Game Totals:")
            lines.append(f"  Brier Score: {fg['brier_score']:.4f}")
            lines.append(f"  Calibration Slope: {fg['calibration_slope']:.3f}")
            lines.append(f"  Win Rate: {fg['win_rate']:.1%}")
            lines.append("")
        
        # ROI Analysis
        lines.append("ROI ANALYSIS:")
        lines.append("-" * 40)
        
        for strategy, metrics in roi_metrics.items():
            if "spread" in strategy:
                bet_type = "Spreads"
            elif "total" in strategy:
                bet_type = "Totals"
            else:
                bet_type = ""
            
            for threshold, data in metrics.items():
                lines.append(f"{bet_type} - {threshold.replace('_', ' ').title()}:")
                lines.append(f"  Bets: {data['count']}")
                lines.append(f"  Win Rate: {data['win_rate']:.1%}")
                lines.append(f"  ROI: {data['roi']:+.2f}%")
                lines.append("")
        
        # Calibration Assessment
        lines.append("CALIBRATION ASSESSMENT:")
        lines.append("-" * 40)
        
        # Check if model is well-calibrated
        if spread_metrics.get("full_game"):
            slope = spread_metrics["full_game"]["calibration_slope"]
            ece = spread_metrics["full_game"]["expected_calibration_error"]
            
            if 0.9 <= slope <= 1.1 and ece < 0.05:
                lines.append("✅ Spread model is WELL CALIBRATED")
            elif 0.8 <= slope <= 1.2 and ece < 0.10:
                lines.append("⚠️  Spread model is MODERATELY CALIBRATED")
            else:
                lines.append("❌ Spread model needs RECALIBRATION")
                if slope < 0.9:
                    lines.append("   - Model is OVERCONFIDENT (reduce confidence)")
                elif slope > 1.1:
                    lines.append("   - Model is UNDERCONFIDENT (increase confidence)")
        
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        lines.append("-" * 40)
        
        # Generate recommendations based on metrics
        recommendations = []
        
        if spread_metrics.get("full_game"):
            fg = spread_metrics["full_game"]
            if fg["calibration_slope"] < 0.9:
                recommendations.append(
                    "• Reduce spread confidence multipliers by "
                    f"{(1 - fg['calibration_slope']) * 100:.0f}%"
                )
            elif fg["calibration_slope"] > 1.1:
                recommendations.append(
                    "• Increase spread confidence multipliers by "
                    f"{(fg['calibration_slope'] - 1) * 100:.0f}%"
                )
        
        if spread_metrics.get("first_half"):
            fh = spread_metrics["first_half"]
            if fh["expected_calibration_error"] > 0.10:
                recommendations.append(
                    "• First half model has high calibration error - "
                    "consider adding FH-specific features"
                )
        
        # ROI recommendations
        if roi_metrics:
            best_strategy = None
            best_roi = -100
            
            for strategy, metrics in roi_metrics.items():
                for threshold, data in metrics.items():
                    if data["roi"] > best_roi and data["count"] >= 20:
                        best_roi = data["roi"]
                        best_strategy = f"{strategy} - {threshold}"
            
            if best_strategy and best_roi > 0:
                recommendations.append(
                    f"• Best strategy: {best_strategy} with {best_roi:+.2f}% ROI"
                )
        
        if recommendations:
            for rec in recommendations:
                lines.append(rec)
        else:
            lines.append("• Model appears well-calibrated, no major adjustments needed")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    validator = ModelValidator()
    
    # Example predictions and actuals (would load from actual data)
    predictions = [
        {
            "spread_prediction": -5.5,
            "market_spread": -3.5,
            "spread_edge": 2.0,
            "spread_pick": "home",
            "spread_win_prob": 0.62,
            "spread_confidence": 0.65,
            "total_prediction": 225,
            "market_total": 220,
            "total_edge": 5.0,
            "total_pick": "OVER",
            "total_win_prob": 0.58,
            "fh_spread_prediction": -2.8,
            "fh_market_spread": -1.5,
            "fh_spread_edge": 1.3,
            "fh_spread_pick": "home",
            "fh_spread_win_prob": 0.55,
        },
        {
            "spread_prediction": 2.1,
            "market_spread": -1.5,
            "spread_edge": 3.6,
            "spread_pick": "away",
            "spread_win_prob": 0.68,
            "spread_confidence": 0.72,
            "total_prediction": 210,
            "market_total": 215,
            "total_edge": -5.0,
            "total_pick": "UNDER",
            "total_win_prob": 0.61,
        },
    ]
    
    actuals = [
        {
            "actual_margin": -8.0,  # Home won by 8
            "actual_total": 228,
            "fh_actual_margin": -4.0,
            "fh_actual_total": 112,
        },
        {
            "actual_margin": 3.0,  # Away won by 3
            "actual_total": 208,
        },
    ]
    
    # Validate spreads
    spread_metrics = validator.validate_spreads(predictions, actuals)
    
    # Validate totals
    total_metrics = validator.validate_totals(predictions, actuals)
    
    # Calculate ROI
    spread_roi = validator.calculate_roi(predictions, actuals, "spread")
    total_roi = validator.calculate_roi(predictions, actuals, "total")
    
    roi_metrics = {
        "spread": spread_roi,
        "total": total_roi,
    }
    
    # Generate report
    report = validator.generate_calibration_report(
        spread_metrics,
        total_metrics,
        roi_metrics
    )
    
    print(report)
