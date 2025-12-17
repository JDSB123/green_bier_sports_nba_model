#!/usr/bin/env python3
"""
Analyze ROI performance and determine if spreads ROI is acceptable.

Calculates:
- Break-even accuracy for -110 odds
- Expected ROI from accuracy
- Statistical significance (t-test)
- High-confidence filter impact
- Required sample size for confidence
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings


def break_even_accuracy(odds: float = -110) -> float:
    """Calculate break-even accuracy for given odds."""
    # For -110: win $100/$110 = $0.909, lose $1
    # Solve: p * 0.909 - (1-p) * 1 = 0
    # p * 0.909 - 1 + p = 0
    # p * 1.909 = 1
    # p = 1 / 1.909 ≈ 0.5238
    if odds < 0:
        win_amount = 100 / abs(odds)
        return 1 / (1 + win_amount)
    else:
        return odds / (100 + odds)


def expected_roi(accuracy: float, odds: float = -110) -> float:
    """Calculate expected ROI from accuracy."""
    if odds < 0:
        win_amount = 100 / abs(odds)
    else:
        win_amount = odds / 100
    
    return accuracy * win_amount - (1 - accuracy) * 1.0


def analyze_spreads_roi(
    accuracy: float,
    roi: float,
    roi_std: float,
    n_samples: int,
    high_conf_accuracy: float = None,
    high_conf_roi: float = None,
    high_conf_n: int = None,
) -> dict:
    """Comprehensive analysis of spreads ROI."""
    be_accuracy = break_even_accuracy(-110)
    expected = expected_roi(accuracy, -110)
    
    results = {
        "all_bets": {},
        "high_confidence": {},
        "recommendations": [],
    }
    
    # All bets analysis
    results["all_bets"] = {
        "accuracy": accuracy,
        "roi": roi,
        "roi_std": roi_std,
        "n_samples": n_samples,
        "break_even_accuracy": be_accuracy,
        "margin_over_break_even": accuracy - be_accuracy,
        "expected_roi_from_accuracy": expected,
        "roi_vs_expected_diff": roi - expected,
    }
    
    # Statistical significance
    if roi_std > 0 and n_samples > 0:
        # Standard error of mean
        se = roi_std / np.sqrt(n_samples)
        # t-statistic for H0: ROI = 0
        t_stat = roi / se if se > 0 else 0
        # Degrees of freedom
        df = n_samples - 1
        # p-value (one-tailed: is ROI significantly > 0?)
        p_value = 1 - stats.t.cdf(t_stat, df) if t_stat > 0 else stats.t.cdf(t_stat, df)
        
        # 95% confidence interval
        t_critical = stats.t.ppf(0.975, df)
        ci_lower = roi - t_critical * se
        ci_upper = roi + t_critical * se
        
        results["all_bets"]["statistical_significance"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant_at_5pct": p_value < 0.05,
            "significant_at_1pct": p_value < 0.01,
            "ci_95pct_lower": ci_lower,
            "ci_95pct_upper": ci_upper,
        }
        
        # Power analysis: what sample size needed for 80% power?
        # Effect size = roi / roi_std
        effect_size = abs(roi) / roi_std if roi_std > 0 else 0
        if effect_size > 0:
            # Using Cohen's d, assuming we want to detect if ROI > 0
            from scipy.stats import norm
            z_alpha = norm.ppf(0.95)  # One-tailed 5% significance
            z_beta = norm.ppf(0.80)   # 80% power
            n_required = ((z_alpha + z_beta) / effect_size) ** 2
            results["all_bets"]["power_analysis"] = {
                "effect_size": effect_size,
                "n_required_for_80pct_power": int(n_required),
                "current_sample_size": n_samples,
                "has_sufficient_power": n_samples >= n_required,
            }
    
    # High-confidence analysis
    if high_conf_accuracy is not None and high_conf_roi is not None and high_conf_n is not None:
        hc_expected = expected_roi(high_conf_accuracy, -110)
        results["high_confidence"] = {
            "accuracy": high_conf_accuracy,
            "roi": high_conf_roi,
            "n_samples": high_conf_n,
            "pct_of_all_bets": high_conf_n / n_samples if n_samples > 0 else 0,
            "expected_roi_from_accuracy": hc_expected,
            "improvement_over_all": high_conf_roi - roi,
            "improvement_pct": ((high_conf_roi - roi) / abs(roi) * 100) if roi != 0 else 0,
        }
    
    # Recommendations
    recommendations = []
    
    if accuracy < be_accuracy:
        recommendations.append(f"[WARN] ACCURACY BELOW BREAK-EVEN ({accuracy:.1%} < {be_accuracy:.1%})")
        recommendations.append("   Model is NOT profitable - needs improvement")
    elif accuracy - be_accuracy < 0.02:
        recommendations.append(f"[WARN] MARGIN IS TIGHT ({accuracy - be_accuracy:.1%} over break-even)")
        recommendations.append("   Consider higher confidence threshold to improve ROI")
    else:
        recommendations.append(f"[OK] Accuracy is {accuracy - be_accuracy:.1%} above break-even")
        recommendations.append(f"   This should generate positive ROI")
    
    if roi > 0:
        if "statistical_significance" in results["all_bets"]:
            sig = results["all_bets"]["statistical_significance"]
            if sig["significant_at_1pct"]:
                recommendations.append(f"[OK] ROI is statistically significant (p < 0.01)")
            elif sig["significant_at_5pct"]:
                recommendations.append(f"[OK] ROI is statistically significant (p < 0.05)")
            else:
                recommendations.append(f"[WARN] ROI not statistically significant (p = {sig['p_value']:.3f})")
                recommendations.append("   Need more samples or higher effect size")
        
        # Compare to expected
        if abs(roi - expected) < 0.01:
            recommendations.append(f"[OK] ROI ({roi:.1%}) matches expected from accuracy ({expected:.1%})")
        elif roi < expected * 0.8:
            recommendations.append(f"[WARN] ROI ({roi:.1%}) is below expected ({expected:.1%})")
            recommendations.append("   May indicate calibration issues - check probability estimates")
        
        # High-confidence filter
        if high_conf_roi and high_conf_roi > roi * 1.2:
            recommendations.append(f"[OK] HIGH-CONFIDENCE FILTER IMPROVES ROI by {high_conf_roi - roi:.1%}")
            recommendations.append(f"   Use threshold (e.g., >60% or <40% prob) to focus on best bets")
            recommendations.append(f"   Trade-off: {high_conf_n} bets ({high_conf_n/n_samples:.1%} of all)")
    else:
        recommendations.append(f"[FAIL] NEGATIVE ROI ({roi:.1%}) - Model is losing money")
        recommendations.append("   Do NOT use for live betting without improvement")
    
    # Industry benchmarks
    recommendations.append("\n[INDUSTRY CONTEXT]:")
    recommendations.append(f"   Professional bettors target: 55-57% accuracy on spreads")
    recommendations.append(f"   Your model: {accuracy:.1%} accuracy")
    recommendations.append(f"   Vegas edge (vig): ~4.55% (breakeven at 52.38%)")
    recommendations.append(f"   Your edge: {roi:.1%} ROI")
    
    results["recommendations"] = recommendations
    
    return results


def print_roi_analysis(results: dict, model_name: str = "Spreads Model"):
    """Pretty print ROI analysis."""
    print("\n" + "=" * 70)
    print(f"ROI ANALYSIS: {model_name.upper()}")
    print("=" * 70)
    
    ab = results["all_bets"]
    print("\n[ALL BETS] Performance:")
    print(f"   Accuracy:           {ab['accuracy']:.1%}")
    print(f"   ROI:                {ab['roi']:+.1%}")
    print(f"   Break-even accuracy:{ab['break_even_accuracy']:.1%}")
    print(f"   Margin over BE:     {ab['margin_over_break_even']:+.1%}")
    print(f"   Expected ROI:       {ab['expected_roi_from_accuracy']:+.1%}")
    print(f"   Sample size:        {ab['n_samples']:,} games")
    
    if "statistical_significance" in ab:
        sig = ab["statistical_significance"]
        print(f"\n   Statistical Test (H0: ROI = 0):")
        print(f"   t-statistic:       {sig['t_statistic']:.3f}")
        print(f"   p-value:           {sig['p_value']:.4f}")
        print(f"   Significant?       {'YES [OK]' if sig['significant_at_5pct'] else 'NO [WARN]'}")
        print(f"   95% CI:            [{sig['ci_95pct_lower']:+.1%}, {sig['ci_95pct_upper']:+.1%}]")
        
        if "power_analysis" in ab:
            pa = ab["power_analysis"]
            print(f"\n   Power Analysis:")
            print(f"   Effect size:       {pa['effect_size']:.3f}")
            print(f"   N required (80%):  {pa['n_required_for_80pct_power']:,}")
            print(f"   Has sufficient?    {'YES [OK]' if pa['has_sufficient_power'] else 'NO [WARN]'}")
    
    if results["high_confidence"]:
        hc = results["high_confidence"]
        print(f"\n[HIGH-CONF] Bets (>60% or <40% prob):")
        print(f"   Accuracy:           {hc['accuracy']:.1%}")
        print(f"   ROI:                {hc['roi']:+.1%}")
        print(f"   Improvement:        {hc['improvement_over_all']:+.1%} ({hc['improvement_pct']:+.0f}%)")
        print(f"   Sample size:        {hc['n_samples']:,} ({hc['pct_of_all_bets']:.1%} of all)")
    
    print("\n[RECOMMENDATIONS]:")
    for rec in results["recommendations"]:
        print(f"   {rec}")
    
    print("=" * 70 + "\n")


def main():
    """Analyze spreads ROI from model comparison results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ROI performance")
    parser.add_argument(
        "--target",
        choices=["spreads", "totals"],
        default="spreads",
        help="Which market to analyze",
    )
    parser.add_argument(
        "--model-type",
        choices=["regression", "logistic", "gradient_boosting"],
        default="regression",
        help="Which model type to analyze",
    )
    args = parser.parse_args()
    
    # Load comparison results
    comp_path = os.path.join(
        settings.data_processed_dir, 
        f"model_comparison_{args.target}.csv"
    )
    
    if not os.path.exists(comp_path):
        print(f"Comparison results not found: {comp_path}")
        print("Run: python scripts/compare_models.py first")
        return 1
    
    df = pd.read_csv(comp_path)
    row = df[df["model_type"] == args.model_type]
    
    if row.empty:
        print(f"Model type '{args.model_type}' not found in results")
        return 1
    
    row = row.iloc[0]
    
    # Parse results (high-confidence data not in CSV, need to reconstruct)
    # For now, we'll use approximations
    accuracy = row["accuracy_mean"]
    roi = row["roi_mean"]
    roi_std = row["roi_std"]
    n_samples = int(row["total_samples"])
    high_conf_acc = row.get("high_conf_acc_mean", None)
    high_conf_roi = row.get("high_conf_roi_mean", None)
    
    # Estimate high-conf n from accuracy difference
    if pd.notna(high_conf_acc) and pd.notna(high_conf_roi):
        # Rough estimate: high-conf typically 20-40% of bets
        high_conf_n = int(n_samples * 0.3)  # Approximate
    else:
        high_conf_acc = high_conf_roi = high_conf_n = None
    
    results = analyze_spreads_roi(
        accuracy=accuracy,
        roi=roi,
        roi_std=roi_std,
        n_samples=n_samples,
        high_conf_accuracy=high_conf_acc,
        high_conf_roi=high_conf_roi,
        high_conf_n=high_conf_n,
    )
    
    model_name = f"{args.target.title()} Model ({args.model_type})"
    print_roi_analysis(results, model_name)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

