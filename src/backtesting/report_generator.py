"""
Report generation for backtesting results.

Generates comprehensive HTML and Markdown reports with visualizations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.backtesting.performance_metrics import MarketPerformance
from src.backtesting.statistical_tests import StatisticalSummary
from src.backtesting.monte_carlo import MonteCarloResults

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate comprehensive backtest reports.
    
    Supports HTML and Markdown output with embedded metrics,
    statistical tests, and visualizations.
    """
    
    def __init__(
        self,
        output_dir: Path,
        version: str = "33.0.15.0",
    ):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to write reports to
            version: Model version for report header
        """
        self.output_dir = Path(output_dir)
        self.version = version
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_full_report(
        self,
        market_performances: Dict[str, MarketPerformance],
        statistical_summaries: Dict[str, StatisticalSummary],
        monte_carlo_results: Optional[Dict[str, MonteCarloResults]] = None,
        predictions_df: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Generate complete backtest report.
        
        Returns:
            Dict with paths to generated files
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        outputs = {}
        
        # Generate Markdown report
        md_path = self.output_dir / f"backtest_report_{timestamp}.md"
        self._write_markdown_report(
            md_path,
            market_performances,
            statistical_summaries,
            monte_carlo_results,
            config,
        )
        outputs["markdown"] = md_path
        
        # Generate HTML report
        html_path = self.output_dir / f"backtest_report_{timestamp}.html"
        self._write_html_report(
            html_path,
            market_performances,
            statistical_summaries,
            monte_carlo_results,
            config,
        )
        outputs["html"] = html_path
        
        # Generate JSON summary
        json_path = self.output_dir / f"backtest_summary_{timestamp}.json"
        self._write_json_summary(
            json_path,
            market_performances,
            statistical_summaries,
            monte_carlo_results,
            config,
        )
        outputs["json"] = json_path
        
        # Save predictions if provided
        if predictions_df is not None and len(predictions_df) > 0:
            csv_path = self.output_dir / f"predictions_{timestamp}.csv"
            predictions_df.to_csv(csv_path, index=False)
            outputs["predictions"] = csv_path
        
        logger.info(f"Generated reports in {self.output_dir}")
        
        return outputs
    
    def _write_markdown_report(
        self,
        path: Path,
        performances: Dict[str, MarketPerformance],
        statistics: Dict[str, StatisticalSummary],
        monte_carlo: Optional[Dict[str, MonteCarloResults]],
        config: Optional[Dict[str, Any]],
    ) -> None:
        """Write Markdown report."""
        lines = []
        
        # Header
        lines.append(f"# NBA Backtest Report v{self.version}")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Configuration
        if config:
            lines.append("## Configuration")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for key, value in config.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")
        
        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        
        # Create summary table
        lines.append("| Market | Bets | Accuracy | ROI | Sharpe | Significant? | Recommendation |")
        lines.append("|--------|------|----------|-----|--------|--------------|----------------|")
        
        for market, perf in performances.items():
            stat = statistics.get(market)
            sig = "Yes" if stat and stat.is_statistically_significant else "No"
            rec = stat.recommendation if stat else "N/A"
            
            # Truncate recommendation for table
            rec_short = rec[:30] + "..." if len(rec) > 30 else rec
            
            lines.append(
                f"| {market} | {perf.n_bets} | {perf.accuracy:.1%} | "
                f"{perf.roi:+.1%} | {perf.sharpe_ratio:.2f} | {sig} | {rec_short} |"
            )
        
        lines.append("")
        
        # Detailed Market Analysis
        lines.append("## Detailed Market Analysis")
        lines.append("")
        
        for market, perf in performances.items():
            lines.append(f"### {market.upper()}")
            lines.append("")
            
            # Basic metrics
            lines.append("#### Performance Metrics")
            lines.append("")
            lines.append(f"- **Total Bets:** {perf.n_bets}")
            lines.append(f"- **Accuracy:** {perf.accuracy:.2%} (95% CI: {perf.accuracy_ci[0]:.2%} - {perf.accuracy_ci[1]:.2%})")
            lines.append(f"- **ROI:** {perf.roi:+.2%} (95% CI: {perf.roi_ci[0]:+.2%} - {perf.roi_ci[1]:+.2%})")
            lines.append(f"- **Total Profit:** {perf.total_profit:+.1f} units")
            lines.append("")
            
            # Risk metrics
            lines.append("#### Risk Metrics")
            lines.append("")
            lines.append(f"- **Sharpe Ratio:** {perf.sharpe_ratio:.3f}")
            lines.append(f"- **Sortino Ratio:** {perf.sortino_ratio:.3f}")
            lines.append(f"- **Max Drawdown:** {perf.max_drawdown:.1f} units ({perf.max_drawdown_pct:.1%})")
            lines.append(f"- **Max Drawdown Duration:** {perf.max_drawdown_duration} bets")
            lines.append("")
            
            # Kelly
            lines.append("#### Kelly Criterion")
            lines.append("")
            lines.append(f"- **Full Kelly:** {perf.kelly_fraction:.1%}")
            lines.append(f"- **Half Kelly (Recommended):** {perf.half_kelly_fraction:.1%}")
            lines.append("")
            
            # High confidence
            if perf.high_conf_n > 0:
                lines.append("#### High Confidence Bets (>60%)")
                lines.append("")
                lines.append(f"- **Count:** {perf.high_conf_n}")
                lines.append(f"- **Accuracy:** {perf.high_conf_accuracy:.1%}")
                lines.append(f"- **ROI:** {perf.high_conf_roi:+.1%}")
                lines.append("")
            
            # Statistical tests
            stat = statistics.get(market)
            if stat:
                lines.append("#### Statistical Tests")
                lines.append("")
                lines.append(f"- **Statistically Significant:** {'Yes' if stat.is_statistically_significant else 'No'}")
                lines.append(f"- **Economically Significant:** {'Yes' if stat.is_economically_significant else 'No'}")
                lines.append("")
                lines.append("| Test | p-value | Significant? | Conclusion |")
                lines.append("|------|---------|--------------|------------|")
                for test in stat.tests:
                    sig = "Yes" if test.is_significant else "No"
                    lines.append(f"| {test.test_name} | {test.p_value:.4f} | {sig} | {test.conclusion[:50]}... |")
                lines.append("")
                
                lines.append(f"**Recommendation:** {stat.recommendation}")
                lines.append("")
        
        # Monte Carlo
        if monte_carlo:
            lines.append("## Monte Carlo Simulations")
            lines.append("")
            
            for market, mc in monte_carlo.items():
                lines.append(f"### {market.upper()}")
                lines.append("")
                lines.append(f"- **Simulations:** {mc.n_simulations:,}")
                lines.append(f"- **Bets per Simulation:** {mc.n_bets}")
                lines.append(f"- **Betting Strategy:** {mc.bet_size_type}")
                lines.append("")
                lines.append("#### Bankroll Projections")
                lines.append("")
                lines.append(f"- **Mean Final Bankroll:** ${mc.mean_final_bankroll:,.0f}")
                lines.append(f"- **Median Final Bankroll:** ${mc.median_final_bankroll:,.0f}")
                lines.append(f"- **5th Percentile:** ${mc.percentile_5:,.0f}")
                lines.append(f"- **95th Percentile:** ${mc.percentile_95:,.0f}")
                lines.append("")
                lines.append("#### Risk Metrics")
                lines.append("")
                lines.append(f"- **Probability of Ruin:** {mc.probability_of_ruin:.1%}")
                lines.append(f"- **Probability of Loss:** {mc.probability_of_loss:.1%}")
                lines.append(f"- **Probability of Profit:** {mc.probability_of_profit:.1%}")
                lines.append(f"- **Probability of Doubling:** {mc.probability_double:.1%}")
                lines.append("")
        
        # Footer
        lines.append("---")
        lines.append(f"*Report generated by NBA Backtest Engine v{self.version}*")
        
        # Write file
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Wrote Markdown report: {path}")
    
    def _write_html_report(
        self,
        path: Path,
        performances: Dict[str, MarketPerformance],
        statistics: Dict[str, StatisticalSummary],
        monte_carlo: Optional[Dict[str, MonteCarloResults]],
        config: Optional[Dict[str, Any]],
    ) -> None:
        """Write HTML report with styling."""
        html = []
        
        # HTML header with styling
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("<meta charset='UTF-8'>")
        html.append("<meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append(f"<title>NBA Backtest Report v{self.version}</title>")
        html.append("<style>")
        html.append("""
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }
            h2 { color: #16213e; margin-top: 30px; }
            h3 { color: #0f3460; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            th, td { border: 1px solid #ddd; padding: 12px 15px; text-align: left; }
            th { background: #1a1a2e; color: white; }
            tr:nth-child(even) { background: #f8f9fa; }
            tr:hover { background: #e8f4f8; }
            .positive { color: #28a745; font-weight: bold; }
            .negative { color: #dc3545; font-weight: bold; }
            .significant { background: #d4edda !important; }
            .not-significant { background: #f8d7da !important; }
            .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .metric { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
            .metric-value { font-size: 24px; font-weight: bold; color: #1a1a2e; }
            .metric-label { font-size: 12px; color: #666; margin-top: 5px; }
            .recommendation { padding: 15px; border-radius: 5px; margin: 15px 0; }
            .recommend { background: #d4edda; border-left: 4px solid #28a745; }
            .caution { background: #fff3cd; border-left: 4px solid #ffc107; }
            .not-recommend { background: #f8d7da; border-left: 4px solid #dc3545; }
            footer { margin-top: 40px; text-align: center; color: #666; font-size: 12px; }
        """)
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append(f"<h1>NBA Backtest Report v{self.version}</h1>")
        html.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Executive Summary
        html.append("<h2>Executive Summary</h2>")
        html.append("<table>")
        html.append("<tr><th>Market</th><th>Bets</th><th>Accuracy</th><th>ROI</th><th>Sharpe</th><th>Status</th></tr>")
        
        for market, perf in performances.items():
            stat = statistics.get(market)
            
            roi_class = "positive" if perf.roi > 0 else "negative"
            status_class = "significant" if stat and stat.is_statistically_significant else "not-significant"
            status = "SIGNIFICANT" if stat and stat.is_statistically_significant else "Not Significant"
            
            html.append(f"<tr class='{status_class}'>")
            html.append(f"<td><strong>{market}</strong></td>")
            html.append(f"<td>{perf.n_bets}</td>")
            html.append(f"<td>{perf.accuracy:.1%}</td>")
            html.append(f"<td class='{roi_class}'>{perf.roi:+.1%}</td>")
            html.append(f"<td>{perf.sharpe_ratio:.2f}</td>")
            html.append(f"<td>{status}</td>")
            html.append("</tr>")
        
        html.append("</table>")
        
        # Detailed Market Cards
        for market, perf in performances.items():
            stat = statistics.get(market)
            
            html.append(f"<div class='card'>")
            html.append(f"<h2>{market.upper()}</h2>")
            
            # Metrics grid
            html.append("<div class='metric-grid'>")
            
            roi_class = "positive" if perf.roi > 0 else "negative"
            
            html.append(f"<div class='metric'><div class='metric-value'>{perf.n_bets}</div><div class='metric-label'>Total Bets</div></div>")
            html.append(f"<div class='metric'><div class='metric-value'>{perf.accuracy:.1%}</div><div class='metric-label'>Accuracy</div></div>")
            html.append(f"<div class='metric'><div class='metric-value {roi_class}'>{perf.roi:+.1%}</div><div class='metric-label'>ROI</div></div>")
            html.append(f"<div class='metric'><div class='metric-value'>{perf.sharpe_ratio:.2f}</div><div class='metric-label'>Sharpe Ratio</div></div>")
            html.append(f"<div class='metric'><div class='metric-value'>{perf.max_drawdown:.0f}</div><div class='metric-label'>Max Drawdown</div></div>")
            html.append(f"<div class='metric'><div class='metric-value'>{perf.kelly_fraction:.1%}</div><div class='metric-label'>Kelly Fraction</div></div>")
            
            html.append("</div>")
            
            # Recommendation
            if stat:
                rec_class = "recommend" if "RECOMMEND" in stat.recommendation else ("caution" if "MONITOR" in stat.recommendation else "not-recommend")
                html.append(f"<div class='recommendation {rec_class}'><strong>Recommendation:</strong> {stat.recommendation}</div>")
            
            html.append("</div>")
        
        # Footer
        html.append(f"<footer>Generated by NBA Backtest Engine v{self.version}</footer>")
        html.append("</body>")
        html.append("</html>")
        
        path.write_text("\n".join(html), encoding="utf-8")
        logger.info(f"Wrote HTML report: {path}")
    
    def _write_json_summary(
        self,
        path: Path,
        performances: Dict[str, MarketPerformance],
        statistics: Dict[str, StatisticalSummary],
        monte_carlo: Optional[Dict[str, MonteCarloResults]],
        config: Optional[Dict[str, Any]],
    ) -> None:
        """Write JSON summary for programmatic access."""
        summary = {
            "version": self.version,
            "generated_at": datetime.now().isoformat(),
            "config": config or {},
            "markets": {},
        }
        
        for market, perf in performances.items():
            market_data = {
                "performance": asdict(perf),
            }
            
            if market in statistics:
                stat = statistics[market]
                # Convert tests to dicts
                stat_dict = asdict(stat)
                stat_dict["tests"] = [asdict(t) for t in stat.tests]
                market_data["statistics"] = stat_dict
            
            if monte_carlo and market in monte_carlo:
                mc = monte_carlo[market]
                mc_dict = asdict(mc)
                # Don't include sample paths in JSON
                mc_dict.pop("sample_paths", None)
                market_data["monte_carlo"] = mc_dict
            
            summary["markets"][market] = market_data
        
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info(f"Wrote JSON summary: {path}")
