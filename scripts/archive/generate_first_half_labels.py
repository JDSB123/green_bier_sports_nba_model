"""Generate heuristic first-half labels when halftime scores are unavailable.

This script reads `data/processed/training_data.csv`, computes approximate
first-half scores and target labels by halving full-game scores and totals,
and writes the augmented dataset to
`data/processed/training_data_fh.csv`.

Note: heuristic only - real halftime data is preferable for accurate FH
models.
"""
import os
import sys
import pandas as pd

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import settings


def main():
    input_path = os.path.join(settings.data_processed_dir, "training_data.csv")
    output_path = os.path.join(settings.data_processed_dir, "training_data_fh.csv")

    if not os.path.exists(input_path):
        print(f"No training data at {input_path}")
        return

    df = pd.read_csv(input_path)

    # Only proceed if full scores present
    if "home_score" not in df.columns or "away_score" not in df.columns:
        print("No full-game score columns found; cannot derive first-half labels.")
        return

    out = df.copy()

    # Heuristic: first-half scores approximated as half the full-game score
    out["home_halftime_score"] = (out["home_score"] * 0.5).round().astype(int)
    out["away_halftime_score"] = (out["away_score"] * 0.5).round().astype(int)

    # First-half winner
    out["fh_home_win"] = (out["home_halftime_score"] > out["away_halftime_score"]).astype(int)

    # Derive first-half lines if full-game lines exist
    if "total_line" in out.columns:
        out["fh_total_line"] = out["total_line"] / 2.0
        out["fh_went_over"] = ((out["home_halftime_score"] + out["away_halftime_score"]) > out["fh_total_line"]).astype(int)

    if "spread_line" in out.columns:
        out["fh_spread_line"] = out["spread_line"] / 2.0
        out["fh_spread_covered"] = ((out["home_halftime_score"] - out["away_halftime_score"]) - out["fh_spread_line"]).apply(lambda x: 1 if x > 0 else (0 if x < 0 else 0))

    # Save to new file
    out.to_csv(output_path, index=False)
    print(f"Wrote first-half augmented training data to {output_path}")


if __name__ == "__main__":
    main()
