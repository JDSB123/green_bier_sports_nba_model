import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.modeling.models import SpreadsModel

df = pd.read_csv('/workspaces/green_bier_sports_nba_model/data/processed/training_data.csv', low_memory=False)

# Ensure labels
if "spread_covered" not in df.columns:
    df["actual_margin"] = df["home_score"] - df["away_score"]
    df["spread_covered"] = df.apply(
        lambda r: int(r["actual_margin"] > -r["spread_line"]) 
        if pd.notna(r.get("spread_line")) else None,
        axis=1
    )

model = SpreadsModel()
features = model.feature_columns

print(f"Checking correlation for {len(features)} features against 'spread_covered'...")

correlations = []
for col in features:
    if col in df.columns:
        # Handle non-numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            corr = df[col].corr(df['spread_covered'])
            correlations.append((col, corr))
        else:
             print(f"Skipping non-numeric: {col}")
    else:
        print(f"Feature missing in CSV: {col}")

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n--- TOP CORRELATIONS ---")
for name, corr in correlations[:10]:
    print(f"{name}: {corr:.4f}")
