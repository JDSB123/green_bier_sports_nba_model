import pandas as pd
import numpy as np

try:
    # Load data
    df = pd.read_csv('data/processed/training_data.csv')
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Define potential features to check for leakage
features_to_check = [
    'spread_line', '1h_spread_line',
    'home_pace', 'away_pace'
]

target = '1h_margin'

print(f"Checking correlations with {target} (Target)...")
print("-" * 50)

for feat in features_to_check:
    if feat in df.columns:
        # Drop NaNs for correlation
        valid = df.dropna(subset=[feat, target])
        if len(valid) == 0:
            print(f"{feat:20s}: No valid data")
            continue

        corr = valid[feat].corr(valid[target])
        print(f"{feat:20s}: {corr:.4f}")

        # Check "Identity" leak
        is_identical = (valid[feat] == valid[target]).mean()
        if is_identical > 0.1:
             print(f"  [ALERT] {feat} is identical to target in {is_identical:.1%} of rows!")
    else:
        print(f"{feat:20s}: Not in CSV")
