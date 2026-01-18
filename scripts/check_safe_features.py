import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/processed/training_data.csv')

# Define potential features to check for leakage
features_to_check = [
    'home_ppg', 'home_ortg', 'home_net_rtg', 'home_elo', 
    'home_l5_margin', 'home_l10_margin',
    'predicted_margin' # Known bad one for reference
]

target = 'home_margin'

print(f"Checking correlations with {target} (Target)...")
print("-" * 50)

for feat in features_to_check:
    if feat in df.columns:
        corr = df[feat].corr(df[target])
        print(f"{feat:20s}: {corr:.4f}")
        
        # Check "Identity" leak (is the value identical to the target?)
        is_identical = (df[feat] == df[target]).mean()
        if is_identical > 0.1:
             print(f"  [ALERT] {feat} is identical to target in {is_identical:.1%} of rows!")
    else:
        print(f"{feat:20s}: Not in CSV")

print("-" * 50)
print("Checking vs Home Score (Raw Output)...")
target_score = 'home_score'
for feat in features_to_check:
    if feat in df.columns:
        corr = df[feat].corr(df[target_score])
        print(f"{feat:20s}: {corr:.4f}")
