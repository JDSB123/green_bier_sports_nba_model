import pandas as pd
import os

df = pd.read_csv('/workspaces/green_bier_sports_nba_model/data/processed/training_data.csv')
# Look for a game later in the season (e.g., row 100) to ensure averages have stabilized
row = df.iloc[100]

print("Date:", row['date'])
print("Home Team:", row['home_team'])
print("Home Score:", row['home_score'])
print("Home PPG (Feature):", row['home_ppg'])
print("Home Win Pct:", row['home_win_pct'])

# Check if PPG looks like an average or a single game score
print(f"Is PPG ({row['home_ppg']}) exactly Score ({row['home_score']})? {row['home_ppg'] == row['home_score']}")
print(f"Diff: {abs(row['home_ppg'] - row['home_score'])}")

# Check for duplicate games (home/away perspective)
game_ids = df['game_id'].dropna().astype(str)
n_rows = len(df)
n_unique_ids = len(game_ids.unique())

print(f"Total Rows: {n_rows}")
print(f"Unique Game IDs: {n_unique_ids}")
print(f"Ratio: {n_rows / n_unique_ids:.2f}")

if n_rows > n_unique_ids:
    print("WARNING: Duplicate games found! Likely Home/Away perspectives.")
    # Check if duplicates are inverted
    counts = game_ids.value_counts()
    print("Most frequent IDs:", counts.head())
