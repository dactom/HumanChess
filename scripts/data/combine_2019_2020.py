import pandas as pd
from pathlib import Path

file_2019 = Path("data/processed/rapid_800_900_positions_2019-01_all.csv")
file_2020 = Path("data/processed/rapid_800_900_positions_2020-01.csv")

output = Path("data/processed/rapid_800_900_positions_2019_2020.csv")

print("Loading 2019...")
df_2019 = pd.read_csv(file_2019)

print(f"2019 positions: {len(df_2019):,}")
print(f"2019 games:     {df_2019['game_id'].nunique():,}")

max_game_id = df_2019["game_id"].max()

print()
print("Loading 2020...")
df_2020 = pd.read_csv(file_2020)

print(f"2020 positions: {len(df_2020):,}")
print(f"2020 games:     {df_2020['game_id'].nunique():,}")

# Give every 2020 game a new unique ID.
df_2020["game_id"] = df_2020["game_id"] + max_game_id

print()
print("Combining datasets...")

combined = pd.concat(
    [df_2019, df_2020],
    ignore_index=True,
)

combined.to_csv(
    output,
    index=False,
)

print()
print(f"Combined positions: {len(combined):,}")
print(f"Combined games:     {combined['game_id'].nunique():,}")
print(f"Saved to:           {output}")
