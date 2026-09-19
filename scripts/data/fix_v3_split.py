import pandas as pd
import torch
from pathlib import Path


CSV_PATH = Path(
    "data/processed/rapid_800_900_positions_2019_2020.csv"
)

PT_PATH = Path(
    "data/processed/rapid_800_900_positions_2019_2020.pt"
)

GAMES_2019 = 19159
GAMES_2020 = 4377

TRAIN_2019 = int(GAMES_2019 * 0.8)
TRAIN_2020 = int(GAMES_2020 * 0.8)

TOTAL_TRAIN = TRAIN_2019 + TRAIN_2020


print("Loading combined CSV...")
df = pd.read_csv(CSV_PATH)

old_ids = df["game_id"].copy()

new_ids = {}


# 2019 training games
next_id = 1

for game_id in range(1, TRAIN_2019 + 1):
    new_ids[game_id] = next_id
    next_id += 1


# 2020 training games
for game_id_2020 in range(1, TRAIN_2020 + 1):
    old_id = GAMES_2019 + game_id_2020
    new_ids[old_id] = next_id
    next_id += 1


# 2019 test games
for game_id in range(TRAIN_2019 + 1, GAMES_2019 + 1):
    new_ids[game_id] = next_id
    next_id += 1


# 2020 test games
for game_id_2020 in range(TRAIN_2020 + 1, GAMES_2020 + 1):
    old_id = GAMES_2019 + game_id_2020
    new_ids[old_id] = next_id
    next_id += 1


df["game_id"] = old_ids.map(new_ids)

if df["game_id"].isna().any():
    raise RuntimeError("Some game IDs were not mapped.")


print("Saving corrected CSV...")
df.to_csv(CSV_PATH, index=False)


print("Updating PT game IDs...")
saved = torch.load(
    PT_PATH,
    map_location="cpu",
    weights_only=False,
)

saved["game_ids"] = torch.tensor(
    df["game_id"].to_numpy(),
    dtype=torch.int64,
)

torch.save(saved, PT_PATH)


print()
print(f"2019 training games: {TRAIN_2019:,}")
print(f"2020 training games: {TRAIN_2020:,}")
print(f"Total training games: {TOTAL_TRAIN:,}")
print()
print(f"Total games: {df['game_id'].nunique():,}")
print(f"Total positions: {len(df):,}")
print("Done.")
