import chess.pgn
from pathlib import Path
from collections import Counter

path = Path("data/filtered/rapid_750_850_test.pgn")

games = 0
white_elos = []
black_elos = []
events = Counter()
time_controls = Counter()
results = Counter()

with path.open(encoding="utf-8") as f:
    while True:
        game = chess.pgn.read_game(f)

        if game is None:
            break

        games += 1
        headers = game.headers

        events[headers.get("Event", "?")] += 1
        time_controls[headers.get("TimeControl", "?")] += 1
        results[headers.get("Result", "?")] += 1

        try:
            white_elos.append(int(headers["WhiteElo"]))
        except (KeyError, ValueError):
            pass

        try:
            black_elos.append(int(headers["BlackElo"]))
        except (KeyError, ValueError):
            pass


print("Games:", games)

print("\nWhite Elo:")
print("  count:", len(white_elos))
print("  min:", min(white_elos))
print("  max:", max(white_elos))

print("\nBlack Elo:")
print("  count:", len(black_elos))
print("  min:", min(black_elos))
print("  max:", max(black_elos))

print("\nEvents:")
for event, count in events.most_common():
    print(f"  {event}: {count}")

print("\nTime controls:")
for time_control, count in time_controls.most_common():
    print(f"  {time_control}: {count}")

print("\nResults:")
for result, count in results.most_common():
    print(f"  {result}: {count}")