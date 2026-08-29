import csv
from collections import Counter
from pathlib import Path

import chess


INPUT_CSV = Path("data/processed/rapid_800_900_positions_test.csv")


def main():
    rows = 0
    game_ids = set()

    player_elos = []
    opponent_elos = []
    plies = []

    side_to_move = Counter()
    results = Counter()

    illegal_moves = 0
    duplicate_positions = 0

    seen_positions = set()

    with INPUT_CSV.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            rows += 1

            game_id = int(row["game_id"])
            fen = row["fen"]
            move_uci = row["uci_move"]
            player_elo = int(row["player_elo"])
            opponent_elo = int(row["opponent_elo"])
            ply = int(row["ply"])
            result = row["result"]

            game_ids.add(game_id)
            player_elos.append(player_elo)
            opponent_elos.append(opponent_elo)
            plies.append(ply)
            results[result] += 1

            board = chess.Board(fen)

            if board.turn == chess.WHITE:
                side_to_move["White"] += 1
            else:
                side_to_move["Black"] += 1

            try:
                move = chess.Move.from_uci(move_uci)

                if move not in board.legal_moves:
                    illegal_moves += 1

            except ValueError:
                illegal_moves += 1

            position_key = (
                fen,
                move_uci,
                player_elo,
                opponent_elo,
            )

            if position_key in seen_positions:
                duplicate_positions += 1
            else:
                seen_positions.add(position_key)

    print(f"Rows: {rows:,}")
    print(f"Unique games: {len(game_ids):,}")

    print("\nPlayer Elo:")
    print(f"  min: {min(player_elos)}")
    print(f"  max: {max(player_elos)}")

    print("\nOpponent Elo:")
    print(f"  min: {min(opponent_elos)}")
    print(f"  max: {max(opponent_elos)}")

    print("\nPly:")
    print(f"  min: {min(plies)}")
    print(f"  max: {max(plies)}")

    print("\nSide to move:")
    for side, count in side_to_move.items():
        print(f"  {side}: {count:,}")

    print("\nResults:")
    for result, count in results.items():
        print(f"  {result}: {count:,}")

    print(f"\nIllegal moves: {illegal_moves:,}")
    print(f"Duplicate position/move rows: {duplicate_positions:,}")

    print(
        f"Average positions per game: "
        f"{rows / len(game_ids):.2f}"
    )


if __name__ == "__main__":
    main()