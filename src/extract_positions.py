import csv
from pathlib import Path

import chess
import chess.pgn


INPUT_PGN = Path("data/filtered/rapid_800_900_test.pgn")
OUTPUT_CSV = Path("data/processed/rapid_800_900_positions_test.csv")


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    games = 0
    positions = 0

    with INPUT_PGN.open("r", encoding="utf-8") as pgn_file, \
         OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "game_id",
                "fen",
                "uci_move",
                "player_elo",
                "opponent_elo",
                "ply",
                "result",
            ],
        )
        writer.writeheader()

        while True:
            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            games += 1

            headers = game.headers

            white_elo = int(headers["WhiteElo"])
            black_elo = int(headers["BlackElo"])
            result = headers.get("Result", "*")

            board = game.board()

            for ply, move in enumerate(game.mainline_moves(), start=1):
                if board.turn == chess.WHITE:
                    player_elo = white_elo
                    opponent_elo = black_elo
                else:
                    player_elo = black_elo
                    opponent_elo = white_elo

                writer.writerow(
                    {
                        "game_id": games,
                        "fen": board.fen(),
                        "uci_move": move.uci(),
                        "player_elo": player_elo,
                        "opponent_elo": opponent_elo,
                        "ply": ply,
                        "result": result,
                    }
                )

                positions += 1
                board.push(move)

    print(f"Processed {games:,} games")
    print(f"Extracted {positions:,} positions")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()