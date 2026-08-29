import argparse
import csv
from pathlib import Path

import chess
import chess.pgn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract training positions from a PGN file."
    )

    parser.add_argument(
        "input_pgn",
        type=Path,
        help="Input PGN file",
    )

    parser.add_argument(
        "output_csv",
        type=Path,
        help="Output CSV file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_pgn = args.input_pgn
    output_csv = args.output_csv

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    games = 0
    positions = 0

    with input_pgn.open(
        "r",
        encoding="utf-8",
    ) as pgn_file, output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

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
            game = chess.pgn.read_game(
                pgn_file
            )

            if game is None:
                break

            games += 1

            headers = game.headers

            white_elo = int(
                headers["WhiteElo"]
            )

            black_elo = int(
                headers["BlackElo"]
            )

            result = headers.get(
                "Result",
                "*",
            )

            board = game.board()

            for ply, move in enumerate(
                game.mainline_moves(),
                start=1,
            ):
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

    print(
        f"Processed {games:,} games"
    )

    print(
        f"Extracted {positions:,} positions"
    )

    print(
        f"Saved to {output_csv}"
    )


if __name__ == "__main__":
    main()