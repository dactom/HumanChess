import csv
import chess

from src.board_encoding import encode_board
from src.move_encoding import (
    move_to_index,
    normalize_move,
)


CSV_PATH = "data/processed/rapid_800_900_positions_test.csv"


def inspect_row(row):
    """
    Convert one CSV row into a complete ML training example.
    """

    # ---------------------------------------------------------
    # 1. Reconstruct the chess position
    # ---------------------------------------------------------

    board = chess.Board(row["fen"])

    # ---------------------------------------------------------
    # 2. Read the move the human actually played
    # ---------------------------------------------------------

    move = chess.Move.from_uci(row["uci_move"])

    # Safety check:
    # the target move should actually be legal in this position.
    if move not in board.legal_moves:
        raise ValueError(
            f"Illegal target move {move} in position {row['fen']}"
        )

    # ---------------------------------------------------------
    # 3. Encode the board
    # ---------------------------------------------------------

    x = encode_board(board)

    # ---------------------------------------------------------
    # 4. Encode the target move
    # ---------------------------------------------------------

    y = move_to_index(
        move,
        board.turn,
    )

    normalized_move = normalize_move(
        move,
        board.turn,
    )

    # ---------------------------------------------------------
    # 5. Display the resulting training example
    # ---------------------------------------------------------

    print("=" * 60)

    print("Game:", row["game_id"])
    print("Ply:", row["ply"])
    print()

    print("Player Elo:", row["player_elo"])
    print("Opponent Elo:", row["opponent_elo"])
    print()

    print(
        "Side to move:",
        "White" if board.turn == chess.WHITE else "Black",
    )

    print()

    print(board)
    print()

    print("Original move:", move)
    print("Normalized move:", normalized_move)
    print("Target class index:", y)

    print()

    print("Input tensor shape:", x.shape)
    print("Input dtype:", x.dtype)
    print("Target type:", type(y))

    print()

    print("Complete ML example:")
    print(f"x = tensor with shape {tuple(x.shape)}")
    print(f"y = {y}")


if __name__ == "__main__":

    with open(
        CSV_PATH,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        # Inspect the first two real training positions.
        for row_number, row in enumerate(reader):

            inspect_row(row)

            if row_number == 1:
                break