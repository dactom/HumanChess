import chess
import pandas as pd
import torch

from src.board_encoding import encode_board
from src.move_encoding import move_to_index


INPUT_CSV = (
    "data/processed/"
    "rapid_800_900_positions_2019_2020.csv"
)

OUTPUT_PT = (
    "data/processed/"
    "rapid_800_900_positions_2019_2020.pt"
)


def main():

    print("Loading CSV...")

    data = pd.read_csv(INPUT_CSV)

    total = len(data)

    print(f"Positions: {total:,}")
    print("Allocating tensors...")

    boards = torch.empty(
        (total, 18, 8, 8),
        dtype=torch.uint8,
    )

    targets = torch.empty(
        total,
        dtype=torch.int64,
    )

    game_ids = torch.empty(
        total,
        dtype=torch.int64,
    )

    print("Encoding positions...")

    for index, row in data.iterrows():

        board = chess.Board(
            row["fen"]
        )

        move = chess.Move.from_uci(
            row["uci_move"]
        )

        x = encode_board(board)

        y = move_to_index(
            move,
            board.turn,
        )

        boards[index] = (
            x.to(torch.uint8)
        )

        targets[index] = y

        game_ids[index] = int(
            row["game_id"]
        )

        if (index + 1) % 10000 == 0:

            print(
                f"{index + 1:,} / "
                f"{total:,} "
                f"("
                f"{100 * (index + 1) / total:.1f}%"
                f")"
            )

    print("Saving...")

    torch.save(
        {
            "boards": boards,
            "targets": targets,
            "game_ids": game_ids,
        },
        OUTPUT_PT,
    )

    print("Done.")

    print(
        f"Saved to: {OUTPUT_PT}"
    )

    print(
        f"Boards shape: "
        f"{boards.shape}"
    )

    print(
        f"Boards dtype: "
        f"{boards.dtype}"
    )

    print(
        f"Targets shape: "
        f"{targets.shape}"
    )


if __name__ == "__main__":
    main()
