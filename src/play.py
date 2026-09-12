import chess
import torch

from src.models.policy_v1 import HumanChessPolicy
from src.models.policy_resnet import HumanChessResNetPolicy

from src.board_encoding import encode_board
from src.move_encoding import move_to_index


# --------------------------------------------------
# Model configuration
# --------------------------------------------------

# Choose:
#   "v1"
#   "resnet"
MODEL_NAME = "resnet"


V1_CHECKPOINT_PATH = (
    "checkpoints/"
    "human_chess_policy_2080ti_test.pt"
)

RESNET_CHECKPOINT_PATH = (
    "checkpoints/"
    "v2/"
    "residual/"
    "best.pt"
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

def load_model():

    if MODEL_NAME == "v1":

        checkpoint_path = (
            V1_CHECKPOINT_PATH
        )

        model = (
            HumanChessPolicy()
        )

    elif MODEL_NAME == "resnet":

        checkpoint_path = (
            RESNET_CHECKPOINT_PATH
        )

        model = (
            HumanChessResNetPolicy()
        )

    else:

        raise ValueError(
            f"Unknown model: {MODEL_NAME}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        checkpoint,
        checkpoint_path,
    )


# --------------------------------------------------
# Move selection
# --------------------------------------------------

def choose_move(
    model,
    board,
):

    position = (
        encode_board(
            board
        )
    )

    position = (
        position.unsqueeze(0)
    )

    with torch.no_grad():

        output = model(
            position
        )

    scores = (
        output[0]
    )

    best_move = None
    best_score = float(
        "-inf"
    )

    for move in board.legal_moves:

        move_index = (
            move_to_index(
                move,
                board.turn,
            )
        )

        score = (
            scores[
                move_index
            ]
            .item()
        )

        if score > best_score:

            best_score = (
                score
            )

            best_move = (
                move
            )

    return best_move


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    (
        model,
        checkpoint,
        checkpoint_path,
    ) = load_model()

    print(
        f"HumanChess loaded"
    )

    print(
        f"Model: "
        f"{MODEL_NAME}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "epoch" in checkpoint
    ):

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "test_accuracy" in checkpoint
    ):

        print(
            f"Model accuracy: "
            f"{checkpoint['test_accuracy'] * 100:.2f}%"
        )

    print()

    board = chess.Board()

    print(
        "You are White."
    )

    print(
        "Enter moves using UCI notation, "
        "for example: e2e4"
    )

    print()

    while not board.is_game_over():

        print(
            board
        )

        print()

        # -------------------------------------------------
        # Human move
        # -------------------------------------------------

        if board.turn == chess.WHITE:

            move_text = (
                input(
                    "Your move: "
                )
                .strip()
            )

            try:

                move = (
                    chess.Move.from_uci(
                        move_text
                    )
                )

            except ValueError:

                print(
                    "Invalid move format."
                )

                print()

                continue

            if (
                move
                not in board.legal_moves
            ):

                print(
                    "Illegal move."
                )

                print()

                continue

            board.push(
                move
            )

        # -------------------------------------------------
        # HumanChess move
        # -------------------------------------------------

        else:

            move = (
                choose_move(
                    model,
                    board,
                )
            )

            print(
                "HumanChess:",
                move,
            )

            print()

            board.push(
                move
            )

    print()

    print(
        board
    )

    print()

    print(
        "Game over"
    )

    print(
        "Result:",
        board.result(),
    )


if __name__ == "__main__":
    main()