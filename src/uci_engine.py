import sys

import chess
import torch

from src.model import HumanChessPolicy
from src.board_encoding import encode_board
from src.move_encoding import move_to_index

CHECKPOINT_PATH = "checkpoints/human_chess_policy_5000games.pt"
#CHECKPOINT_PATH = "checkpoints/human_chess_policy.pt" this is for 500 games

# Tactical penalties
QUEEN_BLUNDER_PENALTY = 4
ROOK_BLUNDER_PENALTY = 1.5

# Controls how adventurous HumanChess is.
#
# Lower = more likely to choose the strongest move
# Higher = more variety / more mistakes
TEMPERATURE = 0.5


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = HumanChessPolicy()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def major_piece_penalty(board, move):
    """
    Check whether playing `move` allows the opponent
    to immediately capture our queen or rook.

    Returns:
        penalty
        reason
    """

    test_board = board.copy()
    our_colour = board.turn

    test_board.push(move)

    highest_penalty = 0.0
    reason = None

    for reply in test_board.legal_moves:

        if not test_board.is_capture(reply):
            continue

        captured_piece = test_board.piece_at(
            reply.to_square
        )

        if captured_piece is None:
            continue

        if captured_piece.color != our_colour:
            continue

        if captured_piece.piece_type == chess.QUEEN:

            if QUEEN_BLUNDER_PENALTY > highest_penalty:
                highest_penalty = QUEEN_BLUNDER_PENALTY

                reason = (
                    f"queen can be captured by "
                    f"{reply.uci()}"
                )

        elif captured_piece.piece_type == chess.ROOK:

            if ROOK_BLUNDER_PENALTY > highest_penalty:
                highest_penalty = ROOK_BLUNDER_PENALTY

                reason = (
                    f"rook can be captured by "
                    f"{reply.uci()}"
                )

    return highest_penalty, reason


def choose_move(model, board):
    position = encode_board(board)
    position = position.unsqueeze(0)

    with torch.no_grad():
        output = model(position)

    scores = output[0]

    candidates = []

    for move in board.legal_moves:

        move_index = move_to_index(
            move,
            board.turn,
        )

        raw_score = scores[move_index].item()

        penalty, reason = major_piece_penalty(
            board,
            move,
        )

        adjusted_score = raw_score - penalty

        candidates.append(
            {
                "move": move,
                "raw_score": raw_score,
                "adjusted_score": adjusted_score,
                "penalty": penalty,
                "reason": reason,
            }
        )

    # ------------------------------------------------
    # Original model probabilities
    # ------------------------------------------------

    raw_scores = torch.tensor(
        [
            candidate["raw_score"]
            for candidate in candidates
        ]
    )

    raw_probabilities = torch.softmax(
        raw_scores,
        dim=0,
    )

    # ------------------------------------------------
    # Adjusted probabilities before temperature
    # ------------------------------------------------

    adjusted_scores = torch.tensor(
        [
            candidate["adjusted_score"]
            for candidate in candidates
        ]
    )

    adjusted_probabilities = torch.softmax(
        adjusted_scores,
        dim=0,
    )

    # ------------------------------------------------
    # Temperature-controlled probabilities
    # ------------------------------------------------

    sampling_probabilities = torch.softmax(
        adjusted_scores / TEMPERATURE,
        dim=0,
    )

    # Store probabilities
    for (
        candidate,
        raw_probability,
        adjusted_probability,
        sampling_probability,
    ) in zip(
        candidates,
        raw_probabilities,
        adjusted_probabilities,
        sampling_probabilities,
    ):

        candidate["raw_probability"] = (
            raw_probability.item()
        )

        candidate["adjusted_probability"] = (
            adjusted_probability.item()
        )

        candidate["sampling_probability"] = (
            sampling_probability.item()
        )

    # ------------------------------------------------
    # Sort only for diagnostic display
    # ------------------------------------------------

    candidates_sorted = sorted(
        candidates,
        key=lambda candidate:
            candidate["sampling_probability"],
        reverse=True,
    )

    print(
        "info string --- HumanChess top moves ---"
    )

    for i, candidate in enumerate(
        candidates_sorted[:10],
        start=1,
    ):
        move = candidate["move"]

        raw_probability = (
            candidate["raw_probability"] * 100
        )

        adjusted_probability = (
            candidate["adjusted_probability"] * 100
        )

        sampling_probability = (
            candidate["sampling_probability"] * 100
        )

        penalty = candidate["penalty"]
        reason = candidate["reason"]

        if penalty == 0:
            status = "OK"
        else:
            status = (
                f"PENALTY {penalty:.1f} - "
                f"{reason}"
            )

        print(
            f"info string "
            f"{i:2d}. {move.uci():5s} "
            f"raw {raw_probability:6.2f}% "
            f"adj {adjusted_probability:6.2f}% "
            f"pick {sampling_probability:6.2f}% "
            f"{status}"
        )

    print(
        f"info string Temperature = "
        f"{TEMPERATURE}"
    )

    # ------------------------------------------------
    # Sample one move
    # ------------------------------------------------

    selected_index = torch.multinomial(
        sampling_probabilities,
        num_samples=1,
    ).item()

    selected_move = candidates[
        selected_index
    ]["move"]

    print(
        f"info string Selected move: "
        f"{selected_move.uci()}"
    )

    print(
        "info string ----------------------------"
    )

    return selected_move


def set_position(board, command):
    parts = command.split()

    if "startpos" in parts:
        board.reset()

        if "moves" in parts:
            moves_index = (
                parts.index("moves") + 1
            )

            for move_text in parts[moves_index:]:
                board.push_uci(move_text)

    elif "fen" in parts:
        fen_index = parts.index("fen") + 1

        if "moves" in parts:
            moves_index = parts.index(
                "moves"
            )

            fen = " ".join(
                parts[
                    fen_index:moves_index
                ]
            )

            board.set_fen(fen)

            for move_text in parts[
                moves_index + 1:
            ]:
                board.push_uci(move_text)

        else:
            fen = " ".join(
                parts[fen_index:]
            )

            board.set_fen(fen)


def main():
    model = None
    board = chess.Board()

    while True:
        line = sys.stdin.readline()

        if not line:
            break

        command = line.strip()

        if command == "uci":
            print("id name HumanChess")
            print("id author Dacian")
            print("uciok")
            sys.stdout.flush()

        elif command == "isready":

            if model is None:
                model = load_model()

            print("readyok")
            sys.stdout.flush()

        elif command == "ucinewgame":
            board.reset()

        elif command.startswith("position"):
            set_position(
                board,
                command,
            )

        elif command.startswith("go"):

            if model is None:
                model = load_model()

            if board.is_game_over():
                print("bestmove 0000")

            else:
                move = choose_move(
                    model,
                    board,
                )

                print(
                    f"info depth 1 multipv 1 "
                    f"pv {move.uci()}"
                )

                print(
                    f"bestmove {move.uci()}"
                )

            sys.stdout.flush()

        elif command == "quit":
            break


if __name__ == "__main__":
    main()