import sys

import chess
import torch

from src.model import HumanChessPolicy
from src.board_encoding import encode_board
from src.move_encoding import move_to_index


CHECKPOINT_PATH = "checkpoints/human_chess_policy.pt"


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


def choose_move(model, board):
    position = encode_board(board)
    position = position.unsqueeze(0)

    with torch.no_grad():
        output = model(position)

    scores = output[0]

    best_move = None
    best_score = float("-inf")

    for move in board.legal_moves:
        move_index = move_to_index(
            move,
            board.turn,
        )

        score = scores[move_index].item()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def set_position(board, command):
    parts = command.split()

    if "startpos" in parts:
        board.reset()

        if "moves" in parts:
            moves_index = parts.index("moves") + 1

            for move_text in parts[moves_index:]:
                board.push_uci(move_text)

    elif "fen" in parts:
        fen_index = parts.index("fen") + 1

        if "moves" in parts:
            moves_index = parts.index("moves")

            fen = " ".join(
                parts[fen_index:moves_index]
            )

            board.set_fen(fen)

            for move_text in parts[moves_index + 1:]:
                board.push_uci(move_text)

        else:
            fen = " ".join(parts[fen_index:])
            board.set_fen(fen)


def main():
    model = load_model()
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
            print("readyok")
            sys.stdout.flush()

        elif command == "ucinewgame":
            board.reset()

        elif command.startswith("position"):
            set_position(board, command)

        elif command.startswith("go"):
            if board.is_game_over():
                print("bestmove 0000")
            else:
                move = choose_move(model, board)
                print(f"bestmove {move.uci()}")

            sys.stdout.flush()

        elif command == "quit":
            break


if __name__ == "__main__":
    main()