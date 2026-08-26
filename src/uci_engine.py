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

def allows_major_piece_capture(board, move):
    """
    Return True if, after playing `move`, the opponent can
    immediately capture our queen or rook.
    """

    test_board = board.copy()
    our_colour = board.turn

    test_board.push(move)

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

        if captured_piece.piece_type in (
            chess.QUEEN,
            chess.ROOK,
        ):
            return True

    return False


def choose_move(model, board):
    position = encode_board(board)
    position = position.unsqueeze(0)

    with torch.no_grad():
        output = model(position)

    scores = output[0]

    candidate_moves = []

    for move in board.legal_moves:
        move_index = move_to_index(
            move,
            board.turn,
        )

        score = scores[move_index].item()

        candidate_moves.append(
            (score, move)
        )

    # Highest neural-network score first
    candidate_moves.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    # Try moves in the model's preferred order.
    # Reject moves that immediately expose
    # our queen or rook to capture.
    for score, move in candidate_moves:

        if not allows_major_piece_capture(
            board,
            move,
        ):
            return move

    # If every legal move fails the safety test,
    # fall back to the model's favourite move.
    return candidate_moves[0][1]



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

            # Load the neural network only after
            # the UCI handshake has started.
            if model is None:
                model = load_model()

            print("readyok")
            sys.stdout.flush()

        elif command == "ucinewgame":
            board.reset()

        elif command.startswith("position"):
            set_position(board, command)

        elif command.startswith("go"):

            # Safety in case a GUI sends "go"
            # before "isready".
            if model is None:
                model = load_model()

            if board.is_game_over():
                print("bestmove 0000")

            else:
                move = choose_move(model, board)

                print(
                    f"info depth 1 multipv 1 "
                    f"pv {move.uci()}"
                )

                print(f"bestmove {move.uci()}")

            sys.stdout.flush()

        elif command == "quit":
            break

if __name__ == "__main__":
    main()