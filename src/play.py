import chess
import torch

from src.model import HumanChessPolicy
from src.board_encoding import encode_board
from src.move_encoding import move_to_index


CHECKPOINT_PATH = "checkpoints/human_chess_policy.pt"


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


def main():
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

    print("HumanChess loaded")
    print(
        f"Model accuracy: "
        f"{checkpoint['test_accuracy'] * 100:.2f}%"
    )
    print()

    board = chess.Board()

    print("You are White.")
    print("Enter moves using UCI notation, for example: e2e4")
    print()

    while not board.is_game_over():

        print(board)
        print()

        # -------------------------------------------------
        # Human move
        # -------------------------------------------------

        if board.turn == chess.WHITE:

            move_text = input("Your move: ").strip()

            try:
                move = chess.Move.from_uci(move_text)
            except ValueError:
                print("Invalid move format.")
                print()
                continue

            if move not in board.legal_moves:
                print("Illegal move.")
                print()
                continue

            board.push(move)

        # -------------------------------------------------
        # HumanChess move
        # -------------------------------------------------

        else:

            move = choose_move(model, board)

            print("HumanChess:", move)
            print()

            board.push(move)

    print()
    print(board)
    print()

    print("Game over")
    print("Result:", board.result())


if __name__ == "__main__":
    main()