from collections import Counter

import chess

from src.move_encoding import normalize_move


class MoveFrequencyPolicy:
    """
    Baseline that predicts the most frequent normalized move
    among the legal moves available in the current position.
    """

    def __init__(self):
        self.move_counts = Counter()


    def observe(self, board, move):
        """
        Add one human move to the frequency statistics.
        """

        normalized_move = normalize_move(
            move,
            board.turn,
        )

        self.move_counts[normalized_move.uci()] += 1


    def predict(self, board):
        """
        Return the legal move with the highest frequency
        in the training data.
        """

        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        best_move = None
        best_count = -1

        for move in legal_moves:

            normalized_move = normalize_move(
                move,
                board.turn,
            )

            count = self.move_counts[
                normalized_move.uci()
            ]

            if count > best_count:
                best_count = count
                best_move = move

        return best_move