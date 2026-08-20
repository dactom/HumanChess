import random


class RandomPolicy:
    """
    Baseline policy that chooses uniformly among all legal moves.
    """

    def predict(self, board):
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        return random.choice(legal_moves)