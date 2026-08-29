import csv
from pathlib import Path

import chess

from src.baselines.random_policy import RandomPolicy
from src.baselines.move_frequency_policy import MoveFrequencyPolicy


DATASET_PATH = Path(
    "data/processed/rapid_800_900_positions_test.csv"
)

TRAIN_FRACTION = 0.8


def load_rows(dataset_path):
    with dataset_path.open("r", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def split_by_game(rows):
    game_ids = sorted(
        {int(row["game_id"]) for row in rows}
    )

    split_index = int(
        len(game_ids) * TRAIN_FRACTION
    )

    train_games = set(game_ids[:split_index])
    test_games = set(game_ids[split_index:])

    train_rows = [
        row
        for row in rows
        if int(row["game_id"]) in train_games
    ]

    test_rows = [
        row
        for row in rows
        if int(row["game_id"]) in test_games
    ]

    return train_rows, test_rows


def train_frequency_policy(policy, rows):
    for row in rows:
        board = chess.Board(row["fen"])
        human_move = chess.Move.from_uci(
            row["uci_move"]
        )

        policy.observe(
            board,
            human_move,
        )


def evaluate_policy(policy, rows):
    total_positions = 0
    correct_predictions = 0
    illegal_predictions = 0

    for row in rows:
        board = chess.Board(row["fen"])
        human_move = chess.Move.from_uci(
            row["uci_move"]
        )

        predicted_move = policy.predict(board)

        if predicted_move is None:
            continue

        total_positions += 1

        if predicted_move not in board.legal_moves:
            illegal_predictions += 1

        if predicted_move == human_move:
            correct_predictions += 1

    accuracy = (
        correct_predictions / total_positions
        if total_positions > 0
        else 0.0
    )

    return {
        "positions": total_positions,
        "correct": correct_predictions,
        "illegal": illegal_predictions,
        "accuracy": accuracy,
    }


def print_result(name, result):
    print(name)
    print("-" * 60)
    print(
        f"Positions evaluated : "
        f"{result['positions']:,}"
    )
    print(
        f"Correct predictions : "
        f"{result['correct']:,}"
    )
    print(
        f"Illegal predictions : "
        f"{result['illegal']:,}"
    )
    print(
        f"Top-1 accuracy      : "
        f"{result['accuracy']:.4%}"
    )
    print()


def main():
    rows = load_rows(DATASET_PATH)

    train_rows, test_rows = split_by_game(
        rows
    )

    print(
        f"Training positions : "
        f"{len(train_rows):,}"
    )

    print(
        f"Test positions     : "
        f"{len(test_rows):,}"
    )

    print()

    random_policy = RandomPolicy()

    frequency_policy = MoveFrequencyPolicy()

    train_frequency_policy(
        frequency_policy,
        train_rows,
    )

    random_result = evaluate_policy(
        random_policy,
        test_rows,
    )

    frequency_result = evaluate_policy(
        frequency_policy,
        test_rows,
    )

    print("=" * 60)
    print("Baseline comparison")
    print("=" * 60)
    print()

    print_result(
        "Random legal-move baseline",
        random_result,
    )

    print_result(
        "Global move-frequency baseline",
        frequency_result,
    )

    improvement = (
        frequency_result["accuracy"]
        - random_result["accuracy"]
    )

    ratio = (
        frequency_result["accuracy"]
        / random_result["accuracy"]
        if random_result["accuracy"] > 0
        else 0.0
    )

    print(
        f"Absolute improvement : "
        f"{improvement:.4%}"
    )

    print(
        f"Accuracy ratio       : "
        f"{ratio:.2f}x"
    )


if __name__ == "__main__":
    main()