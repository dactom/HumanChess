import torch
import chess
from pathlib import Path

from src.dataset import (
    HumanChessDataset,
    split_by_game,
    make_dataloaders,
)
from src.model import HumanChessPolicy
from src.move_encoding import move_to_index


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

DATASET_PATH = (
    "data/processed/rapid_800_900_positions_2019-01_all.csv"
)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15

CHECKPOINT_DIR = Path("checkpoints")

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "human_chess_policy_jan2019_all.pt"
)


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")


# --------------------------------------------------
# Training
# --------------------------------------------------

def train_one_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
):
    model.train()

    total_loss = 0.0
    total_positions = 0

    for batch_number, (x, y) in enumerate(
        train_loader,
        start=1,
    ):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        scores = model(x)
        loss = loss_fn(scores, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_positions += batch_size

        if batch_number % 100 == 0:
            print(
                f"Batch {batch_number:4d} "
                f"| Loss: {loss.item():.4f}"
            )

    average_loss = (
        total_loss / total_positions
    )

    return average_loss


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, test_dataset):
    model.eval()

    correct = 0
    total = 0

    base_dataset = test_dataset.dataset

    with torch.no_grad():

        for index in test_dataset.indices:

            x, y = base_dataset[index]

            fen = (
                base_dataset
                .data
                .iloc[index]["fen"]
            )

            board = chess.Board(fen)

            x = x.to(DEVICE)

            scores = model(
                x.unsqueeze(0)
            ).squeeze(0)

            legal_scores = torch.full_like(
                scores,
                float("-inf"),
            )

            for move in board.legal_moves:

                move_index = move_to_index(
                    move,
                    board.turn,
                )

                legal_scores[move_index] = (
                    scores[move_index]
                )

            predicted_move = (
                legal_scores
                .argmax()
                .item()
            )

            if predicted_move == y:
                correct += 1

            total += 1

    return correct / total


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("HumanChess training")
    print("=" * 60)

    print(f"Device:       {DEVICE}")
    print(f"Dataset:      {DATASET_PATH}")
    print(f"Checkpoint:   {CHECKPOINT_PATH}")
    print()

    dataset = HumanChessDataset(
        DATASET_PATH
    )

    train_dataset, test_dataset = (
        split_by_game(dataset)
    )

    train_loader, _ = make_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
    )

    print(
        f"Total positions:    "
        f"{len(dataset):,}"
    )

    print(
        f"Training positions: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Test positions:     "
        f"{len(test_dataset):,}"
    )

    print(
        f"Batch size:         "
        f"{BATCH_SIZE}"
    )

    print(
        f"Learning rate:      "
        f"{LEARNING_RATE}"
    )

    print(
        f"Epochs:             "
        f"{EPOCHS}"
    )

    print()

    model = HumanChessPolicy().to(
        DEVICE
    )

    loss_fn = (
        torch.nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_accuracy = 0.0
    best_epoch = 0

    # Check whether this particular
    # 5000-game model already exists.
    if CHECKPOINT_PATH.exists():

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "test_accuracy" in checkpoint
        ):
            best_accuracy = (
                checkpoint["test_accuracy"]
            )

            best_epoch = (
                checkpoint["epoch"]
            )

            print(
                f"Existing 5000-game best: "
                f"{best_accuracy * 100:.2f}% "
                f"(epoch {best_epoch})"
            )

            print()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print("=" * 60)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print("=" * 60)

        average_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
        )

        print()
        print(
            "Evaluating test positions..."
        )

        test_accuracy = evaluate(
            model,
            test_dataset,
        )

        print(
            f"Average training loss: "
            f"{average_loss:.4f}"
        )

        print(
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        # Save immediately whenever
        # a new best model is found.
        if test_accuracy > best_accuracy:

            best_accuracy = test_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "test_accuracy":
                        best_accuracy,

                    "epoch":
                        best_epoch,

                    "dataset":
                        DATASET_PATH,

                    "training_positions":
                        len(train_dataset),

                    "test_positions":
                        len(test_dataset),
                },
                CHECKPOINT_PATH,
            )

            print(
                f"New best model saved: "
                f"{best_accuracy * 100:.2f}%"
            )

        print()

    print("=" * 60)

    print(
        f"Best epoch:    "
        f"{best_epoch}"
    )

    print(
        f"Best accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Saved model:   "
        f"{CHECKPOINT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()