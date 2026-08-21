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


DATASET_PATH = (
    "data/processed/rapid_800_900_positions_test.csv"
)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "human_chess_policy.pt"


if torch.xpu.is_available():
    DEVICE = torch.device("xpu")
else:
    DEVICE = torch.device("cpu")


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

        if batch_number % 50 == 0:
            print(
                f"Batch {batch_number:3d} "
                f"| Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / total_positions

    return average_loss


def evaluate(model, test_dataset):
    model.eval()

    correct = 0
    total = 0

    base_dataset = test_dataset.dataset

    with torch.no_grad():
        for index in test_dataset.indices:
            x, y = base_dataset[index]

            fen = base_dataset.data.iloc[index]["fen"]
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

            predicted_move = legal_scores.argmax().item()

            if predicted_move == y:
                correct += 1

            total += 1

    return correct / total


def main():
    print(f"Device:             {DEVICE}")

    dataset = HumanChessDataset(DATASET_PATH)

    train_dataset, test_dataset = split_by_game(
        dataset
    )

    train_loader, _ = make_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
    )

    print(
        f"Training positions: {len(train_dataset):,}"
    )
    print(
        f"Test positions:     {len(test_dataset):,}"
    )
    print(
        f"Batch size:         {BATCH_SIZE}"
    )
    print()

    model = HumanChessPolicy().to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss()

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
    best_state_dict = None

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )

        if (
            isinstance(checkpoint, dict)
            and "test_accuracy" in checkpoint
        ):
            best_accuracy = checkpoint["test_accuracy"]
            best_epoch = checkpoint["epoch"]

            print(
                f"Existing best model: "
                f"{best_accuracy * 100:.2f}% "
                f"(epoch {best_epoch})"
            )
            print()

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")

        average_loss = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
        )

        test_accuracy = evaluate(
            model,
            test_dataset,
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch

            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor
                in model.state_dict().items()
            }

            print(
                f"New best model: "
                f"{best_accuracy * 100:.2f}%"
            )

        print(
            f"Average training loss: "
            f"{average_loss:.4f}"
        )

        print(
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        print()

    if best_state_dict is not None:
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "test_accuracy": best_accuracy,
                "epoch": best_epoch,
            },
            CHECKPOINT_PATH,
        )

        print(
            f"Saved best model from epoch "
            f"{best_epoch}: "
            f"{best_accuracy * 100:.2f}%"
        )
    else:
        print(
            "Existing checkpoint remains the best model."
        )


if __name__ == "__main__":
    main()