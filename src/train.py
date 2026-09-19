import argparse
from pathlib import Path

import chess
import torch

from src.dataset import (
    PreEncodedHumanChessDataset,
    split_by_game,
    make_dataloaders,
)

from src.models.policy_v1 import HumanChessPolicy
from src.models.policy_resnet import HumanChessResNetPolicy

from src.move_encoding import move_to_index


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

DATASET_PATH = (
    "data/processed/"
    "rapid_800_900_positions_2019_2020.csv"
)

PREENCODED_PATH = (
    "data/processed/"
    "rapid_800_900_positions_2019_2020.pt"
)

BATCH_SIZE = 256
LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 15


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

elif (
    hasattr(torch, "xpu")
    and torch.xpu.is_available()
):

    DEVICE = torch.device(
        "xpu"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


# --------------------------------------------------
# Model selection
# --------------------------------------------------

def create_model(model_name):

    if model_name == "v1":
        return HumanChessPolicy()

    if model_name == "resnet":
        return HumanChessResNetPolicy()

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def checkpoint_path_for_model(model_name):

    if model_name == "v1":

        return (
            Path("checkpoints")
            / "human_chess_policy_2080ti_test.pt"
        )

    if model_name == "resnet":

        return (
            Path("checkpoints")
            / "v3"
            / "residual"
            / "best.pt"
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


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

        x = (
            x
            .to(DEVICE)
            .float()
        )

        y = y.to(
            DEVICE
        )

        scores = model(
            x
        )

        loss = loss_fn(
            scores,
            y,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_size = x.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_positions += (
            batch_size
        )

        if batch_number % 100 == 0:

            print(
                f"Batch "
                f"{batch_number:4d} "
                f"| Loss: "
                f"{loss.item():.4f}"
            )

    average_loss = (
        total_loss
        / total_positions
    )

    return average_loss


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(
    model,
    test_dataset,
):

    model.eval()

    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0

    total = 0

    base_dataset = (
        test_dataset.dataset
    )

    indices = (
        test_dataset.indices
    )

    batch_size = (
        BATCH_SIZE
    )

    with torch.no_grad():

        for start in range(
            0,
            len(indices),
            batch_size,
        ):

            batch_indices = indices[
                start:
                start + batch_size
            ]

            xs = []
            ys = []
            boards = []

            for index in batch_indices:

                x, y = (
                    base_dataset[index]
                )

                fen = (
                    base_dataset
                    .data
                    .iloc[index]["fen"]
                )

                board = chess.Board(
                    fen
                )

                xs.append(
                    x
                )

                ys.append(
                    y
                )

                boards.append(
                    board
                )

            x_batch = (
                torch
                .stack(xs)
                .to(DEVICE)
                .float()
            )

            scores_batch = model(
                x_batch
            )

            for scores, y, board in zip(
                scores_batch,
                ys,
                boards,
            ):

                legal_scores = (
                    torch.full_like(
                        scores,
                        float("-inf"),
                    )
                )

                legal_move_count = 0

                for move in board.legal_moves:

                    move_index = (
                        move_to_index(
                            move,
                            board.turn,
                        )
                    )

                    legal_scores[
                        move_index
                    ] = scores[
                        move_index
                    ]

                    legal_move_count += 1

                target = y.item()

                # Top-1
                predicted_move = (
                    legal_scores
                    .argmax()
                    .item()
                )

                if predicted_move == target:
                    correct_top1 += 1

                # Top-3
                top3_count = min(
                    3,
                    legal_move_count,
                )

                top3_moves = (
                    torch.topk(
                        legal_scores,
                        k=top3_count,
                    )
                    .indices
                    .tolist()
                )

                if target in top3_moves:
                    correct_top3 += 1

                # Top-5
                top5_count = min(
                    5,
                    legal_move_count,
                )

                top5_moves = (
                    torch.topk(
                        legal_scores,
                        k=top5_count,
                    )
                    .indices
                    .tolist()
                )

                if target in top5_moves:
                    correct_top5 += 1

                total += 1

    return (
        correct_top1 / total,
        correct_top3 / total,
        correct_top5 / total,
    )


# --------------------------------------------------
# Checkpoint loading
# --------------------------------------------------

def load_checkpoint(
    model,
    checkpoint_path,
):

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        not isinstance(
            checkpoint,
            dict,
        )
        or "model_state_dict"
        not in checkpoint
    ):

        raise ValueError(
            f"Invalid checkpoint: "
            f"{checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    return checkpoint


# --------------------------------------------------
# Command line
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate a "
            "HumanChess policy model."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "v1",
            "resnet",
        ],
        default="v1",
        help=(
            "Model architecture "
            "(default: v1)"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=(
            "Number of training epochs "
            f"(default: {DEFAULT_EPOCHS})"
        ),
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Load the saved checkpoint "
            "and evaluate without training."
        ),
    )

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_args()

    model_name = (
        args.model
    )

    epochs = (
        args.epochs
    )

    checkpoint_path = (
        checkpoint_path_for_model(
            model_name
        )
    )

    checkpoint_dir = (
        checkpoint_path.parent
    )

    print("=" * 60)
    print("HumanChess training")
    print("=" * 60)

    print(
        f"Model:        "
        f"{model_name}"
    )

    print(
        f"Device:       "
        f"{DEVICE}"
    )

    print(
        f"Dataset:      "
        f"{PREENCODED_PATH}"
    )

    print(
        f"Checkpoint:   "
        f"{checkpoint_path}"
    )

    print()

    dataset = (
        PreEncodedHumanChessDataset(
            PREENCODED_PATH,
            DATASET_PATH,
        )
    )

    train_dataset, test_dataset = (
        split_by_game(
            dataset
        )
    )

    train_loader, _ = (
        make_dataloaders(
            train_dataset,
            test_dataset,
            batch_size=BATCH_SIZE,
        )
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

    if not args.evaluate_only:

        print(
            f"Epochs:             "
            f"{epochs}"
        )

    print()

    model = (
        create_model(
            model_name
        )
        .to(DEVICE)
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print()

    # --------------------------------------------------
    # Evaluation-only mode
    # --------------------------------------------------

    if args.evaluate_only:

        print(
            "Loading saved checkpoint..."
        )

        checkpoint = load_checkpoint(
            model,
            checkpoint_path,
        )

        print(
            f"Checkpoint epoch: "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        print()

        print(
            "Evaluating test positions..."
        )

        (
            top1_accuracy,
            top3_accuracy,
            top5_accuracy,
        ) = evaluate(
            model,
            test_dataset,
        )

        print()

        print("=" * 60)
        print("Evaluation results")
        print("=" * 60)

        print(
            f"Top-1 accuracy: "
            f"{top1_accuracy * 100:.2f}%"
        )

        print(
            f"Top-3 accuracy: "
            f"{top3_accuracy * 100:.2f}%"
        )

        print(
            f"Top-5 accuracy: "
            f"{top5_accuracy * 100:.2f}%"
        )

        print("=" * 60)

        return

    # --------------------------------------------------
    # Training mode
    # --------------------------------------------------

    loss_fn = (
        torch.nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_accuracy = 0.0
    best_epoch = 0

    if checkpoint_path.exists():

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and "test_accuracy"
            in checkpoint
        ):

            best_accuracy = (
                checkpoint[
                    "test_accuracy"
                ]
            )

            best_epoch = (
                checkpoint[
                    "epoch"
                ]
            )

            print(
                f"Existing best model: "
                f"{best_accuracy * 100:.2f}% "
                f"(epoch {best_epoch})"
            )

            print()

    for epoch in range(
        1,
        epochs + 1,
    ):

        print("=" * 60)

        print(
            f"Epoch "
            f"{epoch}/{epochs}"
        )

        print("=" * 60)

        average_loss = (
            train_one_epoch(
                model,
                train_loader,
                loss_fn,
                optimizer,
            )
        )

        print()

        print(
            "Evaluating test positions..."
        )

        (
            top1_accuracy,
            top3_accuracy,
            top5_accuracy,
        ) = evaluate(
            model,
            test_dataset,
        )

        print(
            f"Average training loss: "
            f"{average_loss:.4f}"
        )

        print(
            f"Top-1 accuracy: "
            f"{top1_accuracy * 100:.2f}%"
        )

        print(
            f"Top-3 accuracy: "
            f"{top3_accuracy * 100:.2f}%"
        )

        print(
            f"Top-5 accuracy: "
            f"{top5_accuracy * 100:.2f}%"
        )

        if (
            top1_accuracy
            > best_accuracy
        ):

            best_accuracy = (
                top1_accuracy
            )

            best_epoch = (
                epoch
            )

            torch.save(
                {
                    "model_name":
                        model_name,

                    "model_state_dict":
                        model.state_dict(),

                    "test_accuracy":
                        top1_accuracy,

                    "top1_accuracy":
                        top1_accuracy,

                    "top3_accuracy":
                        top3_accuracy,

                    "top5_accuracy":
                        top5_accuracy,

                    "epoch":
                        best_epoch,

                    "dataset":
                        DATASET_PATH,

                    "training_positions":
                        len(
                            train_dataset
                        ),

                    "test_positions":
                        len(
                            test_dataset
                        ),
                },
                checkpoint_path,
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
        f"Best Top-1:    "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Saved model:   "
        f"{checkpoint_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()

