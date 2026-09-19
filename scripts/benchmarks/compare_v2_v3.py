from pathlib import Path

from src.dataset import (
    PreEncodedHumanChessDataset,
    split_by_game,
)
from src.models.policy_resnet import HumanChessResNetPolicy
from src.train import (
    DEVICE,
    evaluate,
    load_checkpoint,
)


CSV_PATH = (
    "data/processed/"
    "rapid_800_900_positions_2019_2020.csv"
)

PT_PATH = (
    "data/processed/"
    "rapid_800_900_positions_2019_2020.pt"
)

CHECKPOINTS = {
    "V2": Path("checkpoints/v2/residual/best.pt"),
    "V3": Path("checkpoints/v3/residual/best.pt"),
}


def main():

    print("=" * 60)
    print("HumanChess V2 vs V3")
    print("=" * 60)

    dataset = PreEncodedHumanChessDataset(
        PT_PATH,
        CSV_PATH,
    )

    _, test_dataset = split_by_game(
        dataset
    )

    print()
    print(f"Device:         {DEVICE}")
    print(f"Test positions: {len(test_dataset):,}")
    print()

    for name, checkpoint_path in CHECKPOINTS.items():

        print("=" * 60)
        print(f"Evaluating {name}")
        print(f"Checkpoint: {checkpoint_path}")
        print("=" * 60)

        model = HumanChessResNetPolicy().to(
            DEVICE
        )

        checkpoint = load_checkpoint(
            model,
            checkpoint_path,
        )

        top1, top3, top5 = evaluate(
            model,
            test_dataset,
        )

        print()
        print(
            f"{name} checkpoint epoch: "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        print(
            f"{name} Top-1: {top1 * 100:.2f}%"
        )

        print(
            f"{name} Top-3: {top3 * 100:.2f}%"
        )

        print(
            f"{name} Top-5: {top5 * 100:.2f}%"
        )

        print()


if __name__ == "__main__":
    main()
