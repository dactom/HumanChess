import chess
import pandas as pd
import torch

from torch.utils.data import (
    Dataset,
    Subset,
    DataLoader,
)

from src.board_encoding import encode_board
from src.move_encoding import move_to_index


# --------------------------------------------------
# Original CSV dataset
# --------------------------------------------------

class HumanChessDataset(Dataset):

    def __init__(self, csv_path):
        self.data = pd.read_csv(
            csv_path
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        board = chess.Board(
            row["fen"]
        )

        move = chess.Move.from_uci(
            row["uci_move"]
        )

        x = encode_board(board)

        y = move_to_index(
            move,
            board.turn,
        )

        return x, y


# --------------------------------------------------
# Pre-encoded dataset
# --------------------------------------------------

class PreEncodedHumanChessDataset(Dataset):

    def __init__(
        self,
        pt_path,
        csv_path,
    ):

        print(
            "Loading pre-encoded dataset..."
        )

        saved = torch.load(
            pt_path,
            map_location="cpu",
        )

        self.boards = saved["boards"]
        self.targets = saved["targets"]
        self.game_ids = saved["game_ids"]

        # FEN is still required during
        # evaluation for legal-move checking.
        self.data = pd.read_csv(
            csv_path,
            usecols=[
                "game_id",
                "fen",
            ],
        )

        print(
            f"Loaded "
            f"{len(self.boards):,} "
            f"pre-encoded positions"
        )

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, index):

        x = self.boards[index]
        y = self.targets[index]

        return x, y


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

def split_by_game(
    dataset,
    train_fraction=0.8,
):

    game_ids = sorted(
        dataset
        .data["game_id"]
        .unique()
    )

    split_index = int(
        len(game_ids)
        * train_fraction
    )

    train_games = set(
        game_ids[:split_index]
    )

    test_games = set(
        game_ids[split_index:]
    )

    train_indices = (
        dataset
        .data
        .index[
            dataset
            .data["game_id"]
            .isin(train_games)
        ]
        .tolist()
    )

    test_indices = (
        dataset
        .data
        .index[
            dataset
            .data["game_id"]
            .isin(test_games)
        ]
        .tolist()
    )

    train_dataset = Subset(
        dataset,
        train_indices,
    )

    test_dataset = Subset(
        dataset,
        test_indices,
    )

    return (
        train_dataset,
        test_dataset,
    )


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

def make_dataloaders(
    train_dataset,
    test_dataset,
    batch_size=64,
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,

        # Keep this at 0 on Windows.
        # Multiple workers caused problems
        # with the large encoded dataset.
        num_workers=0,

        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return (
        train_loader,
        test_loader,
    )