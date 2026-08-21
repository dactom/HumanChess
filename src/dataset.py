import chess
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from src.board_encoding import encode_board
from src.move_encoding import move_to_index


class HumanChessDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        board = chess.Board(row["fen"])
        move = chess.Move.from_uci(row["uci_move"])

        x = encode_board(board)
        y = move_to_index(move, board.turn)

        return x, y

def split_by_game(dataset, train_fraction=0.8):
    game_ids = sorted(dataset.data["game_id"].unique())

    split_index = int(len(game_ids) * train_fraction)

    train_games = set(game_ids[:split_index])
    test_games = set(game_ids[split_index:])

    train_indices = dataset.data.index[
        dataset.data["game_id"].isin(train_games)
    ].tolist()

    test_indices = dataset.data.index[
        dataset.data["game_id"].isin(test_games)
    ].tolist()

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

def make_dataloaders(
    train_dataset,
    test_dataset,
    batch_size=64,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
