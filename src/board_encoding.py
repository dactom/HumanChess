import chess
import torch


NUM_CHANNELS = 18


def square_to_tensor_coords(square: chess.Square, turn: chess.Color):
    """
    Convert a python-chess square into tensor row/column coordinates.

    White to move:
        normal chess orientation.

    Black to move:
        vertically flip the board so that Black's side is at the bottom.

    Files are never reversed:
        a stays a, h stays h.
    """

    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)

    if turn == chess.WHITE:
        row = 7 - rank_index
    else:
        row = rank_index

    col = file_index

    return row, col


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board into an 18x8x8 float32 tensor.

    Channels 0-5:
        pieces belonging to the player to move

    Channels 6-11:
        opponent pieces

    Channels 12-15:
        castling rights

    Channel 16:
        en-passant target square

    Channel 17:
        original player colour
        1 = White to move
        0 = Black to move
    """

    tensor = torch.zeros(
        (NUM_CHANNELS, 8, 8),
        dtype=torch.float32,
    )

    player = board.turn
    opponent = not player

    # ---------------------------------------------------------
    # Piece planes
    # ---------------------------------------------------------

    for square, piece in board.piece_map().items():

        row, col = square_to_tensor_coords(square, player)

        # python-chess piece types are:
        # PAWN   = 1
        # KNIGHT = 2
        # BISHOP = 3
        # ROOK   = 4
        # QUEEN  = 5
        # KING   = 6

        piece_offset = piece.piece_type - 1

        if piece.color == player:
            channel = piece_offset
        else:
            channel = 6 + piece_offset

        tensor[channel, row, col] = 1.0

    # ---------------------------------------------------------
    # Castling rights
    # ---------------------------------------------------------

    if board.has_kingside_castling_rights(player):
        tensor[12, :, :] = 1.0

    if board.has_queenside_castling_rights(player):
        tensor[13, :, :] = 1.0

    if board.has_kingside_castling_rights(opponent):
        tensor[14, :, :] = 1.0

    if board.has_queenside_castling_rights(opponent):
        tensor[15, :, :] = 1.0

    # ---------------------------------------------------------
    # En-passant target
    # ---------------------------------------------------------

    if board.ep_square is not None:
        row, col = square_to_tensor_coords(
            board.ep_square,
            player,
        )

        tensor[16, row, col] = 1.0

    # ---------------------------------------------------------
    # Original player colour
    # ---------------------------------------------------------

    if player == chess.WHITE:
        tensor[17, :, :] = 1.0

    return tensor


def print_plane(plane: torch.Tensor):
    """Simple visual display for checking an 8x8 plane."""

    for row in plane:
        print(" ".join(str(int(value.item())) for value in row))


if __name__ == "__main__":

    board = chess.Board()
    board.push_uci("e2e4")

    print("Position after 1.e4")
    print(board)
    print()

    tensor = encode_board(board)

    print("Tensor shape:", tensor.shape)
    print()

    print("Channel 0: our pawns")
    print_plane(tensor[0])

    print()
    print("Channel 6: their pawns")
    print_plane(tensor[6])

    print()
    print("Channel sums:")

    for channel in range(NUM_CHANNELS):
        print(
            f"{channel:2d}: "
            f"{tensor[channel].sum().item():.0f}"
        )