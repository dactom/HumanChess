import chess


# ---------------------------------------------------------
# Board-orientation helpers
# ---------------------------------------------------------

def flip_square_vertical(square: chess.Square) -> chess.Square:
    """
    Flip a square vertically.

    Examples:
        e7 -> e2
        e8 -> e1
        a6 -> a3

    The file does not change:
        a stays a
        e stays e
        h stays h
    """

    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)

    flipped_rank = 7 - rank_index

    return chess.square(file_index, flipped_rank)


def normalize_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """
    Convert a real chess move to our player-to-move orientation.

    White moves remain unchanged.

    Black moves are vertically flipped.

    Example:
        White e2e4 -> e2e4
        Black e7e5 -> e2e4
    """

    if turn == chess.WHITE:
        return move

    return chess.Move(
        from_square=flip_square_vertical(move.from_square),
        to_square=flip_square_vertical(move.to_square),
        promotion=move.promotion,
    )


def denormalize_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """
    Convert a normalized move back to the real board orientation.

    A vertical flip is its own inverse, so the operation is
    identical to normalize_move().
    """

    if turn == chess.WHITE:
        return move

    return chess.Move(
        from_square=flip_square_vertical(move.from_square),
        to_square=flip_square_vertical(move.to_square),
        promotion=move.promotion,
    )


# ---------------------------------------------------------
# Move vocabulary
# ---------------------------------------------------------

def is_geometrically_possible(
    from_square: chess.Square,
    to_square: chess.Square,
) -> bool:
    """
    Return True if some normal chess piece could geometrically
    move from from_square to to_square on an otherwise empty board.

    This includes:
        rook-like movement
        bishop-like movement
        queen-like movement
        king movement
        knight movement
        pawn straight/diagonal movement
        castling source/destination geometry

    It does NOT check whether the move is legal in an actual position.
    """

    if from_square == to_square:
        return False

    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)

    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)

    file_distance = abs(to_file - from_file)
    rank_distance = abs(to_rank - from_rank)

    # Rook / queen / king / pawn-type geometry
    if from_file == to_file:
        return True

    if from_rank == to_rank:
        return True

    # Bishop / queen geometry
    if file_distance == rank_distance:
        return True

    # Knight geometry
    if (file_distance, rank_distance) in ((1, 2), (2, 1)):
        return True

    return False


def build_move_vocabulary():
    """
    Build a deterministic list containing every move class.

    1792 ordinary source -> destination classes
    + 88 explicit promotion classes
    = 1880 total classes.
    """

    labels = set()

    # -----------------------------------------------------
    # Ordinary source -> destination moves
    # -----------------------------------------------------

    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:

            if is_geometrically_possible(from_square, to_square):

                label = (
                    chess.square_name(from_square)
                    + chess.square_name(to_square)
                )

                labels.add(label)

    # -----------------------------------------------------
    # Promotion moves
    # -----------------------------------------------------
    #
    # Because all positions are normalized to the player-to-move
    # perspective, promotion always goes from rank 7 to rank 8.
    #
    # Examples:
    #     e7e8q
    #     e7e8r
    #     e7e8b
    #     e7e8n
    #
    # Capturing promotions are included as well:
    #     e7d8q
    #     e7f8n
    # -----------------------------------------------------

    promotion_pieces = [
        chess.QUEEN,
        chess.ROOK,
        chess.BISHOP,
        chess.KNIGHT,
    ]

    from_rank = 6  # python-chess rank index for rank 7
    to_rank = 7    # python-chess rank index for rank 8

    for from_file in range(8):

        # Pawn can promote straight ahead or while capturing
        # one file left/right.

        possible_to_files = [
            from_file - 1,
            from_file,
            from_file + 1,
        ]

        for to_file in possible_to_files:

            if not 0 <= to_file < 8:
                continue

            from_square = chess.square(from_file, from_rank)
            to_square = chess.square(to_file, to_rank)

            for promotion_piece in promotion_pieces:

                move = chess.Move(
                    from_square,
                    to_square,
                    promotion=promotion_piece,
                )

                labels.add(move.uci())

    # Sorting makes the mapping deterministic.
    # The same move will get the same index every time we run
    # the program.

    return sorted(labels)


MOVE_LABELS = build_move_vocabulary()

MOVE_TO_INDEX = {
    label: index
    for index, label in enumerate(MOVE_LABELS)
}

NUM_MOVE_CLASSES = len(MOVE_LABELS)


# ---------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------

def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """
    Convert a real chess move into our normalized move-class index.
    """

    normalized = normalize_move(move, turn)
    label = normalized.uci()

    if label not in MOVE_TO_INDEX:
        raise ValueError(f"Move not in vocabulary: {label}")

    return MOVE_TO_INDEX[label]


def index_to_move(index: int, turn: chess.Color) -> chess.Move:
    """
    Convert a move-class index back into a real chess move.
    """

    if not 0 <= index < NUM_MOVE_CLASSES:
        raise ValueError(f"Invalid move index: {index}")

    label = MOVE_LABELS[index]

    normalized_move = chess.Move.from_uci(label)

    return denormalize_move(normalized_move, turn)


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Number of move classes:", NUM_MOVE_CLASSES)
    print()

    # -----------------------------------------------------
    # Test 1: equivalent White and Black pawn moves
    # -----------------------------------------------------

    white_move = chess.Move.from_uci("e2e4")
    black_move = chess.Move.from_uci("e7e5")

    white_index = move_to_index(
        white_move,
        chess.WHITE,
    )

    black_index = move_to_index(
        black_move,
        chess.BLACK,
    )

    print("White move:", white_move)
    print("Black move:", black_move)

    print("White normalized:",
          normalize_move(white_move, chess.WHITE))

    print("Black normalized:",
          normalize_move(black_move, chess.BLACK))

    print()

    print("White index:", white_index)
    print("Black index:", black_index)

    print()

    print(
        "Same class:",
        white_index == black_index,
    )

    # -----------------------------------------------------
    # Test 2: knight symmetry
    # -----------------------------------------------------

    print()

    white_knight = chess.Move.from_uci("g1f3")
    black_knight = chess.Move.from_uci("g8f6")

    print(
        "White knight normalized:",
        normalize_move(white_knight, chess.WHITE),
    )

    print(
        "Black knight normalized:",
        normalize_move(black_knight, chess.BLACK),
    )

    print(
        "Same knight class:",
        move_to_index(white_knight, chess.WHITE)
        == move_to_index(black_knight, chess.BLACK),
    )

    # -----------------------------------------------------
    # Test 3: round-trip
    # -----------------------------------------------------

    print()

    index = move_to_index(
        black_move,
        chess.BLACK,
    )

    reconstructed = index_to_move(
        index,
        chess.BLACK,
    )

    print("Original Black move:", black_move)
    print("Reconstructed move:", reconstructed)

    # -----------------------------------------------------
    # Test 4: promotion
    # -----------------------------------------------------

    print()

    promotion = chess.Move.from_uci("e7e8q")

    promotion_index = move_to_index(
        promotion,
        chess.WHITE,
    )

    print("Promotion:", promotion)
    print("Promotion index:", promotion_index)

    reconstructed_promotion = index_to_move(
        promotion_index,
        chess.WHITE,
    )

    print(
        "Reconstructed promotion:",
        reconstructed_promotion,
    )