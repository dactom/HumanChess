import sys
from pathlib import Path

import chess
import chess.pgn


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def human_chess_color(game):
    white = game.headers.get("White", "").lower()
    black = game.headers.get("Black", "").lower()

    if "humanchess" in white:
        return chess.WHITE

    if "humanchess" in black:
        return chess.BLACK

    return None


def mate_in_one_replies(board):
    mates = []

    for move in board.legal_moves:
        test_board = board.copy()
        test_board.push(move)

        if test_board.is_checkmate():
            mates.append(move)

    return mates


def safe_alternatives_against_mate(board):
    safe_moves = []

    for move in board.legal_moves:
        test_board = board.copy()
        test_board.push(move)

        replies = mate_in_one_replies(
            test_board
        )

        if not replies:
            safe_moves.append(move)

    return safe_moves


def analyse_major_piece_move(
    board_before,
    move,
):
    moved_piece = board_before.piece_at(
        move.from_square
    )

    if moved_piece is None:
        return None

    if moved_piece.piece_type not in (
        chess.QUEEN,
        chess.ROOK,
    ):
        return None

    captured_piece = None

    if board_before.is_capture(move):
        captured_piece = board_before.piece_at(
            move.to_square
        )

    board_after = board_before.copy()
    board_after.push(move)

    immediate_captures = []

    for reply in board_after.legal_moves:

        if not board_after.is_capture(reply):
            continue

        if reply.to_square != move.to_square:
            continue

        piece_on_target = board_after.piece_at(
            move.to_square
        )

        if piece_on_target is None:
            continue

        if (
            piece_on_target.piece_type
            != moved_piece.piece_type
        ):
            continue

        immediate_captures.append(reply)

    if not immediate_captures:
        return None

    moved_value = PIECE_VALUES[
        moved_piece.piece_type
    ]

    if captured_piece is None:
        captured_value = 0
    else:
        captured_value = PIECE_VALUES[
            captured_piece.piece_type
        ]

    if captured_piece is None:
        classification = "CLEAN HANG"

    elif captured_value < moved_value:
        classification = "BAD TRADE"

    else:
        classification = "FAIR TRADE"

    return {
        "piece":
            moved_piece.piece_type,

        "classification":
            classification,

        "captured_piece":
            captured_piece,

        "reply":
            immediate_captures[0],
    }


def analyse_pgn(
    pgn_path,
):
    games = 0
    human_moves = 0

    mate_positions = 0
    avoidable_mates = 0
    unavoidable_mates = 0

    major_candidates = 0
    clean_hangs = 0
    bad_trades = 0
    fair_trades = 0

    queen_failures = 0
    rook_failures = 0

    with open(
        pgn_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as pgn_file:

        while True:

            game = chess.pgn.read_game(
                pgn_file
            )

            if game is None:
                break

            human_color = human_chess_color(
                game
            )

            if human_color is None:
                continue

            games += 1

            board = game.board()

            for move in game.mainline_moves():

                moving_color = board.turn

                if moving_color == human_color:

                    human_moves += 1

                    # ----------------------------------
                    # Major-piece failure analysis
                    # ----------------------------------

                    result = (
                        analyse_major_piece_move(
                            board,
                            move,
                        )
                    )

                    if result is not None:

                        major_candidates += 1

                        classification = (
                            result[
                                "classification"
                            ]
                        )

                        piece = (
                            result[
                                "piece"
                            ]
                        )

                        if (
                            classification
                            == "CLEAN HANG"
                        ):

                            clean_hangs += 1

                            if (
                                piece
                                == chess.QUEEN
                            ):
                                queen_failures += 1

                            elif (
                                piece
                                == chess.ROOK
                            ):
                                rook_failures += 1

                        elif (
                            classification
                            == "BAD TRADE"
                        ):

                            bad_trades += 1

                            if (
                                piece
                                == chess.QUEEN
                            ):
                                queen_failures += 1

                            elif (
                                piece
                                == chess.ROOK
                            ):
                                rook_failures += 1

                        elif (
                            classification
                            == "FAIR TRADE"
                        ):

                            fair_trades += 1

                    # ----------------------------------
                    # Mate-in-one analysis
                    # ----------------------------------

                    board_before_move = (
                        board.copy()
                    )

                    board.push(
                        move
                    )

                    mate_replies = (
                        mate_in_one_replies(
                            board
                        )
                    )

                    if mate_replies:

                        mate_positions += 1

                        safe_moves = (
                            safe_alternatives_against_mate(
                                board_before_move
                            )
                        )

                        if safe_moves:
                            avoidable_mates += 1
                        else:
                            unavoidable_mates += 1

                    continue

                board.push(
                    move
                )

    major_failures = (
        clean_hangs
        + bad_trades
    )

    if human_moves > 0:

        avoidable_mate_rate = (
            avoidable_mates
            / human_moves
            * 100
        )

        major_failure_rate = (
            major_failures
            / human_moves
            * 100
        )

    else:

        avoidable_mate_rate = 0.0
        major_failure_rate = 0.0

    print("=" * 60)
    print("HumanChess Tactical Failure Analysis")
    print("=" * 60)

    print(
        f"PGN: {pgn_path}"
    )

    print()

    print(
        f"Games:              "
        f"{games}"
    )

    print(
        f"HumanChess moves:   "
        f"{human_moves}"
    )

    print()

    print("-" * 60)
    print("Mate-in-one")
    print("-" * 60)

    print(
        f"Mate-in-1 positions: "
        f"{mate_positions}"
    )

    print(
        f"Avoidable:           "
        f"{avoidable_mates}"
    )

    print(
        f"Unavoidable:         "
        f"{unavoidable_mates}"
    )

    print(
        f"Avoidable mate rate: "
        f"{avoidable_mate_rate:.2f}%"
    )

    print()

    print("-" * 60)
    print("Major-piece failures")
    print("-" * 60)

    print(
        f"Candidate positions: "
        f"{major_candidates}"
    )

    print(
        f"Clean hangs:         "
        f"{clean_hangs}"
    )

    print(
        f"Bad trades:          "
        f"{bad_trades}"
    )

    print(
        f"Fair trades ignored: "
        f"{fair_trades}"
    )

    print(
        f"Major failures:      "
        f"{major_failures}"
    )

    print(
        f"Queen failures:      "
        f"{queen_failures}"
    )

    print(
        f"Rook failures:       "
        f"{rook_failures}"
    )

    print(
        f"Major failure rate:  "
        f"{major_failure_rate:.2f}%"
    )

    print("=" * 60)


def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python "
            "scripts\\benchmarks\\analyse_tactics.py "
            "<pgn_file>"
        )

        sys.exit(1)

    pgn_path = Path(
        sys.argv[1]
    )

    if not pgn_path.exists():

        print(
            f"PGN file not found: "
            f"{pgn_path}"
        )

        sys.exit(1)

    analyse_pgn(
        pgn_path
    )


if __name__ == "__main__":
    main()