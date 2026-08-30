import argparse
import sys


MIN_ELO = 800
MAX_ELO = 900


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-kept",
        type=int,
        default=None,
        help="Stop after this many matching games. "
             "Default: process the entire input file.",
    )

    return parser.parse_args()


def rating_ok(value):
    try:
        rating = int(value)
        return MIN_ELO <= rating <= MAX_ELO
    except (TypeError, ValueError):
        return False


def header_value(line):
    first_quote = line.find('"')
    last_quote = line.rfind('"')

    if first_quote == -1 or last_quote <= first_quote:
        return ""

    return line[first_quote + 1:last_quote]


def keep_headers(headers):
    event = headers.get("Event", "")
    white_elo = headers.get("WhiteElo")
    black_elo = headers.get("BlackElo")
    white_title = headers.get("WhiteTitle", "")
    black_title = headers.get("BlackTitle", "")

    if "Rated Rapid game" not in event:
        return False

    if not rating_ok(white_elo):
        return False

    if not rating_ok(black_elo):
        return False

    if white_title == "BOT" or black_title == "BOT":
        return False

    return True


def main():
    args = parse_args()

    total = 0
    kept = 0

    game_lines = []
    headers = {}

    for line in sys.stdin:

        if line.startswith("["):
            game_lines.append(line)

            if line.startswith("[Event "):
                headers["Event"] = header_value(line)

            elif line.startswith("[WhiteElo "):
                headers["WhiteElo"] = header_value(line)

            elif line.startswith("[BlackElo "):
                headers["BlackElo"] = header_value(line)

            elif line.startswith("[WhiteTitle "):
                headers["WhiteTitle"] = header_value(line)

            elif line.startswith("[BlackTitle "):
                headers["BlackTitle"] = header_value(line)

        elif line.strip() == "":
            game_lines.append(line)

        else:
            game_lines.append(line)

            # Movetext line marks the end of the game.
            total += 1

            if keep_headers(headers):
                kept += 1

                sys.stdout.writelines(game_lines)
                sys.stdout.write("\n")

            if (
                args.max_kept is not None
                and kept >= args.max_kept
            ):
                print(
                    f"Reached limit of {kept:,} kept games.",
                    file=sys.stderr,
                )
                return

            game_lines = []
            headers = {}

            if total % 100000 == 0:
                print(
                    f"Processed {total:,} games, "
                    f"kept {kept:,}",
                    file=sys.stderr,
                )

    print(
        f"Finished. Processed {total:,} games, "
        f"kept {kept:,}.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()