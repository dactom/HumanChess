from pathlib import Path
import subprocess
import sys

import zstandard as zstd


INPUT = Path(r"data\raw\lichess\lichess_db_standard_rated_2020-01.pgn.zst")
OUTPUT = Path(r"data\filtered\rapid_800_900_2020-01.pgn")
FILTER = Path(r"scripts\data\filter_lichess.py")


def main():
    print(f"Input : {INPUT}")
    print(f"Output: {OUTPUT}")
    print()

    with INPUT.open("rb") as compressed, OUTPUT.open("wb") as output:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(compressed) as reader:
            process = subprocess.Popen(
                [sys.executable, str(FILTER)],
                stdin=subprocess.PIPE,
                stdout=output,
            )

            if process.stdin is None:
                raise RuntimeError("Could not open filter input pipe.")

            while True:
                chunk = reader.read(1024 * 1024)

                if not chunk:
                    break

                process.stdin.write(chunk)

            process.stdin.close()
            return_code = process.wait()

    if return_code != 0:
        raise SystemExit(return_code)

    print()
    print("Filtering completed successfully.")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
