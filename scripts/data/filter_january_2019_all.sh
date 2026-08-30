#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

INPUT="data/raw/lichess/lichess_db_standard_rated_2019-01.pgn.zst"
OUTPUT="data/filtered/rapid_800_900_2019-01_all.pgn"

echo "Filtering all January 2019 games..."
echo
echo "Input:"
echo "  $INPUT"
echo
echo "Output:"
echo "  $OUTPUT"
echo

zstdcat "$INPUT" \
  | python3 scripts/data/filter_lichess.py \
  > "$OUTPUT"

echo
echo "Finished."
echo
echo "Filtered PGN:"
echo "  $OUTPUT"