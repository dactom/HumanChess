#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

zstdcat data/raw/lichess/lichess_db_standard_rated_2019-01.pgn.zst \
  | python3 scripts/data/filter_lichess.py --max-kept 5000 \
  > data/filtered/rapid_800_900_5000.pgn