#!/usr/bin/env bash
set -euo pipefail

zstdcat data/raw/lichess/lichess_db_standard_rated_2019-01.pgn.zst \
  | python3 scripts/filter_lichess.py \
  > data/filtered/rapid_800_900_5000.pgn
