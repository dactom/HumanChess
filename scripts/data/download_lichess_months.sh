#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/home/dacian/lichessdb"
BASE_URL="https://database.lichess.org/standard"

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  $0 YYYY-MM [YYYY-MM ...]"
    echo
    echo "Examples:"
    echo "  $0 2019-02"
    echo "  $0 2019-02 2019-03 2019-04"
    exit 1
fi

mkdir -p "$DEST_DIR"

for MONTH in "$@"; do

    if ! [[ "$MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
        echo "Invalid month: $MONTH"
        echo "Expected format: YYYY-MM"
        exit 1
    fi

    FILENAME="lichess_db_standard_rated_${MONTH}.pgn.zst"
    URL="$BASE_URL/$FILENAME"

    echo
    echo "Downloading:"
    echo "  $FILENAME"
    echo "Destination:"
    echo "  $DEST_DIR"
    echo

    wget \
        -c \
        --show-progress \
        -P "$DEST_DIR" \
        "$URL"

done

echo
echo "Download complete."
echo "Files stored in:"
echo "  $DEST_DIR"
