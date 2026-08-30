#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  $0 YYYY-MM"
    echo "  $0 YYYY-MM --verify"
    exit 1
fi

MONTH="$1"

if ! [[ "$MONTH" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
    echo "Invalid month: $MONTH"
    echo "Expected format: YYYY-MM"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

FILENAME="lichess_db_standard_rated_${MONTH}.pgn.zst"
LOCAL_FILE="$PROJECT_DIR/data/raw/lichess/$FILENAME"

BASE_URL="https://database.lichess.org/standard"
COUNTS_URL="$BASE_URL/counts.txt"

OFFICIAL_COUNT="$(
    curl -fsSL "$COUNTS_URL" |
    awk -v filename="$FILENAME" '
        $1 == filename { print $2 }
    '
)"

echo
echo "Lichess database information"
echo "----------------------------"
echo "Month:       $MONTH"
echo "Filename:    $FILENAME"

if [ -n "$OFFICIAL_COUNT" ]; then
    printf "Games:       %'d\n" "$OFFICIAL_COUNT"
else
    echo "Games:       not found"
fi

echo "URL:         $BASE_URL/$FILENAME"

if [ -f "$LOCAL_FILE" ]; then
    echo "Local file:  yes"
    echo "Location:    $LOCAL_FILE"
    echo "Size:        $(du -h "$LOCAL_FILE" | cut -f1)"
else
    echo "Local file:  no"
fi

if [ "${2:-}" = "--verify" ]; then
    if [ ! -f "$LOCAL_FILE" ]; then
        echo
        echo "Cannot verify: local file does not exist."
        exit 1
    fi

    echo
    echo "Counting games in local compressed file..."
    echo "This requires reading the entire database."

    LOCAL_COUNT="$(
        zstdcat "$LOCAL_FILE" |
        awk '/^\[Event "/ { count++ } END { print count+0 }'
    )"

    printf "Local count: %'d\n" "$LOCAL_COUNT"
fi