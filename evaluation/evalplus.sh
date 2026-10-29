#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 humaneval|mbpp samples.jsonl" >&2
    exit 1
fi
exec python -m tipcoder_eval evalplus --dataset "$1" --samples "$2"
